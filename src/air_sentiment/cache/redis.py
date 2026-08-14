"""Redis-backed cache, shared across workers and restarts.

Every operation here fails open. A cache is an optimisation, so an outage must
cost latency — the recompute a miss implies — and never availability: a Redis
blip that turned into a 500 would make the service *less* reliable than running
with no cache at all. Nothing below raises; failures are logged and reported as
a miss.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final

from air_sentiment.observability.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis

logger = get_logger(__name__)

#: Deliberately short. A cache lookup sits in front of every request, so a slow
#: Redis must behave like a miss rather than eat the caller's latency budget.
_DEFAULT_TIMEOUT_S: Final[float] = 1.0


class RedisCache:
    """Cache backend over ``redis.asyncio``.

    The client is built on first use rather than in ``__init__`` so that
    constructing the cache never performs I/O and never depends on Redis being
    up at import or startup time.
    """

    def __init__(
        self,
        url: str,
        *,
        client: AsyncRedis[str] | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._url = url
        self._client = client
        self._timeout_s = timeout_s
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        client = await self._client_or_none()
        if client is None:
            return None
        try:
            return _as_text(await client.get(key))
        except Exception as exc:
            self._degraded("get", key, exc)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        client = await self._client_or_none()
        if client is None:
            return
        try:
            if ttl_seconds > 0:
                await client.set(key, value, ex=ttl_seconds)
            else:
                await client.set(key, value)
        except Exception as exc:
            self._degraded("set", key, exc)

    async def delete(self, key: str) -> None:
        client = await self._client_or_none()
        if client is None:
            return
        try:
            await client.delete(key)
        except Exception as exc:
            self._degraded("delete", key, exc)

    async def incr(self, key: str, ttl_seconds: int) -> int:
        """Increment a counter, giving it a TTL only on the first write.

        ``SET NX EX`` seeds the window and ``INCR`` moves it, both inside one
        pipeline. Doing it as ``INCR`` followed by a conditional ``EXPIRE``
        would leave a window in which a crash between the two commands strands
        an immortal counter — one that then rejects that caller forever.

        Returns 0 when Redis is unreachable, which reads as "not counted" and
        lets the caller fail open.
        """
        client = await self._client_or_none()
        if client is None:
            return 0
        try:
            pipe = client.pipeline(transaction=True)
            async with pipe:
                if ttl_seconds > 0:
                    pipe.set(key, 0, ex=ttl_seconds, nx=True)
                pipe.incr(key)
                results = await pipe.execute()
            return int(results[-1])
        except Exception as exc:
            self._degraded("incr", key, exc)
            return 0

    async def ping(self) -> bool:
        client = await self._client_or_none()
        if client is None:
            return False
        try:
            return bool(await client.ping())
        except Exception as exc:
            self._degraded("ping", None, exc)
            return False

    async def aclose(self) -> None:
        self._closed = True
        client, self._client = self._client, None
        if client is None:
            return
        try:
            # redis-py renamed close() to aclose() in 5.0.1; resolve at runtime
            # so either generation of the client shuts down cleanly.
            closer = getattr(client, "aclose", client.close)
            await closer()
        except Exception as exc:
            self._degraded("aclose", None, exc)

    # ── internals ─────────────────────────────────────────────────────────

    async def _client_or_none(self) -> AsyncRedis[str] | None:
        """Return the client, building it on first use.

        The lock is held across construction so a burst of concurrent requests
        opens one connection pool rather than one each.
        """
        client = self._client
        if client is not None:
            return client
        if self._closed:
            return None
        async with self._connect_lock:
            client = self._client
            if client is not None or self._closed:
                return client
            client = self._create_client()
            self._client = client
        return client

    def _create_client(self) -> AsyncRedis[str] | None:
        """Construct the client, or None if it cannot be built.

        `redis` is an optional dependency, so a missing package is a
        configuration problem to log — not an ImportError raised at request
        time from the middle of the pipeline.
        """
        try:
            from redis.asyncio import Redis

            return Redis.from_url(
                self._url,
                decode_responses=True,
                socket_timeout=self._timeout_s,
                socket_connect_timeout=self._timeout_s,
            )
        except Exception as exc:
            self._degraded("connect", None, exc)
            return None

    def _degraded(self, op: str, key: str | None, exc: BaseException) -> None:
        # Keys are hashed digests by construction, so they are safe to log.
        logger.warning(
            "cache.redis_degraded",
            op=op,
            key=key,
            error=str(exc),
            error_type=type(exc).__name__,
        )


def _as_text(raw: Any) -> str | None:
    """Normalise a value read back from Redis.

    ``decode_responses=True`` yields str, but a caller may inject a client
    configured otherwise, and bytes here would poison every consumer downstream.
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
