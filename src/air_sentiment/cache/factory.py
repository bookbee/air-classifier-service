"""Cache backend selection.

Which backend is running is a deployment decision, not a code path: everything
downstream sees only :class:`~air_sentiment.cache.base.CacheBackend`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from air_sentiment.cache.base import CacheBackend
from air_sentiment.cache.memory import MemoryCache
from air_sentiment.cache.redis import RedisCache
from air_sentiment.observability.logging import get_logger

if TYPE_CHECKING:
    from air_sentiment.config import Settings

logger = get_logger(__name__)

__all__ = ["NullCache", "build_cache"]


class NullCache:
    """A cache that stores nothing and always misses.

    Useful twice over: load tests measure the cold path rather than the hit
    rate, and turning it on in staging proves the service is still correct with
    an empty cache — which is what a cold start or a flushed Redis looks like.
    """

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def incr(self, key: str, ttl_seconds: int) -> int:
        # Nothing is retained, so nothing is counted; callers fail open.
        return 0

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


async def build_cache(settings: Settings) -> CacheBackend:
    """Construct the configured backend.

    Never raises on an unreachable Redis: the probe below is a startup warning,
    not a gate. A service that refused to boot without its cache would have
    turned an optimisation into a dependency.
    """
    backend = settings.cache.backend
    if backend == "redis":
        cache = RedisCache(settings.cache.redis_url)
        reachable = await cache.ping()
        logger.info(
            "cache.configured",
            backend=backend,
            target=_safe_target(settings.cache.redis_url),
            reachable=reachable,
        )
        return cache

    if backend == "none":
        logger.info("cache.configured", backend=backend)
        return NullCache()

    logger.info("cache.configured", backend="memory", max_entries=settings.cache.max_entries)
    return MemoryCache(max_entries=settings.cache.max_entries)


def _safe_target(url: str) -> str:
    """Host and port only — a Redis URL may carry a password."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "invalid-url"
    host = parts.hostname or "?"
    return f"{host}:{parts.port}" if parts.port else host
