"""Per-key request quotas.

Counters live on the :class:`~air_classifier.cache.base.CacheBackend` rather than
in process memory so that N replicas share one budget. An in-process counter
would silently multiply every published limit by the replica count, which tends
to go unnoticed until a downstream provider notices it first.

With the memory backend the same code is an in-process limiter, which is exactly
what a single-replica or test deployment wants.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Final

from pydantic import BaseModel, Field

from air_classifier.cache.base import CacheBackend
from air_classifier.config import Settings
from air_classifier.observability.logging import get_logger
from air_classifier.security.api_keys import ApiKeyRecord

logger = get_logger(__name__)

#: Limits are published per minute, so the window is a minute.
WINDOW_SECONDS: Final[float] = 60.0


class RateLimitResult(BaseModel):
    """Everything the HTTP layer needs to answer, and to explain the answer."""

    allowed: bool
    limit: int = Field(ge=0, description="Requests admitted per window.")
    remaining: int = Field(ge=0)
    retry_after_s: float = Field(ge=0.0, description="Zero when allowed.")
    reset_at_s: float = Field(ge=0.0, description="Unix time at which the window rolls over.")

    def headers(self) -> dict[str, str]:
        """``RateLimit-*`` response headers, plus ``Retry-After`` when denied."""
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(self.remaining),
            "RateLimit-Reset": str(max(0, math.ceil(self.reset_at_s - time.time()))),
        }
        if not self.allowed:
            # Retry-After is integer seconds, and a floor of 1 stops a client
            # from retrying immediately on a sub-second remainder.
            headers["Retry-After"] = str(max(1, math.ceil(self.retry_after_s)))
        return headers


class TokenBucket:
    """A per-key request budget over a fixed window.

    Named for the shape it presents — a sustained rate plus burst headroom — not
    for its mechanics. ``CacheBackend`` exposes ``incr`` and nothing else atomic,
    so underneath this is a fixed-window counter, and fixed windows have a known
    weakness: a caller that spends its whole budget in the last moment of one
    window and again in the first moment of the next passes twice the limit
    across that straddle. That is accepted here. These limits exist to bound
    *sustained* load on metered providers, and a momentary doubling is what the
    bulkhead and the circuit breaker are for.

    A true sliding window costs a wider cache contract: either a sorted set per
    key with a range trim (Redis-only, extra round trips or a Lua script), or two
    counters read together and weighted. The second is the cheap upgrade if the
    straddle ever matters — read the previous window's count alongside the
    current one and prorate it by how far into the window the request lands —
    and it needs only a `get` beside the `incr` this already does.
    """

    def __init__(
        self,
        cache: CacheBackend,
        *,
        prefix: str = "",
        window_s: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache = cache
        self._prefix = prefix
        self._window_s = window_s
        self._clock = clock

    async def consume(self, key_id: str, *, limit: int) -> RateLimitResult:
        """Charge one request against ``key_id``'s budget for the current window."""
        now = self._clock()
        window_index = int(now // self._window_s)
        reset_at = (window_index + 1) * self._window_s
        effective = max(1, limit)
        key = f"{self._prefix}rl:{window_index}:{key_id}"
        # One second beyond the window so a counter cannot expire before its own
        # window closes on a replica whose clock runs slightly ahead.
        ttl = int(self._window_s) + 1

        try:
            used = await self._cache.incr(key, ttl)
        except Exception:
            # The cache contract says backends swallow their own failures, but a
            # limiter that turns a Redis blip into a service-wide 429 storm is
            # worse than one that briefly over-admits. Fail open, loudly.
            logger.warning("rate_limit_backend_failed", key_id=key_id, exc_info=True)
            return RateLimitResult(
                allowed=True,
                limit=effective,
                remaining=effective,
                retry_after_s=0.0,
                reset_at_s=reset_at,
            )

        allowed = used <= effective
        return RateLimitResult(
            allowed=allowed,
            limit=effective,
            remaining=max(0, effective - used),
            retry_after_s=0.0 if allowed else max(0.0, reset_at - now),
            reset_at_s=reset_at,
        )


class RateLimiter:
    """Resolves a key's limits from config or its record, then applies them."""

    def __init__(self, cache: CacheBackend, settings: Settings) -> None:
        self._settings = settings
        self._bucket = TokenBucket(cache, prefix=settings.cache.key_prefix)

    async def check(
        self,
        key_id: str,
        *,
        rpm: int | None = None,
        burst: int | None = None,
    ) -> RateLimitResult:
        """Charge one request against ``key_id``.

        ``rpm`` is the sustained allowance and ``burst`` the headroom above it,
        so a window admits ``rpm + burst``. Either falling back to ``None`` takes
        the service default from ``security``.
        """
        security = self._settings.security
        effective_rpm = security.default_rate_limit_rpm if rpm is None else rpm
        effective_burst = security.default_burst if burst is None else burst
        limit = max(1, effective_rpm + effective_burst)

        result = await self._bucket.consume(key_id, limit=limit)
        if not result.allowed:
            logger.info(
                "rate_limit_exceeded",
                key_id=key_id,
                limit=limit,
                retry_after_s=round(result.retry_after_s, 3),
            )
        return result

    async def check_key(self, record: ApiKeyRecord) -> RateLimitResult:
        """Apply a record's own overrides. The integration point for auth deps."""
        return await self.check(record.id, rpm=record.rate_limit_rpm, burst=record.burst)
