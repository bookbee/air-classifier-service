"""Composed call protection.

Timeout, retry, circuit breaker and bulkhead are only useful together, and the
order they compose in changes their meaning. Call sites should never hand-roll
that composition: one that gets the nesting subtly wrong looks correct in review
and misbehaves exactly when the dependency is already in trouble.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from air_classifier.observability.logging import get_logger
from air_classifier.resilience.breaker import CircuitBreaker, CircuitOpenError
from air_classifier.resilience.bulkhead import Bulkhead, BulkheadFullError

if TYPE_CHECKING:
    from air_classifier.config import ResilienceSettings

logger = get_logger(__name__)

#: Rejections that already mean "do not go near this dependency right now".
#: Retrying them turns a fail-fast into a slow-fail and defeats both guards.
_NEVER_RETRY: Final[tuple[type[Exception], ...]] = (CircuitOpenError, BulkheadFullError)

#: Guards against ``2 ** n`` being evaluated on an n that ``min`` discards anyway.
_MAX_SHIFT: Final[int] = 16


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to re-attempt, how long to wait, and for what.

    ``attempts`` is the *total* number of tries including the first, so
    ``attempts=1`` disables retrying. Note that
    :attr:`~air_classifier.config.ResilienceSettings.retry_attempts` counts
    re-tries; :meth:`from_settings` bridges the two.
    """

    attempts: int
    base_delay_ms: int
    max_delay_ms: int
    retry_on: tuple[type[Exception], ...]

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1 (it counts the first try)")
        if self.base_delay_ms < 1 or self.max_delay_ms < 1:
            raise ValueError("delays must be >= 1ms")
        if self.max_delay_ms < self.base_delay_ms:
            raise ValueError("max_delay_ms must be >= base_delay_ms")

    @classmethod
    def from_settings(
        cls, settings: ResilienceSettings, *, retry_on: tuple[type[Exception], ...]
    ) -> RetryPolicy:
        return cls(
            attempts=settings.retry_attempts + 1,
            base_delay_ms=settings.retry_base_delay_ms,
            max_delay_ms=settings.retry_max_delay_ms,
            retry_on=retry_on,
        )

    def should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, _NEVER_RETRY):
            return False
        return isinstance(exc, self.retry_on)

    def delay_for(self, retry_index: int) -> float:
        """Seconds to wait before retry number ``retry_index`` (0-based).

        Full jitter — a uniform draw from ``[0, backoff]`` rather than the
        backoff itself. Deterministic exponential backoff keeps every caller
        that failed in the same instant synchronised, so they retry in the same
        instant too and hit the recovering dependency as one wave.
        """
        exponential_ms = self.base_delay_ms * 2 ** min(retry_index, _MAX_SHIFT)
        capped_ms = min(exponential_ms, self.max_delay_ms)
        return random.uniform(0.0, capped_ms) / 1000.0  # noqa: S311 — jitter, not a secret


#: Used when a caller passes no policy: one attempt, nothing retried.
_SINGLE_ATTEMPT: Final[RetryPolicy] = RetryPolicy(
    attempts=1, base_delay_ms=1, max_delay_ms=1, retry_on=()
)


async def guarded[T](
    fn: Callable[..., Awaitable[T]],
    *,
    timeout_s: float,
    breaker: CircuitBreaker | None = None,
    bulkhead: Bulkhead | None = None,
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> T:
    """Run ``fn(**kwargs)`` behind every guard the caller supplied.

    The layers nest, outermost first:

    1. **Bulkhead.** A call that cannot get a concurrency slot must never reach
       the dependency, so nothing else runs until the slot is held.
    2. **Timeout**, wrapping the retry loop rather than each attempt. A
       per-attempt timeout silently multiplies by the attempt count, which is a
       latency budget the caller never agreed to; outside the loop it caps
       total wall-clock, which is what a latency budget actually means.
    3. **Retry loop**, inside the breaker.
    4. **Breaker**, consulted per attempt. Because the retries sit inside it,
       one logical call that fails every attempt spends that many failures of
       the budget — which is the correct reading: each attempt was a genuine
       observation of an unhealthy dependency, not one caller's bad luck. It
       also means a breaker that opens mid-retry stops the remaining attempts
       immediately instead of letting them run into a dependency that has
       already been cut off.

    Positional arguments are deliberately not forwarded; bind them with
    :func:`functools.partial` so every guarded call site reads the same way.

    Raises:
        BulkheadFullError: No concurrency slot within the acquisition timeout.
        CircuitOpenError: The breaker is open.
        TimeoutError: ``timeout_s`` elapsed across all attempts. Recorded as a
            breaker failure — a dependency that hangs has failed, even though
            it never raised.
    """
    if bulkhead is None:
        return await _timed(fn, timeout_s=timeout_s, breaker=breaker, retry=retry, kwargs=kwargs)
    async with bulkhead.acquire():
        return await _timed(fn, timeout_s=timeout_s, breaker=breaker, retry=retry, kwargs=kwargs)


async def _timed[T](
    fn: Callable[..., Awaitable[T]],
    *,
    timeout_s: float,
    breaker: CircuitBreaker | None,
    retry: RetryPolicy | None,
    kwargs: dict[str, Any],
) -> T:
    try:
        async with asyncio.timeout(timeout_s):
            return await _attempt(fn, breaker=breaker, retry=retry, kwargs=kwargs)
    except TimeoutError:
        # The inner call was cancelled, so the breaker's guard saw a
        # CancelledError and deliberately declined to judge. Judge it here.
        if breaker is not None:
            await breaker.record_failure()
        logger.warning("guard.timeout", timeout_s=timeout_s, call=_describe(fn))
        raise


async def _attempt[T](
    fn: Callable[..., Awaitable[T]],
    *,
    breaker: CircuitBreaker | None,
    retry: RetryPolicy | None,
    kwargs: dict[str, Any],
) -> T:
    policy = retry if retry is not None else _SINGLE_ATTEMPT
    attempt = 0
    while True:
        try:
            if breaker is None:
                return await fn(**kwargs)
            async with breaker.guard():
                return await fn(**kwargs)
        except Exception as exc:
            attempt += 1
            if attempt >= policy.attempts or not policy.should_retry(exc):
                raise
            delay = policy.delay_for(attempt - 1)
            logger.debug(
                "guard.retry",
                call=_describe(fn),
                attempt=attempt,
                of=policy.attempts,
                delay_s=round(delay, 4),
                error=type(exc).__name__,
            )
            await asyncio.sleep(delay)


def _describe(fn: Callable[..., Awaitable[Any]]) -> str:
    """Readable name for a callable that is usually a ``functools.partial``."""
    target = getattr(fn, "func", fn)
    name = getattr(target, "__qualname__", None)
    return name if isinstance(name, str) else type(target).__name__
