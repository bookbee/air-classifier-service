"""Composition of timeout, retry, breaker and bulkhead."""

from __future__ import annotations

import asyncio
import functools

import pytest

from air_classifier.config import ResilienceSettings
from air_classifier.resilience.breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
)
from air_classifier.resilience.bulkhead import Bulkhead, BulkheadFullError
from air_classifier.resilience.guard import RetryPolicy, guarded


class RetryableError(Exception):
    """A transient failure the policy is willing to re-attempt."""


class FatalError(Exception):
    """A failure no amount of retrying will fix."""


def _policy(
    *,
    attempts: int = 3,
    base_delay_ms: int = 1,
    max_delay_ms: int = 2,
    retry_on: tuple[type[Exception], ...] = (RetryableError,),
) -> RetryPolicy:
    return RetryPolicy(
        attempts=attempts,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
        retry_on=retry_on,
    )


async def _echo(*, value: str) -> str:
    return value


# ── RetryPolicy ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("attempts", "base_delay_ms", "max_delay_ms"),
    [(0, 1, 2), (1, 0, 2), (1, 1, 0), (1, 100, 50)],
)
def test_policy_rejects_nonsense_configuration(
    attempts: int, base_delay_ms: int, max_delay_ms: int
) -> None:
    with pytest.raises(ValueError):
        _policy(attempts=attempts, base_delay_ms=base_delay_ms, max_delay_ms=max_delay_ms)


def test_from_settings_converts_retries_into_attempts() -> None:
    settings = ResilienceSettings()
    policy = RetryPolicy.from_settings(settings, retry_on=(RetryableError,))

    assert policy.attempts == settings.retry_attempts + 1
    assert policy.base_delay_ms == settings.retry_base_delay_ms
    assert policy.max_delay_ms == settings.retry_max_delay_ms


def test_backoff_is_bounded_by_the_ceiling() -> None:
    policy = _policy(base_delay_ms=100, max_delay_ms=400)

    # Uncapped the sequence would be 100, 200, 400, 800, 1600...
    for retry_index, ceiling_ms in [(0, 100), (1, 200), (2, 400), (3, 400), (60, 400)]:
        samples = [policy.delay_for(retry_index) for _ in range(200)]
        assert all(0.0 <= s <= ceiling_ms / 1000 for s in samples)


def test_backoff_is_jittered_not_deterministic() -> None:
    policy = _policy(base_delay_ms=100, max_delay_ms=400)
    samples = [policy.delay_for(2) for _ in range(200)]

    # Full jitter: a uniform draw, not the backoff itself. Identical values
    # across callers are exactly the synchronisation this is meant to break.
    assert len(set(samples)) > 100
    assert min(samples) < 0.1 < max(samples)


def test_never_retries_fail_fast_rejections() -> None:
    policy = _policy(retry_on=(Exception,))

    assert policy.should_retry(RetryableError())
    assert not policy.should_retry(CircuitOpenError("x"))
    assert not policy.should_retry(BulkheadFullError("x", limit=1, waited_s=0.0))


# ── Retry loop ────────────────────────────────────────────────────────────────


async def test_succeeds_without_guards() -> None:
    assert await guarded(_echo, timeout_s=1.0, value="hi") == "hi"


async def test_retries_until_success() -> None:
    calls = 0

    async def _flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableError
        return "ok"

    assert await guarded(_flaky, timeout_s=1.0, retry=_policy(attempts=3)) == "ok"
    assert calls == 3


async def test_exhausts_attempts_then_raises_the_real_error() -> None:
    calls = 0

    async def _always_fails() -> str:
        nonlocal calls
        calls += 1
        raise RetryableError

    with pytest.raises(RetryableError):
        await guarded(_always_fails, timeout_s=1.0, retry=_policy(attempts=3))
    assert calls == 3


async def test_non_retryable_error_is_raised_without_burning_attempts() -> None:
    calls = 0

    async def _fatal() -> str:
        nonlocal calls
        calls += 1
        raise FatalError

    with pytest.raises(FatalError):
        await guarded(_fatal, timeout_s=1.0, retry=_policy(attempts=5))
    assert calls == 1, "a FatalError is not in retry_on; it must not be re-attempted"


async def test_no_policy_means_one_attempt() -> None:
    calls = 0

    async def _fails() -> str:
        nonlocal calls
        calls += 1
        raise RetryableError

    with pytest.raises(RetryableError):
        await guarded(_fails, timeout_s=1.0)
    assert calls == 1


async def test_positional_arguments_go_through_partial() -> None:
    async def _add(a: int, b: int, *, c: int) -> int:
        return a + b + c

    bound = functools.partial(_add, 1, 2)
    assert await guarded(bound, timeout_s=1.0, c=3) == 6


# ── Timeout ───────────────────────────────────────────────────────────────────


async def test_timeout_caps_total_wall_clock_across_attempts() -> None:
    calls = 0

    async def _slow() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1.0)
        return "never"

    with pytest.raises(TimeoutError):
        await guarded(
            _slow,
            timeout_s=0.03,
            retry=_policy(attempts=5, base_delay_ms=1, max_delay_ms=1),
        )

    # One attempt already blew the whole budget, so the remaining four never
    # ran: the ceiling is total, not per-attempt.
    assert calls == 1


async def test_timeout_counts_as_a_breaker_failure() -> None:
    breaker = CircuitBreaker("test", threshold=1, cooldown_s=30.0)

    async def _hang() -> str:
        await asyncio.sleep(1.0)
        return "never"

    with pytest.raises(TimeoutError):
        await guarded(_hang, timeout_s=0.01, breaker=breaker)

    assert breaker.state is BreakerState.OPEN, "a dependency that hangs has failed"


# ── Breaker composition ───────────────────────────────────────────────────────


async def test_each_attempt_votes_with_the_breaker() -> None:
    breaker = CircuitBreaker("test", threshold=3, cooldown_s=30.0)

    async def _fails() -> str:
        raise RetryableError

    # Retries sit inside the breaker, so one logical call spends three failures.
    with pytest.raises((RetryableError, CircuitOpenError)):
        await guarded(_fails, timeout_s=1.0, breaker=breaker, retry=_policy(attempts=3))

    assert breaker.state is BreakerState.OPEN


async def test_open_breaker_short_circuits_and_is_not_retried() -> None:
    breaker = CircuitBreaker("test", threshold=1, cooldown_s=30.0)
    await breaker.record_failure()
    calls = 0

    async def _never_called() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        await guarded(
            _never_called,
            timeout_s=1.0,
            breaker=breaker,
            retry=_policy(attempts=5, retry_on=(Exception,)),
        )

    assert calls == 0
    assert breaker.consecutive_failures == 1, "a refusal is not a new failure"


async def test_recovered_breaker_lets_traffic_through_again() -> None:
    breaker = CircuitBreaker("test", threshold=1, cooldown_s=0.02)
    await breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        await guarded(_echo, timeout_s=1.0, breaker=breaker, value="hi")

    await asyncio.sleep(0.03)
    assert await guarded(_echo, timeout_s=1.0, breaker=breaker, value="hi") == "hi"
    assert breaker.state is BreakerState.CLOSED


# ── Bulkhead composition ──────────────────────────────────────────────────────


async def test_bulkhead_rejection_never_reaches_the_dependency() -> None:
    bulkhead = Bulkhead("test", limit=1, acquire_timeout_s=0.02)
    breaker = CircuitBreaker("test", threshold=1, cooldown_s=30.0)
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    async def _hold() -> str:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return "held"

    holder = asyncio.create_task(guarded(_hold, timeout_s=1.0, bulkhead=bulkhead))
    await entered.wait()

    with pytest.raises(BulkheadFullError):
        await guarded(_hold, timeout_s=1.0, bulkhead=bulkhead, breaker=breaker)

    assert calls == 1
    # A full compartment says nothing about the dependency's health.
    assert breaker.state is BreakerState.CLOSED

    release.set()
    assert await holder == "held"
    assert bulkhead.in_flight == 0


async def test_all_layers_together() -> None:
    bulkhead = Bulkhead("test", limit=2, acquire_timeout_s=1.0)
    breaker = CircuitBreaker("test", threshold=10, cooldown_s=30.0)
    seen: dict[str, int] = {}

    async def _flaky(*, value: str) -> str:
        seen[value] = seen.get(value, 0) + 1
        if seen[value] < 2:
            raise RetryableError
        return value

    results = await asyncio.gather(
        *(
            guarded(
                _flaky,
                timeout_s=1.0,
                breaker=breaker,
                bulkhead=bulkhead,
                retry=_policy(attempts=3),
                value=f"v{i}",
            )
            for i in range(4)
        )
    )

    assert sorted(results) == ["v0", "v1", "v2", "v3"]
    assert bulkhead.in_flight == 0
    assert breaker.state is BreakerState.CLOSED
