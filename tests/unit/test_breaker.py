"""Circuit breaker state machine."""

from __future__ import annotations

import asyncio
import time

import pytest

from air_sentiment.constants import Provider
from air_sentiment.observability import metrics
from air_sentiment.resilience.breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
)


class BoomError(Exception):
    """Stand-in for a dependency failure."""


async def _fail() -> None:
    raise BoomError


async def _ok() -> str:
    return "ok"


def _breaker(
    *, threshold: int = 3, cooldown_s: float = 0.02, half_open_max: int = 1
) -> CircuitBreaker:
    return CircuitBreaker(
        "test", threshold=threshold, cooldown_s=cooldown_s, half_open_max=half_open_max
    )


# ── Construction ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("threshold", "cooldown_s", "half_open_max"),
    [(0, 1.0, 1), (1, 0.0, 1), (1, -1.0, 1), (1, 1.0, 0)],
)
def test_rejects_nonsense_configuration(
    threshold: int, cooldown_s: float, half_open_max: int
) -> None:
    with pytest.raises(ValueError):
        _breaker(threshold=threshold, cooldown_s=cooldown_s, half_open_max=half_open_max)


def test_starts_closed() -> None:
    breaker = _breaker()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.is_closed
    assert breaker.consecutive_failures == 0


# ── Opening ───────────────────────────────────────────────────────────────────


async def test_opens_exactly_at_threshold() -> None:
    breaker = _breaker(threshold=3)

    for expected_failures in (1, 2):
        with pytest.raises(BoomError):
            await breaker.call(_fail)
        assert breaker.state is BreakerState.CLOSED
        assert breaker.consecutive_failures == expected_failures

    with pytest.raises(BoomError):
        await breaker.call(_fail)
    assert breaker.state is BreakerState.OPEN
    assert not breaker.is_closed


async def test_success_resets_the_failure_run() -> None:
    breaker = _breaker(threshold=3)

    for _ in range(2):
        with pytest.raises(BoomError):
            await breaker.call(_fail)
    assert await breaker.call(_ok) == "ok"
    assert breaker.consecutive_failures == 0

    # Two more failures must not be enough now that the run was broken.
    for _ in range(2):
        with pytest.raises(BoomError):
            await breaker.call(_fail)
    assert breaker.state is BreakerState.CLOSED


async def test_refuses_calls_while_open() -> None:
    breaker = _breaker(threshold=1, cooldown_s=30.0)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    calls = 0

    async def _counted() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError) as excinfo:
        await breaker.call(_counted)

    assert calls == 0, "an open breaker must not reach the dependency"
    assert excinfo.value.breaker_name == "test"
    assert 0 < excinfo.value.retry_after_s <= 30.0


# ── Recovery ──────────────────────────────────────────────────────────────────


async def test_half_open_probe_success_closes() -> None:
    breaker = _breaker(threshold=1, cooldown_s=0.02)
    with pytest.raises(BoomError):
        await breaker.call(_fail)
    assert breaker.state is BreakerState.OPEN

    await asyncio.sleep(0.03)
    # Cooldown elapsed: the breaker advertises itself as probeable so a
    # selector will actually send the probe.
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.is_closed

    assert await breaker.call(_ok) == "ok"
    assert breaker.state is BreakerState.CLOSED


async def test_half_open_probe_failure_reopens() -> None:
    breaker = _breaker(threshold=1, cooldown_s=0.02)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    await asyncio.sleep(0.03)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    assert breaker.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.call(_ok)


async def test_half_open_admits_only_the_configured_number_of_probes() -> None:
    breaker = _breaker(threshold=1, cooldown_s=0.02, half_open_max=1)
    with pytest.raises(BoomError):
        await breaker.call(_fail)
    await asyncio.sleep(0.03)

    released = asyncio.Event()
    admitted = asyncio.Event()

    async def _slow() -> str:
        admitted.set()
        await released.wait()
        return "ok"

    probe = asyncio.create_task(breaker.call(_slow))
    await admitted.wait()

    # The second caller must fail fast rather than queue behind the probe.
    with pytest.raises(CircuitOpenError):
        await breaker.call(_ok)

    released.set()
    assert await probe == "ok"
    assert breaker.state is BreakerState.CLOSED


async def test_half_open_slot_is_returned_after_a_failed_probe() -> None:
    breaker = _breaker(threshold=1, cooldown_s=0.01, half_open_max=1)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    # Three failed probes in a row: each one must get its slot back, otherwise
    # the breaker wedges itself shut forever.
    for _ in range(3):
        await asyncio.sleep(0.02)
        with pytest.raises(BoomError):
            await breaker.call(_fail)

    await asyncio.sleep(0.02)
    assert await breaker.call(_ok) == "ok"
    assert breaker.state is BreakerState.CLOSED


async def test_reset_forces_closed() -> None:
    breaker = _breaker(threshold=1, cooldown_s=30.0)
    with pytest.raises(BoomError):
        await breaker.call(_fail)
    assert breaker.state is BreakerState.OPEN

    await breaker.reset()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0
    assert await breaker.call(_ok) == "ok"


# ── Cancellation and manual bookkeeping ───────────────────────────────────────


async def test_cancellation_is_not_a_dependency_failure() -> None:
    breaker = _breaker(threshold=1, cooldown_s=30.0)
    started = asyncio.Event()

    async def _hang() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(breaker.call(_hang))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker.state is BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


async def test_cancelled_probe_returns_its_half_open_slot() -> None:
    breaker = _breaker(threshold=1, cooldown_s=0.01, half_open_max=1)
    with pytest.raises(BoomError):
        await breaker.call(_fail)
    await asyncio.sleep(0.02)

    started = asyncio.Event()

    async def _hang() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(breaker.call(_hang))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await breaker.call(_ok) == "ok"


async def test_record_helpers_drive_the_same_machine() -> None:
    breaker = _breaker(threshold=2, cooldown_s=30.0)

    await breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    await breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    await breaker.reset()
    await breaker.record_failure()
    await breaker.record_success()
    assert breaker.consecutive_failures == 0


async def test_guard_context_manager_yields_and_records() -> None:
    breaker = _breaker(threshold=1, cooldown_s=30.0)

    async with breaker.guard():
        pass
    assert breaker.state is BreakerState.CLOSED

    with pytest.raises(BoomError):
        async with breaker.guard():
            raise BoomError
    assert breaker.state is BreakerState.OPEN


# ── Clocks ────────────────────────────────────────────────────────────────────


async def test_cooldown_ignores_wall_clock_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    breaker = _breaker(threshold=1, cooldown_s=30.0)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    # An NTP correction, a DST jump, an operator running `date`. None of it may
    # let a breaker out of its cooldown early.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3600.0)

    assert breaker.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.call(_ok)


async def test_concurrent_callers_open_the_breaker_once() -> None:
    breaker = _breaker(threshold=3, cooldown_s=30.0)

    results = await asyncio.gather(
        *(breaker.call(_fail) for _ in range(10)), return_exceptions=True
    )

    assert breaker.state is BreakerState.OPEN
    # Everything that got in failed for the real reason or was refused; nothing
    # escaped as a bookkeeping error.
    assert all(isinstance(r, BoomError | CircuitOpenError) for r in results)
    assert sum(isinstance(r, CircuitOpenError) for r in results) > 0


# ── Observability ─────────────────────────────────────────────────────────────


async def test_transitions_drive_the_provider_gauge() -> None:
    def gauge() -> float | None:
        return metrics.get_registry().get_sample_value(
            "air_sentiment_breaker_state", {"provider": Provider.OLLAMA.value}
        )

    breaker = CircuitBreaker(Provider.OLLAMA.value, threshold=1, cooldown_s=0.02)

    with pytest.raises(BoomError):
        await breaker.call(_fail)
    assert gauge() == metrics.BreakerState.OPEN

    await asyncio.sleep(0.03)
    assert await breaker.call(_ok) == "ok"
    assert gauge() == metrics.BreakerState.CLOSED


async def test_a_breaker_that_is_not_a_provider_publishes_nothing() -> None:
    # The gauge's only label is `provider`; inventing a value for it would put
    # junk series in the dashboard rather than report anything useful.
    breaker = _breaker(threshold=1, cooldown_s=30.0)
    with pytest.raises(BoomError):
        await breaker.call(_fail)

    assert (
        metrics.get_registry().get_sample_value("air_sentiment_breaker_state", {"provider": "test"})
        is None
    )
