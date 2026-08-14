"""Bounded-concurrency compartment."""

from __future__ import annotations

import asyncio

import pytest

from air_sentiment.resilience.bulkhead import Bulkhead, BulkheadFullError


class BoomError(Exception):
    """Stand-in for a dependency failure."""


@pytest.mark.parametrize(("limit", "timeout_s"), [(0, 1.0), (1, 0.0), (1, -1.0)])
def test_rejects_nonsense_configuration(limit: int, timeout_s: float) -> None:
    with pytest.raises(ValueError):
        Bulkhead("test", limit=limit, acquire_timeout_s=timeout_s)


async def test_reports_occupancy() -> None:
    bulkhead = Bulkhead("test", limit=2, acquire_timeout_s=0.05)
    assert bulkhead.in_flight == 0
    assert bulkhead.available == 2

    async with bulkhead.acquire():
        assert bulkhead.in_flight == 1
        assert bulkhead.available == 1
        async with bulkhead.acquire():
            assert bulkhead.in_flight == 2
            assert bulkhead.available == 0

    assert bulkhead.in_flight == 0
    assert bulkhead.available == 2


async def test_rejects_when_full_rather_than_queueing_forever() -> None:
    bulkhead = Bulkhead("test", limit=1, acquire_timeout_s=0.02)
    release = asyncio.Event()

    async def _hold() -> None:
        async with bulkhead.acquire():
            await release.wait()

    holder = asyncio.create_task(_hold())
    await asyncio.sleep(0)  # let the holder take the only slot

    with pytest.raises(BulkheadFullError) as excinfo:
        async with bulkhead.acquire():
            pytest.fail("the compartment was full; this block must not run")

    assert excinfo.value.bulkhead_name == "test"
    assert excinfo.value.limit == 1
    assert 0.015 <= excinfo.value.waited_s < 1.0

    release.set()
    await holder


async def test_waiter_proceeds_once_a_slot_frees() -> None:
    bulkhead = Bulkhead("test", limit=1, acquire_timeout_s=0.5)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def _hold() -> None:
        async with bulkhead.acquire():
            entered.set()
            await release.wait()

    async def _wait() -> str:
        async with bulkhead.acquire():
            return "in"

    holder = asyncio.create_task(_hold())
    await entered.wait()
    waiter = asyncio.create_task(_wait())
    await asyncio.sleep(0)

    assert not waiter.done()
    release.set()
    await holder
    assert await waiter == "in"


async def test_slot_is_released_when_the_body_raises() -> None:
    bulkhead = Bulkhead("test", limit=1, acquire_timeout_s=0.02)

    with pytest.raises(BoomError):
        async with bulkhead.acquire():
            raise BoomError

    assert bulkhead.in_flight == 0
    async with bulkhead.acquire():
        assert bulkhead.in_flight == 1


async def test_slot_is_released_when_the_body_is_cancelled() -> None:
    bulkhead = Bulkhead("test", limit=1, acquire_timeout_s=0.05)
    entered = asyncio.Event()

    async def _hang() -> None:
        async with bulkhead.acquire():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(_hang())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert bulkhead.available == 1
    async with bulkhead.acquire():
        pass


async def test_limit_is_never_exceeded_under_load() -> None:
    bulkhead = Bulkhead("test", limit=3, acquire_timeout_s=1.0)
    peak = 0

    async def _work() -> None:
        nonlocal peak
        async with bulkhead.acquire():
            peak = max(peak, bulkhead.in_flight)
            await asyncio.sleep(0.001)

    await asyncio.gather(*(_work() for _ in range(25)))

    assert peak == 3
    assert bulkhead.in_flight == 0
