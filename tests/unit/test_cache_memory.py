"""MemoryCache: TTL, LRU, and counter behaviour."""

from __future__ import annotations

import asyncio

import pytest

from air_classifier.cache.base import CacheBackend
from air_classifier.cache.memory import MemoryCache


class FakeClock:
    """Monotonic clock the test drives, so TTL tests never sleep."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build(max_entries: int = 10, clock: FakeClock | None = None) -> MemoryCache:
    return MemoryCache(max_entries=max_entries, clock=clock or FakeClock())


async def test_satisfies_backend_protocol() -> None:
    assert isinstance(build(), CacheBackend)


async def test_rejects_zero_capacity() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        MemoryCache(max_entries=0)


async def test_set_then_get_roundtrips() -> None:
    cache = build()
    await cache.set("k", "v", 60)
    assert await cache.get("k") == "v"


async def test_missing_key_is_a_miss() -> None:
    assert await build().get("absent") is None


async def test_set_overwrites_existing_value() -> None:
    cache = build()
    await cache.set("k", "first", 60)
    await cache.set("k", "second", 60)
    assert await cache.get("k") == "second"
    assert cache.size == 1


async def test_delete_removes_entry() -> None:
    cache = build()
    await cache.set("k", "v", 60)
    await cache.delete("k")
    assert await cache.get("k") is None


async def test_delete_of_absent_key_is_silent() -> None:
    await build().delete("absent")


# ── TTL ───────────────────────────────────────────────────────────────────


async def test_entry_expires_after_ttl() -> None:
    clock = FakeClock()
    cache = build(clock=clock)
    await cache.set("k", "v", 30)

    clock.advance(29)
    assert await cache.get("k") == "v"

    clock.advance(1)
    assert await cache.get("k") is None


async def test_expired_entry_is_dropped_on_read() -> None:
    clock = FakeClock()
    cache = build(clock=clock)
    await cache.set("k", "v", 5)
    clock.advance(10)

    await cache.get("k")

    assert cache.size == 0


async def test_write_sweeps_expired_cold_entries() -> None:
    """An idle key nobody reads again must not hold its slot forever."""
    clock = FakeClock()
    cache = build(max_entries=100, clock=clock)
    for i in range(4):
        await cache.set(f"stale-{i}", "v", 5)

    clock.advance(10)
    await cache.set("fresh", "v", 60)

    assert cache.size == 1
    assert await cache.get("fresh") == "v"


async def test_non_positive_ttl_never_expires() -> None:
    clock = FakeClock()
    cache = build(clock=clock)
    await cache.set("k", "v", 0)

    clock.advance(1_000_000)

    assert await cache.get("k") == "v"


# ── LRU ───────────────────────────────────────────────────────────────────


async def test_evicts_least_recently_used_at_capacity() -> None:
    cache = build(max_entries=3)
    for key in ("a", "b", "c"):
        await cache.set(key, key, 60)

    await cache.set("d", "d", 60)

    assert cache.size == 3
    assert await cache.get("a") is None
    assert await cache.get("d") == "d"


async def test_read_refreshes_recency() -> None:
    cache = build(max_entries=3)
    for key in ("a", "b", "c"):
        await cache.set(key, key, 60)

    await cache.get("a")
    await cache.set("d", "d", 60)

    assert await cache.get("a") == "a"
    assert await cache.get("b") is None


async def test_overwrite_refreshes_recency() -> None:
    cache = build(max_entries=3)
    for key in ("a", "b", "c"):
        await cache.set(key, key, 60)

    await cache.set("a", "a2", 60)
    await cache.set("d", "d", 60)

    assert await cache.get("a") == "a2"
    assert await cache.get("b") is None


async def test_capacity_holds_under_churn() -> None:
    cache = build(max_entries=5)
    for i in range(200):
        await cache.set(f"k{i}", "v", 60)
    assert cache.size == 5


# ── Counters ──────────────────────────────────────────────────────────────


async def test_incr_counts_from_one() -> None:
    cache = build()
    assert await cache.incr("c", 60) == 1
    assert await cache.incr("c", 60) == 2
    assert await cache.get("c") == "2"


async def test_incr_window_does_not_slide() -> None:
    """The TTL is the rate-limit window: later hits must not extend it."""
    clock = FakeClock()
    cache = build(clock=clock)

    await cache.incr("c", 60)
    clock.advance(59)
    assert await cache.incr("c", 60) == 2

    clock.advance(1)
    assert await cache.incr("c", 60) == 1


async def test_incr_over_non_numeric_value_restarts() -> None:
    cache = build()
    await cache.set("c", "not-a-number", 60)
    assert await cache.incr("c", 60) == 1


async def test_incr_is_atomic_under_concurrency() -> None:
    cache = build(max_entries=64)
    tasks = [cache.incr("c", 60) for _ in range(200)]

    seen = await asyncio.gather(*tasks)

    assert sorted(seen) == list(range(1, 201))
    assert await cache.get("c") == "200"


async def test_concurrent_mixed_writes_stay_within_capacity() -> None:
    cache = build(max_entries=4)
    await asyncio.gather(*(cache.set(f"k{i}", "v", 60) for i in range(50)))
    assert cache.size == 4


# ── Lifecycle ─────────────────────────────────────────────────────────────


async def test_ping_is_always_true() -> None:
    assert await build().ping() is True


async def test_aclose_drops_contents() -> None:
    cache = build()
    await cache.set("k", "v", 60)

    await cache.aclose()

    assert cache.size == 0
    assert await cache.get("k") is None
