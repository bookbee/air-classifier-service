"""In-process TTL + LRU cache — the default backend.

Chosen as the default so a fresh checkout runs with nothing to operate. It is
per-process: two uvicorn workers keep two independent caches, so the hit rate is
lower than a shared Redis would give. That is an acceptable trade because a miss
only ever costs a recompute — no answer depends on the cache being warm.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from typing import Final

from air_classifier.observability.logging import get_logger

logger = get_logger(__name__)

#: How many cold entries a write inspects for expiry. Bounded so a write stays
#: O(1); whatever the scan misses is caught by lazy expiry on read, and the LRU
#: cap bounds the footprint regardless.
_SWEEP_BUDGET: Final[int] = 8


@dataclass(slots=True)
class _Entry:
    value: str
    #: Monotonic deadline, or None for an entry only the LRU can remove.
    expires_at: float | None

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class MemoryCache:
    """TTL + LRU map behind a single lock.

    The lock is what makes :meth:`incr` usable for rate limiting: read, add, and
    write-back happen without another task observing the counter mid-update.
    """

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args:
        max_entries: Hard ceiling; the least-recently-used entry goes first.
        clock: Monotonic time source, injectable for tests. Wall-clock time
            can jump backwards across an NTP correction and strand entries
            past their deadline, so it is not used here.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._clock = clock
        # Ordered oldest-first, so the head is always the eviction victim.
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Live entry count, including any not yet swept."""
        return len(self._entries)

    async def get(self, key: str) -> str | None:
        now = self._clock()
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired(now):
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        now = self._clock()
        async with self._lock:
            self._store(key, value, self._deadline(now, ttl_seconds), now)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)

    async def incr(self, key: str, ttl_seconds: int) -> int:
        now = self._clock()
        async with self._lock:
            current = self._entries.get(key)
            if current is not None and current.is_expired(now):
                current = None
            count = self._current_count(key, current) + 1
            # The TTL *is* the window, so it is fixed at the first write:
            # refreshing it on every hit would let steady traffic keep a
            # counter — and the limit it enforces — alive indefinitely.
            deadline = (
                current.expires_at if current is not None else self._deadline(now, ttl_seconds)
            )
            self._store(key, str(count), deadline, now)
            return count

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        async with self._lock:
            self._entries.clear()

    # ── internals ─────────────────────────────────────────────────────────

    def _store(self, key: str, value: str, deadline: float | None, now: float) -> None:
        self._entries[key] = _Entry(value=value, expires_at=deadline)
        self._entries.move_to_end(key)
        self._sweep_expired(now)
        self._evict_overflow()

    def _sweep_expired(self, now: float) -> None:
        """Drop expired entries from the cold end.

        Without this an idle key that nobody reads again would hold its slot
        until the LRU pressure reached it, which may be never.
        """
        for key in list(islice(self._entries, _SWEEP_BUDGET)):
            if self._entries[key].is_expired(now):
                del self._entries[key]

    def _evict_overflow(self) -> None:
        while len(self._entries) > self._max_entries:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug("cache.evicted", key=evicted, reason="capacity")

    def _current_count(self, key: str, entry: _Entry | None) -> int:
        if entry is None:
            return 0
        try:
            return int(entry.value)
        except ValueError:
            # A counter key reused for opaque text. Restarting the count keeps
            # the caller moving; raising here would take the request with it.
            logger.warning("cache.counter_value_invalid", key=key)
            return 0

    @staticmethod
    def _deadline(now: float, ttl_seconds: int) -> float | None:
        # A non-positive TTL means "no expiry"; the LRU cap still bounds it.
        return now + ttl_seconds if ttl_seconds > 0 else None
