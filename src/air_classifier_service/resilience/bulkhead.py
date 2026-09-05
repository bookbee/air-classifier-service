"""Bounded-concurrency compartment.

A bulkhead caps how many calls may be in flight against one dependency at once.
The cap is the point: an unbounded caller will happily hold thousands of tasks
open against a stalled provider, and the memory and file descriptors that costs
take down the endpoints that had nothing to do with it.

The acquisition timeout matters as much as the limit. Waiting indefinitely for a
slot converts a saturated dependency into unbounded queueing, which is the same
failure the limit was meant to prevent, one level up.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from air_classifier_service.observability.logging import get_logger

if TYPE_CHECKING:
    from air_classifier_service.config import ResilienceSettings

logger = get_logger(__name__)


class BulkheadFullError(Exception):
    """Raised when no slot became free within the acquisition timeout."""

    def __init__(self, name: str, *, limit: int, waited_s: float) -> None:
        super().__init__(f"bulkhead {name!r} full: {limit} slots busy after {waited_s:.3f}s")
        self.bulkhead_name = name
        self.limit = limit
        self.waited_s = waited_s


class Bulkhead:
    """A named semaphore with a bounded wait."""

    def __init__(self, name: str, *, limit: int = 8, acquire_timeout_s: float = 2.0) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if acquire_timeout_s <= 0:
            raise ValueError("acquire_timeout_s must be > 0")

        self._name = name
        self._limit = limit
        self._acquire_timeout_s = acquire_timeout_s
        self._semaphore = asyncio.Semaphore(limit)
        self._in_flight = 0

    @classmethod
    def from_settings(cls, name: str, settings: ResilienceSettings) -> Bulkhead:
        """Build from the tuned values in :mod:`air_classifier_service.config`."""
        return cls(
            name,
            limit=settings.llm_concurrency,
            acquire_timeout_s=settings.bulkhead_acquire_timeout_s,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def available(self) -> int:
        return self._limit - self._in_flight

    def __repr__(self) -> str:
        return f"Bulkhead(name={self._name!r}, in_flight={self._in_flight}/{self._limit})"

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Hold one slot for the duration of the block.

        Raises:
            BulkheadFullError: No slot freed within ``acquire_timeout_s``.
        """
        started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(self._acquire_timeout_s):
                await self._semaphore.acquire()
        except TimeoutError as exc:
            waited = asyncio.get_running_loop().time() - started
            logger.warning(
                "bulkhead.rejected",
                bulkhead=self._name,
                limit=self._limit,
                waited_s=round(waited, 4),
            )
            raise BulkheadFullError(self._name, limit=self._limit, waited_s=waited) from exc

        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1
            self._semaphore.release()
