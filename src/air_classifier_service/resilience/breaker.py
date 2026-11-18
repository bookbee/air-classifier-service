"""Async circuit breaker.

A breaker exists to stop a service from queueing work against a dependency that
has already told it, repeatedly, that it cannot serve. Without one, a slow
provider converts into a thread/task pile-up and the outage spreads inward.

The state machine::

    CLOSED --(threshold consecutive failures)--> OPEN
    OPEN --(cooldown elapsed, on next call)--> HALF_OPEN
    HALF_OPEN --(probe succeeds)--> CLOSED
    HALF_OPEN --(probe fails)--> OPEN

All timing is on ``time.monotonic``. Wall-clock is not usable here: an NTP step
or a DST-adjacent correction must never open or close a breaker.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from air_classifier_service.constants import Provider
from air_classifier_service.observability.logging import get_logger

if TYPE_CHECKING:
    from air_classifier_service.config import ResilienceSettings

logger = get_logger(__name__)


class BreakerState(StrEnum):
    """Values match the ``air_classifier_breaker_state`` gauge's documented order."""

    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


class CircuitOpenError(Exception):
    """Raised in place of calling a dependency the breaker has cut off.

    ``retry_after_s`` is the remaining cooldown, so callers can surface a
    meaningful ``Retry-After`` rather than inviting an immediate re-attempt.
    """

    def __init__(self, name: str, *, retry_after_s: float = 0.0) -> None:
        super().__init__(f"circuit breaker {name!r} is open")
        self.breaker_name = name
        self.retry_after_s = retry_after_s


class CircuitBreaker:
    """Per-dependency failure gate.

    One instance guards one logical dependency (a provider, an endpoint). Share
    it across all callers of that dependency — a per-request breaker learns
    nothing.
    """

    def __init__(
        self,
        name: str,
        *,
        threshold: int = 5,
        cooldown_s: float = 30.0,
        half_open_max: int = 1,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        if half_open_max < 1:
            raise ValueError("half_open_max must be >= 1")

        self._name = name
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._half_open_max = half_open_max

        self._lock = asyncio.Lock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = 0

        # Seed the gauge. Without this a provider that has never failed has no
        # series at all, so a breaker dashboard is blank until the first outage
        # — exactly when nobody wants to be wondering whether the panel is
        # broken or the provider is fine.
        _publish_state(self._name, BreakerState.CLOSED)

    @classmethod
    def from_settings(cls, name: str, settings: ResilienceSettings) -> CircuitBreaker:
        """Build from the tuned values in :mod:`air_classifier_service.config`."""
        return cls(
            name,
            threshold=settings.breaker_threshold,
            cooldown_s=settings.breaker_cooldown_s,
            half_open_max=settings.breaker_half_open_max,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> BreakerState:
        """Externally visible state.

        Once the cooldown has elapsed this reports ``HALF_OPEN`` even though no
        probe has been admitted yet: the breaker is willing to be tested, and a
        selector that could not see that would never send the probe that lets
        the breaker recover.
        """
        if self._state is BreakerState.OPEN and self._cooldown_remaining() <= 0.0:
            return BreakerState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        """Whether a call would be admitted.

        Half-open counts as closed for selection purposes, for the reason given
        on :attr:`state`.
        """
        return self.state is not BreakerState.OPEN

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self._name!r}, state={self.state.value}, "
            f"failures={self._consecutive_failures})"
        )

    # ── Call wrapping ─────────────────────────────────────────────────────

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """Admit one call and record its outcome.

        Any :class:`Exception` escaping the block counts as a failure. Anything
        deriving from :class:`BaseException` — cancellation, in practice — does
        not: the caller withdrawing says nothing about the dependency's health.
        """
        probe = await self._acquire()
        try:
            yield
        except Exception:
            await self._settle(probe=probe, success=False)
            raise
        except BaseException:
            # Return the half-open slot without taking the lock: awaiting
            # anything while unwinding a cancellation risks never finishing.
            if probe:
                self._half_open_in_flight -= 1
            raise
        else:
            await self._settle(probe=probe, success=True)

    async def call[T](self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` under :meth:`guard`."""
        async with self.guard():
            return await fn(*args, **kwargs)

    # ── Manual bookkeeping ────────────────────────────────────────────────

    async def record_success(self) -> None:
        """Register a healthy outcome observed outside :meth:`guard`."""
        async with self._lock:
            self._on_success()

    async def record_failure(self) -> None:
        """Register a failed outcome observed outside :meth:`guard`.

        Used for failures the guard cannot see as exceptions from the wrapped
        call — a wall-clock timeout being the important one.
        """
        async with self._lock:
            self._on_failure()

    async def reset(self) -> None:
        """Force back to CLOSED. For operator intervention and test setup."""
        async with self._lock:
            self._consecutive_failures = 0
            self._half_open_in_flight = 0
            self._transition(BreakerState.CLOSED)

    # ── Internals ─────────────────────────────────────────────────────────

    async def _acquire(self) -> bool:
        """Admission check. Returns whether this call is a half-open probe."""
        async with self._lock:
            if self._state is BreakerState.OPEN:
                remaining = self._cooldown_remaining()
                if remaining > 0.0:
                    raise CircuitOpenError(self._name, retry_after_s=remaining)
                self._transition(BreakerState.HALF_OPEN)

            if self._state is BreakerState.HALF_OPEN:
                # Excess probes fail fast rather than queue. A recovering
                # dependency that gets the full backlog thrown at it the
                # instant the cooldown expires simply breaks again.
                if self._half_open_in_flight >= self._half_open_max:
                    raise CircuitOpenError(self._name, retry_after_s=self._cooldown_s)
                self._half_open_in_flight += 1
                return True

            return False

    async def _settle(self, *, probe: bool, success: bool) -> None:
        async with self._lock:
            if probe:
                self._half_open_in_flight -= 1
            if success:
                self._on_success()
            else:
                self._on_failure()

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        if self._state is BreakerState.HALF_OPEN:
            self._transition(BreakerState.CLOSED)

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        # A failed probe re-opens immediately: the cooldown just bought the
        # dependency time and it is still unwell, so a second chance now would
        # only cost another failed request.
        if self._state is BreakerState.HALF_OPEN or (
            self._state is BreakerState.CLOSED and self._consecutive_failures >= self._threshold
        ):
            self._open()

    def _open(self) -> None:
        self._opened_at = time.monotonic()
        self._transition(BreakerState.OPEN)

    def _transition(self, new_state: BreakerState) -> None:
        if new_state is self._state:
            return
        previous = self._state
        self._state = new_state
        if new_state is BreakerState.HALF_OPEN:
            self._half_open_in_flight = 0
        elif new_state is BreakerState.CLOSED:
            self._opened_at = None
            self._consecutive_failures = 0
        logger.info(
            "breaker.transition",
            breaker=self._name,
            from_state=previous.value,
            to_state=new_state.value,
            consecutive_failures=self._consecutive_failures,
        )
        _publish_state(self._name, new_state)

    def _cooldown_remaining(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self._cooldown_s - (time.monotonic() - self._opened_at))


def _publish_state(name: str, state: BreakerState) -> None:
    """Best-effort gauge update.

    ``air_classifier_breaker_state`` is labelled by provider, so a breaker
    guarding something else has nowhere to report and is skipped rather than
    inventing a label value. The import is lazy and failures are swallowed: a
    breaker that could not flip because the metrics registry was unhappy would
    turn an observability problem into an availability one.
    """
    try:
        provider = Provider(name)
    except ValueError:
        return

    try:
        from air_classifier_service.observability import metrics

        metrics.set_breaker_state(provider, metrics.BreakerState.from_name(state.name))
    except Exception:
        logger.debug("breaker.metric_unavailable", breaker=name, state=state.value)
