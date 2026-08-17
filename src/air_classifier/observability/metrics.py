"""The Prometheus catalogue from docs/02-lld.md §8.

Metric objects live in a module-private :class:`~prometheus_client.CollectorRegistry`
rather than the process-wide default one. The default registry raises on duplicate
registration, which makes a re-import (test collection, ``importlib.reload``, a
uvicorn reloader) explode and makes assertions order-dependent because every test
module shares the same counters.

Call sites never build label strings by hand. Every metric is fronted by a typed
recording helper below, so a label value can only ever come from an enum member or
a value the caller already holds — which is what keeps cardinality bounded and
dashboards from silently splitting in two after a typo.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum, StrEnum
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from air_classifier.constants import EscalationReason, PIIKind, Provider, Tier

# ── Label vocabularies ────────────────────────────────────────────────────────


class TierOutcome(StrEnum):
    """What the policy did with a tier's verdict."""

    STOP = "stop"
    ESCALATE = "escalate"
    FAILED = "failed"


class CacheEvent(StrEnum):
    """Verdict-cache outcomes. `store` is a write, not a lookup."""

    HIT = "hit"
    MISS = "miss"
    STORE = "store"


class TokenKind(StrEnum):
    """Token classes billed at different rates."""

    IN = "in"
    OUT = "out"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"


class BreakerState(IntEnum):
    """Circuit-breaker state as a gauge value.

    Numeric rather than a label so a dashboard can alert on ``> 0`` without
    enumerating states, and so a provider's state is one time series instead of
    three that have to be summed.
    """

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2

    @classmethod
    def from_name(cls, name: str) -> BreakerState:
        """Map a breaker's own state name onto the gauge value."""
        return cls.__members__[name.strip().upper()]


# ── Buckets ───────────────────────────────────────────────────────────────────

#: Whole-request latency. `Options.latency_budget_ms` caps at 30s, so the tail
#: bucket sits there — prometheus' 10s default would file every budget-exhausted
#: request into +Inf and hide exactly the tail we care about.
REQUEST_DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
)

#: Per-tier latency spans four orders of magnitude: T0 is a lexicon scan measured
#: in tens of microseconds, T3 is a cloud round trip measured in seconds. One
#: linear bucket set cannot serve both ends — every T0 observation would land in
#: the first bucket and every T3 one in +Inf, making both P95s meaningless. These
#: are roughly logarithmic from 100µs to 20s so each tier gets usable resolution
#: in its own range.
TIER_DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
)

# ── Registry and metrics ──────────────────────────────────────────────────────

_REGISTRY: Final[CollectorRegistry] = CollectorRegistry()

REQUESTS_TOTAL: Final = Counter(
    "air_classifier_requests_total",
    "Requests served, by endpoint, HTTP status, and API key id.",
    ("endpoint", "status", "api_key_id"),
    registry=_REGISTRY,
)

REQUEST_DURATION_SECONDS: Final = Histogram(
    "air_classifier_request_duration_seconds",
    "End-to-end request latency.",
    ("endpoint",),
    buckets=REQUEST_DURATION_BUCKETS,
    registry=_REGISTRY,
)

TIER_DECISIONS_TOTAL: Final = Counter(
    "air_classifier_tier_decisions_total",
    "Policy decisions taken after a tier ran.",
    ("tier", "outcome"),
    registry=_REGISTRY,
)

ESCALATIONS_TOTAL: Final = Counter(
    "air_classifier_escalations_total",
    "Climbs from one rung to the next, by reason code.",
    ("from_tier", "to_tier", "reason"),
    registry=_REGISTRY,
)

TIER_DURATION_SECONDS: Final = Histogram(
    "air_classifier_tier_duration_seconds",
    "Time spent inside a single tier.",
    ("tier",),
    buckets=TIER_DURATION_BUCKETS,
    registry=_REGISTRY,
)

DECIDED_BY_TOTAL: Final = Counter(
    "air_classifier_decided_by_total",
    "Which rung produced the answer the caller received.",
    ("tier",),
    registry=_REGISTRY,
)

LLM_TOKENS_TOTAL: Final = Counter(
    "air_classifier_llm_tokens_total",
    "Tokens consumed, by provider, model, and billing class.",
    ("provider", "model", "kind"),
    registry=_REGISTRY,
)

LLM_COST_USD_TOTAL: Final = Counter(
    "air_classifier_llm_cost_usd_total",
    "Estimated spend in USD, by provider and model.",
    ("provider", "model"),
    registry=_REGISTRY,
)

CACHE_EVENTS_TOTAL: Final = Counter(
    "air_classifier_cache_events_total",
    "Verdict cache hits, misses, and stores.",
    ("result",),
    registry=_REGISTRY,
)

BREAKER_STATE: Final = Gauge(
    "air_classifier_breaker_state",
    "Circuit breaker state per provider: 0 closed, 1 half_open, 2 open.",
    ("provider",),
    registry=_REGISTRY,
)

DEGRADED_RESPONSES_TOTAL: Final = Counter(
    "air_classifier_degraded_responses_total",
    "Responses returned with degraded=true, by reason code.",
    ("reason",),
    registry=_REGISTRY,
)

PII_REDACTIONS_TOTAL: Final = Counter(
    "air_classifier_pii_redactions_total",
    "Spans redacted before any tier saw the text, by kind.",
    ("kind",),
    registry=_REGISTRY,
)


def get_registry() -> CollectorRegistry:
    """The registry every metric in this service is registered against."""
    return _REGISTRY


def render_latest() -> tuple[bytes, str]:
    """Scrape payload and its content type, ready for the `/metrics` response."""
    payload: bytes = generate_latest(_REGISTRY)
    content_type: str = CONTENT_TYPE_LATEST
    return payload, content_type


# ── Recording helpers ─────────────────────────────────────────────────────────


def record_request(
    *,
    endpoint: str,
    status: int,
    api_key_id: str,
    duration_s: float | None = None,
) -> None:
    """Count one served request, and optionally time it.

    ``api_key_id`` is the key's *identifier* — never the key itself, and never a
    hash the key can be recovered from. Anything else turns a scrape endpoint
    into a credential leak, and metrics endpoints are routinely less guarded
    than the API they describe.
    """
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=str(status), api_key_id=api_key_id).inc()
    if duration_s is not None:
        REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration_s)


def record_tier_decision(tier: Tier, outcome: TierOutcome) -> None:
    """One policy decision about one rung."""
    TIER_DECISIONS_TOTAL.labels(tier=tier.value, outcome=outcome.value).inc()


def record_escalation(from_tier: Tier, to_tier: Tier, reason: EscalationReason) -> None:
    """One climb up the ladder."""
    ESCALATIONS_TOTAL.labels(
        from_tier=from_tier.value, to_tier=to_tier.value, reason=reason.value
    ).inc()


def record_decided_by(tier: Tier) -> None:
    """The rung whose verdict was returned.

    Divided by ``requests_total`` this is the tier-distribution KPI — the number
    that says whether the ladder is earning its keep.
    """
    DECIDED_BY_TOTAL.labels(tier=tier.value).inc()


def record_llm_usage(
    provider: Provider,
    model: str,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Token and spend accounting for one provider call.

    Zero-valued classes are skipped rather than incremented by zero: an
    always-zero series for a provider that has no prompt cache is noise on every
    dashboard that groups by ``kind``.
    """
    counts = (
        (TokenKind.IN, tokens_in),
        (TokenKind.OUT, tokens_out),
        (TokenKind.CACHE_READ, cache_read),
        (TokenKind.CACHE_WRITE, cache_write),
    )
    for kind, count in counts:
        if count > 0:
            LLM_TOKENS_TOTAL.labels(provider=provider.value, model=model, kind=kind.value).inc(
                count
            )
    if cost_usd > 0.0:
        LLM_COST_USD_TOTAL.labels(provider=provider.value, model=model).inc(cost_usd)


def record_cache_event(result: CacheEvent) -> None:
    """One verdict-cache lookup or write."""
    CACHE_EVENTS_TOTAL.labels(result=result.value).inc()


def set_breaker_state(provider: Provider, state: BreakerState) -> None:
    """Publish a breaker's current state."""
    BREAKER_STATE.labels(provider=provider.value).set(float(state.value))


def record_degraded(reason: EscalationReason) -> None:
    """One response returned with ``degraded=true``."""
    DEGRADED_RESPONSES_TOTAL.labels(reason=reason.value).inc()


def record_pii_redaction(kind: PIIKind, count: int = 1) -> None:
    """Redacted spans of one kind. Counts spans, not requests."""
    if count > 0:
        PII_REDACTIONS_TOTAL.labels(kind=kind.value).inc(count)


@contextmanager
def tier_timer(tier: Tier) -> Iterator[None]:
    """Time one tier, whether it returns a verdict or raises.

    A tier that times out is still latency the caller paid for, so the
    observation happens in ``finally`` and the exception propagates untouched —
    the orchestrator, not this timer, decides what a failure means.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        TIER_DURATION_SECONDS.labels(tier=tier.value).observe(time.perf_counter() - started)
