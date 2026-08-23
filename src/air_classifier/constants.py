"""Stable identifiers shared across the service.

Everything in this module is part of the public surface: these values appear in
API responses, in Prometheus metric labels, and in configuration keys. Renaming
a member is a breaking change for callers and dashboards alike.

`EscalationReason` in particular is deliberately a *closed* set. It is used as a
Prometheus label value, and an open-ended reason string would let label
cardinality grow without bound.

**Label compatibility.** `Label`, `Tone`, `Topic`, `Intent`, `Urgency`,
`RoutingPriority`, `Actionability` and `RatingAgreement` are this service's
response vocabulary, consumed by callers
that may branch on specific values. Adding a member is safe; removing one,
changing its meaning, or a retraining that shifts which inputs map to which
label (even with the member names untouched) is not — a cached verdict from
before the change would misrepresent one from after it. Any of those bumps
`cache.base.CACHE_SCHEMA_VERSION`, the mechanism that currently plays the role
a separately-versioned taxonomy contract would in a larger deployment.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Tier(StrEnum):
    """The rungs of the escalation ladder, cheapest first."""

    T0_RULES = "t0_rules"
    T1_CLASSIFIER = "t1_classifier"
    T2_LOCAL_LLM = "t2_local_llm"
    T3_CLOUD_LLM = "t3_cloud_llm"

    @property
    def rank(self) -> int:
        """Position in the ladder. Higher means more expensive."""
        return _TIER_RANK[self]

    @classmethod
    def ordered(cls) -> tuple[Tier, ...]:
        """All tiers, cheapest first."""
        return _TIER_ORDER

    def next_tier(self) -> Tier | None:
        """The next rung up, or None if this is the top."""
        index = self.rank + 1
        return _TIER_ORDER[index] if index < len(_TIER_ORDER) else None


_TIER_ORDER: Final[tuple[Tier, ...]] = (
    Tier.T0_RULES,
    Tier.T1_CLASSIFIER,
    Tier.T2_LOCAL_LLM,
    Tier.T3_CLOUD_LLM,
)
_TIER_RANK: Final[dict[Tier, int]] = {tier: i for i, tier in enumerate(_TIER_ORDER)}


class Label(StrEnum):
    """Sentiment labels.

    UNKNOWN is not a classification — it is what a tier returns when it declines
    to judge, and it always escalates.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Provider(StrEnum):
    """LLM backends. Ollama is local; the rest are remote and metered."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class EscalationReason(StrEnum):
    """Closed set of reason codes for a policy decision.

    Used both in `escalation_trace[].escalated_because` and as a Prometheus
    label. Adding a member is fine; making this open-ended is not.
    """

    # ── Stop reasons ──────────────────────────────────────────────────────
    CONFIDENT = "confident"
    TIER_CEILING = "tier_ceiling"
    BUDGET_EXHAUSTED = "budget_exhausted"
    COST_CEILING = "cost_ceiling"
    NEXT_TIER_UNAVAILABLE = "next_tier_unavailable"
    CACHE_HIT = "cache_hit"

    # ── Escalation reasons ────────────────────────────────────────────────
    MIN_TIER_FLOOR = "min_tier_floor"
    TIER_FAILED = "tier_failed"
    LOW_CONFIDENCE = "low_confidence"
    NARROW_MARGIN = "narrow_margin"
    TIER_DISAGREEMENT = "tier_disagreement"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    NO_SHORTCUT_MATCHED = "no_shortcut_matched"

    # ── Complexity triggers ───────────────────────────────────────────────
    CONTRASTIVE_CONJUNCTION = "contrastive_conjunction"
    POLARITY_FLIP = "polarity_flip"
    EMOJI_TEXT_CONFLICT = "emoji_text_conflict"
    NEGATION_DENSITY = "negation_density"
    LONG_INPUT = "long_input"


#: Reasons that terminate the ladder. Everything else continues it.
STOP_REASONS: Final[frozenset[EscalationReason]] = frozenset(
    {
        EscalationReason.CONFIDENT,
        EscalationReason.TIER_CEILING,
        EscalationReason.BUDGET_EXHAUSTED,
        EscalationReason.COST_CEILING,
        EscalationReason.NEXT_TIER_UNAVAILABLE,
        EscalationReason.CACHE_HIT,
    }
)

#: The subset of `STOP_REASONS` an `escalation_trace` step renders as `null`
#: (schemas/verdict.py::EscalationStep.build) rather than the reason code
#: itself. `CONFIDENT`, `CACHE_HIT` and a caller-chosen ceiling
#: (`TIER_CEILING`/`BUDGET_EXHAUSTED`/`COST_CEILING`) are the ladder finishing
#: on its own terms — nothing for a caller to act on. `NEXT_TIER_UNAVAILABLE`
#: is the opposite: the policy *wanted* to climb and a dependency wasn't
#: there, which is exactly the gap `pipeline.orchestrator` also flags as
#: `degraded` and worth surfacing rather than folding into the same silence
#: a confident answer gets.
SILENT_STOP_REASONS: Final[frozenset[EscalationReason]] = STOP_REASONS - {
    EscalationReason.NEXT_TIER_UNAVAILABLE
}


class Urgency(StrEnum):
    """How soon a piece of text needs a human."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Actionability(StrEnum):
    """Whether a piece of text implies work for someone."""

    NONE = "none"
    INFORMATIONAL = "informational"
    ACTIONABLE = "actionable"
    BLOCKING = "blocking"


class Tone(StrEnum):
    """The emotional register of the text — finer-grained than sentiment.

    Distinct from `Emotion` (schemas/common.py), which is free-named and
    multi-valued with per-item intensity; this is a single closed-vocabulary
    label with one confidence, the same shape as `Label` and `Urgency`,
    because it answers one question: which one word best names the author's
    register.
    """

    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    DISAPPOINTED = "disappointed"
    CONFUSED = "confused"
    ANXIOUS = "anxious"
    SARCASTIC = "sarcastic"
    APPRECIATIVE = "appreciative"
    URGENT = "urgent"


class Topic(StrEnum):
    """What the text is about. Multi-label — a text may raise several."""

    DELIVERY = "delivery"
    PAYMENT = "payment"
    PRODUCT_QUALITY = "product_quality"
    RETURN = "return"
    REFUND = "refund"
    PRICING = "pricing"


class Intent(StrEnum):
    """What the author wants done. Single-label."""

    COMPLAINT = "complaint"
    INFORMATION = "information"
    REFUND_REQUEST = "refund_request"
    RETURN_REQUEST = "return_request"
    ORDER_TRACKING = "order_tracking"
    REPLACEMENT_REQUEST = "replacement_request"
    CANCELLATION = "cancellation"
    RECOMMENDATION = "recommendation"
    PRAISE = "praise"


class RoutingPriority(StrEnum):
    """Incident-style priority for `routing.priority`.

    A direct rank mapping of `Urgency` (`CRITICAL`->`P1` ... `LOW`->`P4`) rather
    than a second judgement: two independently-derived urgency signals on one
    response would invite them to disagree, which is a worse outcome than one
    signal expressed twice.
    """

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class RatingAgreement(StrEnum):
    """Whether a text's prose matches the star rating given alongside it."""

    AGREES = "agrees"
    PROSE_MORE_POSITIVE = "prose_more_positive"
    PROSE_MORE_NEGATIVE = "prose_more_negative"


class PIIKind(StrEnum):
    """Categories the redactor recognises."""

    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    NATIONAL_ID = "national_id"
    IP_ADDRESS = "ip_address"
    URL_CREDENTIALS = "url_credentials"


#: Placeholder written in place of redacted spans, e.g. ``[REDACTED:email]``.
REDACTION_TEMPLATE: Final[str] = "[REDACTED:{kind}]"

#: Header names, defined once so middleware and clients cannot drift apart.
HEADER_API_KEY: Final[str] = "X-API-Key"
HEADER_REQUEST_ID: Final[str] = "X-Request-ID"
HEADER_IDEMPOTENCY_KEY: Final[str] = "Idempotency-Key"

#: Base URI for RFC 9457 `type` values.
ERROR_TYPE_BASE: Final[str] = "https://air.dev/errors"
