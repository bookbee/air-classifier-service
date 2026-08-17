"""Stable identifiers shared across the service.

Everything in this module is part of the public surface: these values appear in
API responses, in Prometheus metric labels, and in configuration keys. Renaming
a member is a breaking change for callers and dashboards alike.

`EscalationReason` in particular is deliberately a *closed* set. It is used as a
Prometheus label value, and an open-ended reason string would let label
cardinality grow without bound.
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
    OLLAMA = "ollama"


class ResourceKind(StrEnum):
    """Which endpoint family a request arrived through."""

    SENTIMENT = "sentiment"
    FEEDBACK = "feedback"
    REVIEWS = "reviews"


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


class Urgency(StrEnum):
    """How soon a piece of feedback needs a human."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Actionability(StrEnum):
    """Whether feedback implies work."""

    NONE = "none"
    INFORMATIONAL = "informational"
    ACTIONABLE = "actionable"
    BLOCKING = "blocking"


class RatingAgreement(StrEnum):
    """Whether a review's prose matches its star rating."""

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
