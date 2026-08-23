"""Models shared by requests, responses, and the internals in between.

These are the contract. OpenAPI, request validation, and the JSON Schema handed
to LLM providers as a structured-output constraint are all derived from this
module — none of them is maintained by hand alongside it.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from air_classifier.constants import Label, PIIKind, RatingAgreement, Tier

# ── Primitives ────────────────────────────────────────────────────────────────


class LanguageInfo(BaseModel):
    """Detected language of the input."""

    code: str = Field(description="ISO 639-1 code, or 'und' when undetermined.")
    confidence: float = Field(ge=0.0, le=1.0)

    @classmethod
    def undetermined(cls) -> LanguageInfo:
        return cls(code="und", confidence=0.0)


class SentimentResult(BaseModel):
    """The core judgement every tier produces."""

    label: Label
    polarity: float = Field(
        ge=-1.0, le=1.0, description="Continuous valence: -1 fully negative, +1 fully positive."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=500)


class Aspect(BaseModel):
    """Sentiment toward one named facet of the subject."""

    name: str = Field(min_length=1, max_length=64)
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = Field(default=None, max_length=280)


class Emotion(BaseModel):
    """A named affect with an intensity."""

    name: str = Field(min_length=1, max_length=32)
    score: float = Field(ge=0.0, le=1.0)


class LabeledScore[T](BaseModel):
    """A closed-vocabulary label with the confidence behind it.

    The shape `/v1/classify`'s `tone`, `urgency`, `intent` and each entry of
    `topics` all share — `T` is `Tone`, `Urgency`, `Intent` or `Topic`
    respectively (see `constants.py`). One generic rather than four near-
    identical models, the same reasoning as `schemas/classify.py`'s
    `ClassifyBatchItem`/`ClassifyBatchResponse` share one shape rather than
    repeating it per field.
    """

    label: T
    confidence: float = Field(ge=0.0, le=1.0)


class RatingConsistency(BaseModel):
    """Whether a review's prose matches the stars it carries."""

    agreement: RatingAgreement
    #: Signed gap between prose polarity and normalised rating, both on -1..+1.
    delta: float = Field(ge=-2.0, le=2.0)
    normalised_rating: float = Field(ge=-1.0, le=1.0)


# ── Safety ────────────────────────────────────────────────────────────────────


class PIIFinding(BaseModel):
    kind: PIIKind
    count: int = Field(ge=1)


class SafetyReport(BaseModel):
    """What preprocessing did to the text before any tier saw it."""

    pii_redacted: bool = False
    pii_findings: list[PIIFinding] = Field(default_factory=list)
    flagged: bool = False
    flag_reason: str | None = None
    truncated: bool = False


# ── Usage & cost ──────────────────────────────────────────────────────────────


class TierUsage(BaseModel):
    """Resource consumption attributable to a single tier."""

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    provider: str | None = None
    model: str | None = None

    def __add__(self, other: TierUsage) -> TierUsage:
        return TierUsage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            provider=other.provider or self.provider,
            model=other.model or self.model,
        )


class RequestUsage(BaseModel):
    """Roll-up across every tier that ran."""

    tiers_run: int = Field(default=0, ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    est_cost_usd: float = Field(default=0.0, ge=0.0)


# ── Complexity features ───────────────────────────────────────────────────────


class ComplexityFeatures(BaseModel):
    """Cheap lexical signals used to route *whom to ask*, not to classify.

    These are heuristics, and they are deliberately tuned toward escalation:
    being liberal costs one cheap tier hop, being conservative costs accuracy.
    """

    token_count: int = Field(default=0, ge=0)
    char_count: int = Field(default=0, ge=0)
    negation_count: int = Field(default=0, ge=0)
    contrastive_count: int = Field(default=0, ge=0)
    intensifier_count: int = Field(default=0, ge=0)
    exclamation_count: int = Field(default=0, ge=0)
    positive_hits: int = Field(default=0, ge=0)
    negative_hits: int = Field(default=0, ge=0)
    emoji_polarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    lexical_polarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    uppercase_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    question_only: bool = False
    has_url_or_code: bool = False

    @property
    def polarity_flip(self) -> bool:
        """Both polarities present in the same text."""
        return self.positive_hits > 0 and self.negative_hits > 0

    @property
    def emoji_text_conflict(self) -> bool:
        """Emoji sentiment opposes the lexical sentiment with meaningful strength."""
        return (
            abs(self.emoji_polarity) >= 0.3
            and abs(self.lexical_polarity) >= 0.3
            and (self.emoji_polarity * self.lexical_polarity) < 0
        )


# ── Options ───────────────────────────────────────────────────────────────────


class Options(BaseModel):
    """Per-request controls over the ladder.

    ``model_fields_set`` is load-bearing here: the API needs to distinguish
    "caller explicitly asked for redact_pii=false" from "caller said nothing and
    got the default", because an API key carrying ``force_pii_redaction``
    rejects the former and silently permits the latter.
    """

    model_config = ConfigDict(extra="forbid")

    min_tier: Tier | None = Field(
        default=None, description="Force the ladder to reach at least this rung."
    )
    max_tier: Tier = Field(
        default=Tier.T3_CLOUD_LLM, description="Hard ceiling; also a privacy and cost lever."
    )
    #: The real governor on how far the ladder climbs: the orchestrator gives each
    #: rung min(its configured timeout, what is left of this). 8s is sized so a
    #: request can reach the cloud tier on the default path — T0+T1 cost ~2ms,
    #: T2 runs ~1.5s at p50, leaving room for a T3 call. Lower it to trade
    #: accuracy for latency; the ladder simply stops earlier and says so in the
    #: trace via `budget_exhausted`.
    latency_budget_ms: int = Field(default=8000, ge=50, le=30_000)
    max_cost_usd: float = Field(default=0.05, ge=0.0, le=10.0)
    include_trace: bool = True
    redact_pii: bool = True
    aspects: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Aspects to score explicitly, when the text has any.",
    )

    @model_validator(mode="after")
    def _tier_bounds(self) -> Self:
        if self.min_tier is not None and self.min_tier.rank > self.max_tier.rank:
            raise ValueError("min_tier cannot exceed max_tier")
        return self

    @property
    def redact_pii_was_explicit(self) -> bool:
        return "redact_pii" in self.model_fields_set

    @property
    def max_tier_was_explicit(self) -> bool:
        return "max_tier" in self.model_fields_set


# ── Capabilities ──────────────────────────────────────────────────────────────


class TierCapability(BaseModel):
    tier: Tier
    enabled: bool
    available: bool
    model_version: str
    detail: dict[str, Any] = Field(default_factory=dict)


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    tiers: list[TierCapability]
    supported_languages: list[str]
    max_batch_items: int
    max_text_chars: int
