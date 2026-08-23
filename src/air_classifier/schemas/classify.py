"""Request and response shapes for ``/v1/classify`` — the whole analysis surface.

This used to be one of four resource shapes (`/sentiment`, `/feedback`,
`/reviews`, plus this unified route); the other three are gone, and this
module absorbed what made each of them worth having. ``ClassifyRequest``
carries every input any of the three ever accepted — a review's `rating`, a
feedback item's `channel`, both optional and both irrelevant unless the text
actually has one — and ``ClassificationResponse`` carries every output any of
them ever produced: `/reviews`' `aspects` and `rating_consistency`,
`/feedback`'s `actionability`, plus the tone/topics/intent/routing/
requires_human this route always had. One shape, not three, because a caller
choosing between three response contracts on one endpoint was always the same
confusion three endpoints were.
"""

from __future__ import annotations

from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from air_classifier.constants import (
    Actionability,
    Intent,
    RoutingPriority,
    Tier,
    Tone,
    Topic,
    Urgency,
)
from air_classifier.schemas.common import (
    Aspect,
    Emotion,
    LabeledScore,
    LanguageInfo,
    Options,
    RatingConsistency,
    RequestUsage,
    SafetyReport,
    SentimentResult,
)
from air_classifier.schemas.errors import ProblemDetail
from air_classifier.schemas.verdict import EscalationStep

__all__ = [
    "MAX_BATCH_ITEMS_CEILING",
    "MAX_TEXT_CHARS",
    "ClassificationResponse",
    "ClassifyBatchItem",
    "ClassifyBatchRequest",
    "ClassifyBatchResponse",
    "ClassifyRequest",
    "ModelInfo",
    "RoutingInfo",
]

#: Absolute schema ceiling. The *effective* cap is `app.max_batch_items`,
#: enforced in the route so operators can lower it without a schema change.
MAX_BATCH_ITEMS_CEILING: Final = 1000
MAX_TEXT_CHARS: Final = 20_000


class ClassifyRequest(BaseModel):
    """One piece of text, plus every optional field any prior resource used.

    Only `text` is required. The rest are hints: fill in `rating` for
    something that came from a review, `channel` for something that came from
    a support form, or nothing at all for plain text with no known origin —
    each is read only where it is relevant (`rating` by the T0 rating
    shortcut and `rating_consistency`; `channel` by the urgency heuristic) and
    ignored everywhere else.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="The text to analyse.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Situational context (locale, channel, product). Available to rules.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-defined key/values. Echoed back; never interpreted.",
    )
    options: Options = Field(default_factory=Options)

    # ── Formerly /feedback's ─────────────────────────────────────────────
    channel: str | None = Field(
        default=None, max_length=64, description="app | email | support | survey | ..."
    )
    user_segment: str | None = Field(default=None, max_length=64)
    subject: str | None = Field(default=None, max_length=500)

    # ── Formerly /reviews' ───────────────────────────────────────────────
    rating: float | None = Field(default=None, ge=0, description="Stars as given by the author.")
    rating_scale_max: float = Field(default=5.0, gt=0, le=100)
    product_id: str | None = Field(default=None, max_length=128)
    verified_purchase: bool | None = None
    title: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _rating_within_scale(self) -> Self:
        if self.rating is not None and self.rating > self.rating_scale_max:
            raise ValueError("rating cannot exceed rating_scale_max")
        return self


class RoutingInfo(BaseModel):
    """Where this should go and how fast.

    Both heuristic, both deterministic — see
    `pipeline/service.py::_derive_routing`.
    """

    queue: str | None = Field(
        default=None, description="Team or queue implied by the content, when inferable."
    )
    priority: RoutingPriority


class ModelInfo(BaseModel):
    """Which model answered, and which build of this service shaped the reply.

    Not a replacement for `model_versions` (every tier that ran, kept below
    for the same reason it always has been) — this is the single-value
    summary the flat response shape calls for. `version` is this service's
    own release, not the model's: the model string already names its own
    version where one exists (`gemini:gemini-2.5-flash`), and a second
    version number on the same line would be ambiguous about which build it
    refers to.
    """

    name: str
    version: str


class ClassificationResponse(BaseModel):
    """The verdict. Every field a caller of any prior resource could rely on."""

    request_id: str

    #: The full judgement — label, continuous polarity, confidence and an
    #: optional rationale — not the label+confidence-only shape `tone`,
    #: `urgency`, `intent` and each `topics` entry use. Formerly `/sentiment`'s
    #: whole response; `polarity` in particular is worth keeping at full
    #: precision for a caller averaging or trending across many verdicts.
    sentiment: SentimentResult
    #: `None` when the ladder never reached a tier that can judge it — see
    #: this module's docstring and `pipeline/service.py::_shape`.
    tone: LabeledScore[Tone] | None = None
    urgency: LabeledScore[Urgency]
    topics: list[LabeledScore[Topic]] = Field(default_factory=list)
    intent: LabeledScore[Intent] | None = None
    routing: RoutingInfo
    requires_human: bool
    actionability: Actionability

    #: Formerly `/reviews`' — per-facet sentiment, populated by an LLM tier
    #: when the text discusses distinct facets; empty otherwise.
    aspects: list[Aspect] = Field(default_factory=list)
    #: Formerly `/reviews`' — `None` unless the request carried a `rating`.
    rating_consistency: RatingConsistency | None = None
    emotions: list[Emotion] = Field(default_factory=list)

    language: LanguageInfo
    model: ModelInfo

    # ── Observability, unconditionally present ──────────────────────────
    decided_by: Tier = Field(description="The rung whose verdict was returned.")
    degraded: bool = Field(
        default=False,
        description="True when a tier failed and a lower rung's answer was used instead.",
    )
    escalation_trace: list[EscalationStep] = Field(
        default_factory=list,
        description="Why the ladder climbed as far as it did. Suppressible via options.",
    )
    safety: SafetyReport = Field(default_factory=SafetyReport)
    usage: RequestUsage = Field(default_factory=RequestUsage)
    cached: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    model_versions: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Echoed back from the request, unmodified."
    )


class ClassifyBatchRequest(BaseModel):
    """A set of items analysed in one call."""

    model_config = ConfigDict(extra="forbid")

    items: list[ClassifyRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS_CEILING)
    options: Options | None = Field(
        default=None, description="Applied to items that did not specify their own."
    )


class ClassifyBatchItem(BaseModel):
    """One item's outcome.

    A batch is not all-or-nothing: one bad item yields a problem in that slot
    while its siblings still return results, so a caller never loses 99 good
    answers to 1 malformed input.
    """

    index: int = Field(ge=0)
    ok: bool
    result: ClassificationResponse | None = None
    error: ProblemDetail | None = None


class ClassifyBatchResponse(BaseModel):
    """Envelope for a batch call."""

    request_id: str
    items: list[ClassifyBatchItem]
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    usage: RequestUsage = Field(default_factory=RequestUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
