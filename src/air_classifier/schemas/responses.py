"""Response models.

The envelope is identical across all three resources; `/feedback` and `/reviews`
add their own blocks alongside it. A caller that understands only the core can
ignore the extras safely — which is what makes adding a fourth resource a
non-breaking change.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from air_classifier.constants import Actionability, Tier, Urgency
from air_classifier.schemas.common import (
    Aspect,
    Emotion,
    LanguageInfo,
    RatingConsistency,
    RequestUsage,
    SafetyReport,
    SentimentResult,
)
from air_classifier.schemas.errors import ProblemDetail
from air_classifier.schemas.verdict import EscalationStep


class BaseAnalysisResponse(BaseModel):
    """The shared envelope."""

    request_id: str
    sentiment: SentimentResult
    emotions: list[Emotion] = Field(default_factory=list)
    aspects: list[Aspect] = Field(default_factory=list)
    language: LanguageInfo
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


class SentimentResponse(BaseAnalysisResponse):
    """Generic text result."""


class FeedbackResponse(BaseAnalysisResponse):
    """Feedback result, with triage signals."""

    urgency: Urgency = Urgency.LOW
    actionability: Actionability = Actionability.INFORMATIONAL
    suggested_route: str | None = Field(
        default=None, description="Team or queue implied by the content, when inferable."
    )


class ReviewResponse(BaseAnalysisResponse):
    """Review result, with rating cross-check."""

    rating_consistency: RatingConsistency | None = Field(
        default=None, description="Null when the request carried no rating."
    )


class BatchItemResult[ResponseT: BaseAnalysisResponse](BaseModel):
    """One item's outcome.

    A batch is not all-or-nothing: one bad item yields a problem in that slot
    while its siblings still return results, so a caller never loses 99 good
    answers to 1 malformed input.
    """

    index: int = Field(ge=0)
    ok: bool
    result: ResponseT | None = None
    error: ProblemDetail | None = None


class BatchResponse[ResponseT: BaseAnalysisResponse](BaseModel):
    """Envelope for a batch call."""

    request_id: str
    items: list[BatchItemResult[ResponseT]]
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    usage: RequestUsage = Field(default_factory=RequestUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)


SentimentBatchResponse = BatchResponse[SentimentResponse]
FeedbackBatchResponse = BatchResponse[FeedbackResponse]
ReviewBatchResponse = BatchResponse[ReviewResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "air-classifier"
    version: str


class ReadyComponent(BaseModel):
    name: str
    ready: bool
    detail: str | None = None


class ReadyResponse(BaseModel):
    """Readiness.

    ``status`` is ``degraded`` — not ``not_ready`` — when an upper tier is down
    but a lower one can still answer. Taking the pod out of rotation for a cloud
    outage would be exactly the wrong reaction.
    """

    status: str
    components: list[ReadyComponent]
