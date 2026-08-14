"""Request models.

Three resources rather than one endpoint with a `type` discriminator: they share
a core but differ in required inputs and returned fields, so each gets an honest
OpenAPI schema instead of a union where every field is conditionally optional.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from air_sentiment.schemas.common import Options

#: Absolute schema ceiling. The *effective* cap is `app.max_batch_items`, enforced
#: in the route so operators can lower it without a schema change.
MAX_BATCH_ITEMS_CEILING = 1000
MAX_TEXT_CHARS = 20_000


class BaseAnalysisRequest(BaseModel):
    """Fields common to every analysis endpoint."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="The text to analyse.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Situational context (locale, channel, product). Available to rules.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-defined key/values. Echoed back; never interpreted.",
    )
    options: Options = Field(default_factory=Options)


class SentimentRequest(BaseAnalysisRequest):
    """Generic text with no domain assumptions."""


class FeedbackRequest(BaseAnalysisRequest):
    """User feedback, which additionally gets urgency and routing analysis."""

    channel: str | None = Field(
        default=None, max_length=64, description="app | email | support | survey | ..."
    )
    user_segment: str | None = Field(default=None, max_length=64)
    subject: str | None = Field(default=None, max_length=500)


class ReviewRequest(BaseAnalysisRequest):
    """A product review, which additionally gets aspects and rating consistency."""

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

    @property
    def normalised_rating(self) -> float | None:
        """Rating mapped onto the -1..+1 polarity scale, or None when absent."""
        if self.rating is None:
            return None
        return (self.rating / self.rating_scale_max) * 2.0 - 1.0


class BatchRequest[RequestT: BaseAnalysisRequest](BaseModel):
    """A set of items analysed in one call.

    ``options`` here is a batch-level default; an item that carries its own
    ``options`` wins, so a caller can cap one expensive item without lowering the
    ceiling for the rest.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[RequestT] = Field(min_length=1, max_length=MAX_BATCH_ITEMS_CEILING)
    options: Options | None = Field(
        default=None, description="Applied to items that did not specify their own."
    )


SentimentBatchRequest = BatchRequest[SentimentRequest]
FeedbackBatchRequest = BatchRequest[FeedbackRequest]
ReviewBatchRequest = BatchRequest[ReviewRequest]
