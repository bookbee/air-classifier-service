"""Customer-summary refresh — the stateless `/v1/summary/refresh` contract.

This service holds no customer data. A caller stores a :class:`CustomerSummary`
itself, sends it back as ``existing_summary`` alongside a batch of new
feedback/review texts, and receives an updated one to store in its place. See
`pipeline/summary_service.py` for how the two are reconciled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from air_classifier.constants import Label, Topic, Urgency
from air_classifier.schemas.classify import (
    MAX_BATCH_ITEMS_CEILING,
    MAX_TEXT_CHARS,
    ClassifyBatchItem,
)
from air_classifier.schemas.common import Options, RequestUsage

__all__ = [
    "RECENT_RATINGS_WINDOW",
    "CustomerSummary",
    "LLMSummaryOutput",
    "SummaryItem",
    "SummaryRefreshRequest",
    "SummaryRefreshResponse",
    "SummaryRollup",
]

#: Ratings kept verbatim for a "recent trend" view, newest last. Five is enough
#: to see a shift without the caller having to resend or this service having
#: to store the customer's full rating history. Public — `pipeline/summary_service.py`
#: reuses it when sliding the window on merge.
RECENT_RATINGS_WINDOW: Final = 5


class SummaryItem(BaseModel):
    """One new feedback or review text to fold into a customer's summary.

    Shares its optional fields with :class:`~air_classifier.schemas.classify.ClassifyRequest`
    but is its own model, not a subclass: `ClassifyRequest` is `extra="forbid"`
    and already a large, stable contract, and `source_type`/`occurred_at` are
    concepts specific to this endpoint, not to single-text classification.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="The text to analyse.")
    source_type: Literal["feedback", "review"] = Field(
        description="Where this text came from — governs `rollup.counts_by_source_type` only."
    )
    channel: str | None = Field(
        default=None, max_length=64, description="app | email | support | survey | ..."
    )
    subject: str | None = Field(default=None, max_length=500)
    rating: float | None = Field(default=None, ge=0, description="Stars as given by the author.")
    rating_scale_max: float = Field(default=5.0, gt=0, le=100)
    product_id: str | None = Field(default=None, max_length=128)
    occurred_at: datetime | None = Field(
        default=None,
        description="When this was authored. Governs rollup ordering; omit to sort last.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rating_within_scale(self) -> Self:
        if self.rating is not None and self.rating > self.rating_scale_max:
            raise ValueError("rating cannot exceed rating_scale_max")
        return self


class SummaryRollup(BaseModel):
    """Deterministic, cheap-to-recompute aggregate over every item folded in so far.

    Every histogram is kept in full rather than truncated to a "top N": each of
    `Label`, `Urgency` and `Topic` is a small closed vocabulary, so the full map
    costs nothing and a truncated one would silently lose history whenever a
    value fell out of the window and later reappeared.
    """

    model_config = ConfigDict(extra="forbid")

    total_items: int = Field(default=0, ge=0, description="Every item folded in so far.")
    counts_by_source_type: dict[str, int] = Field(
        default_factory=dict, description='e.g. {"feedback": 3, "review": 1}.'
    )
    sentiment_counts: dict[Label, int] = Field(
        default_factory=dict, description="Full histogram — every Label seen, none truncated."
    )
    urgency_counts: dict[Urgency, int] = Field(
        default_factory=dict, description="Full histogram — every Urgency seen, none truncated."
    )
    topic_counts: dict[Topic, int] = Field(
        default_factory=dict, description="Full histogram — every Topic seen, none truncated."
    )

    rating_count: int = Field(default=0, ge=0, description="Items that carried a rating.")
    rating_sum: float = Field(default=0.0, description="Sum of every rating seen, lifetime.")
    average_rating: float | None = Field(
        default=None, description="rating_sum / rating_count, on the scale each item supplied."
    )
    recent_ratings: list[float] = Field(
        default_factory=list,
        max_length=RECENT_RATINGS_WINDOW,
        description=(
            f"The last (up to) {RECENT_RATINGS_WINDOW} ratings, oldest first — kept as actual "
            "values, not an average, because a stateless refresh cannot reconstruct one from "
            "the other."
        ),
    )
    recent_average_rating: float | None = Field(default=None, description="Mean of recent_ratings.")

    first_seen_at: datetime | None = Field(
        default=None, description="`occurred_at` of the earliest item folded in, across all calls."
    )
    last_seen_at: datetime | None = Field(
        default=None, description="`occurred_at` of the most recent item folded in."
    )


class CustomerSummary(BaseModel):
    """The full rolling state a caller stores and resends on the next refresh.

    Round-trips byte-for-byte: store exactly what this service returns, and
    send it back exactly as `existing_summary` next time — this service never
    reads or writes it anywhere else.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(
        default="", max_length=500, description="2-4 sentence synthesis. Empty before any refresh."
    )
    rollup: SummaryRollup = Field(default_factory=SummaryRollup)
    version: int = Field(
        default=1,
        description=(
            "Bumped when this shape's *meaning* changes (a rollup field is redefined, a "
            "histogram's vocabulary shifts). A caller-stored summary at an older version "
            "should be rebuilt from scratch rather than merged into."
        ),
    )


class SummaryRefreshRequest(BaseModel):
    """A batch of new items to fold onto a customer's summary, stateless.

    This service persists nothing about `customer_id` or the resulting
    `summary` — the caller owns storage and resends `existing_summary` on the
    next call.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(
        min_length=1,
        max_length=128,
        description="Opaque identifier, echoed back only. Never looked up or stored.",
    )
    existing_summary: CustomerSummary | None = Field(
        default=None, description="Omit for a customer's first-ever summary."
    )
    items: list[SummaryItem] = Field(min_length=1, max_length=MAX_BATCH_ITEMS_CEILING)
    options: Options | None = Field(
        default=None, description="Applied to every item; see /v1/classify/batch's own field."
    )


class SummaryRefreshResponse(BaseModel):
    """The updated state to store, plus the individual verdict for every item sent.

    Persist `summary` in place of whatever `existing_summary` was on this call,
    and send it back as `existing_summary` on the next refresh for this
    customer.
    """

    request_id: str
    customer_id: str = Field(description="Echoed back from the request, unmodified.")
    summary: CustomerSummary = Field(
        description="Store this; resend it as existing_summary on the next refresh."
    )
    items: list[ClassifyBatchItem] = Field(
        description="One entry per request item, in order — same shape as /classify/batch's."
    )
    degraded: bool = Field(
        default=False,
        description="True if the narrative refresh failed; rollup and items are still current.",
    )
    usage: RequestUsage = Field(
        default_factory=RequestUsage, description="Summed across items classified successfully."
    )
    latency_ms: float = Field(default=0.0, ge=0.0)


class LLMSummaryOutput(BaseModel):
    """The narrative-refresh call's structured-output contract.

    Kept out of `schemas/verdict.py` — that module is scoped to the ladder's
    own internal currency, and this is a different call for a different task.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(max_length=500, description="2-4 sentences.")
