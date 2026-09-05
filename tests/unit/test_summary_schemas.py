"""The `/v1/summary/refresh` contract: validation and round-tripping.

`CustomerSummary` is the shape a caller stores between calls — a round trip
through JSON has to come back byte-for-byte equivalent, or "send it back next
time" silently loses information.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from air_classifier_service.schemas.summary import (
    RECENT_RATINGS_WINDOW,
    CustomerSummary,
    SummaryItem,
    SummaryRefreshRequest,
    SummaryRollup,
)


def test_source_type_is_required_and_closed() -> None:
    with pytest.raises(ValidationError):
        SummaryItem(text="fine")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SummaryItem(text="fine", source_type="survey")  # not one of the two allowed


def test_rating_within_scale_is_accepted() -> None:
    item = SummaryItem(text="great", source_type="review", rating=4, rating_scale_max=5)
    assert item.rating == 4


def test_rating_above_scale_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SummaryItem(text="great", source_type="review", rating=6, rating_scale_max=5)


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        SummaryItem.model_validate({"text": "x", "source_type": "feedback", "unexpected": "field"})


def test_recent_ratings_field_is_capped_at_the_window() -> None:
    with pytest.raises(ValidationError):
        SummaryRollup(recent_ratings=[1.0] * (RECENT_RATINGS_WINDOW + 1))


def test_customer_summary_round_trips_through_json() -> None:
    summary = CustomerSummary(
        narrative="A generally satisfied customer with one recent complaint.",
        rollup=SummaryRollup(
            total_items=3,
            counts_by_source_type={"feedback": 2, "review": 1},
            recent_ratings=[4.0, 5.0],
            average_rating=4.5,
        ),
        version=1,
    )

    restored = CustomerSummary.model_validate_json(summary.model_dump_json())

    assert restored == summary


def test_existing_summary_is_optional() -> None:
    request = SummaryRefreshRequest(
        customer_id="cust_1",
        items=[SummaryItem(text="first ever", source_type="feedback")],
    )
    assert request.existing_summary is None


def test_items_batch_requires_at_least_one() -> None:
    with pytest.raises(ValidationError):
        SummaryRefreshRequest(customer_id="cust_1", items=[])
