"""Request options and LLM-output → verdict mapping (schemas/common.py, schemas/verdict.py)."""

from __future__ import annotations

import pydantic
import pytest

from air_classifier_service.constants import Label, Tier, Topic
from air_classifier_service.schemas.common import LabeledScore, Options, TierUsage
from air_classifier_service.schemas.verdict import LLMSentimentOutput

# ── Options.aspects (SIT CL-006d) ─────────────────────────────────────────────


@pytest.mark.parametrize("aspects", [[""], ["   "], ["battery", ""]])
def test_a_blank_aspect_name_is_rejected(aspects: list[str]) -> None:
    with pytest.raises(pydantic.ValidationError):
        Options(aspects=aspects)


def test_aspect_names_are_trimmed() -> None:
    assert Options(aspects=[" battery ", "camera"]).aspects == ["battery", "camera"]


# ── Topics (SIT CL-006e) ──────────────────────────────────────────────────────


def test_a_zero_confidence_topic_is_not_reported() -> None:
    output = LLMSentimentOutput(
        label=Label.POSITIVE,
        polarity=0.8,
        confidence=0.9,
        rationale="Praises support.",
        topics=[
            LabeledScore[Topic](label=Topic.PRODUCT_QUALITY, confidence=0.0),
            LabeledScore[Topic](label=Topic.DELIVERY, confidence=0.4),
        ],
    )

    verdict = output.to_verdict(
        Tier.T2_LOCAL_LLM, model_version="m", latency_ms=1.0, usage=TierUsage()
    )

    assert [topic.label for topic in verdict.topics] == [Topic.DELIVERY]
