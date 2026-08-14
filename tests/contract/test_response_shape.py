"""Golden fixtures: what a caller actually receives.

``test_openapi.py`` pins what the service *says* it returns. This suite pins
what it *does* return, by driving all three resources through the assembled
application — real middleware, real dependencies, real Tier 0 and Tier 1 — and
reading the JSON off the wire rather than a model instance. The distinction
matters: a response model can be correct while the route serialises it through
the wrong envelope, returns a bare ``Response``, or drops a block that only
appears on one resource.

The deployment is deliberately two rungs deep — Ollama off, no cloud
credentials — for two reasons. It opens no sockets, so the suite is hermetic
without mocking anything. And it makes the ladder *deterministic*: T0 and T1 are
pure functions of the input, so the trace below is the same on every machine,
which is what lets these be fixtures rather than smoke tests.

What is asserted is the shape and the invariants — field names, the trace's
columns, which resource carries which extra block. Concrete polarities are not:
they belong to the tiers' own suites, and pinning them here would make a lexicon
tweak fail an HTTP contract test for no reason.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest
from fastapi.testclient import TestClient

from air_sentiment.config import (
    AppSettings,
    CacheSettings,
    OllamaSettings,
    ProvidersSettings,
    SecuritySettings,
    Settings,
)
from air_sentiment.constants import Tier
from air_sentiment.main import create_app
from air_sentiment.schemas.responses import (
    FeedbackResponse,
    ReviewResponse,
    SentimentResponse,
)

#: Mixed polarity with a contrastive conjunction: T0 has no shortcut for it and
#: T1 is honestly unconfident, so the ladder climbs and the trace has more than
#: one row. A single-polarity string would be decided at T0 and prove nothing
#: about escalation.
MIXED: Final[str] = "The battery life is superb but the camera is a letdown."

#: docs/01-hld.md §4, response envelope. Listed literally rather than derived
#: from the model, because the point is to catch the model and the document
#: disagreeing — deriving it would make the assertion vacuous.
DOCUMENTED_ENVELOPE: Final[frozenset[str]] = frozenset(
    {
        "request_id",
        "sentiment",
        "emotions",
        "aspects",
        "language",
        "decided_by",
        "degraded",
        "escalation_trace",
        "safety",
        "usage",
        "cached",
        "latency_ms",
        "model_versions",
    }
)

#: One row of `escalation_trace`. `succeeded` is not in the §4 sample but is
#: part of the published `EscalationStep`, and a trace that hides a failed rung
#: would misreport a degraded answer as a clean one.
TRACE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"tier", "label", "confidence", "latency_ms", "succeeded", "escalated_because"}
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """The assembled app, started.

    ``allow_unauthenticated`` rather than a configured key: identity is
    ``test_api_deps``'s subject, and every route here would otherwise carry a
    header that has nothing to do with the response shape under test.
    """
    settings = Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(ollama=OllamaSettings(enabled=False)),
        cache=CacheSettings(backend="memory"),
        security=SecuritySettings(allow_unauthenticated=True),
    )
    # The context manager is what runs the lifespan — without it there is no
    # ladder on `app.state` and every route 500s on a wiring error.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def post(client: TestClient, path: str, **body: Any) -> dict[str, Any]:
    """POST a body and return the decoded envelope, asserting it was served."""
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


# ── The shared envelope ───────────────────────────────────────────────────────


def assert_core_envelope(payload: dict[str, Any]) -> None:
    """Every documented field, present and of the documented type."""
    assert set(payload) >= DOCUMENTED_ENVELOPE

    assert payload["request_id"].startswith("req_")

    sentiment = payload["sentiment"]
    assert set(sentiment) == {"label", "polarity", "confidence", "rationale"}
    assert sentiment["label"] in {"positive", "negative", "neutral", "mixed", "unknown"}
    assert -1.0 <= sentiment["polarity"] <= 1.0
    assert 0.0 <= sentiment["confidence"] <= 1.0

    assert set(payload["language"]) == {"code", "confidence"}
    assert isinstance(payload["language"]["code"], str)

    assert payload["decided_by"] in {tier.value for tier in Tier}
    assert isinstance(payload["degraded"], bool)
    assert isinstance(payload["cached"], bool)
    assert payload["latency_ms"] >= 0.0

    assert isinstance(payload["emotions"], list)
    assert isinstance(payload["aspects"], list)
    assert isinstance(payload["model_versions"], dict)

    assert set(payload["safety"]) == {
        "pii_redacted",
        "pii_findings",
        "flagged",
        "flag_reason",
        "truncated",
    }
    assert set(payload["usage"]) == {
        "tiers_run",
        "tokens_in",
        "tokens_out",
        "cache_read_tokens",
        "est_cost_usd",
    }


def assert_trace(payload: dict[str, Any]) -> None:
    """The trace is populated, well-formed, and ordered like the ladder."""
    trace = payload["escalation_trace"]
    assert trace, "the trace is the service's explainability contract; it is never empty"

    ranks = []
    for step in trace:
        assert set(step) == TRACE_COLUMNS
        assert step["tier"] in {tier.value for tier in Tier}
        assert 0.0 <= step["confidence"] <= 1.0
        assert step["latency_ms"] >= 0.0
        assert isinstance(step["succeeded"], bool)
        assert step["escalated_because"] is None or isinstance(step["escalated_because"], str)
        ranks.append(Tier(step["tier"]).rank)

    # Cheapest first, each rung at most once: the trace is a climb, not a set.
    assert ranks == sorted(ranks)
    assert len(ranks) == len(set(ranks))

    # Every rung but the last says why the ladder kept going; the last is where
    # it stopped, and "you didn't escalate" is said with null.
    for step in trace[:-1]:
        assert step["escalated_because"], f"{step['tier']} escalated without saying why"

    # The rung that answered is the deepest one that succeeded.
    decided = [step for step in trace if step["succeeded"]][-1]
    assert payload["decided_by"] == decided["tier"]


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("/v1/sentiment", SentimentResponse),
        ("/v1/feedback", FeedbackResponse),
        ("/v1/reviews", ReviewResponse),
    ],
)
def test_every_resource_returns_the_same_core_envelope(
    client: TestClient, path: str, model: type[Any]
) -> None:
    """One envelope, three resources — the property that makes a fourth cheap."""
    payload = post(client, path, text=MIXED)

    assert_core_envelope(payload)
    assert_trace(payload)
    # Nothing extra and nothing missing relative to the published model, so a
    # field cannot reach a caller without also reaching the OpenAPI document.
    assert set(payload) == set(model.model_fields)


@pytest.mark.parametrize("path", ["/v1/sentiment", "/v1/feedback", "/v1/reviews"])
def test_metadata_is_echoed_back_unmodified(client: TestClient, path: str) -> None:
    """`metadata` is the caller's, and it comes back as it went in."""
    sent = {"tenant": "acme", "attempt": 3, "nested": {"a": [1, 2]}}
    payload = post(client, path, text=MIXED, metadata=sent)

    assert payload["metadata"] == sent


# ── Resource-specific blocks ──────────────────────────────────────────────────


def test_sentiment_carries_no_resource_extras(client: TestClient) -> None:
    """The unopinionated base: the envelope and nothing else."""
    payload = post(client, "/v1/sentiment", text=MIXED)

    assert set(payload) == set(SentimentResponse.model_fields)
    assert not ({"urgency", "actionability", "suggested_route"} & set(payload))
    assert "rating_consistency" not in payload


def test_feedback_adds_its_triage_block(client: TestClient) -> None:
    payload = post(
        client,
        "/v1/feedback",
        text="The app crashes every time I try to pay. Fix this now.",
        channel="support",
    )

    assert_core_envelope(payload)
    assert payload["urgency"] in {"low", "medium", "high", "critical"}
    assert payload["actionability"] in {"informational", "actionable", "blocking"}
    assert payload["suggested_route"] is None or isinstance(payload["suggested_route"], str)
    # Triage is derived, never a tier's opinion, so it is present regardless of
    # how far the ladder climbed.
    assert "rating_consistency" not in payload


def test_a_rated_review_gets_rating_consistency(client: TestClient) -> None:
    """Prose against stars, on one shared -1..+1 scale."""
    payload = post(client, "/v1/reviews", text=MIXED, rating=3)

    assert_core_envelope(payload)
    consistency = payload["rating_consistency"]
    assert consistency is not None
    assert set(consistency) == {"agreement", "delta", "normalised_rating"}
    assert consistency["agreement"] in {"agrees", "prose_more_positive", "prose_more_negative"}
    # 3 of 5 stars → 0.2 on the polarity scale.
    assert consistency["normalised_rating"] == pytest.approx(0.2)
    assert consistency["delta"] == pytest.approx(
        payload["sentiment"]["polarity"] - consistency["normalised_rating"], abs=1e-3
    )


def test_an_unrated_review_gets_null_rating_consistency(client: TestClient) -> None:
    """Null rather than a guess: with no stars there is nothing to compare."""
    payload = post(client, "/v1/reviews", text=MIXED)

    assert_core_envelope(payload)
    assert "rating_consistency" in payload
    assert payload["rating_consistency"] is None


def test_the_rating_scale_is_respected(client: TestClient) -> None:
    """A 10-point scale normalises on its own maximum, not on 5."""
    payload = post(client, "/v1/reviews", text=MIXED, rating=10, rating_scale_max=10)

    consistency = payload["rating_consistency"]
    assert consistency is not None
    assert consistency["normalised_rating"] == pytest.approx(1.0)


# ── Trace suppression ─────────────────────────────────────────────────────────


def test_the_trace_can_be_suppressed(client: TestClient) -> None:
    """`include_trace: false` empties the trace without changing the verdict."""
    with_trace = post(client, "/v1/sentiment", text=MIXED)
    without = post(client, "/v1/sentiment", text=MIXED, options={"include_trace": False})

    assert without["escalation_trace"] == []
    # The field stays present — a caller's parser should not have to branch —
    # and the answer is unchanged.
    assert "escalation_trace" in without
    assert without["decided_by"] == with_trace["decided_by"]
    assert without["sentiment"]["label"] == with_trace["sentiment"]["label"]
