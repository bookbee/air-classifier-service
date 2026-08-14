"""The HTTP-facing seam: `AnalysisService` and its per-resource shaping.

The orchestrator is a scripted stub throughout. That is the point of the seam —
everything asserted here is preprocessing, caching, response shaping and batch
isolation, none of which should depend on which rung answered or on anything
being reachable over a network.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from air_sentiment.cache.memory import MemoryCache
from air_sentiment.config import AppSettings, ClassifierSettings, Settings
from air_sentiment.constants import (
    Actionability,
    Label,
    RatingAgreement,
    ResourceKind,
    Tier,
    Urgency,
)
from air_sentiment.pipeline.orchestrator import NoVerdictError, Orchestrator
from air_sentiment.pipeline.service import (
    AnalysisService,
    _derive_route,
    _derive_urgency,
    _shape_feedback,
    _shape_review,
    _shape_sentiment,
)
from air_sentiment.pipeline.tiers.base import TierContext
from air_sentiment.schemas.common import (
    Aspect,
    Emotion,
    LanguageInfo,
    Options,
    RequestUsage,
    SafetyReport,
    SentimentResult,
    TierUsage,
)
from air_sentiment.schemas.errors import ProblemDetail
from air_sentiment.schemas.requests import FeedbackRequest, ReviewRequest, SentimentRequest
from air_sentiment.schemas.responses import BaseAnalysisResponse
from air_sentiment.schemas.verdict import (
    EscalationStep,
    OrchestrationResult,
    TierVerdict,
)

# ── Scripted ladder ───────────────────────────────────────────────────────────


class StubOrchestrator:
    """An orchestrator that returns what it was told to, and records what it saw.

    Typed as the real :class:`Orchestrator` at the call site via ``cast``: the
    service holds the concrete class, but touches only ``run``.
    """

    def __init__(
        self,
        result: OrchestrationResult | None = None,
        *,
        fail_on: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result if result is not None else make_result()
        self._fail_on = fail_on
        self._error = error or NoVerdictError("every rung failed")
        self.contexts: list[TierContext] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def run(self, ctx: TierContext) -> OrchestrationResult:
        self.contexts.append(ctx)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # A yield point, so bounded concurrency is actually observable.
            await asyncio.sleep(0)
            if self._fail_on is not None and self._fail_on in ctx.text:
                raise self._error
            return self._result
        finally:
            self.in_flight -= 1

    @property
    def calls(self) -> int:
        return len(self.contexts)


def make_verdict(
    *,
    tier: Tier = Tier.T1_CLASSIFIER,
    label: Label = Label.POSITIVE,
    polarity: float = 0.8,
    confidence: float = 0.9,
    aspects: list[Aspect] | None = None,
    emotions: list[Emotion] | None = None,
    usage: TierUsage | None = None,
) -> TierVerdict:
    return TierVerdict(
        tier=tier,
        model_version=f"{tier.value}-v1",
        result=SentimentResult(label=label, polarity=polarity, confidence=confidence),
        aspects=aspects or [],
        emotions=emotions or [],
        usage=usage or TierUsage(),
    )


def make_result(
    *,
    verdicts: list[TierVerdict] | None = None,
    degraded: bool = False,
) -> OrchestrationResult:
    chosen = verdicts or [make_verdict()]
    steps = [
        EscalationStep(
            tier=verdict.tier,
            label=verdict.result.label,
            confidence=verdict.result.confidence,
            latency_ms=1.0,
            succeeded=verdict.succeeded,
        )
        for verdict in chosen
    ]
    return OrchestrationResult(best=chosen[-1], verdicts=chosen, steps=steps, degraded=degraded)


def build_service(
    orchestrator: StubOrchestrator,
    settings: Settings,
    cache: MemoryCache | None = None,
) -> AnalysisService:
    return AnalysisService(cast("Orchestrator", orchestrator), settings, cache)


def envelope(
    *,
    sentiment: SentimentResult | None = None,
    aspects: list[Aspect] | None = None,
) -> BaseAnalysisResponse:
    """A minimal common envelope, for exercising the shapers on their own."""
    return BaseAnalysisResponse(
        request_id="req_test",
        sentiment=sentiment or SentimentResult(label=Label.POSITIVE, polarity=0.8, confidence=0.9),
        aspects=aspects or [],
        language=LanguageInfo(code="en", confidence=0.9),
        decided_by=Tier.T1_CLASSIFIER,
        safety=SafetyReport(),
        usage=RequestUsage(),
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────


async def test_redaction_happens_before_any_tier_sees_the_text(settings: Settings) -> None:
    """The whole reason redaction lives above the orchestrator."""
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    response = await service.analyse(
        SentimentRequest(text="Mail me at alice@example.com about the refund"),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(),
    )

    seen = stub.contexts[0].text
    assert "alice@example.com" not in seen
    assert "[REDACTED:email]" in seen
    assert response.safety.pii_redacted is True
    assert [finding.kind.value for finding in response.safety.pii_findings] == ["email"]


async def test_redaction_is_skipped_when_the_caller_opted_out(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    response = await service.analyse(
        SentimentRequest(text="Mail me at alice@example.com"),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(redact_pii=False),
    )

    assert "alice@example.com" in stub.contexts[0].text
    assert response.safety.pii_redacted is False


async def test_text_is_capped_at_the_classifier_window(settings: Settings) -> None:
    tight = settings.model_copy(update={"classifier": ClassifierSettings(max_input_chars=64)})
    stub = StubOrchestrator()
    service = build_service(stub, tight)

    original = "great " * 200
    response = await service.analyse(
        SentimentRequest(text=original),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(),
    )

    assert len(stub.contexts[0].text) <= 64
    assert stub.contexts[0].original_length == len(original)
    assert response.safety.truncated is True


async def test_resource_fields_carry_the_review_inputs(settings: Settings) -> None:
    """The tiers read `rating`/`rating_scale_max` out of `resource_fields`."""
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    await service.analyse(
        ReviewRequest(text="Sturdy and cheap", rating=4, rating_scale_max=5, title="Good buy"),
        resource=ResourceKind.REVIEWS,
        request_id="req_1",
        options=Options(),
    )

    fields = stub.contexts[0].resource_fields
    assert fields["rating"] == 4
    assert fields["rating_scale_max"] == 5
    assert fields["title"] == "Good buy"


async def test_latency_budget_becomes_a_deadline(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    await service.analyse(
        SentimentRequest(text="fine"),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(latency_budget_ms=500),
    )

    remaining = stub.contexts[0].remaining_ms
    assert 0 < remaining <= 500


# ── Envelope ──────────────────────────────────────────────────────────────────


async def test_envelope_reports_every_tier_that_ran(settings: Settings) -> None:
    verdicts = [
        make_verdict(tier=Tier.T0_RULES, label=Label.UNKNOWN, confidence=0.0),
        make_verdict(tier=Tier.T1_CLASSIFIER, usage=TierUsage(tokens_in=10, tokens_out=4)),
        make_verdict(
            tier=Tier.T2_LOCAL_LLM,
            usage=TierUsage(tokens_in=100, tokens_out=20, cost_usd=0.002),
        ),
    ]
    stub = StubOrchestrator(make_result(verdicts=verdicts))
    service = build_service(stub, settings)

    response = await service.analyse(
        SentimentRequest(text="mixed feelings, but mostly good"),
        resource=ResourceKind.SENTIMENT,
        request_id="req_9",
        options=Options(),
    )

    assert response.request_id == "req_9"
    assert response.decided_by is Tier.T2_LOCAL_LLM
    assert set(response.model_versions) == {"t0_rules", "t1_classifier", "t2_local_llm"}
    assert response.usage.tiers_run == 3
    assert response.usage.tokens_in == 110
    assert response.usage.tokens_out == 24
    assert response.usage.est_cost_usd == pytest.approx(0.002)
    assert response.latency_ms >= 0.0


async def test_metadata_is_echoed_verbatim(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings)
    metadata: dict[str, Any] = {"tenant": "acme", "nested": {"a": [1, 2]}, "n": 3}

    response = await service.analyse(
        SentimentRequest(text="all good", metadata=metadata),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(),
    )

    assert response.metadata == metadata


async def test_trace_is_included_by_default_and_suppressible(settings: Settings) -> None:
    verdicts = [
        make_verdict(tier=Tier.T0_RULES, label=Label.UNKNOWN, confidence=0.0),
        make_verdict(tier=Tier.T1_CLASSIFIER),
    ]
    stub = StubOrchestrator(make_result(verdicts=verdicts))
    service = build_service(stub, settings)
    request = SentimentRequest(text="good enough")

    with_trace = await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_1", options=Options()
    )
    assert [step.tier for step in with_trace.escalation_trace] == [
        Tier.T0_RULES,
        Tier.T1_CLASSIFIER,
    ]

    without = await service.analyse(
        request,
        resource=ResourceKind.SENTIMENT,
        request_id="req_2",
        options=Options(include_trace=False),
    )
    assert without.escalation_trace == []
    # Suppression is a shaping choice, not a different analysis.
    assert without.sentiment == with_trace.sentiment


async def test_degraded_flag_is_carried_through(settings: Settings) -> None:
    stub = StubOrchestrator(make_result(degraded=True))
    service = build_service(stub, settings)

    response = await service.analyse(
        SentimentRequest(text="ok"),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(),
    )

    assert response.degraded is True


# ── Cache ─────────────────────────────────────────────────────────────────────


async def test_cache_hit_skips_the_ladder(settings: Settings, memory_cache: MemoryCache) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings, memory_cache)
    request = SentimentRequest(text="the battery lasts for days")

    first = await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_1", options=Options()
    )
    second = await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_2", options=Options()
    )

    assert stub.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.request_id == "req_2"
    assert second.sentiment == first.sentiment
    assert second.decided_by == first.decided_by
    assert [step.tier for step in second.escalation_trace] == [
        step.tier for step in first.escalation_trace
    ]


async def test_cache_hit_reshapes_rather_than_replays(
    settings: Settings, memory_cache: MemoryCache
) -> None:
    """`include_trace` and `metadata` are per-request, not part of the key."""
    stub = StubOrchestrator()
    service = build_service(stub, settings, memory_cache)

    await service.analyse(
        SentimentRequest(text="same text", metadata={"caller": "first"}),
        resource=ResourceKind.SENTIMENT,
        request_id="req_1",
        options=Options(),
    )
    replayed = await service.analyse(
        SentimentRequest(text="same text", metadata={"caller": "second"}),
        resource=ResourceKind.SENTIMENT,
        request_id="req_2",
        options=Options(include_trace=False),
    )

    assert stub.calls == 1
    assert replayed.cached is True
    assert replayed.metadata == {"caller": "second"}
    assert replayed.escalation_trace == []


async def test_options_that_change_the_answer_change_the_key(
    settings: Settings, memory_cache: MemoryCache
) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings, memory_cache)
    request = SentimentRequest(text="worth every penny")

    await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_1", options=Options()
    )
    await service.analyse(
        request,
        resource=ResourceKind.SENTIMENT,
        request_id="req_2",
        options=Options(max_tier=Tier.T1_CLASSIFIER),
    )

    assert stub.calls == 2


async def test_degraded_results_are_not_cached(
    settings: Settings, memory_cache: MemoryCache
) -> None:
    """A provider blip must not pin the fallback answer for the whole TTL."""
    stub = StubOrchestrator(make_result(degraded=True))
    service = build_service(stub, settings, memory_cache)
    request = SentimentRequest(text="cannot fault it")

    await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_1", options=Options()
    )
    await service.analyse(
        request, resource=ResourceKind.SENTIMENT, request_id="req_2", options=Options()
    )

    assert stub.calls == 2


async def test_no_cache_backend_is_a_supported_shape(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings, None)
    request = SentimentRequest(text="no cache here")

    for index in range(2):
        result = await service.analyse(
            request,
            resource=ResourceKind.SENTIMENT,
            request_id=f"req_{index}",
            options=Options(),
        )
        assert result.cached is False
    assert stub.calls == 2


# ── Batch ─────────────────────────────────────────────────────────────────────


async def test_batch_isolates_one_bad_item(settings: Settings) -> None:
    """One item raising must not cost the caller its siblings' answers."""
    stub = StubOrchestrator(fail_on="boom")
    service = build_service(stub, settings)

    items = await service.analyse_batch(
        [
            SentimentRequest(text="first is fine"),
            SentimentRequest(text="second goes boom"),
            SentimentRequest(text="third is fine"),
        ],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
        instance="/v1/sentiment/batch",
    )

    assert [item.index for item in items] == [0, 1, 2]
    assert [item.ok for item in items] == [True, False, True]
    assert items[0].result is not None
    assert items[2].result is not None

    problem = items[1].error
    assert problem is not None
    assert problem.status == 503
    assert problem.type.endswith("/no-verdict")
    assert problem.instance == "/v1/sentiment/batch"
    assert problem.request_id == "req_batch"
    assert items[1].result is None


async def test_batch_hides_unexpected_failures_behind_a_500(settings: Settings) -> None:
    stub = StubOrchestrator(fail_on="boom", error=RuntimeError("psycopg2://user:pw@host"))
    service = build_service(stub, settings)

    items = await service.analyse_batch(
        [SentimentRequest(text="boom")],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
    )

    problem = items[0].error
    assert problem is not None
    assert problem.status == 500
    assert problem.detail is not None
    assert "psycopg2" not in problem.detail


async def test_batch_lets_an_exception_render_its_own_problem(settings: Settings) -> None:
    """How a 403 raised by the caller-supplied clamp reaches the right slot."""

    class PolicyRefusalError(Exception):
        def to_problem(
            self, *, instance: str | None = None, request_id: str | None = None
        ) -> ProblemDetail:
            return ProblemDetail.build(
                slug="insufficient-scope",
                title="Insufficient scope",
                status=403,
                instance=instance,
                request_id=request_id,
            )

    def refuse(options: Options) -> Options:
        raise PolicyRefusalError

    stub = StubOrchestrator()
    service = build_service(stub, settings)

    items = await service.analyse_batch(
        [SentimentRequest(text="anything")],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
        clamp=refuse,
    )

    problem = items[0].error
    assert problem is not None
    assert problem.status == 403
    assert problem.request_id == "req_batch"
    assert stub.calls == 0


async def test_batch_concurrency_is_bounded(settings: Settings) -> None:
    bounded = settings.model_copy(update={"app": AppSettings(env="test", batch_concurrency=2)})
    stub = StubOrchestrator()
    service = build_service(stub, bounded)

    items = await service.analyse_batch(
        [SentimentRequest(text=f"item number {i}") for i in range(12)],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
    )

    assert len(items) == 12
    assert all(item.ok for item in items)
    assert stub.max_in_flight <= 2


async def test_batch_item_options_beat_the_batch_default(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    await service.analyse_batch(
        [
            SentimentRequest(text="takes the default"),
            SentimentRequest(text="brings its own", options=Options(max_tier=Tier.T0_RULES)),
        ],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
        options=Options(max_tier=Tier.T2_LOCAL_LLM),
    )

    by_text = {ctx.text: ctx.options.max_tier for ctx in stub.contexts}
    assert by_text["takes the default"] is Tier.T2_LOCAL_LLM
    assert by_text["brings its own"] is Tier.T0_RULES


async def test_batch_clamp_applies_to_item_options(settings: Settings) -> None:
    """An item's own options are a way to ask for less, never for more."""
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    def ceiling(options: Options) -> Options:
        return options.model_copy(update={"max_tier": Tier.T1_CLASSIFIER})

    await service.analyse_batch(
        [SentimentRequest(text="reach for the cloud", options=Options(max_tier=Tier.T3_CLOUD_LLM))],
        resource=ResourceKind.SENTIMENT,
        request_id="req_batch",
        clamp=ceiling,
    )

    assert stub.contexts[0].options.max_tier is Tier.T1_CLASSIFIER


async def test_batch_returns_the_resources_own_response_type(settings: Settings) -> None:
    stub = StubOrchestrator()
    service = build_service(stub, settings)

    items = await service.analyse_batch(
        [ReviewRequest(text="Solid build", rating=5, rating_scale_max=5)],
        resource=ResourceKind.REVIEWS,
        request_id="req_batch",
    )

    result = items[0].result
    assert result is not None
    assert result.rating_consistency is not None


# ── Feedback shaping ──────────────────────────────────────────────────────────


async def test_feedback_response_carries_derived_triage(settings: Settings) -> None:
    stub = StubOrchestrator(
        make_result(verdicts=[make_verdict(label=Label.NEGATIVE, polarity=-0.8)])
    )
    service = build_service(stub, settings)

    response = await service.analyse(
        FeedbackRequest(text="Total outage, none of our team can log in", channel="support"),
        resource=ResourceKind.FEEDBACK,
        request_id="req_1",
        options=Options(),
    )

    assert response.urgency is Urgency.CRITICAL
    assert response.actionability is Actionability.BLOCKING
    assert response.suggested_route == "reliability"


@pytest.mark.parametrize(
    ("label", "polarity", "text", "channel", "expected"),
    [
        (Label.POSITIVE, 0.9, "Love the new dashboard", None, Urgency.LOW),
        (Label.NEUTRAL, 0.0, "The interface is fine", None, Urgency.LOW),
        (Label.NEGATIVE, -0.4, "The checkout flow is confusing", None, Urgency.MEDIUM),
        (Label.NEGATIVE, -0.9, "Everything is fine but slow", None, Urgency.HIGH),
        (Label.NEGATIVE, -0.5, "There has been an outage all morning", None, Urgency.CRITICAL),
        (Label.POSITIVE, 0.6, "I was charged twice", None, Urgency.CRITICAL),
    ],
)
def test_urgency_derivation(
    label: Label, polarity: float, text: str, channel: str | None, expected: Urgency
) -> None:
    sentiment = SentimentResult(label=label, polarity=polarity, confidence=0.9)
    assert _derive_urgency(sentiment=sentiment, text=text, channel=channel) is expected


def test_channel_raises_and_lowers_urgency_symmetrically() -> None:
    """A ticket someone opened outranks the same words in a solicited survey."""
    sentiment = SentimentResult(label=Label.NEGATIVE, polarity=-0.5, confidence=0.9)
    text = "The export is broken"

    assert _derive_urgency(sentiment=sentiment, text=text, channel=None) is Urgency.HIGH
    assert _derive_urgency(sentiment=sentiment, text=text, channel="support") is Urgency.CRITICAL
    assert _derive_urgency(sentiment=sentiment, text=text, channel="survey") is Urgency.MEDIUM


def test_cue_matching_respects_word_boundaries() -> None:
    """`download` is not `down`, and a substring match would make it one."""
    sentiment = SentimentResult(label=Label.POSITIVE, polarity=0.7, confidence=0.9)
    assert (
        _derive_urgency(sentiment=sentiment, text="The download was quick", channel=None)
        is Urgency.LOW
    )


def test_actionability_grades_praise_and_silence_apart() -> None:
    praise = envelope(sentiment=SentimentResult(label=Label.POSITIVE, polarity=0.9, confidence=0.9))
    shaped = _shape_feedback(praise, text="Genuinely delightful", channel=None)
    assert shaped.actionability is Actionability.INFORMATIONAL

    neutral = envelope(sentiment=SentimentResult(label=Label.NEUTRAL, polarity=0.0, confidence=0.6))
    quiet = _shape_feedback(neutral, text="Received the newsletter", channel=None)
    assert quiet.actionability is Actionability.NONE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I was charged twice and still want a refund", "billing"),
        ("Someone got into my account, my password leaked", "security"),
        ("The parcel never arrived, terrible delivery", "shipping"),
        ("Nothing in particular to report", None),
    ],
)
def test_route_derivation(text: str, expected: str | None) -> None:
    assert _derive_route(text=text, aspects=[]) == expected


def test_route_derivation_reads_aspect_names_too() -> None:
    """A tier that named the facet has already done this matching, with context."""
    aspects = [Aspect(name="delivery", label=Label.NEGATIVE, confidence=0.8)]
    assert _derive_route(text="Took forever to turn up", aspects=aspects) == "shipping"


# ── Review shaping ────────────────────────────────────────────────────────────


def test_rating_consistency_agrees() -> None:
    base = envelope(sentiment=SentimentResult(label=Label.POSITIVE, polarity=0.9, confidence=0.9))
    shaped = _shape_review(base, rating=5, rating_scale_max=5)

    assert shaped.rating_consistency is not None
    assert shaped.rating_consistency.agreement is RatingAgreement.AGREES
    assert shaped.rating_consistency.normalised_rating == pytest.approx(1.0)
    assert shaped.rating_consistency.delta == pytest.approx(-0.1)


def test_rating_consistency_prose_more_positive() -> None:
    base = envelope(sentiment=SentimentResult(label=Label.POSITIVE, polarity=0.8, confidence=0.9))
    shaped = _shape_review(base, rating=2, rating_scale_max=5)

    assert shaped.rating_consistency is not None
    assert shaped.rating_consistency.agreement is RatingAgreement.PROSE_MORE_POSITIVE
    assert shaped.rating_consistency.normalised_rating == pytest.approx(-0.2)
    assert shaped.rating_consistency.delta == pytest.approx(1.0)


def test_rating_consistency_prose_more_negative() -> None:
    """Five stars and a scathing paragraph — a mis-click, or sarcasm."""
    base = envelope(sentiment=SentimentResult(label=Label.NEGATIVE, polarity=-0.3, confidence=0.9))
    shaped = _shape_review(base, rating=5, rating_scale_max=5)

    assert shaped.rating_consistency is not None
    assert shaped.rating_consistency.agreement is RatingAgreement.PROSE_MORE_NEGATIVE
    assert shaped.rating_consistency.delta == pytest.approx(-1.3)


def test_rating_consistency_is_null_without_a_rating() -> None:
    shaped = _shape_review(envelope(), rating=None, rating_scale_max=5)
    assert shaped.rating_consistency is None


def test_rating_consistency_handles_a_non_five_scale() -> None:
    base = envelope(sentiment=SentimentResult(label=Label.POSITIVE, polarity=0.5, confidence=0.9))
    shaped = _shape_review(base, rating=75, rating_scale_max=100)

    assert shaped.rating_consistency is not None
    assert shaped.rating_consistency.normalised_rating == pytest.approx(0.5)
    assert shaped.rating_consistency.agreement is RatingAgreement.AGREES


async def test_review_rating_reaches_the_shaper_from_the_request(settings: Settings) -> None:
    stub = StubOrchestrator(
        make_result(verdicts=[make_verdict(label=Label.NEGATIVE, polarity=-0.6)])
    )
    service = build_service(stub, settings)

    response = await service.analyse(
        ReviewRequest(text="Broke after a week", rating=5, rating_scale_max=5),
        resource=ResourceKind.REVIEWS,
        request_id="req_1",
        options=Options(),
    )

    assert response.rating_consistency is not None
    assert response.rating_consistency.agreement is RatingAgreement.PROSE_MORE_NEGATIVE


# ── Sentiment shaping ─────────────────────────────────────────────────────────


def test_sentiment_shaper_is_the_envelope_unchanged() -> None:
    base = envelope(aspects=[Aspect(name="battery", label=Label.POSITIVE, confidence=0.7)])
    shaped = _shape_sentiment(base)

    assert shaped.model_dump() == base.model_dump()


async def test_no_verdict_propagates_from_a_single_call(settings: Settings) -> None:
    """The service's only path to a 5xx; the route's handler owns the mapping."""
    stub = StubOrchestrator(fail_on="anything")
    service = build_service(stub, settings)

    with pytest.raises(NoVerdictError):
        await service.analyse(
            SentimentRequest(text="anything at all"),
            resource=ResourceKind.SENTIMENT,
            request_id="req_1",
            options=Options(),
        )
