"""SummaryService: the rollup merge (pure) and the narrative refresh (guarded LLM call).

The rollup-merge tests go straight at the private module functions — the same
convention `test_analysis_service.py` uses for `_derive_*` — because they are
pure and deterministic, and testing them through the full `refresh()` call
would only add ladder noise around what is actually being asserted. The
narrative-refresh tests go through `SummaryService.refresh` itself, driven by
a scripted orchestrator and a scripted provider mirroring
`test_t2_local_llm.py`'s `StubProvider`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from air_classifier_service.config import Settings
from air_classifier_service.constants import Label, Provider, Tier, Topic, Urgency
from air_classifier_service.pipeline.orchestrator import Orchestrator
from air_classifier_service.pipeline.service import AnalysisService
from air_classifier_service.pipeline.summary_service import (
    SummaryService,
    _merge_rollup,
    _succeeded_pairs,
    _sum_usage,
    _to_classify_request,
)
from air_classifier_service.pipeline.tiers.base import TierContext
from air_classifier_service.providers.base import (
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier_service.resilience.breaker import CircuitBreaker
from air_classifier_service.resilience.bulkhead import Bulkhead
from air_classifier_service.schemas.classify import (
    ClassificationResponse,
    ClassifyBatchItem,
    ModelInfo,
    RoutingInfo,
)
from air_classifier_service.schemas.common import (
    LabeledScore,
    LanguageInfo,
    RequestUsage,
    SentimentResult,
)
from air_classifier_service.schemas.errors import ProblemDetail
from air_classifier_service.schemas.summary import (
    CustomerSummary,
    LLMSummaryOutput,
    SummaryItem,
    SummaryRefreshRequest,
    SummaryRollup,
)
from air_classifier_service.schemas.verdict import EscalationStep, OrchestrationResult, TierVerdict

TEXT_A = "a plain, undecorated piece of feedback text"


def _response(
    *,
    label: Label = Label.POSITIVE,
    urgency: Urgency = Urgency.LOW,
    topics: list[Topic] | None = None,
    usage: RequestUsage | None = None,
) -> ClassificationResponse:
    return ClassificationResponse(
        request_id="req_1",
        sentiment=SentimentResult(label=label, polarity=0.5, confidence=0.9),
        urgency=LabeledScore(label=urgency, confidence=1.0),
        topics=[LabeledScore(label=t, confidence=0.7) for t in (topics or [])],
        routing=RoutingInfo(priority="P4"),
        requires_human=False,
        actionability="informational",
        language=LanguageInfo(code="en", confidence=0.9),
        model=ModelInfo(name="test", version="0.0.0"),
        decided_by=Tier.T1_CLASSIFIER,
        usage=usage or RequestUsage(),
    )


def _item(
    text: str = "some feedback",
    *,
    source_type: str = "feedback",
    rating: float | None = None,
    occurred_at: datetime | None = None,
) -> SummaryItem:
    return SummaryItem(text=text, source_type=source_type, rating=rating, occurred_at=occurred_at)


def _batch_item(
    index: int, result: ClassificationResponse | None, *, ok: bool = True
) -> ClassifyBatchItem:
    return ClassifyBatchItem(
        index=index,
        ok=ok,
        result=result,
        error=None if ok else ProblemDetail.build(slug="x", title="x", status=500),
    )


# ── Rollup merge (pure) ─────────────────────────────────────────────────────


def test_merge_from_scratch_counts_the_first_batch() -> None:
    pairs = [
        (_item("a", source_type="feedback"), _response(label=Label.POSITIVE)),
        (_item("b", source_type="review", rating=4), _response(label=Label.NEGATIVE)),
    ]

    rollup = _merge_rollup(None, pairs)

    assert rollup.total_items == 2
    assert rollup.counts_by_source_type == {"feedback": 1, "review": 1}
    assert rollup.sentiment_counts == {Label.POSITIVE: 1, Label.NEGATIVE: 1}
    assert rollup.rating_count == 1
    assert rollup.average_rating == 4.0


def test_merge_accumulates_onto_an_existing_rollup() -> None:
    existing = SummaryRollup(total_items=1, sentiment_counts={Label.POSITIVE: 1})
    pairs = [(_item("a"), _response(label=Label.POSITIVE))]

    rollup = _merge_rollup(existing, pairs)

    assert rollup.total_items == 2
    assert rollup.sentiment_counts == {Label.POSITIVE: 2}
    # The existing rollup passed in is not mutated in place.
    assert existing.total_items == 1


def test_topic_histogram_is_never_truncated() -> None:
    """All six Topic members can survive across repeated merges — no top-N loss."""
    rollup: SummaryRollup | None = None
    for topic in Topic:
        rollup = _merge_rollup(rollup, [(_item(), _response(topics=[topic]))])

    assert rollup is not None
    assert set(rollup.topic_counts) == set(Topic)
    assert all(count == 1 for count in rollup.topic_counts.values())


def test_recent_ratings_window_slides_and_caps() -> None:
    rollup: SummaryRollup | None = None
    for rating in [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]:
        rollup = _merge_rollup(rollup, [(_item(rating=rating), _response())])

    assert rollup is not None
    assert rollup.recent_ratings == [2.0, 3.0, 4.0, 5.0, 5.0]
    assert rollup.rating_count == 6
    assert rollup.recent_average_rating == pytest.approx(sum([2, 3, 4, 5, 5]) / 5)


def test_first_and_last_seen_track_occurred_at() -> None:
    early = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 6, 1, tzinfo=UTC)
    pairs = [
        (_item("a", occurred_at=late), _response()),
        (_item("b", occurred_at=early), _response()),
    ]

    rollup = _merge_rollup(None, pairs)

    assert rollup.first_seen_at == early
    assert rollup.last_seen_at == late


def test_items_with_no_rating_do_not_touch_rating_fields() -> None:
    rollup = _merge_rollup(None, [(_item(rating=None), _response())])

    assert rollup.rating_count == 0
    assert rollup.average_rating is None
    assert rollup.recent_ratings == []


# ── Pairing: failure isolation and ordering ─────────────────────────────────


def test_failed_items_are_excluded_from_the_pairs() -> None:
    items = [_item("a"), _item("b")]
    batch = [_batch_item(0, _response()), _batch_item(1, None, ok=False)]

    pairs = _succeeded_pairs(items, batch)

    assert len(pairs) == 1
    assert pairs[0][0].text == "a"


def test_items_with_no_occurred_at_do_not_crash_the_sort() -> None:
    """Regression: sorting `None` against `None` must not raise TypeError."""
    items = [_item("a"), _item("b"), _item("c")]
    batch = [_batch_item(i, _response()) for i in range(3)]

    pairs = _succeeded_pairs(items, batch)

    assert len(pairs) == 3


def test_dated_items_sort_chronologically_and_undated_sort_last() -> None:
    now = datetime.now(UTC)
    items = [
        _item("undated"),
        _item("later", occurred_at=now),
        _item("earlier", occurred_at=now - timedelta(days=1)),
    ]
    batch = [_batch_item(i, _response()) for i in range(3)]

    pairs = _succeeded_pairs(items, batch)

    assert [item.text for item, _ in pairs] == ["earlier", "later", "undated"]


# ── _to_classify_request / _sum_usage ────────────────────────────────────────


def test_to_classify_request_carries_the_shared_fields() -> None:
    item = _item("hello", rating=4)
    request = _to_classify_request(item)

    assert request.text == "hello"
    assert request.rating == 4


def test_sum_usage_only_counts_succeeded_items() -> None:
    ok = _response(usage=RequestUsage(tokens_in=10, tokens_out=5))
    batch = [_batch_item(0, ok), _batch_item(1, None, ok=False)]

    usage = _sum_usage(batch)

    assert usage.tokens_in == 10
    assert usage.tokens_out == 5


# ── refresh(): narrative call, guarded ───────────────────────────────────────


class ScriptedOrchestrator:
    """Keyed by text rather than call order — `analyse_batch` runs concurrently."""

    def __init__(self, results_by_text: dict[str, OrchestrationResult]) -> None:
        self._results_by_text = results_by_text

    async def run(self, ctx: TierContext) -> OrchestrationResult:
        return self._results_by_text[ctx.text]


def _orchestration_result(label: Label = Label.POSITIVE) -> OrchestrationResult:
    """A minimal, always-usable ladder result.

    `AnalysisService._shape` derives the actual `ClassificationResponse`
    (urgency, routing, actionability, ...) from this verdict itself — the
    refresh tests below don't assert on those derived heuristics, only on
    what `SummaryService` does with whatever comes back.
    """
    verdict = TierVerdict(
        tier=Tier.T1_CLASSIFIER,
        result=SentimentResult(label=label, polarity=0.5, confidence=0.9),
    )
    step = EscalationStep(
        tier=verdict.tier,
        label=verdict.result.label,
        confidence=verdict.result.confidence,
        latency_ms=1.0,
        succeeded=True,
    )
    return OrchestrationResult(best=verdict, verdicts=[verdict], steps=[step], degraded=False)


class StubProvider:
    name: Provider = Provider.ANTHROPIC

    def __init__(self, *outcomes: ProviderResult | Exception, configured: bool = True) -> None:
        self._outcomes = list(outcomes)
        self._configured = configured
        self.calls: list[dict[str, Any]] = []

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def health_detail(self) -> str | None:
        return None

    async def health(self) -> bool:
        return True

    async def classify(self, **kwargs: Any) -> ProviderResult:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0) if self._outcomes else _provider_result()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


def _provider_result(narrative: str = "Updated narrative.") -> ProviderResult:
    return ProviderResult(
        parsed=LLMSummaryOutput(narrative=narrative),
        provider=Provider.ANTHROPIC,
        model="llama3.2:3b",
        tokens_in=10,
        tokens_out=5,
        latency_ms=5.0,
        cost_usd=0.0,
    )


def _service(
    texts: list[str], provider: StubProvider | None, *, settings: Settings | None = None
) -> SummaryService:
    resolved = settings or Settings()
    orchestrator = ScriptedOrchestrator({text: _orchestration_result() for text in texts})
    analysis = AnalysisService(cast("Orchestrator", orchestrator), resolved, None, None)
    return SummaryService(
        analysis,
        provider,
        resolved,
        bulkhead=Bulkhead("summary-test", limit=4, acquire_timeout_s=1.0),
        breaker=CircuitBreaker("summary-test", threshold=5, cooldown_s=30.0),
    )


async def test_first_time_summary_gets_version_one() -> None:
    service = _service([TEXT_A], StubProvider())
    request = SummaryRefreshRequest(
        customer_id="cust_1", items=[_item(TEXT_A, source_type="feedback")]
    )

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.summary.version == 1
    assert response.summary.rollup.total_items == 1
    assert response.degraded is False


async def test_the_narrative_call_forwards_the_request_id() -> None:
    provider = StubProvider()
    service = _service([TEXT_A], provider)
    request = SummaryRefreshRequest(
        customer_id="cust_1", items=[_item(TEXT_A, source_type="feedback")]
    )

    await service.refresh(request, request_id="req_1", options=None)

    assert [call["request_id"] for call in provider.calls] == ["req_1"]


async def test_narrative_refresh_success_updates_the_narrative() -> None:
    provider = StubProvider(_provider_result("Improving trend, one complaint resolved."))
    service = _service([TEXT_A], provider)
    request = SummaryRefreshRequest(customer_id="cust_1", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.summary.narrative == "Improving trend, one complaint resolved."
    assert response.degraded is False


async def test_one_schema_failure_is_repaired() -> None:
    provider = StubProvider(ProviderSchemaError("narrative: field required"), _provider_result())
    service = _service([TEXT_A], provider)
    request = SummaryRefreshRequest(customer_id="cust_1", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.degraded is False
    assert len(provider.calls) == 2


async def test_two_schema_failures_degrade_and_keep_the_prior_narrative() -> None:
    provider = StubProvider(ProviderSchemaError("bad once"), ProviderSchemaError("bad twice"))
    service = _service([TEXT_A], provider)
    existing = CustomerSummary(narrative="The prior narrative.")
    request = SummaryRefreshRequest(
        customer_id="cust_1", existing_summary=existing, items=[_item(TEXT_A)]
    )

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.degraded is True
    assert response.summary.narrative == "The prior narrative."
    assert response.summary.rollup.total_items == 1  # the rollup still updated


async def test_provider_unavailable_degrades() -> None:
    provider = StubProvider(ProviderUnavailableError("air-llm down"))
    service = _service([TEXT_A], provider)
    request = SummaryRefreshRequest(customer_id="cust_1", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.degraded is True


async def test_unconfigured_provider_degrades_without_calling_it() -> None:
    provider = StubProvider(configured=False)
    service = _service([TEXT_A], provider)
    request = SummaryRefreshRequest(customer_id="cust_1", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.degraded is True
    assert provider.calls == []


async def test_no_provider_at_all_degrades() -> None:
    """`main.py` always builds a `SummaryService`, provider or not."""
    service = _service([TEXT_A], None)
    request = SummaryRefreshRequest(customer_id="cust_1", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_1", options=None)

    assert response.degraded is True
    assert response.summary.rollup.total_items == 1  # rollup is unaffected


async def test_request_id_and_customer_id_are_echoed_back() -> None:
    service = _service([TEXT_A], StubProvider())
    request = SummaryRefreshRequest(customer_id="cust_42", items=[_item(TEXT_A)])

    response = await service.refresh(request, request_id="req_99", options=None)

    assert response.request_id == "req_99"
    assert response.customer_id == "cust_42"
