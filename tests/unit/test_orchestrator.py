"""The ladder, driven entirely by scripted fakes.

Every rung here is a :class:`FakeTier` with a programmed verdict, latency or
explosion. No real tier, no model, no network — which is the point: the
orchestrator's semantics are pinned while they are still cheap to change, and
they stay pinned when the real rungs arrive.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from air_classifier_service.config import AppSettings, Settings, TiersSettings
from air_classifier_service.constants import Label, Tier
from air_classifier_service.observability import metrics
from air_classifier_service.pipeline import orchestrator as orchestrator_module
from air_classifier_service.pipeline.orchestrator import NoVerdictError, Orchestrator
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.tiers.base import (
    Tier as TierProtocol,
)
from air_classifier_service.pipeline.tiers.base import (
    TierContext,
    TierError,
    TierUnavailableError,
)
from air_classifier_service.schemas.common import (
    ComplexityFeatures,
    LanguageInfo,
    Options,
    SafetyReport,
    SentimentResult,
    TierUsage,
)
from air_classifier_service.schemas.verdict import TierVerdict

# ── Scripted rungs ────────────────────────────────────────────────────────────


@dataclass
class FakeTier:
    """One rung of the ladder, scripted.

    Implements the :class:`Tier` protocol structurally rather than by
    inheritance — if this class stops satisfying it, the protocol changed and
    the real rungs need to change with it.
    """

    tier: Tier
    verdict: TierVerdict | None = None
    raises: Exception | None = None
    delay_s: float = 0.0
    available: bool = True
    probe_raises: bool = False
    model_version: str = "fake-1"
    #: Every context this rung was handed, in order — the immutability evidence.
    seen: list[TierContext] = field(default_factory=list)

    async def is_available(self) -> bool:
        if self.probe_raises:
            raise RuntimeError("probe exploded")
        return self.available

    async def unavailable_reason(self) -> str | None:
        return None if self.available else "scripted unavailable"

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        self.seen.append(ctx)
        if self.delay_s > 0.0:
            await asyncio.sleep(self.delay_s)
        if self.raises is not None:
            raise self.raises
        return self.verdict if self.verdict is not None else TierVerdict.abstained(self.tier)

    def supports_language(self, code: str) -> bool:
        return True

    @property
    def calls(self) -> int:
        return len(self.seen)


def verdict(
    tier: Tier,
    *,
    label: Label = Label.POSITIVE,
    confidence: float = 0.99,
    margin: float = 0.9,
    cost_usd: float = 0.0,
) -> TierVerdict:
    """A verdict that stops the ladder unless a test weakens it."""
    return TierVerdict(
        tier=tier,
        model_version="fake-1",
        result=SentimentResult(label=label, polarity=0.7, confidence=confidence),
        label_margin=margin,
        latency_ms=1.0,
        usage=TierUsage(cost_usd=cost_usd),
    )


def unsure(tier: Tier, **kwargs: float) -> TierVerdict:
    """A usable verdict that every rung's floor rejects, so the ladder climbs."""
    return verdict(tier, confidence=0.4, margin=0.05, **kwargs)


# ── Harness ───────────────────────────────────────────────────────────────────


def tuned_settings(*, timeout_ms: int = 1000) -> Settings:
    """Default thresholds, but per-rung timeouts a scripted fake can live with.

    The shipped T0 timeout is 5ms, which says something true about a lexicon
    scan and nothing useful about a coroutine on a loaded CI box.
    """
    base = TiersSettings()
    return Settings(
        app=AppSettings(env="test"),
        tiers=TiersSettings(
            t0_rules=base.t0_rules.model_copy(update={"timeout_ms": timeout_ms}),
            t1_classifier=base.t1_classifier.model_copy(update={"timeout_ms": timeout_ms}),
            t2_local_llm=base.t2_local_llm.model_copy(update={"timeout_ms": timeout_ms}),
            t3_cloud_llm=base.t3_cloud_llm.model_copy(update={"timeout_ms": timeout_ms}),
        ),
    )


def make_ctx(*, options: Options | None = None, budget_ms: float = 5000.0) -> TierContext:
    return TierContext(
        request_id="req_fake",
        text="the battery lasts all day",
        original_length=25,
        language=LanguageInfo(code="en", confidence=0.99),
        features=ComplexityFeatures(),
        safety=SafetyReport(),
        options=options or Options(),
        deadline_at=time.monotonic() + budget_ms / 1000.0,
    )


def build(*tiers: FakeTier, settings: Settings | None = None) -> Orchestrator:
    resolved = settings or tuned_settings()
    return Orchestrator(list(tiers), EscalationPolicy(resolved), resolved)


def sample(name: str, **labels: str) -> float:
    """Current value of one metric series, treating 'never observed' as zero."""
    value = metrics.get_registry().get_sample_value(name, labels)
    return 0.0 if value is None else value


# ── The protocol ──────────────────────────────────────────────────────────────


def test_the_fake_satisfies_the_tier_protocol() -> None:
    assert isinstance(FakeTier(tier=Tier.T0_RULES), TierProtocol)


# ── Ordering and stopping ─────────────────────────────────────────────────────


async def test_rungs_run_cheapest_first_regardless_of_registration_order() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=unsure(Tier.T1_CLASSIFIER))
    t2 = FakeTier(tier=Tier.T2_LOCAL_LLM, verdict=verdict(Tier.T2_LOCAL_LLM))

    result = await build(t2, t0, t1).run(make_ctx())

    assert [v.tier for v in result.verdicts] == [
        Tier.T0_RULES,
        Tier.T1_CLASSIFIER,
        Tier.T2_LOCAL_LLM,
    ]
    assert result.best.tier is Tier.T2_LOCAL_LLM
    assert result.degraded is False


async def test_a_confident_rung_ends_the_ladder() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=verdict(Tier.T0_RULES, confidence=0.99))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1).run(make_ctx())

    assert t1.calls == 0
    assert result.best.tier is Tier.T0_RULES
    assert result.steps[-1].escalated_because is None


async def test_the_trace_has_one_step_per_rung_that_ran() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=unsure(Tier.T1_CLASSIFIER))
    t2 = FakeTier(tier=Tier.T2_LOCAL_LLM, verdict=verdict(Tier.T2_LOCAL_LLM))
    t3 = FakeTier(tier=Tier.T3_CLOUD_LLM, verdict=verdict(Tier.T3_CLOUD_LLM))

    result = await build(t0, t1, t2, t3).run(make_ctx())

    assert [step.tier for step in result.steps] == [
        Tier.T0_RULES,
        Tier.T1_CLASSIFIER,
        Tier.T2_LOCAL_LLM,
    ]
    # T0 reports no runner-up distribution, so only its confidence is evidence.
    assert result.steps[0].escalated_because == "low_confidence"
    assert result.steps[1].escalated_because == "low_confidence,narrow_margin"
    assert result.steps[2].escalated_because is None
    assert t3.calls == 0


async def test_only_registered_rungs_are_run() -> None:
    """A ladder with holes in it is still a ladder, in the same order."""
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=unsure(Tier.T1_CLASSIFIER))
    t3 = FakeTier(tier=Tier.T3_CLOUD_LLM, verdict=verdict(Tier.T3_CLOUD_LLM))

    result = await build(t3, t1).run(make_ctx())

    assert [v.tier for v in result.verdicts] == [Tier.T1_CLASSIFIER, Tier.T3_CLOUD_LLM]


# ── Caller bounds ─────────────────────────────────────────────────────────────


async def test_min_tier_skips_the_cheaper_rungs_entirely() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=verdict(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))
    t2 = FakeTier(tier=Tier.T2_LOCAL_LLM, verdict=verdict(Tier.T2_LOCAL_LLM))

    result = await build(t0, t1, t2).run(make_ctx(options=Options(min_tier=Tier.T2_LOCAL_LLM)))

    assert (t0.calls, t1.calls, t2.calls) == (0, 0, 1)
    assert result.best.tier is Tier.T2_LOCAL_LLM


async def test_max_tier_is_a_hard_ceiling_even_for_a_poor_verdict() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=unsure(Tier.T1_CLASSIFIER))
    t2 = FakeTier(tier=Tier.T2_LOCAL_LLM, verdict=verdict(Tier.T2_LOCAL_LLM))

    result = await build(t0, t1, t2).run(make_ctx(options=Options(max_tier=Tier.T1_CLASSIFIER)))

    assert t2.calls == 0
    assert result.best.tier is Tier.T1_CLASSIFIER
    assert result.steps[-1].escalated_because is None


async def test_the_cost_ceiling_stops_the_ladder() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES, cost_usd=0.02))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1).run(make_ctx(options=Options(max_cost_usd=0.01)))

    assert t1.calls == 0
    assert result.best.tier is Tier.T0_RULES


# ── Budget ────────────────────────────────────────────────────────────────────


async def test_a_rung_is_not_started_when_the_budget_is_already_gone() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=verdict(Tier.T0_RULES))

    with pytest.raises(NoVerdictError):
        await build(t0).run(make_ctx(budget_ms=-1.0))

    assert t0.calls == 0


async def test_budget_exhaustion_mid_ladder_keeps_the_verdict_already_won() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER), delay_s=1.0)
    t2 = FakeTier(tier=Tier.T2_LOCAL_LLM, verdict=verdict(Tier.T2_LOCAL_LLM))

    result = await build(t0, t1, t2).run(make_ctx(budget_ms=60.0))

    assert t2.calls == 0
    assert result.degraded is True
    assert result.best.tier is Tier.T0_RULES


async def test_the_per_tier_timeout_is_clamped_by_the_remaining_budget() -> None:
    """The rung's own 1000ms timeout must not outlive a 50ms request budget."""
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=verdict(Tier.T0_RULES), delay_s=5.0)

    started = time.monotonic()
    with pytest.raises(NoVerdictError):
        await build(t0).run(make_ctx(budget_ms=50.0))

    assert time.monotonic() - started < 1.0


async def test_a_rung_that_overruns_its_own_timeout_fails_and_the_ladder_continues() -> None:
    settings = tuned_settings(timeout_ms=20)
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=verdict(Tier.T0_RULES), delay_s=5.0)
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1, settings=settings).run(make_ctx())

    assert result.degraded is True
    assert result.best.tier is Tier.T1_CLASSIFIER
    assert result.verdicts[0].succeeded is False
    assert "timed out" in (result.verdicts[0].error or "")


# ── Failure is not an exception ───────────────────────────────────────────────


async def test_a_tier_error_becomes_a_failed_verdict_and_degrades_the_response() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, raises=TierError("lexicon file missing"))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1).run(make_ctx())

    assert result.degraded is True
    assert result.best.tier is Tier.T1_CLASSIFIER
    assert result.steps[0].succeeded is False
    assert result.steps[0].escalated_because == "tier_failed"
    assert "lexicon file missing" in (result.verdicts[0].error or "")


async def test_an_unexpected_exception_is_contained_just_the_same() -> None:
    """A defect in a rung is still not the caller's problem."""
    t0 = FakeTier(tier=Tier.T0_RULES, raises=ValueError("someone forgot a None check"))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1).run(make_ctx())

    assert result.degraded is True
    assert result.best.tier is Tier.T1_CLASSIFIER
    assert "ValueError" in (result.verdicts[0].error or "")


async def test_a_rung_returning_a_failed_verdict_also_degrades() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=TierVerdict.failed(Tier.T0_RULES, "no shortcut"))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))

    result = await build(t0, t1).run(make_ctx())

    assert result.degraded is True


async def test_every_rung_failing_is_the_only_path_to_no_verdict() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, raises=TierError("down"))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, raises=TierUnavailableError("breaker open"))

    with pytest.raises(NoVerdictError):
        await build(t0, t1).run(make_ctx(options=Options(max_tier=Tier.T1_CLASSIFIER)))

    assert t1.calls == 1


async def test_an_abstaining_ladder_yields_no_verdict() -> None:
    """Abstention is not failure, but it is not an answer either."""
    t0 = FakeTier(tier=Tier.T0_RULES)
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER)

    with pytest.raises(NoVerdictError):
        await build(t0, t1).run(make_ctx(options=Options(max_tier=Tier.T1_CLASSIFIER)))


# ── Lookahead ─────────────────────────────────────────────────────────────────


async def test_an_unavailable_next_rung_stops_the_ladder_early() -> None:
    """Stopping here is not the same as stopping confidently.

    T0's own verdict is `unsure` — it wanted to climb — and the only thing
    that stopped it was T1 reporting unavailable. That is a capability gap,
    not a clean finish: `degraded` must say so, and unlike every other stop
    reason the trace does not fold it into `null`, because a caller (or
    DevOps, reading the same field) needs to be able to tell "confident" and
    "stuck below where the ladder wanted to be" apart.
    """
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER), available=False)

    result = await build(t0, t1).run(make_ctx())

    assert t1.calls == 0
    assert result.best.tier is Tier.T0_RULES
    assert result.steps[-1].escalated_because == "next_tier_unavailable"
    assert result.degraded is True


async def test_a_probe_that_raises_counts_as_unavailable() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER), probe_raises=True)

    result = await build(t0, t1).run(make_ctx())

    assert t1.calls == 0
    assert result.best.tier is Tier.T0_RULES


# ── Immutability ──────────────────────────────────────────────────────────────


async def test_context_is_threaded_immutably() -> None:
    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES, cost_usd=0.001))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))
    ctx = make_ctx()

    await build(t0, t1).run(ctx)

    assert t0.seen[0] is ctx
    assert [v.tier for v in t1.seen[0].prior_verdicts] == [Tier.T0_RULES]
    assert t1.seen[0].cost_spent_usd == pytest.approx(0.001)
    # The caller's context is unchanged by any of it.
    assert ctx.prior_verdicts == []
    assert ctx.cost_spent_usd == 0.0


# ── Instrumentation ───────────────────────────────────────────────────────────


async def test_the_ladder_emits_its_metrics() -> None:
    decided = sample("air_classifier_decided_by_total", tier="t1_classifier")
    climbed = sample(
        "air_classifier_escalations_total",
        from_tier="t0_rules",
        to_tier="t1_classifier",
        reason="low_confidence",
    )
    stopped = sample("air_classifier_tier_decisions_total", tier="t1_classifier", outcome="stop")
    timed = sample("air_classifier_tier_duration_seconds_count", tier="t0_rules")

    t0 = FakeTier(tier=Tier.T0_RULES, verdict=unsure(Tier.T0_RULES))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))
    await build(t0, t1).run(make_ctx())

    assert sample("air_classifier_decided_by_total", tier="t1_classifier") == decided + 1
    assert (
        sample(
            "air_classifier_escalations_total",
            from_tier="t0_rules",
            to_tier="t1_classifier",
            reason="low_confidence",
        )
        == climbed + 1
    )
    assert (
        sample("air_classifier_tier_decisions_total", tier="t1_classifier", outcome="stop")
        == stopped + 1
    )
    assert sample("air_classifier_tier_duration_seconds_count", tier="t0_rules") == timed + 1


async def test_a_failed_rung_is_counted_as_failed_not_as_an_escalation() -> None:
    def failures() -> float:
        return sample("air_classifier_tier_decisions_total", tier="t0_rules", outcome="failed")

    before = failures()
    t0 = FakeTier(tier=Tier.T0_RULES, raises=TierError("down"))
    t1 = FakeTier(tier=Tier.T1_CLASSIFIER, verdict=verdict(Tier.T1_CLASSIFIER))
    await build(t0, t1).run(make_ctx())

    assert failures() == before + 1


# ── Tier-agnosticism ──────────────────────────────────────────────────────────


def test_the_loop_never_names_a_concrete_rung() -> None:
    """The moment this fails, adding a fifth rung stops being a registration."""
    source = Path(inspect.getfile(orchestrator_module)).read_text(encoding="utf-8")

    for member in Tier:
        assert f"Tier.{member.name}" not in source
        assert member.value not in source
