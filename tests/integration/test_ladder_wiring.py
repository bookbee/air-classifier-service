"""The ladder, assembled.

Every other test in this suite exercises one rung, one policy rule or one
provider adapter behind a fake. This file is the first that puts the real
Tier 0 and Tier 1 implementations, the real :class:`EscalationPolicy` and the
real :class:`Orchestrator` together and drives text through them, so it is the
only place that can catch the failures nobody sees from inside a single module:
a constructor the orchestrator cannot call, a tier that misses the protocol by
one method, a trace whose rows do not line up with the verdicts that produced
them.

Tiers 2 and 3 are scripted stubs rather than the real rungs, and that is a hard
requirement rather than a convenience. ``LocalLLMTier.is_available()`` issues a
real HTTP request to air-model-gateway's configured base URL, so a "real" wiring test
would pass or fail depending on whether the developer running it happens to
have air-model-gateway up. Nothing here opens a socket.

The inputs are built by the same preprocessing the API layer runs — normalise,
detect language, score the lexicon, extract features — because the features
object is what drives escalation rules 9 and 11. Handing the ladder a default
``ComplexityFeatures()`` would test the loop while quietly disabling half the
policy.
"""

from __future__ import annotations

import time

import pytest

from air_classifier_service.config import Settings
from air_classifier_service.constants import EscalationReason, Label, Provider
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.pipeline.features import extract_features
from air_classifier_service.pipeline.orchestrator import NoVerdictError, Orchestrator
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.tiers.base import (
    Tier,
    TierContext,
    TierError,
    TierUnavailableError,
)
from air_classifier_service.pipeline.tiers.scorers import ScoreResult
from air_classifier_service.pipeline.tiers.t0_rules import RulesTier
from air_classifier_service.pipeline.tiers.t1_classifier import ClassifierTier
from air_classifier_service.pipeline.tiers.t2_local_llm import LocalLLMTier
from air_classifier_service.pipeline.tiers.t3_cloud_llm import CloudLLMTier
from air_classifier_service.preprocess.language import detect_language
from air_classifier_service.preprocess.lexicon import score_text
from air_classifier_service.preprocess.normalize import normalize
from air_classifier_service.providers.base import LLMProvider
from air_classifier_service.providers.model_gateway_provider import ModelGatewayProvider
from air_classifier_service.schemas.common import Options, SafetyReport, SentimentResult
from air_classifier_service.schemas.verdict import OrchestrationResult, TierVerdict

pytestmark = pytest.mark.integration

# ── Building a request the way the API layer will ─────────────────────────────


def build_context(
    settings: Settings,
    text: str,
    *,
    options: Options | None = None,
    budget_s: float = 30.0,
    **resource_fields: object,
) -> TierContext:
    """Run the real preprocessing chain and hand back what the ladder receives.

    ``budget_s`` is generous on purpose: this file asserts on routing, and a
    deadline tight enough to expire mid-ladder would turn every assertion into a
    race against the machine the suite happens to run on. Budget exhaustion has
    its own test, which forces the condition rather than hoping for it.
    """
    normalised = normalize(text, max_chars=settings.classifier.max_input_chars)
    lexicon = score_text(normalised.text)
    return TierContext(
        request_id="req_ladder_wiring",
        text=normalised.text,
        original_length=len(text),
        language=detect_language(normalised.text),
        features=extract_features(normalised.text, lexicon_score=lexicon),
        safety=SafetyReport(truncated=normalised.truncated),
        resource_fields=resource_fields,
        options=options or Options(),
        deadline_at=time.monotonic() + budget_s,
    )


class ScriptedTier:
    """A rung that answers from a script instead of a model.

    Deliberately written as a plain class rather than a subclass of anything:
    if it satisfies :class:`Tier` it does so structurally, which is the same
    thing the real rungs have to do.
    """

    def __init__(
        self,
        tier: TierId,
        *,
        label: Label = Label.MIXED,
        confidence: float = 0.9,
        polarity: float = 0.0,
        available: bool = True,
        raises: Exception | None = None,
    ) -> None:
        self.tier = tier
        self.model_version = f"scripted/{tier.value}"
        self._label = label
        self._confidence = confidence
        self._polarity = polarity
        self._available = available
        self._raises = raises
        self.calls = 0

    async def is_available(self) -> bool:
        return self._available

    async def unavailable_reason(self) -> str | None:
        return None if self._available else "scripted unavailable"

    def supports_language(self, code: str) -> bool:
        return True

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return TierVerdict(
            tier=self.tier,
            model_version=self.model_version,
            result=SentimentResult(
                label=self._label, polarity=self._polarity, confidence=self._confidence
            ),
            # The same derivation the LLM rungs use, so a scripted verdict clears
            # the same margin floor a real one would.
            label_margin=max(0.0, min(1.0, (self._confidence - 0.5) * 2.0)),
        )


def build_ladder(settings: Settings, *tiers: Tier) -> Orchestrator:
    return Orchestrator(list(tiers), EscalationPolicy(settings), settings)


def full_ladder(
    settings: Settings, *, classifier: ClassifierTier | None = None
) -> tuple[Orchestrator, ScriptedTier, ScriptedTier]:
    """Real T0 and T1, scripted T2 and T3 — the standard arrangement here.

    T1 is the real tier on whatever backend `build_scorer` finds, which in a
    checkout without the ONNX artifact is the lexicon fallback.
    """
    local = ScriptedTier(TierId.T2_LOCAL_LLM, label=Label.MIXED, confidence=0.81)
    cloud = ScriptedTier(TierId.T3_CLOUD_LLM, label=Label.MIXED, confidence=0.93)
    orchestrator = build_ladder(
        settings, RulesTier(settings), classifier or ClassifierTier(settings), local, cloud
    )
    return orchestrator, local, cloud


class ConfidentModelScorer:
    """Stands in for the ONNX model: a real backend, sure of a clear sentence."""

    name = "stub-model-1.0"
    available = True

    def score(self, text: str) -> ScoreResult:
        return ScoreResult.from_probabilities(
            {Label.POSITIVE: 0.96, Label.NEGATIVE: 0.02, Label.NEUTRAL: 0.02}
        )


# ── Coherence of the trace ────────────────────────────────────────────────────


def assert_trace_is_coherent(result: OrchestrationResult) -> None:
    """Invariants the caller-visible trace must hold whatever the ladder did.

    These are the properties an API consumer reads the trace for. Every one of
    them is a cross-module claim — the orchestrator builds the rows, the policy
    supplies the reasons, and the tiers supply the verdicts — so none of them
    can be established by a unit test of any single module.
    """
    assert len(result.steps) == len(result.verdicts), "one trace row per rung that ran"

    for step, verdict in zip(result.steps, result.verdicts, strict=True):
        assert step.tier is verdict.tier
        assert step.label is verdict.result.label
        assert step.succeeded == verdict.succeeded
        assert step.confidence == pytest.approx(verdict.result.confidence)

    ranks = [step.tier.rank for step in result.steps]
    assert ranks == sorted(set(ranks)), "rungs run once each, cheapest first"

    # A non-terminal row exists precisely because the ladder climbed past it, so
    # it must say why. Only the last row may be silent.
    for step in result.steps[:-1]:
        assert step.escalated_because, f"{step.tier.value} escalated without a reason"

    assert result.best in result.verdicts
    assert result.best.is_usable
    usable = [v for v in result.verdicts if v.is_usable]
    assert result.best.tier.rank == max(v.tier.rank for v in usable), (
        "the dearest rung that committed to a label wins"
    )
    assert result.degraded == any(not v.succeeded for v in result.verdicts)


# ── The composition claims ────────────────────────────────────────────────────


def test_every_real_rung_satisfies_the_tier_protocol(settings: Settings) -> None:
    """Structural conformance, checked at runtime against all four real classes.

    ``mypy`` already proves this for the rungs that assert it internally, but
    only for the ones that remembered to. The runtime check is uniform and it is
    what the orchestrator actually depends on.

    The Tier 2 and Tier 3 rungs are *constructed* here but never probed:
    construction touches nothing, whereas ``is_available()`` on Tier 2 would open
    a socket to a local daemon.
    """
    candidates: list[object] = [
        RulesTier(settings),
        ClassifierTier(settings),
        LocalLLMTier(settings, ModelGatewayProvider(settings, Provider.OLLAMA)),
        CloudLLMTier(settings, ModelGatewayProvider(settings, Provider.ANTHROPIC)),
    ]

    rungs: list[Tier] = []
    for candidate in candidates:
        assert isinstance(candidate, Tier), f"{type(candidate).__name__} does not satisfy Tier"
        rungs.append(candidate)

    assert {rung.tier for rung in rungs} == set(TierId.ordered()), (
        "the four rungs must cover the ladder exactly once each"
    )
    for rung in rungs:
        assert rung.model_version, f"{type(rung).__name__} reports no model_version"


def test_every_provider_satisfies_the_llm_provider_protocol(settings: Settings) -> None:
    """Constructors only — no ``health()``, so no network from this test."""
    for provider in (
        ModelGatewayProvider(settings, Provider.OLLAMA),
        ModelGatewayProvider(settings, Provider.ANTHROPIC),
    ):
        assert isinstance(provider, LLMProvider), f"{type(provider).__name__} is not an LLMProvider"


async def test_registration_order_does_not_decide_the_climbing_order(
    settings: Settings,
) -> None:
    """The policy orders the ladder; the constructor argument merely supplies it.

    The rungs are registered backwards here. If the loop ever iterated its own
    argument, the ladder would start at Tier 3 and this would be a one-step run
    ending at the most expensive rung.
    """
    local = ScriptedTier(TierId.T2_LOCAL_LLM, label=Label.MIXED, confidence=0.81)
    cloud = ScriptedTier(TierId.T3_CLOUD_LLM, label=Label.MIXED, confidence=0.93)
    orchestrator = build_ladder(
        settings, cloud, local, ClassifierTier(settings), RulesTier(settings)
    )

    result = await orchestrator.run(
        build_context(settings, "The battery life is great but the camera is awful.")
    )

    assert [step.tier for step in result.steps] == list(TierId.ordered())
    assert_trace_is_coherent(result)


# ── End to end ────────────────────────────────────────────────────────────────


async def test_a_strong_one_word_verdict_stops_at_tier_zero(settings: Settings) -> None:
    """The cheapest rung answers, and nothing above it is consulted or billed."""
    orchestrator, local, cloud = full_ladder(settings)

    result = await orchestrator.run(build_context(settings, "Terrible."))

    assert result.best.tier is TierId.T0_RULES
    assert result.best.result.label is Label.NEGATIVE
    assert result.best.detail["rule"] == "lexicon"
    assert [step.tier for step in result.steps] == [TierId.T0_RULES]
    assert result.steps[0].escalated_because is None, "a terminal row explains nothing"
    assert local.calls == cloud.calls == 0
    assert result.degraded is False
    assert_trace_is_coherent(result)


async def test_a_trivially_positive_sentence_stops_at_the_classifier(
    settings: Settings,
) -> None:
    """Tier 0 has no shortcut for a seven-token sentence, so Tier 1 settles it.

    The point of the assertion on ``local.calls`` is cost: this is the shape of
    input that must never reach an LLM, because paying for one would mean the
    two cheap rungs are not earning their place. That claim is about a
    model-backed Tier 1; the lexicon fallback deliberately does not settle it
    (next test).
    """
    classifier = ClassifierTier(settings, scorer=ConfidentModelScorer())
    orchestrator, local, cloud = full_ladder(settings, classifier=classifier)

    result = await orchestrator.run(
        build_context(settings, "Absolutely fantastic, superb service, I love it.")
    )

    assert [step.tier for step in result.steps] == [TierId.T0_RULES, TierId.T1_CLASSIFIER]
    assert result.steps[0].label is Label.UNKNOWN
    assert result.steps[0].escalated_because == EscalationReason.NO_SHORTCUT_MATCHED.value

    assert result.best.tier is TierId.T1_CLASSIFIER
    assert result.best.result.label is Label.POSITIVE
    floors = settings.tiers.for_tier(TierId.T1_CLASSIFIER)
    assert result.best.result.confidence >= floors.confidence_floor
    assert result.best.label_margin >= floors.margin_floor
    assert result.degraded is False
    assert local.calls == cloud.calls == 0
    assert_trace_is_coherent(result)


async def test_on_the_lexicon_fallback_the_same_sentence_climbs(settings: Settings) -> None:
    """SIT CL-005: the fallback never stops the ladder on its own say-so."""
    orchestrator, local, _cloud = full_ladder(settings)

    result = await orchestrator.run(
        build_context(settings, "Absolutely fantastic, superb service, I love it.")
    )

    assert local.calls == 1
    assert result.steps[1].escalated_because is not None
    assert "low_confidence" in result.steps[1].escalated_because
    # The scripted LLMs answer `mixed`, so T2 disagrees with T1 and T3 settles it;
    # either way an LLM decided, not the fallback.
    assert result.best.tier.rank >= TierId.T2_LOCAL_LLM.rank
    assert result.degraded is False
    assert_trace_is_coherent(result)


async def test_a_lexicon_verdict_that_has_to_be_returned_is_degraded(
    settings: Settings,
) -> None:
    """With no LLM rung to climb to, the fallback's answer stands — marked degraded."""
    orchestrator = build_ladder(settings, RulesTier(settings), ClassifierTier(settings))

    result = await orchestrator.run(
        build_context(settings, "Absolutely fantastic, superb service, I love it.")
    )

    assert result.best.tier is TierId.T1_CLASSIFIER
    assert result.best.result.label is Label.POSITIVE
    assert result.degraded is True


async def test_a_contrastive_sentence_climbs_the_whole_ladder(settings: Settings) -> None:
    """Two strong signals in one sentence is the case the ladder exists for.

    It also pins a consequence of rule 9 that is easy to miss: the complexity
    triggers read the request's features, not the deciding rung's competence, so
    a contrastive sentence keeps climbing even after Tier 2 answers it *above*
    its own confidence and margin floors. The ladder is bounded by the ceiling
    rather than by satisfaction.
    """
    orchestrator, local, cloud = full_ladder(settings)
    ctx = build_context(settings, "The battery life is great but the camera is awful.")
    assert ctx.features.contrastive_count > 0 and ctx.features.polarity_flip

    result = await orchestrator.run(ctx)

    assert [step.tier for step in result.steps] == list(TierId.ordered())
    assert local.calls == cloud.calls == 1

    classifier_step = result.steps[1]
    assert classifier_step.escalated_because is not None
    reasons = classifier_step.escalated_because.split(",")
    assert EscalationReason.CONTRASTIVE_CONJUNCTION.value in reasons
    assert EscalationReason.POLARITY_FLIP.value in reasons

    local_reasons = (result.steps[2].escalated_because or "").split(",")
    assert EscalationReason.CONTRASTIVE_CONJUNCTION.value in local_reasons
    local_verdict = result.verdicts[2]
    assert (
        local_verdict.result.confidence
        >= settings.tiers.for_tier(TierId.T2_LOCAL_LLM).confidence_floor
    ), "climbed on structure, not on doubt"

    assert result.best.tier is TierId.T3_CLOUD_LLM
    assert result.steps[-1].escalated_because is None, "the ceiling is a stop, not a reason"
    assert_trace_is_coherent(result)


async def test_decided_by_is_the_verdict_the_caller_is_actually_given(
    settings: Settings,
) -> None:
    """`best` must be the winning verdict itself, not a rung id that resembles it.

    Tier 3 is scripted to abstain, which makes the dearest rung the one that did
    *not* decide — the case where an implementation that trusted "last rung run"
    would hand back the wrong answer.
    """
    local = ScriptedTier(TierId.T2_LOCAL_LLM, label=Label.NEGATIVE, confidence=0.88, polarity=-0.7)
    cloud = ScriptedTier(TierId.T3_CLOUD_LLM, label=Label.UNKNOWN, confidence=0.0)
    orchestrator = build_ladder(
        settings, RulesTier(settings), ClassifierTier(settings), local, cloud
    )

    result = await orchestrator.run(
        build_context(settings, "The battery life is great but the camera is awful.")
    )

    assert cloud.calls == 1, "the top rung ran"
    assert result.best.tier is TierId.T2_LOCAL_LLM, "but the top rung did not decide"
    assert result.best.model_version == local.model_version
    assert result.best.result.label is Label.NEGATIVE
    assert result.verdicts[-1].tier is TierId.T3_CLOUD_LLM
    assert not result.verdicts[-1].is_usable
    assert_trace_is_coherent(result)


async def test_a_rated_review_is_settled_by_tier_zero_without_prose(
    settings: Settings,
) -> None:
    """Resource fields survive the trip from the context into the rules rung."""
    orchestrator, local, cloud = full_ladder(settings)

    result = await orchestrator.run(
        build_context(settings, "Perfect.", rating=5, rating_scale_max=5)
    )

    assert result.best.tier is TierId.T0_RULES
    assert result.best.detail["rule"] == "rating"
    assert result.best.result.label is Label.POSITIVE
    assert local.calls == cloud.calls == 0
    assert_trace_is_coherent(result)


# ── Degrading rather than failing ─────────────────────────────────────────────


async def test_a_broken_rung_degrades_the_run_without_sinking_it(
    settings: Settings,
) -> None:
    """Tier 2 raises and Tier 3 is down: the caller still gets Tier 1's answer."""
    local = ScriptedTier(TierId.T2_LOCAL_LLM, raises=TierError("ollama exploded"))
    cloud = ScriptedTier(TierId.T3_CLOUD_LLM, available=False)
    orchestrator = build_ladder(
        settings, RulesTier(settings), ClassifierTier(settings), local, cloud
    )

    result = await orchestrator.run(
        build_context(settings, "The battery life is great but the camera is awful.")
    )

    assert result.degraded is True
    assert cloud.calls == 0, "an unavailable rung is never asked to analyse"
    assert result.verdicts[-1].succeeded is False
    assert result.verdicts[-1].error is not None
    assert result.best.tier is TierId.T1_CLASSIFIER
    assert_trace_is_coherent(result)


async def test_an_unexpected_defect_is_contained_like_any_other_failure(
    settings: Settings,
) -> None:
    """A rung raising something that is not a TierError must not escape the loop."""
    local = ScriptedTier(TierId.T2_LOCAL_LLM, raises=RuntimeError("nobody planned this"))
    cloud = ScriptedTier(TierId.T3_CLOUD_LLM, label=Label.MIXED, confidence=0.93)
    orchestrator = build_ladder(
        settings, RulesTier(settings), ClassifierTier(settings), local, cloud
    )

    result = await orchestrator.run(
        build_context(settings, "The battery life is great but the camera is awful.")
    )

    assert result.degraded is True
    assert result.best.tier is TierId.T3_CLOUD_LLM, "the ladder climbed past the defect"
    assert_trace_is_coherent(result)


async def test_no_usable_verdict_anywhere_is_the_one_path_to_an_error(
    settings: Settings,
) -> None:
    """Tier 0 abstains and everything above it fails — the only 5xx in the service."""
    orchestrator = build_ladder(
        settings,
        RulesTier(settings),
        ScriptedTier(TierId.T1_CLASSIFIER, raises=TierError("scorer gone")),
        ScriptedTier(TierId.T2_LOCAL_LLM, raises=TierUnavailableError("daemon down")),
        ScriptedTier(TierId.T3_CLOUD_LLM, raises=TierError("no provider")),
    )

    with pytest.raises(NoVerdictError):
        await orchestrator.run(build_context(settings, "The package arrived on Tuesday."))


# ── Caller-supplied bounds ────────────────────────────────────────────────────


async def test_max_tier_pins_the_ladder_to_its_cheapest_rung(settings: Settings) -> None:
    orchestrator, local, cloud = full_ladder(settings)

    result = await orchestrator.run(
        build_context(settings, "Terrible.", options=Options(max_tier=TierId.T0_RULES))
    )

    assert [step.tier for step in result.steps] == [TierId.T0_RULES]
    assert result.best.tier is TierId.T0_RULES
    assert local.calls == cloud.calls == 0


async def test_min_tier_skips_the_rungs_the_caller_has_written_off(
    settings: Settings,
) -> None:
    """`min_tier` removes cheaper rungs entirely rather than running and ignoring them.

    "Terrible." is text Tier 0 settles on its own, which is what makes it the
    right probe: the floor has to be strong enough to skip a rung that would
    have answered.
    """
    orchestrator, local, cloud = full_ladder(settings)

    result = await orchestrator.run(
        build_context(settings, "Terrible.", options=Options(min_tier=TierId.T2_LOCAL_LLM))
    )

    assert [step.tier for step in result.steps] == [TierId.T2_LOCAL_LLM]
    assert local.calls == 1
    assert cloud.calls == 0, "the floor starts the ladder; it does not force it to the top"
    assert result.best.tier is TierId.T2_LOCAL_LLM
    assert_trace_is_coherent(result)


async def test_an_exhausted_latency_budget_stops_the_climb(settings: Settings) -> None:
    """A deadline already in the past ends the ladder at the rung that has run.

    Forced rather than raced: the context is built with a deadline behind us, so
    the assertion does not depend on how fast the machine is.
    """
    orchestrator, local, cloud = full_ladder(settings)
    ctx = build_context(settings, "Terrible, broken, awful.", budget_s=30.0).model_copy(
        update={"deadline_at": time.monotonic() - 1.0}
    )

    with pytest.raises(NoVerdictError):
        await orchestrator.run(ctx)

    assert local.calls == cloud.calls == 0, "no rung is started on a spent budget"


async def test_a_zero_cost_allowance_pins_the_ladder_to_its_first_rung(
    settings: Settings,
) -> None:
    """`max_cost_usd=0` is a stop condition by definition, even at zero spend."""
    orchestrator, local, cloud = full_ladder(settings)

    result = await orchestrator.run(
        build_context(settings, "Terrible.", options=Options(max_cost_usd=0.0))
    )

    assert [step.tier for step in result.steps] == [TierId.T0_RULES]
    assert local.calls == cloud.calls == 0
