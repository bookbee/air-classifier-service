"""The escalation truth table from docs/02-lld.md §5, executed.

This file is the spec table in runnable form. Every numbered rule gets two
tests: one proving it fires exactly when it should, and one proving a
higher-priority rule pre-empts it. If a change to `policy.py` does not break
this file, it did not change the ladder's semantics; if it does, the table in
the design document is what has to be amended first.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pytest

from air_classifier_service.config import (
    AppSettings,
    ClassifierSettings,
    Settings,
    TierSettings,
    TiersSettings,
)
from air_classifier_service.constants import EscalationReason as Reason
from air_classifier_service.constants import Label, Tier
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.tiers.base import TierContext
from air_classifier_service.schemas.common import (
    ComplexityFeatures,
    LanguageInfo,
    Options,
    SafetyReport,
    SentimentResult,
    TierUsage,
)
from air_classifier_service.schemas.verdict import TierVerdict

# ── Fixtures and builders ─────────────────────────────────────────────────────


@pytest.fixture
def policy(settings: Settings) -> EscalationPolicy:
    return EscalationPolicy(settings)


def make_ctx(
    *,
    options: Options | None = None,
    features: ComplexityFeatures | None = None,
    language: LanguageInfo | None = None,
    priors: Sequence[TierVerdict] = (),
    cost_spent_usd: float = 0.0,
    remaining_ms: float = 5000.0,
) -> TierContext:
    return TierContext(
        request_id="req_test",
        text="the battery lasts all day",
        original_length=25,
        language=language or LanguageInfo(code="en", confidence=0.99),
        features=features or ComplexityFeatures(),
        safety=SafetyReport(),
        options=options or Options(),
        prior_verdicts=list(priors),
        deadline_at=time.monotonic() + remaining_ms / 1000.0,
        cost_spent_usd=cost_spent_usd,
    )


def verdict(
    tier: Tier,
    *,
    label: Label = Label.POSITIVE,
    confidence: float = 0.99,
    margin: float = 0.9,
    cost_usd: float = 0.0,
) -> TierVerdict:
    """A verdict that clears every floor unless a test lowers one."""
    return TierVerdict(
        tier=tier,
        model_version="test-1",
        result=SentimentResult(label=label, polarity=0.8, confidence=confidence),
        label_margin=margin,
        usage=TierUsage(cost_usd=cost_usd),
    )


# ── 1 · tier ceiling ──────────────────────────────────────────────────────────


def test_rule1_ceiling_stops_even_a_worthless_verdict(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        verdict(Tier.T3_CLOUD_LLM, confidence=0.01, margin=0.0),
        make_ctx(),
        next_available=True,
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.TIER_CEILING]


def test_rule1_honours_a_caller_lowered_ceiling(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(max_tier=Tier.T1_CLASSIFIER))

    decision = policy.decide(verdict(Tier.T1_CLASSIFIER, confidence=0.1), ctx, next_available=True)

    assert decision.reasons == [Reason.TIER_CEILING]


def test_rule1_preempts_every_stop_condition_below_it(policy: EscalationPolicy) -> None:
    ctx = make_ctx(remaining_ms=-1.0, cost_spent_usd=99.0)

    decision = policy.decide(verdict(Tier.T3_CLOUD_LLM, confidence=0.0), ctx, next_available=False)

    assert decision.reasons == [Reason.TIER_CEILING]


# ── 2 · budget exhausted ──────────────────────────────────────────────────────


def test_rule2_an_exhausted_budget_stops_the_ladder(policy: EscalationPolicy) -> None:
    ctx = make_ctx(remaining_ms=-1.0)

    decision = policy.decide(verdict(Tier.T0_RULES, confidence=0.1), ctx, next_available=True)

    assert decision.escalate is False
    assert decision.reasons == [Reason.BUDGET_EXHAUSTED]


def test_rule2_is_preempted_by_the_ceiling(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(max_tier=Tier.T0_RULES), remaining_ms=-1.0)

    decision = policy.decide(verdict(Tier.T0_RULES, confidence=0.1), ctx, next_available=True)

    assert decision.reasons == [Reason.TIER_CEILING]


# ── 3 · cost ceiling ──────────────────────────────────────────────────────────


#: What the policy is told the next (paid) rung costs in the rule-3 tests.
_T3_COST = 0.005


def test_rule3_an_unaffordable_next_rung_stops_the_ladder(policy: EscalationPolicy) -> None:
    ctx = make_ctx(cost_spent_usd=0.046)  # 0.046 + 0.005 > the default 0.05

    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.1),
        ctx,
        next_available=True,
        next_cost_usd=_T3_COST,
    )

    assert decision.escalate is False
    # The quality reasons ride along: the trace says what the money would have bought.
    assert decision.reasons == [Reason.COST_CEILING, Reason.LOW_CONFIDENCE]


def test_rule3_is_checked_before_the_paid_call_not_after(policy: EscalationPolicy) -> None:
    """SIT CL-001: nothing spent yet, but the next rung alone exceeds the ceiling."""
    ctx = make_ctx(options=Options(max_cost_usd=0.001))

    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER, confidence=0.1),
        ctx,
        next_available=True,
        next_cost_usd=_T3_COST,
    )

    assert decision.escalate is False
    assert decision.primary_reason is Reason.COST_CEILING


def test_rule3_a_zero_ceiling_still_reaches_a_free_rung(policy: EscalationPolicy) -> None:
    """SIT CL-002: `max_cost_usd: 0` means "free tiers only", not "T0 only"."""
    ctx = make_ctx(options=Options(max_cost_usd=0.0))

    decision = policy.decide(
        verdict(Tier.T0_RULES, label=Label.UNKNOWN, confidence=0.0),
        ctx,
        next_available=True,
        next_cost_usd=0.0,
    )

    assert decision.escalate is True


def test_rule3_a_zero_ceiling_stops_before_a_paid_rung(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(max_cost_usd=0.0))

    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.1),
        ctx,
        next_available=True,
        next_cost_usd=_T3_COST,
    )

    assert decision.escalate is False
    assert decision.primary_reason is Reason.COST_CEILING


def test_rule3_does_not_misreport_a_confident_verdict(policy: EscalationPolicy) -> None:
    """A rung that did not want to climb stopped because it was sure, not because of money."""
    ctx = make_ctx(options=Options(max_cost_usd=0.0))

    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER), ctx, next_available=True, next_cost_usd=_T3_COST
    )

    assert decision.reasons == [Reason.CONFIDENT]


def test_can_afford_prices_each_rung_from_config(policy: EscalationPolicy) -> None:
    capped = make_ctx(options=Options(max_cost_usd=0.001))

    assert policy.can_afford(Tier.T1_CLASSIFIER, capped) is True
    assert policy.can_afford(Tier.T2_LOCAL_LLM, capped) is True
    assert policy.can_afford(Tier.T3_CLOUD_LLM, capped) is False
    assert policy.can_afford(Tier.T3_CLOUD_LLM, make_ctx()) is True


def test_rule3_does_not_fire_when_the_next_rung_fits(policy: EscalationPolicy) -> None:
    ctx = make_ctx(cost_spent_usd=0.044)  # 0.044 + 0.005 <= 0.05

    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.1),
        ctx,
        next_available=True,
        next_cost_usd=_T3_COST,
    )

    assert decision.escalate is True


def test_rule3_is_preempted_by_the_budget(policy: EscalationPolicy) -> None:
    ctx = make_ctx(cost_spent_usd=1.0, remaining_ms=-1.0)

    decision = policy.decide(verdict(Tier.T0_RULES, confidence=0.1), ctx, next_available=True)

    assert decision.reasons == [Reason.BUDGET_EXHAUSTED]


# ── 4 · next tier unavailable ─────────────────────────────────────────────────


def test_rule4_nowhere_to_climb_stops_the_ladder(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        verdict(Tier.T0_RULES, confidence=0.1), make_ctx(), next_available=False
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


def test_rule4_is_reported_ahead_of_the_cost_ceiling(policy: EscalationPolicy) -> None:
    """Nothing to climb to is the binding fact; what it would have cost is moot."""
    ctx = make_ctx(cost_spent_usd=0.06)

    decision = policy.decide(verdict(Tier.T0_RULES, confidence=0.1), ctx, next_available=False)

    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


# ── 5 · min_tier floor ────────────────────────────────────────────────────────


def test_rule5_a_confident_verdict_below_the_floor_still_climbs(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(min_tier=Tier.T2_LOCAL_LLM))

    decision = policy.decide(verdict(Tier.T1_CLASSIFIER), ctx, next_available=True)

    assert decision.escalate is True
    assert decision.reasons == [Reason.MIN_TIER_FLOOR]


def test_rule5_stops_once_the_floor_is_reached(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(min_tier=Tier.T2_LOCAL_LLM))

    decision = policy.decide(verdict(Tier.T2_LOCAL_LLM), ctx, next_available=True)

    assert decision.escalate is False
    assert decision.reasons == [Reason.CONFIDENT]


def test_rule5_is_preempted_by_an_unavailable_next_tier(policy: EscalationPolicy) -> None:
    """A floor is a preference; a missing rung is a fact."""
    ctx = make_ctx(options=Options(min_tier=Tier.T2_LOCAL_LLM))

    decision = policy.decide(verdict(Tier.T1_CLASSIFIER), ctx, next_available=False)

    assert decision.escalate is False
    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


# ── 6 · tier failed / abstained ───────────────────────────────────────────────


def test_rule6_a_failed_verdict_climbs_on_that_reason_alone(policy: EscalationPolicy) -> None:
    failed = TierVerdict.failed(Tier.T1_CLASSIFIER, "boom")

    decision = policy.decide(failed, make_ctx(), next_available=True)

    assert decision.escalate is True
    # Its zero confidence is an artefact of failing, not further evidence.
    assert decision.reasons == [Reason.TIER_FAILED]


def test_rule6_an_abstention_is_reported_as_such(policy: EscalationPolicy) -> None:
    decision = policy.decide(TierVerdict.abstained(Tier.T0_RULES), make_ctx(), next_available=True)

    assert decision.escalate is True
    assert decision.reasons == [Reason.NO_SHORTCUT_MATCHED]


def test_rule6_reports_the_floor_first_when_both_apply(policy: EscalationPolicy) -> None:
    ctx = make_ctx(options=Options(min_tier=Tier.T3_CLOUD_LLM))

    failed = TierVerdict.failed(Tier.T1_CLASSIFIER, "boom")

    decision = policy.decide(failed, ctx, next_available=True)

    assert decision.reasons == [Reason.MIN_TIER_FLOOR, Reason.TIER_FAILED]


def test_rule6_is_preempted_by_a_stop_condition(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        TierVerdict.failed(Tier.T1_CLASSIFIER, "boom"), make_ctx(), next_available=False
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


# ── 7 · low confidence ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tier", "confidence", "expected"),
    [
        (Tier.T0_RULES, 0.94, True),
        (Tier.T0_RULES, 0.95, False),
        (Tier.T1_CLASSIFIER, 0.79, True),
        (Tier.T1_CLASSIFIER, 0.80, False),
        (Tier.T2_LOCAL_LLM, 0.71, True),
        (Tier.T2_LOCAL_LLM, 0.72, False),
    ],
)
def test_rule7_fires_against_the_configured_floor_for_that_rung(
    policy: EscalationPolicy, tier: Tier, confidence: float, expected: bool
) -> None:
    decision = policy.decide(verdict(tier, confidence=confidence), make_ctx(), next_available=True)

    assert (Reason.LOW_CONFIDENCE in decision.reasons) is expected


def test_rule7_is_preempted_by_a_failed_verdict(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        TierVerdict.failed(Tier.T1_CLASSIFIER, "boom"), make_ctx(), next_available=True
    )

    assert Reason.LOW_CONFIDENCE not in decision.reasons


# ── 8 · narrow margin ─────────────────────────────────────────────────────────


def test_rule8_a_close_runner_up_climbs(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER, confidence=0.9, margin=0.24), make_ctx(), next_available=True
    )

    assert decision.escalate is True
    assert decision.reasons == [Reason.NARROW_MARGIN]


def test_rule8_a_zero_margin_floor_disables_the_check(policy: EscalationPolicy) -> None:
    """T0 reports no distribution, so its margin of 0.0 must not be evidence."""
    decision = policy.decide(
        verdict(Tier.T0_RULES, confidence=0.99, margin=0.0), make_ctx(), next_available=True
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.CONFIDENT]


def test_rule8_is_reported_after_low_confidence(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER, confidence=0.5, margin=0.1), make_ctx(), next_available=True
    )

    assert decision.reasons == [Reason.LOW_CONFIDENCE, Reason.NARROW_MARGIN]


# ── 9 · complexity triggers ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (ComplexityFeatures(contrastive_count=1, token_count=9), Reason.CONTRASTIVE_CONJUNCTION),
        (ComplexityFeatures(positive_hits=1, negative_hits=1), Reason.POLARITY_FLIP),
        (
            ComplexityFeatures(emoji_polarity=0.8, lexical_polarity=-0.8),
            Reason.EMOJI_TEXT_CONFLICT,
        ),
        (ComplexityFeatures(negation_count=2), Reason.NEGATION_DENSITY),
        (ComplexityFeatures(token_count=121), Reason.LONG_INPUT),
    ],
)
def test_rule9_each_trigger_escalates_a_confident_verdict(
    policy: EscalationPolicy, features: ComplexityFeatures, expected: Reason
) -> None:
    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER), make_ctx(features=features), next_available=True
    )

    assert decision.escalate is True
    assert decision.reasons == [expected]


@pytest.mark.parametrize(
    "features",
    [
        ComplexityFeatures(contrastive_count=1, token_count=8),
        ComplexityFeatures(negation_count=1),
        ComplexityFeatures(token_count=120),
        ComplexityFeatures(positive_hits=2, negative_hits=0),
        ComplexityFeatures(emoji_polarity=0.8, lexical_polarity=0.8),
    ],
)
def test_rule9_stays_quiet_just_below_each_threshold(
    policy: EscalationPolicy, features: ComplexityFeatures
) -> None:
    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER), make_ctx(features=features), next_available=True
    )

    assert decision.escalate is False


def test_rule9_thresholds_come_from_settings_not_from_code() -> None:
    tuned = Settings(
        app=AppSettings(env="test"),
        tiers=TiersSettings(negation_threshold=5, contrastive_min_tokens=50, long_input_tokens=10),
    )
    features = ComplexityFeatures(negation_count=4, contrastive_count=1, token_count=11)

    decision = EscalationPolicy(tuned).decide(
        verdict(Tier.T1_CLASSIFIER), make_ctx(features=features), next_available=True
    )

    # Only the retuned long-input rule can fire on this feature set.
    assert decision.reasons == [Reason.LONG_INPUT]


def test_rule9_is_preempted_by_a_stop_condition(policy: EscalationPolicy) -> None:
    ctx = make_ctx(features=ComplexityFeatures(positive_hits=1, negative_hits=1))

    decision = policy.decide(verdict(Tier.T1_CLASSIFIER), ctx, next_available=False)

    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


# ── 10 · tier disagreement ────────────────────────────────────────────────────


def test_rule10_contradicting_a_cheaper_rung_climbs(policy: EscalationPolicy) -> None:
    prior = verdict(Tier.T0_RULES, label=Label.POSITIVE)
    current = verdict(Tier.T1_CLASSIFIER, label=Label.NEGATIVE)

    decision = policy.decide(current, make_ctx(priors=[prior, current]), next_available=True)

    assert decision.escalate is True
    assert decision.reasons == [Reason.TIER_DISAGREEMENT]


def test_rule10_agreement_is_not_a_reason(policy: EscalationPolicy) -> None:
    prior = verdict(Tier.T0_RULES, label=Label.POSITIVE)
    current = verdict(Tier.T1_CLASSIFIER, label=Label.POSITIVE)

    decision = policy.decide(current, make_ctx(priors=[prior, current]), next_available=True)

    assert decision.escalate is False


def test_rule10_never_compares_a_verdict_with_itself(policy: EscalationPolicy) -> None:
    """`prior_verdicts` already contains the verdict under judgement."""
    current = verdict(Tier.T1_CLASSIFIER, label=Label.NEGATIVE)

    decision = policy.decide(current, make_ctx(priors=[current]), next_available=True)

    assert decision.escalate is False


def test_rule10_ignores_priors_that_committed_to_nothing(policy: EscalationPolicy) -> None:
    priors = [
        TierVerdict.failed(Tier.T0_RULES, "boom"),
        TierVerdict.abstained(Tier.T0_RULES),
    ]

    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER, label=Label.NEGATIVE),
        make_ctx(priors=priors),
        next_available=True,
    )

    assert decision.escalate is False


def test_rule10_is_preempted_by_a_stop_condition(policy: EscalationPolicy) -> None:
    prior = verdict(Tier.T0_RULES, label=Label.POSITIVE)
    current = verdict(Tier.T1_CLASSIFIER, label=Label.NEGATIVE)
    ctx = make_ctx(priors=[prior, current], remaining_ms=-1.0)

    decision = policy.decide(current, ctx, next_available=True)

    assert decision.reasons == [Reason.BUDGET_EXHAUSTED]


# ── 11 · unsupported language ─────────────────────────────────────────────────


def test_rule11_a_cheap_rung_out_of_its_depth_climbs(policy: EscalationPolicy) -> None:
    ctx = make_ctx(language=LanguageInfo(code="fr", confidence=0.99))

    decision = policy.decide(verdict(Tier.T1_CLASSIFIER), ctx, next_available=True)

    assert decision.escalate is True
    assert decision.reasons == [Reason.UNSUPPORTED_LANGUAGE]


def test_rule11_treats_the_llm_rungs_as_multilingual(policy: EscalationPolicy) -> None:
    ctx = make_ctx(language=LanguageInfo(code="fr", confidence=0.99))

    decision = policy.decide(verdict(Tier.T2_LOCAL_LLM), ctx, next_available=True)

    assert decision.escalate is False


@pytest.mark.parametrize(
    "language",
    [LanguageInfo(code="und", confidence=0.0), LanguageInfo(code="fr", confidence=0.2)],
)
def test_rule11_does_not_fire_on_a_guess(policy: EscalationPolicy, language: LanguageInfo) -> None:
    decision = policy.decide(
        verdict(Tier.T0_RULES), make_ctx(language=language), next_available=True
    )

    assert decision.escalate is False


def test_rule11_reads_the_supported_list_from_settings() -> None:
    tuned = Settings(
        app=AppSettings(env="test"),
        classifier=ClassifierSettings(supported_languages=["en", "FR"]),
    )
    ctx = make_ctx(language=LanguageInfo(code="fr", confidence=0.99))

    decision = EscalationPolicy(tuned).decide(verdict(Tier.T1_CLASSIFIER), ctx, next_available=True)

    assert decision.escalate is False


def test_rule11_is_preempted_by_a_stop_condition(policy: EscalationPolicy) -> None:
    ctx = make_ctx(language=LanguageInfo(code="fr", confidence=0.99), cost_spent_usd=1.0)

    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER), ctx, next_available=True, next_cost_usd=_T3_COST
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.COST_CEILING, Reason.UNSUPPORTED_LANGUAGE]


# ── Fall-through ──────────────────────────────────────────────────────────────


def test_nothing_firing_is_a_confident_stop(policy: EscalationPolicy) -> None:
    decision = policy.decide(verdict(Tier.T1_CLASSIFIER), make_ctx(), next_available=True)

    assert decision.escalate is False
    assert decision.reasons == [Reason.CONFIDENT]


def test_reasons_accumulate_in_rule_order(policy: EscalationPolicy) -> None:
    ctx = make_ctx(
        options=Options(min_tier=Tier.T3_CLOUD_LLM),
        features=ComplexityFeatures(negation_count=3),
        language=LanguageInfo(code="de", confidence=0.99),
        priors=[verdict(Tier.T0_RULES, label=Label.POSITIVE)],
    )

    decision = policy.decide(
        verdict(Tier.T1_CLASSIFIER, label=Label.NEGATIVE, confidence=0.3, margin=0.05),
        ctx,
        next_available=True,
    )

    assert decision.reasons == [
        Reason.MIN_TIER_FLOOR,
        Reason.LOW_CONFIDENCE,
        Reason.NARROW_MARGIN,
        Reason.NEGATION_DENSITY,
        Reason.TIER_DISAGREEMENT,
        Reason.UNSUPPORTED_LANGUAGE,
    ]
    assert decision.primary_reason is Reason.MIN_TIER_FLOOR


# ── select_best ───────────────────────────────────────────────────────────────


def test_select_best_returns_none_when_nothing_is_usable(policy: EscalationPolicy) -> None:
    assert policy.select_best([]) is None
    assert (
        policy.select_best(
            [TierVerdict.failed(Tier.T0_RULES, "boom"), TierVerdict.abstained(Tier.T1_CLASSIFIER)]
        )
        is None
    )


def test_select_best_prefers_the_dearest_rung_over_the_most_confident(
    policy: EscalationPolicy,
) -> None:
    cheap = verdict(Tier.T0_RULES, confidence=0.99)
    dear = verdict(Tier.T2_LOCAL_LLM, confidence=0.66)

    assert policy.select_best([cheap, dear]) is dear


def test_select_best_breaks_ties_within_a_rung_by_confidence(policy: EscalationPolicy) -> None:
    low = verdict(Tier.T3_CLOUD_LLM, confidence=0.60)
    high = verdict(Tier.T3_CLOUD_LLM, confidence=0.90)

    assert policy.select_best([low, high]) is high


def test_select_best_never_returns_a_failed_verdict(policy: EscalationPolicy) -> None:
    usable = verdict(Tier.T0_RULES, confidence=0.5)
    failed = TierVerdict.failed(Tier.T3_CLOUD_LLM, "boom")

    assert policy.select_best([usable, failed]) is usable


# ── enabled_tiers_for ─────────────────────────────────────────────────────────


def test_enabled_tiers_defaults_to_the_whole_ladder(policy: EscalationPolicy) -> None:
    assert policy.enabled_tiers_for(Options()) == Tier.ordered()


def test_enabled_tiers_applies_both_caller_bounds(policy: EscalationPolicy) -> None:
    options = Options(min_tier=Tier.T1_CLASSIFIER, max_tier=Tier.T2_LOCAL_LLM)

    assert policy.enabled_tiers_for(options) == (Tier.T1_CLASSIFIER, Tier.T2_LOCAL_LLM)


def test_enabled_tiers_omits_rungs_switched_off_in_configuration() -> None:
    tuned = Settings(
        app=AppSettings(env="test"),
        tiers=TiersSettings(t2_local_llm=TierSettings(enabled=False)),
    )

    assert EscalationPolicy(tuned).enabled_tiers_for(Options()) == (
        Tier.T0_RULES,
        Tier.T1_CLASSIFIER,
        Tier.T3_CLOUD_LLM,
    )


def test_enabled_tiers_can_be_empty() -> None:
    """Bounds that exclude every enabled rung leave no ladder to run."""
    tuned = Settings(
        app=AppSettings(env="test"),
        tiers=TiersSettings(
            t2_local_llm=TierSettings(enabled=False), t3_cloud_llm=TierSettings(enabled=False)
        ),
    )
    options = Options(min_tier=Tier.T2_LOCAL_LLM)

    assert EscalationPolicy(tuned).enabled_tiers_for(options) == ()


# ── next_tier_unavailable reports only when it is actually the reason ──────────


def test_a_confident_rung_stops_as_confident_even_with_nowhere_to_climb(
    policy: EscalationPolicy,
) -> None:
    """Rule 4 must not mask the fact that we were finished anyway.

    Explainability is a first-class output of this service, so the trace has to
    distinguish "we would have escalated but the next rung is down" from "we had
    no reason to escalate". Reporting the former when the latter is true sends an
    operator hunting for an outage that never happened.
    """
    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.99, margin=0.9), make_ctx(), next_available=False
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.CONFIDENT]


def test_an_unconfident_rung_with_nowhere_to_climb_says_so(policy: EscalationPolicy) -> None:
    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.1, margin=0.0), make_ctx(), next_available=False
    )

    assert decision.escalate is False
    assert decision.reasons == [Reason.NEXT_TIER_UNAVAILABLE]


def test_the_same_verdict_climbs_when_the_next_rung_is_up(policy: EscalationPolicy) -> None:
    """The only difference between this and the test above is availability."""
    decision = policy.decide(
        verdict(Tier.T2_LOCAL_LLM, confidence=0.1, margin=0.0), make_ctx(), next_available=True
    )

    assert decision.escalate is True
    assert Reason.LOW_CONFIDENCE in decision.reasons
