"""The escalation policy — docs/02-lld.md §5, made executable.

Every decision about whether to climb the ladder is taken here, and nowhere
else. The orchestrator supplies facts (a verdict, the context it was produced
in, whether a next rung could serve) and receives a verdict-shaped answer; it
never inspects a confidence or a threshold itself. That separation is what makes
the ladder tunable from configuration alone.

Two properties of :meth:`EscalationPolicy.decide` matter more than the
arithmetic:

* **Stop conditions win, in the documented order.** A ceiling, an exhausted
  budget, an unaffordable next rung or a missing one all mean the same thing —
  there is nowhere to go — and no amount of low confidence can override them.
  The tier ceiling and latency budget are checked first; the cost and
  availability stops are checked after the quality rules, because both depend
  on which rung is next and both should only be reported when the ladder
  actually wanted to climb.
* **Escalation reasons accumulate.** A narrow-margin verdict on a contrastive
  sentence reports both, because the trace exists to explain the climb and one
  reason rarely tells the whole story. The first reason is the primary one, so
  the ordering of the checks is itself part of the contract.
"""

from __future__ import annotations

from typing import Final

from air_classifier_service.config import Settings, TierSettings
from air_classifier_service.constants import EscalationReason, Tier
from air_classifier_service.pipeline.tiers.base import TierContext
from air_classifier_service.schemas.common import ComplexityFeatures, Options
from air_classifier_service.schemas.verdict import EscalationDecision, TierVerdict

__all__ = ["EscalationPolicy"]

#: Below this, language identification is guesswork, and guesswork is not
#: grounds for spending a tier hop.
_LANGUAGE_CONFIDENCE_FLOOR: Final[float] = 0.5


class EscalationPolicy:
    """Decides whether a verdict is good enough to return."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ── The truth table ───────────────────────────────────────────────────────

    def decide(
        self,
        verdict: TierVerdict,
        ctx: TierContext,
        *,
        next_available: bool,
        next_cost_usd: float = 0.0,
    ) -> EscalationDecision:
        """Should the ladder climb past `verdict`?

        `ctx` is the context *after* the verdict was folded in, so
        ``ctx.cost_spent_usd`` already includes what this rung spent — a tier
        that blows the cost ceiling stops the ladder rather than being billed
        for and then escalated past anyway.

        `next_available` is the orchestrator's lookahead: there is a further
        rung, it is enabled, and it says it can serve. The policy cannot probe
        for that itself without holding tier instances, which would make it as
        untestable as the thing it governs. `next_cost_usd` is that rung's
        expected cost (:meth:`expected_cost_usd`), for the same reason.
        """
        options = ctx.options
        floors = self._settings.tiers.for_tier(verdict.tier)

        # 1 — ceiling reached; nothing above this rung is permitted.
        if verdict.tier.rank >= options.max_tier.rank:
            return EscalationDecision.stop(EscalationReason.TIER_CEILING)

        # 2 — the caller's latency budget is spent; a further rung cannot finish.
        if ctx.budget_exhausted:
            return EscalationDecision.stop(EscalationReason.BUDGET_EXHAUSTED)

        # The quality rules (5-11) are evaluated before the "nowhere to climb to"
        # check, even though that check can only ever stop the ladder. Knowing
        # *whether we wanted to climb* is what makes the trace honest: reporting
        # `next_tier_unavailable` on a rung that was perfectly confident tells an
        # operator we were blocked when in fact we were simply finished, and
        # sends them hunting for an outage that never happened. All of these
        # rules are pure computation, so evaluating them costs nothing.
        reasons = self._quality_reasons(verdict, ctx, floors=floors)

        if not reasons:
            return EscalationDecision.stop(EscalationReason.CONFIDENT)

        # 4 — we wanted to climb, but there is nowhere to climb to: no next rung,
        # or it is down. The outcome is the same stop either way; what differs is
        # that the trace now says so only when it is true.
        if not next_available:
            return EscalationDecision.stop(EscalationReason.NEXT_TIER_UNAVAILABLE)

        # 3 — we wanted to climb, and the next rung would cost more than the
        # caller has left. Checked against that rung's *expected* cost before it
        # runs: a ceiling checked only against money already spent is passed by
        # every first paid call, however far over the ceiling that call goes
        # (SIT CL-001). It sits here rather than with the hard ceilings above
        # because it depends on which rung is next, and because a free rung
        # (cost 0) stays reachable under `max_cost_usd: 0` (CL-002). The
        # quality reasons ride along, so the trace says what the money would
        # have bought.
        if ctx.cost_spent_usd + next_cost_usd > options.max_cost_usd:
            return EscalationDecision.stop(EscalationReason.COST_CEILING, *reasons)

        return EscalationDecision.climb(*reasons)

    # ── Answer quality ────────────────────────────────────────────────────────

    def is_settled(self, verdict: TierVerdict) -> bool:
        """Whether `verdict` meets its own rung's confidence and margin floors.

        Deliberately narrower than the quality rules: complexity triggers,
        `min_tier`, disagreement and language say *whom else to ask*, not that
        this answer is poor — a confident T3 answer to a sentence with "but" in
        it is the ladder working, not a weak verdict. The orchestrator combines
        this with how the ladder stopped to decide whether a degraded result is
        still safe to cache (a dead rung hopped on the way to a confident one).
        """
        if not verdict.is_usable:
            return False
        floors = self._settings.tiers.for_tier(verdict.tier)
        if verdict.result.confidence < floors.confidence_floor:
            return False
        return floors.margin_floor <= 0.0 or verdict.label_margin >= floors.margin_floor

    # ── Rules 5-11 ────────────────────────────────────────────────────────────

    def _quality_reasons(
        self, verdict: TierVerdict, ctx: TierContext, *, floors: TierSettings
    ) -> list[EscalationReason]:
        """Rules 5-11, in reporting order: every reason this rung wants to climb."""
        options = ctx.options
        reasons: list[EscalationReason] = []

        # 5 — the caller demanded a floor this rung sits below.
        if options.min_tier is not None and verdict.tier.rank < options.min_tier.rank:
            reasons.append(EscalationReason.MIN_TIER_FLOOR)

        # 6 — the rung produced nothing to judge. Confidence and margin are
        # meaningless on a verdict that never happened, so the checks below are
        # skipped rather than reported as further evidence.
        if not verdict.succeeded:
            reasons.append(EscalationReason.TIER_FAILED)
        elif not verdict.is_usable:
            reasons.append(EscalationReason.NO_SHORTCUT_MATCHED)
        else:
            # 7 — not sure enough for this rung.
            if verdict.result.confidence < floors.confidence_floor:
                reasons.append(EscalationReason.LOW_CONFIDENCE)

            # 8 — the runner-up was too close. A margin floor of zero disables the
            # check, which is how rungs that report no distribution opt out.
            if floors.margin_floor > 0.0 and verdict.label_margin < floors.margin_floor:
                reasons.append(EscalationReason.NARROW_MARGIN)

            # 9 — the text has structure a cheap rung reliably mangles.
            reasons.extend(self._complexity_reasons(ctx.features))

            # 10 — this rung contradicts one below it; someone better breaks the tie.
            if self._disagrees_with_prior(verdict, ctx):
                reasons.append(EscalationReason.TIER_DISAGREEMENT)

            # 11 — this rung was never trained or prompted for this language.
            if self._language_unsupported(verdict, ctx):
                reasons.append(EscalationReason.UNSUPPORTED_LANGUAGE)
        return reasons

    # ── Rule 9 ────────────────────────────────────────────────────────────────

    def _complexity_reasons(self, features: ComplexityFeatures) -> list[EscalationReason]:
        """Structural triggers, in the order they are reported.

        These route *whom to ask*; they classify nothing. Being liberal costs one
        cheap tier hop and being conservative costs accuracy, so the thresholds
        are tuned toward escalating.
        """
        tiers = self._settings.tiers
        reasons: list[EscalationReason] = []

        # "but" in a two-word text is not a contrast worth an LLM.
        if features.contrastive_count > 0 and features.token_count > tiers.contrastive_min_tokens:
            reasons.append(EscalationReason.CONTRASTIVE_CONJUNCTION)
        if features.polarity_flip:
            reasons.append(EscalationReason.POLARITY_FLIP)
        if features.emoji_text_conflict:
            reasons.append(EscalationReason.EMOJI_TEXT_CONFLICT)
        if features.negation_count >= tiers.negation_threshold:
            reasons.append(EscalationReason.NEGATION_DENSITY)
        if features.token_count > tiers.long_input_tokens:
            reasons.append(EscalationReason.LONG_INPUT)
        return reasons

    # ── Rule 10 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _disagrees_with_prior(verdict: TierVerdict, ctx: TierContext) -> bool:
        """Whether the nearest cheaper rung that committed to a label disagrees.

        Only rungs strictly below this one count: ``prior_verdicts`` already
        contains the verdict being judged, and a verdict cannot disagree with
        itself.
        """
        for prior in reversed(ctx.prior_verdicts):
            if prior.tier.rank >= verdict.tier.rank or not prior.is_usable:
                continue
            return prior.result.label is not verdict.result.label
        return False

    # ── Rule 11 ───────────────────────────────────────────────────────────────

    def _language_unsupported(self, verdict: TierVerdict, ctx: TierContext) -> bool:
        """Whether the deciding rung is out of its linguistic depth.

        The lexical and classifier rungs are trained on an explicit language
        list; the LLM rungs are prompted, not trained, and are treated as
        multilingual. An undetermined or low-confidence detection is not
        evidence of anything and never triggers a climb.
        """
        if verdict.tier.rank > Tier.T1_CLASSIFIER.rank:
            return False
        language = ctx.language
        if language.code == "und" or language.confidence < _LANGUAGE_CONFIDENCE_FLOOR:
            return False
        supported = {code.lower() for code in self._settings.classifier.supported_languages}
        return language.code.lower() not in supported

    # ── Cost ──────────────────────────────────────────────────────────────────

    def expected_cost_usd(self, tier: Tier) -> float:
        """What one call to `tier` is expected to cost (config, per rung)."""
        return self._settings.tiers.for_tier(tier).expected_cost_usd

    def can_afford(self, tier: Tier, ctx: TierContext) -> bool:
        """Whether running `tier` fits in what the caller has left to spend.

        The orchestrator asks this before *every* rung, including the first:
        rule 3 only guards a climb, and a request pinned to a paid rung
        (`min_tier: t3_cloud_llm`) never climbs into it.
        """
        return ctx.cost_spent_usd + self.expected_cost_usd(tier) <= ctx.options.max_cost_usd

    # ── Selection and laddering ───────────────────────────────────────────────

    @staticmethod
    def select_best(verdicts: list[TierVerdict]) -> TierVerdict | None:
        """The verdict to return to the caller, or None if there is none.

        The highest rung that committed to a label wins, because that is the
        rung the ladder paid the most to consult; ties are broken by confidence.
        A failed or abstaining verdict can never win — which is why the only
        path to a 503 is every rung producing one.
        """
        usable = [verdict for verdict in verdicts if verdict.is_usable]
        if not usable:
            return None
        return max(usable, key=lambda verdict: (verdict.tier.rank, verdict.result.confidence))

    def enabled_tiers_for(self, options: Options) -> tuple[Tier, ...]:
        """The rungs this request may use, cheapest first.

        Both bounds are applied here rather than inside the loop: `min_tier`
        removes cheaper rungs entirely (the caller has already decided they are
        not worth asking) and `max_tier` removes dearer ones. An empty result is
        legitimate — a request whose bounds exclude every enabled rung has no
        ladder to run, and gets the same 503 as one where every rung failed.
        """
        floor = options.min_tier.rank if options.min_tier is not None else 0
        ceiling = options.max_tier.rank
        return tuple(
            tier for tier in self._settings.enabled_tiers() if floor <= tier.rank <= ceiling
        )
