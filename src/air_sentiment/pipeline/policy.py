"""The escalation policy — docs/02-lld.md §5, made executable.

Every decision about whether to climb the ladder is taken here, and nowhere
else. The orchestrator supplies facts (a verdict, the context it was produced
in, whether a next rung could serve) and receives a verdict-shaped answer; it
never inspects a confidence or a threshold itself. That separation is what makes
the ladder tunable from configuration alone.

Two properties of :meth:`EscalationPolicy.decide` matter more than the
arithmetic:

* **Stop conditions are checked first, in the documented order.** A ceiling, an
  exhausted budget, a spent cost allowance or a missing next rung all mean the
  same thing — there is nowhere to go — and no amount of low confidence can
  override them.
* **Escalation reasons accumulate.** A narrow-margin verdict on a contrastive
  sentence reports both, because the trace exists to explain the climb and one
  reason rarely tells the whole story. The first reason is the primary one, so
  the ordering of the checks is itself part of the contract.
"""

from __future__ import annotations

from typing import Final

from air_sentiment.config import Settings
from air_sentiment.constants import EscalationReason, Tier
from air_sentiment.pipeline.tiers.base import TierContext
from air_sentiment.schemas.common import ComplexityFeatures, Options
from air_sentiment.schemas.verdict import EscalationDecision, TierVerdict

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
        self, verdict: TierVerdict, ctx: TierContext, *, next_available: bool
    ) -> EscalationDecision:
        """Should the ladder climb past `verdict`?

        `ctx` is the context *after* the verdict was folded in, so
        ``ctx.cost_spent_usd`` already includes what this rung spent — a tier
        that blows the cost ceiling stops the ladder rather than being billed
        for and then escalated past anyway.

        `next_available` is the orchestrator's lookahead: there is a further
        rung, it is enabled, and it says it can serve. The policy cannot probe
        for that itself without holding tier instances, which would make it as
        untestable as the thing it governs.
        """
        options = ctx.options
        floors = self._settings.tiers.for_tier(verdict.tier)

        # 1 — ceiling reached; nothing above this rung is permitted.
        if verdict.tier.rank >= options.max_tier.rank:
            return EscalationDecision.stop(EscalationReason.TIER_CEILING)

        # 2 — the caller's latency budget is spent; a further rung cannot finish.
        if ctx.budget_exhausted:
            return EscalationDecision.stop(EscalationReason.BUDGET_EXHAUSTED)

        # 3 — the caller's money is spent. Note that `max_cost_usd == 0` pins the
        # ladder to its first rung by definition, which is the point of it.
        if ctx.cost_spent_usd >= options.max_cost_usd:
            return EscalationDecision.stop(EscalationReason.COST_CEILING)

        # The quality rules (5-11) are evaluated before the "nowhere to climb to"
        # check, even though that check can only ever stop the ladder. Knowing
        # *whether we wanted to climb* is what makes the trace honest: reporting
        # `next_tier_unavailable` on a rung that was perfectly confident tells an
        # operator we were blocked when in fact we were simply finished, and
        # sends them hunting for an outage that never happened. All of these
        # rules are pure computation, so evaluating them costs nothing.
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

        if not reasons:
            return EscalationDecision.stop(EscalationReason.CONFIDENT)

        # 4 — we wanted to climb, but there is nowhere to climb to: no next rung,
        # or it is down. The outcome is the same stop either way; what differs is
        # that the trace now says so only when it is true.
        if not next_available:
            return EscalationDecision.stop(EscalationReason.NEXT_TIER_UNAVAILABLE)

        return EscalationDecision.climb(*reasons)

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
