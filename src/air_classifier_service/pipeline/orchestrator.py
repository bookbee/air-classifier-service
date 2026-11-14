"""The ladder — docs/02-lld.md §4.

The loop below is the whole service in twenty lines: ask the cheapest rung, let
the policy judge the answer, climb only if it says so. Three properties are
worth defending in review, because losing any of them turns a cost-aware ladder
back into a pile of special cases:

* **Tier-agnostic.** Nothing here names a concrete rung or a :class:`Tier`
  member. Rungs arrive as protocol implementations and their order comes from
  the policy, so adding a fifth one is a registration rather than a refactor —
  and the whole ladder can be exercised with scripted fakes and no network.
* **Immutably threaded context.** Each rung receives a context carrying its
  predecessors' verdicts and their accumulated cost, built with
  :meth:`TierContext.with_verdict`. A rung cannot corrupt state for the ones
  that follow it, and the caller's original context survives the run untouched.
* **A failing rung is not an exception.** Timeouts, :class:`TierError` and
  outright defects all become a failed verdict; the ladder continues and
  ``degraded`` is set. "Degrade, don't fail" is not a slogan here, it is this
  ``try`` block.

``degraded`` covers a second case beyond an outright failure: a verdict that
succeeded but was not good enough, with nothing further up the ladder able to
serve. T0 and T1 have no external dependency and cannot land here; T2 and T3
can, whenever a downstream service is down or not yet reachable. That is a
capability gap, not a defect in this rung, and it gets the same
``degraded: true`` treatment plus a distinct ``ladder.capacity_gap`` warning —
console-visible and, via the response's own ``escalation_trace``, visible to
the caller too, so either side can tell "confident" and "stuck below where the
ladder wanted to be" apart.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from air_classifier_service.config import Settings
from air_classifier_service.constants import EscalationReason
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.observability.logging import get_logger, safe_text_fields
from air_classifier_service.observability.metrics import (
    TierOutcome,
    record_decided_by,
    record_escalation,
    record_tier_decision,
    tier_timer,
)
from air_classifier_service.observability.tracing import span
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.tiers.base import Tier, TierContext, TierError
from air_classifier_service.schemas.verdict import (
    EscalationStep,
    OrchestrationResult,
    TierVerdict,
)

__all__ = ["NoVerdictError", "Orchestrator"]

_log = get_logger(__name__)


class NoVerdictError(Exception):
    """No rung produced a usable answer.

    A provider outage does not raise it — that is a lower rung's verdict with
    ``degraded=true``. Only an empty ladder, or one where every rung failed or
    abstained, gets here.

    It has two causes that a caller must be able to tell apart (SIT CL-003):

    * ``retryable=True`` — a rung failed or was unavailable on the way. That can
      clear on its own: the 503 the HTTP layer renders it as.
    * ``retryable=False`` — nothing failed; the caller's own options (`max_tier`,
      `min_tier`, `max_cost_usd`, `latency_budget_ms`) stopped the ladder before
      any rung could answer. The same request fails the same way every time, so
      it is a 422 naming the ``limit`` instead of a 5xx anyone retries or pages on.
    """

    def __init__(self, message: str, *, retryable: bool = True, limit: str | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        #: The option-derived stop that left the ladder empty-handed, when
        #: ``retryable`` is false: an :class:`EscalationReason` value, or
        #: ``"tier_bounds"`` when `min_tier`/`max_tier` exclude every enabled rung.
        self.limit = limit

    def limit_detail(self) -> str:
        """A caller-facing detail naming the option to relax."""
        hint = _LIMIT_HINTS.get(self.limit or "", "the request's options")
        return f"No tier could produce a verdict within {hint}. Relax it and resend."


_LIMIT_HINTS: dict[str, str] = {
    "tier_ceiling": "options.max_tier",
    "tier_bounds": "options.min_tier/options.max_tier",
    "cost_ceiling": "options.max_cost_usd",
    "budget_exhausted": "options.latency_budget_ms",
}


class Orchestrator:
    """Runs the escalation ladder for one request."""

    def __init__(self, tiers: Sequence[Tier], policy: EscalationPolicy, settings: Settings) -> None:
        # Keyed by rung id so the policy's ordering, not the constructor
        # argument's, decides who runs and in what sequence.
        self._tiers: dict[TierId, Tier] = {tier.tier: tier for tier in tiers}
        self._policy = policy
        self._settings = settings

    async def run(self, ctx: TierContext) -> OrchestrationResult:
        """Climb until the policy says stop, then return the best answer."""
        ladder = self._ladder_for(ctx)
        verdicts: list[TierVerdict] = []
        steps: list[EscalationStep] = []
        degraded = False
        # The caller-imposed stop that ended the ladder, if one did — what a
        # `NoVerdictError` names when nothing failed on the way.
        limit: str | None = None if ladder else "tier_bounds"

        # The decision taken on the last rung that ran — what `settled` reads.
        last_reason: EscalationReason | None = None

        position = 0
        while position < len(ladder):
            tier = ladder[position]
            timeout_s = self._timeout_s(tier, ctx)
            if timeout_s <= 0.0:
                # Starting a rung that cannot finish buys nothing but latency.
                _log.warning(
                    "tier.skipped", request_id=ctx.request_id, tier=tier.tier.value, reason="budget"
                )
                limit = EscalationReason.BUDGET_EXHAUSTED.value
                break
            if not self._policy.can_afford(tier.tier, ctx):
                # Only reachable for the first rung run: every later one was
                # already priced by the policy before it said "climb".
                _log.info(
                    "tier.skipped",
                    request_id=ctx.request_id,
                    tier=tier.tier.value,
                    reason="cost_ceiling",
                    expected_cost_usd=self._policy.expected_cost_usd(tier.tier),
                    max_cost_usd=ctx.options.max_cost_usd,
                )
                limit = EscalationReason.COST_CEILING.value
                break

            async with span(
                f"tier.{tier.tier.value}",
                tier=tier.tier.value,
                request_id=ctx.request_id,
                timeout_s=timeout_s,
            ):
                with tier_timer(tier.tier):
                    verdict = await self._analyse(tier, ctx, timeout_s)

            degraded = degraded or not verdict.succeeded
            verdicts.append(verdict)
            ctx = ctx.with_verdict(verdict)

            # The lookahead spans the *rest of the ladder*, not just the adjacent
            # rung: a dead rung one hop up must not mask a healthy one two hops
            # up. `next_position` is the absolute index of the nearest rung above
            # `position` that can serve, or None if nothing above can.
            next_position = await self._next_available_position(ladder, position + 1)
            next_tier = ladder[next_position] if next_position is not None else None
            decision = self._policy.decide(
                verdict,
                ctx,
                next_available=next_tier is not None,
                next_cost_usd=(
                    self._policy.expected_cost_usd(next_tier.tier) if next_tier is not None else 0.0
                ),
            )
            steps.append(EscalationStep.build(verdict, decision))
            last_reason = decision.primary_reason
            record_tier_decision(tier.tier, _outcome_of(verdict, escalate=decision.escalate))
            _log.info(
                "tier.decision",
                request_id=ctx.request_id,
                tier=tier.tier.value,
                label=verdict.result.label.value,
                confidence=round(verdict.result.confidence, 4),
                margin=round(verdict.label_margin, 4),
                latency_ms=round(verdict.latency_ms, 2),
                succeeded=verdict.succeeded,
                escalate=decision.escalate,
                reasons=[reason.value for reason in decision.reasons],
                **safe_text_fields(ctx.text, self._settings),
            )

            # A verdict that succeeded but was not good enough, with nothing
            # further up the ladder to hand it to, is a capability gap — the
            # caller is getting a worse answer than the ladder's own thresholds
            # call acceptable, purely because a dependency was not there. That
            # is "degrade, don't fail" too, not only an outright tier failure:
            # `degraded` (and hence the response `degraded` field) must say so,
            # and this is the one stop reason `EscalationStep.build` does not
            # collapse to null, so a caller sees it in the same place.
            capacity_gap = (
                not decision.escalate
                and decision.primary_reason is EscalationReason.NEXT_TIER_UNAVAILABLE
            )
            if capacity_gap:
                degraded = True
                _log.warning(
                    "ladder.capacity_gap",
                    request_id=ctx.request_id,
                    stopped_at=tier.tier.value,
                    confidence=round(verdict.result.confidence, 4),
                    reasons=[reason.value for reason in decision.reasons],
                )

            if not decision.escalate:
                if decision.primary_reason in _CALLER_LIMITS:
                    limit = decision.primary_reason.value
                break
            # The policy only escalates when the lookahead found a servable rung,
            # so both are guaranteed here — this is not a second check.
            assert next_position is not None
            assert next_tier is not None
            for reason in decision.reasons:
                record_escalation(tier.tier, next_tier.tier, reason)
            for skipped in ladder[position + 1 : next_position]:
                # A rung the ladder wanted and could not use, because it was
                # down, is the same outage whether its call failed or the
                # lookahead stepped over it — so it marks the answer degraded
                # either way (SIT CL-006i). Before this, a T2 whose gateway call
                # failed made the response degraded but a T2 the probe already
                # knew was down did not.
                degraded = True
                _log.warning(
                    "tier.skipped",
                    request_id=ctx.request_id,
                    tier=skipped.tier.value,
                    reason="unavailable",
                )
            position = next_position

        best = self._policy.select_best(verdicts)
        if best is None:
            message = f"no usable verdict after {len(verdicts)} of {len(ladder)} tiers"
            if degraded or limit is None:
                raise NoVerdictError(message, retryable=True)
            raise NoVerdictError(f"{message}; stopped by {limit}", retryable=False, limit=limit)

        record_decided_by(best.tier)
        # Settled: the answer meets its own rung's floors, and the ladder did not
        # stop *short* of where it wanted to be — no later rung was tried and
        # failed, and it did not stop for want of one. A result that is degraded
        # only because a dead rung was hopped on the way is still settled.
        settled = (
            best is verdicts[-1]
            and not best.fallback
            and last_reason is not EscalationReason.NEXT_TIER_UNAVAILABLE
            and self._policy.is_settled(best)
        )
        return OrchestrationResult(
            best=best,
            verdicts=verdicts,
            steps=steps,
            degraded=degraded or best.fallback,
            settled=settled,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ladder_for(self, ctx: TierContext) -> tuple[Tier, ...]:
        """Registered rungs the request's options permit, cheapest first."""
        return tuple(
            self._tiers[tier_id]
            for tier_id in self._policy.enabled_tiers_for(ctx.options)
            if tier_id in self._tiers
        )

    def _timeout_s(self, tier: Tier, ctx: TierContext) -> float:
        """Whichever runs out first: this rung's own timeout or the request's budget."""
        configured_ms = float(self._settings.tiers.for_tier(tier.tier).timeout_ms)
        return min(configured_ms, ctx.remaining_ms) / 1000.0

    async def _analyse(self, tier: Tier, ctx: TierContext, timeout_s: float) -> TierVerdict:
        """Run one rung, converting every failure mode into a failed verdict."""
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(tier.analyse(ctx), timeout=timeout_s)
        except TimeoutError:
            error = f"timed out after {timeout_s * 1000:.0f}ms"
            _log.warning(
                "tier.timeout",
                request_id=ctx.request_id,
                tier=tier.tier.value,
                timeout_s=timeout_s,
            )
        except TierError as exc:
            error = f"{type(exc).__name__}: {exc}"
            _log.warning(
                "tier.failed", request_id=ctx.request_id, tier=tier.tier.value, error=str(exc)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # Logged at error level with the traceback: the ladder survives it,
            # but nobody intended it, so it should page a human eventually.
            _log.error(
                "tier.defect",
                request_id=ctx.request_id,
                tier=tier.tier.value,
                error=str(exc),
                exc_info=True,
            )
        return TierVerdict.failed(
            tier.tier,
            error,
            model_version=tier.model_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _next_available_position(
        self, ladder: Sequence[Tier], from_position: int
    ) -> int | None:
        """Index of the nearest servable rung at or after `from_position`.

        Walks the ladder in order rather than probing every remaining rung up
        front: most requests climb at most once, so paying for a probe past the
        rung that actually answers is wasted work most of the time.
        """
        for index in range(from_position, len(ladder)):
            if await self._can_serve(ladder[index]):
                return index
        return None

    @staticmethod
    async def _can_serve(tier: Tier | None) -> bool:
        """Lookahead for the policy: is there a next rung, and can it serve?

        A probe that raises is a probe that failed, and the contract says it must
        not raise — so treating it as unavailable is both safe and a fair
        reading of what just happened.
        """
        if tier is None:
            return False
        try:
            return await tier.is_available()
        except Exception as exc:
            _log.warning("tier.probe_failed", tier=tier.tier.value, error=str(exc))
            return False


#: Stops the caller's own options cause. A ladder that ends empty-handed on one
#: of these, with nothing failing on the way, fails the same way every time.
_CALLER_LIMITS: frozenset[EscalationReason] = frozenset(
    {
        EscalationReason.TIER_CEILING,
        EscalationReason.BUDGET_EXHAUSTED,
        EscalationReason.COST_CEILING,
    }
)


def _outcome_of(verdict: TierVerdict, *, escalate: bool) -> TierOutcome:
    """How a rung's turn is counted in `tier_decisions_total`.

    A failed rung is reported as failed rather than as the escalation it also
    caused, so the metric separates "the ladder chose to climb" from "the ladder
    had to".
    """
    if not verdict.succeeded:
        return TierOutcome.FAILED
    return TierOutcome.ESCALATE if escalate else TierOutcome.STOP
