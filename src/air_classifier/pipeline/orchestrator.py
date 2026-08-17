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
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from air_classifier.config import Settings
from air_classifier.constants import Tier as TierId
from air_classifier.observability.logging import get_logger, safe_text_fields
from air_classifier.observability.metrics import (
    TierOutcome,
    record_decided_by,
    record_escalation,
    record_tier_decision,
    tier_timer,
)
from air_classifier.observability.tracing import span
from air_classifier.pipeline.policy import EscalationPolicy
from air_classifier.pipeline.tiers.base import Tier, TierContext, TierError
from air_classifier.schemas.verdict import (
    EscalationStep,
    OrchestrationResult,
    TierVerdict,
)

__all__ = ["NoVerdictError", "Orchestrator"]

_log = get_logger(__name__)


class NoVerdictError(Exception):
    """No rung produced a usable answer.

    The single path to a 5xx in this service. A provider outage does not raise
    it — that is a lower rung's verdict with ``degraded=true``. Only an empty
    ladder, or one where every rung failed or abstained, gets here.
    """


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

        for position, tier in enumerate(ladder):
            timeout_s = self._timeout_s(tier, ctx)
            if timeout_s <= 0.0:
                # Starting a rung that cannot finish buys nothing but latency.
                _log.warning(
                    "tier.skipped", request_id=ctx.request_id, tier=tier.tier.value, reason="budget"
                )
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

            next_tier = ladder[position + 1] if position + 1 < len(ladder) else None
            decision = self._policy.decide(
                verdict, ctx, next_available=await self._can_serve(next_tier)
            )
            steps.append(EscalationStep.build(verdict, decision))
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

            if not decision.escalate:
                break
            if next_tier is not None:
                for reason in decision.reasons:
                    record_escalation(tier.tier, next_tier.tier, reason)

        best = self._policy.select_best(verdicts)
        if best is None:
            raise NoVerdictError(f"no usable verdict after {len(verdicts)} of {len(ladder)} tiers")

        record_decided_by(best.tier)
        return OrchestrationResult(best=best, verdicts=verdicts, steps=steps, degraded=degraded)

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


def _outcome_of(verdict: TierVerdict, *, escalate: bool) -> TierOutcome:
    """How a rung's turn is counted in `tier_decisions_total`.

    A failed rung is reported as failed rather than as the escalation it also
    caused, so the metric separates "the ladder chose to climb" from "the ladder
    had to".
    """
    if not verdict.succeeded:
        return TierOutcome.FAILED
    return TierOutcome.ESCALATE if escalate else TierOutcome.STOP
