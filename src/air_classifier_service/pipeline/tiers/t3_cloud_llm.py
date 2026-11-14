"""Tier 3 — the metered cloud rung, and the last one.

Everything below this tier is cheap enough to run speculatively; this one is
not, which shapes both of its behaviours.

**Arbitration is a second opinion, not a retry.** When the fast model comes back
under ``arbitration_floor`` it is telling us it does not know, and the honest
response to that is a stronger model rather than the same one again. It happens
at most once — there is no loop here — and both calls' tokens and cost are
counted, because a spend ceiling that only saw the second call would be a lie.

**Failure degrades, it never propagates.** A refusal, an open breaker, or air-model-gateway
being unreachable becomes a :class:`TierError`, which the orchestrator turns
into a failed verdict and a lower rung's answer. There is no path from this
module to a 5xx.

This rung does not select *among* cloud providers — that failover is
air-model-gateway's job, expressed as a model class whose chain the gateway owns
(``config.py``'s ``ModelGatewaySettings.cloud_model``/``arbiter_model``), not a
``ProviderRegistry`` here. What is still
this rung's job: deciding *whether* the fast model's answer was good enough, and
asking for a second, stronger opinion when it was not.
"""

from __future__ import annotations

import time
from typing import Final

from air_classifier_service.config import ModelGatewaySettings, Settings
from air_classifier_service.constants import Tier
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.observability.metrics import record_llm_usage
from air_classifier_service.observability.tracing import span
from air_classifier_service.pipeline.tiers.base import TierContext, TierError, TierUnavailableError
from air_classifier_service.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
)
from air_classifier_service.providers.prompts.loader import (
    repair_instruction,
    system_prompt,
    system_prompt_version,
    user_content,
)
from air_classifier_service.resilience.breaker import CircuitBreaker, CircuitOpenError
from air_classifier_service.resilience.bulkhead import Bulkhead, BulkheadFullError
from air_classifier_service.resilience.guard import guarded
from air_classifier_service.schemas.common import TierUsage
from air_classifier_service.schemas.verdict import LLMSentimentOutput, TierVerdict

logger = get_logger(__name__)

#: Floor on the per-call timeout. ``asyncio.timeout`` fires immediately on a
#: non-positive deadline, so a nearly-spent budget would abort before the request
#: was even sent — an expensive way to learn nothing.
_MIN_TIMEOUT_S: Final[float] = 0.05


class CloudLLMTier:
    """The top rung: a metered cloud model, with a stronger one behind it."""

    tier: Tier = Tier.T3_CLOUD_LLM

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        *,
        bulkhead: Bulkhead | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._config: ModelGatewaySettings = settings.model_gateway
        self._tier_settings = settings.tiers.for_tier(self.tier)
        dependency = provider.name.value
        self._bulkhead = bulkhead or Bulkhead.from_settings(dependency, settings.resilience)
        self._breaker = breaker or CircuitBreaker.from_settings(dependency, settings.resilience)
        # The class asked for (`auto`); the concrete model is per answer.
        self.model_version: str = self._config.cloud_model.value

    async def is_available(self) -> bool:
        if not self._provider.configured or not self._breaker.is_closed:
            return False
        try:
            return await self._provider.health()
        except Exception as exc:
            logger.warning("t3.availability_probe_failed", error=type(exc).__name__)
            return False

    async def unavailable_reason(self) -> str | None:
        """Same order `is_available` checks in, so the two cannot disagree
        about *which* condition is the one actually blocking this rung."""
        if not self._provider.configured:
            # `configured` is a gateway-level check (MODEL_GATEWAY__ENABLED /
            # MODEL_GATEWAY__TOKEN), not per-vendor — naming the breaker label here
            # would send an operator looking for a knob that does not exist.
            return "air-model-gateway not configured"
        if not self._breaker.is_closed:
            return f"circuit breaker open for {self._provider.name.value} — retries after cooldown"
        return self._provider.health_detail

    def supports_language(self, code: str) -> bool:
        """Every language, which is a large part of why this rung exists.

        The classifier below it is English-only, so an unsupported language is
        one of the escalation reasons that lands work here.
        """
        return True

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        if ctx.budget_exhausted:
            raise TierUnavailableError("latency budget exhausted before the cloud call")

        started = time.perf_counter()
        async with span(
            "tier.t3_cloud_llm",
            model=self._config.cloud_model,
            prompt_version=system_prompt_version(),
        ):
            primary = await self._call(
                ctx,
                model=self._config.cloud_model,
                max_tokens=self._config.max_tokens,
                user_text=user_content(ctx),
            )
            output = _output_of(primary)
            final, usage = await self._arbitrate(ctx, primary=primary, output=output)

        if final is not primary:
            output = _output_of(final)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return output.to_verdict(
            self.tier,
            model_version=f"{final.provider.value}:{final.model}",
            latency_ms=latency_ms,
            usage=usage,
        )

    # ── Arbitration ───────────────────────────────────────────────────────

    async def _arbitrate(
        self,
        ctx: TierContext,
        *,
        primary: ProviderResult,
        output: LLMSentimentOutput,
    ) -> tuple[ProviderResult, TierUsage]:
        """Take a second opinion from the arbiter model, once, if it is warranted.

        Returns whichever result stands plus the usage of *every* call made, so
        the cost ceiling sees the true spend rather than the winning half of it.
        """
        usage = primary.to_usage()
        if not self._should_arbitrate(ctx, primary=primary, output=output):
            return primary, usage

        # The arbiter is shown what the fast model concluded, threaded through
        # the same `prior_verdicts` channel the lower tiers use: "haiku said
        # negative at 0.51" is the whole reason this call is happening.
        hinted = ctx.with_verdict(
            output.to_verdict(
                self.tier,
                model_version=primary.model,
                latency_ms=primary.latency_ms,
                usage=usage,
            )
        )

        try:
            arbiter = await self._call(
                hinted,
                model=self._config.arbiter_model,
                max_tokens=self._config.arbiter_max_tokens,
                user_text=user_content(hinted),
            )
        except TierError as exc:
            # A failed second opinion is not a failed request: the primary answer
            # is unconfident, not unusable, and returning it beats degrading to a
            # rung that knows less.
            logger.warning(
                "t3.arbitration_failed",
                arbiter_model=self._config.arbiter_model,
                error=type(exc).__name__,
            )
            return primary, usage

        logger.info(
            "t3.arbitrated",
            from_model=primary.model,
            to_model=arbiter.model,
            from_confidence=round(output.confidence, 3),
            to_confidence=round(_output_of(arbiter).confidence, 3),
        )
        return arbiter, usage + arbiter.to_usage()

    def _should_arbitrate(
        self,
        ctx: TierContext,
        *,
        primary: ProviderResult,
        output: LLMSentimentOutput,
    ) -> bool:
        config = self._config
        if not config.arbitration_enabled or output.confidence >= config.arbitration_floor:
            return False
        if ctx.budget_exhausted:
            logger.info("t3.arbitration_skipped", reason="budget_exhausted")
            return False
        # The policy prices a rung before it runs, which covers this rung's first
        # call but not its second. The arbiter is priced the same way here —
        # what has been spent plus what the arbiter is expected to cost — so the
        # second opinion cannot push a request past a ceiling the caller set
        # (SIT CL-001: checking only the spend so far let it through every time).
        spent = ctx.cost_spent_usd + primary.cost_usd
        expected = config.arbiter_expected_cost_usd
        if spent + expected > ctx.options.max_cost_usd:
            logger.info(
                "t3.arbitration_skipped",
                reason="cost_ceiling",
                spent_usd=spent,
                expected_usd=expected,
            )
            return False
        return True

    # ── Calling ───────────────────────────────────────────────────────────

    async def _call(
        self,
        ctx: TierContext,
        *,
        model: str,
        max_tokens: int,
        user_text: str,
    ) -> ProviderResult:
        """One logical call, with a single bounded repair retry on a bad parse.

        Only a schema failure is retried, and only once. A transport failure is
        air-model-gateway's own retry budget to spend, and a second bad parse is a model
        that cannot follow the schema today — looping on it burns the latency
        budget to arrive at the same escalation.
        """
        try:
            return await self._guarded_call(
                ctx, model=model, max_tokens=max_tokens, user_text=user_text
            )
        except ProviderSchemaError as exc:
            # The validation detail quotes the model's own output back, which is
            # derived from the caller's text — it goes to the model in the repair
            # instruction, never to the log stream.
            logger.warning("t3.schema_repair", model=model, error=type(exc).__name__)
            correction = repair_instruction(str(exc))

        try:
            return await self._guarded_call(
                ctx,
                model=model,
                max_tokens=max_tokens,
                user_text=user_text + correction,
            )
        except ProviderSchemaError as exc:
            raise TierError(f"cloud provider could not satisfy the schema: {exc}") from exc

    async def _guarded_call(
        self,
        ctx: TierContext,
        *,
        model: str,
        max_tokens: int,
        user_text: str,
    ) -> ProviderResult:
        timeout_s = self._timeout_s(ctx)
        try:
            result = await guarded(
                self._provider.classify,
                timeout_s=timeout_s,
                breaker=self._breaker,
                bulkhead=self._bulkhead,
                system_prompt=system_prompt(),
                user_content=user_text,
                output_model=LLMSentimentOutput,
                model=model,
                max_tokens=max_tokens,
                cache_prefix=True,
                remaining_ms=timeout_s * 1000.0,
                request_id=ctx.request_id,
            )
        except ProviderRefusalError as exc:
            raise TierError(f"cloud provider refused: {exc}") from exc
        except ProviderSchemaError:
            raise
        except CircuitOpenError as exc:
            raise TierUnavailableError(str(exc)) from exc
        except BulkheadFullError as exc:
            raise TierUnavailableError(str(exc)) from exc
        except ProviderError as exc:
            raise TierError(f"cloud provider failed: {type(exc).__name__}: {exc}") from exc

        _record(result)
        return result

    def _timeout_s(self, ctx: TierContext) -> float:
        """The tightest of the three budgets that apply to this call.

        The caller's remaining latency budget is one of them, so a request that
        has already spent most of its allowance does not get to spend a fresh
        eight seconds here on top.
        """
        return max(
            _MIN_TIMEOUT_S,
            min(self._tier_settings.timeout_ms / 1000.0, ctx.remaining_ms / 1000.0),
        )


def _output_of(result: ProviderResult) -> LLMSentimentOutput:
    """Narrow the provider's validated model back to the one we asked for."""
    parsed = result.parsed
    if not isinstance(parsed, LLMSentimentOutput):
        raise TierError(f"cloud provider returned {type(parsed).__name__}, not a sentiment output")
    return parsed


def _record(result: ProviderResult) -> None:
    # `result.provider` is who actually answered, which may differ from this
    # tier's own provider label when the gateway's chain failed over vendors.
    record_llm_usage(
        result.provider,
        result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cache_read=result.cache_read_tokens,
        cache_write=result.cache_write_tokens,
        cost_usd=result.cost_usd,
    )
