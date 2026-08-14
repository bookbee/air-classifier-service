"""Tier 3 — the metered cloud rung, and the last one.

Everything below this tier is cheap enough to run speculatively; this one is
not, which shapes all three of its behaviours.

**Selection happens before the call.** The registry consults breaker state and
cached health first, so an outage moves traffic wholesale instead of charging
every request one failed attempt.

**Arbitration is a second opinion, not a retry.** When the fast model comes back
under ``arbitration_floor`` it is telling us it does not know, and the honest
response to that is a stronger model rather than the same one again. It happens
at most once — there is no loop here — and both calls' tokens and cost are
counted, because a spend ceiling that only saw the second call would be a lie.

**Failure degrades, it never propagates.** A refusal, an open breaker, or no
provider at all becomes a :class:`TierError`, which the orchestrator turns into a
failed verdict and a lower rung's answer. There is no path from this module to a
5xx.
"""

from __future__ import annotations

import time
from typing import Final

from air_sentiment.config import Settings
from air_sentiment.constants import Provider, Tier
from air_sentiment.observability.logging import get_logger
from air_sentiment.observability.metrics import record_llm_usage
from air_sentiment.observability.tracing import span
from air_sentiment.pipeline.tiers.base import TierContext, TierError, TierUnavailableError
from air_sentiment.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
)
from air_sentiment.providers.prompts.loader import (
    repair_instruction,
    system_prompt,
    user_content,
)
from air_sentiment.providers.registry import (
    CloudProviderSettings,
    NoProviderAvailableError,
    ProviderRegistry,
)
from air_sentiment.resilience.breaker import CircuitOpenError
from air_sentiment.resilience.bulkhead import Bulkhead, BulkheadFullError
from air_sentiment.resilience.guard import guarded
from air_sentiment.schemas.common import TierUsage
from air_sentiment.schemas.verdict import LLMSentimentOutput, TierVerdict

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
        registry: ProviderRegistry,
        bulkhead: Bulkhead | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._bulkhead = bulkhead
        self._tier_settings = settings.tiers.for_tier(self.tier)
        self.model_version = _default_model_version(settings)

    async def is_available(self) -> bool:
        try:
            return await self._registry.available()
        except Exception as exc:
            logger.warning("t3.availability_probe_failed", error=type(exc).__name__)
            return False

    def supports_language(self, code: str) -> bool:
        """Every language, which is a large part of why this rung exists.

        The classifier below it is English-only, so an unsupported language is
        one of the escalation reasons that lands work here.
        """
        return True

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        if ctx.budget_exhausted:
            raise TierUnavailableError("latency budget exhausted before the cloud call")

        try:
            provider, model = await self._registry.select(ctx)
        except NoProviderAvailableError as exc:
            raise TierUnavailableError(str(exc)) from exc

        config = self._registry.config_for(provider.name)
        started = time.perf_counter()

        async with span("tier.t3_cloud_llm", provider=provider.name.value, model=model):
            primary = await self._call(
                provider,
                ctx,
                model=model,
                max_tokens=config.max_tokens,
                user_text=user_content(ctx),
            )
            output = _output_of(primary)
            final, usage = await self._arbitrate(
                provider, ctx, config, primary=primary, output=output
            )

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
        provider: LLMProvider,
        ctx: TierContext,
        config: CloudProviderSettings,
        *,
        primary: ProviderResult,
        output: LLMSentimentOutput,
    ) -> tuple[ProviderResult, TierUsage]:
        """Take a second opinion from the arbiter model, once, if it is warranted.

        Returns whichever result stands plus the usage of *every* call made, so
        the cost ceiling sees the true spend rather than the winning half of it.
        """
        usage = primary.to_usage()
        if not self._should_arbitrate(ctx, config, primary=primary, output=output):
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
                provider,
                hinted,
                model=config.arbiter_model,
                max_tokens=config.arbiter_max_tokens,
                user_text=user_content(hinted),
            )
        except TierError as exc:
            # A failed second opinion is not a failed request: the primary answer
            # is unconfident, not unusable, and returning it beats degrading to a
            # rung that knows less.
            logger.warning(
                "t3.arbitration_failed",
                arbiter_model=config.arbiter_model,
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
        config: CloudProviderSettings,
        *,
        primary: ProviderResult,
        output: LLMSentimentOutput,
    ) -> bool:
        if not config.arbitration_enabled or output.confidence >= config.arbitration_floor:
            return False
        if ctx.budget_exhausted:
            logger.info("t3.arbitration_skipped", reason="budget_exhausted")
            return False
        # The policy's cost ceiling applies between rungs, which is one call too
        # late for a rung that makes two: the spend the first call has *already*
        # incurred is checked here so the arbiter cannot push a request past a
        # ceiling the caller set.
        spent = ctx.cost_spent_usd + primary.cost_usd
        if spent >= ctx.options.max_cost_usd:
            logger.info("t3.arbitration_skipped", reason="cost_ceiling", spent_usd=spent)
            return False
        return True

    # ── Calling ───────────────────────────────────────────────────────────

    async def _call(
        self,
        provider: LLMProvider,
        ctx: TierContext,
        *,
        model: str,
        max_tokens: int,
        user_text: str,
    ) -> ProviderResult:
        """One logical call, with a single bounded repair retry on a bad parse.

        Only a schema failure is retried, and only once. A transport failure is
        the SDK's own retry budget to spend, and a second bad parse is a model
        that cannot follow the schema today — looping on it burns the latency
        budget to arrive at the same escalation.
        """
        try:
            return await self._guarded_call(
                provider, ctx, model=model, max_tokens=max_tokens, user_text=user_text
            )
        except ProviderSchemaError as exc:
            # The validation detail quotes the model's own output back, which is
            # derived from the caller's text — it goes to the model in the repair
            # instruction, never to the log stream.
            logger.warning("t3.schema_repair", model=model, error=type(exc).__name__)
            correction = repair_instruction(str(exc))

        try:
            return await self._guarded_call(
                provider,
                ctx,
                model=model,
                max_tokens=max_tokens,
                user_text=user_text + correction,
            )
        except ProviderSchemaError as exc:
            raise TierError(f"cloud provider could not satisfy the schema: {exc}") from exc

    async def _guarded_call(
        self,
        provider: LLMProvider,
        ctx: TierContext,
        *,
        model: str,
        max_tokens: int,
        user_text: str,
    ) -> ProviderResult:
        try:
            result = await guarded(
                provider.classify,
                timeout_s=self._timeout_s(ctx),
                breaker=self._registry.breaker_for(provider.name),
                bulkhead=self._bulkhead,
                system_prompt=system_prompt(),
                user_content=user_text,
                output_model=LLMSentimentOutput,
                model=model,
                max_tokens=max_tokens,
                cache_prefix=True,
            )
        except ProviderRefusalError as exc:
            raise TierError(f"cloud provider refused: {exc}") from exc
        except ProviderSchemaError:
            raise
        except CircuitOpenError as exc:
            self._registry.invalidate_health(provider.name)
            raise TierUnavailableError(str(exc)) from exc
        except BulkheadFullError as exc:
            raise TierUnavailableError(str(exc)) from exc
        except ProviderError as exc:
            self._registry.invalidate_health(provider.name)
            raise TierError(f"cloud provider failed: {type(exc).__name__}: {exc}") from exc

        _record(provider.name, result)
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


def _record(provider: Provider, result: ProviderResult) -> None:
    record_llm_usage(
        provider,
        result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cache_read=result.cache_read_tokens,
        cache_write=result.cache_write_tokens,
        cost_usd=result.cost_usd,
    )


def _default_model_version(settings: Settings) -> str:
    """What ``/v1/capabilities`` reports for this rung.

    The head of the preference order rather than whichever provider served the
    last request: this describes the configuration, while a verdict's own
    ``model_version`` records the model that actually answered.
    """
    for name in settings.providers.preference:
        if name is Provider.ANTHROPIC:
            return f"{name.value}:{settings.providers.anthropic.model}"
        if name is Provider.OPENAI:
            return f"{name.value}:{settings.providers.openai.model}"
    return "cloud:unconfigured"
