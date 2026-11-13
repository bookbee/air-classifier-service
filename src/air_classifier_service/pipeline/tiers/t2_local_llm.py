"""Tier 2 — the local LLM.

The rung that makes the ladder pay: a 3B model on our own hardware reasons about
contrast, negation and sarcasm well enough to settle most of what Tier 1 cannot,
at no per-token cost. Everything here exists to keep that bargain honest — a
local model that is slow, saturated, or babbling must fail *fast* and hand the
request up to Tier 3, because a local tier that hangs is strictly worse than one
that is absent.

Hence three guards, none of them optional:

* A **bulkhead**, because local inference saturates CPU. Unbounded queueing in
  front of a box that can serve two concurrent generations turns a latency spike
  into an outage, and does it to endpoints that never touched this tier.
* A **circuit breaker**, so a stopped daemon costs one probe per cooldown rather
  than one connection refusal per request.
* Exactly **one** repair retry on a schema violation. Small models fail
  validation in ways they cannot fix; looping against one is a documented way to
  burn a latency budget to no effect, so a second failure escalates instead.
"""

from __future__ import annotations

import time
from typing import Final

from air_classifier_service.config import Settings
from air_classifier_service.constants import Tier
from air_classifier_service.observability.logging import get_logger, safe_text_fields
from air_classifier_service.observability.metrics import record_llm_usage
from air_classifier_service.observability.tracing import span
from air_classifier_service.pipeline.tiers.base import (
    TierContext,
    TierError,
    TierUnavailableError,
)
from air_classifier_service.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier_service.providers.prompts.loader import (
    compact_system_prompt,
    compact_system_prompt_version,
    repair_instruction,
    user_content,
)
from air_classifier_service.resilience.breaker import CircuitBreaker, CircuitOpenError
from air_classifier_service.resilience.bulkhead import Bulkhead, BulkheadFullError
from air_classifier_service.resilience.guard import guarded
from air_classifier_service.schemas.verdict import LLMSentimentOutput, TierVerdict

logger = get_logger(__name__)

#: Output ceiling for one verdict. The response object is small by construction —
#: a bounded rationale, at most five emotions and ten aspects — so this is a
#: runaway guard rather than a budget: a model that starts repeating itself is
#: cut off instead of being allowed to fill the context window.
_MAX_OUTPUT_TOKENS: Final[int] = 768


class LocalLLMTier:
    """Rung 2: a locally served instruction-tuned model."""

    tier: Tier = Tier.T2_LOCAL_LLM

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
        self._model = settings.air_llm.local_model
        # The class asked for (`local`); the concrete model is per answer.
        self.model_version: str = self._model.value

        # Both guards are named after the *dependency*, not the rung: they exist
        # to protect one daemon, so anything else that reaches it must be able to
        # share them — and the breaker gauge is labelled by provider.
        dependency = provider.name.value
        self._bulkhead = bulkhead or Bulkhead.from_settings(dependency, settings.resilience)
        self._breaker = breaker or CircuitBreaker.from_settings(dependency, settings.resilience)

    async def is_available(self) -> bool:
        """Whether the daemon is reachable *and* the breaker will admit a call.

        The breaker is consulted first: it answers without I/O, and an open
        breaker means the answer is no however healthy the daemon looks.
        """
        if not self._provider.configured or not self._breaker.is_closed:
            return False
        try:
            return await self._provider.health()
        except Exception:
            # The protocol says health() must not raise. If one does, it has
            # already told us what we needed to know.
            logger.debug("tier.t2.health_raised", tier=self.tier.value)
            return False

    async def unavailable_reason(self) -> str | None:
        """Same order `is_available` checks in, so the two cannot disagree
        about *which* condition is the one actually blocking this rung."""
        if not self._provider.configured:
            # `configured` is an air-llm-level check (AIR_LLM__ENABLED /
            # AIR_LLM__TOKEN), not per-vendor — naming the breaker label here
            # would send an operator looking for a knob that does not exist.
            return "air-llm not configured"
        if not self._breaker.is_closed:
            return f"circuit breaker open for {self._provider.name.value} — retries after cooldown"
        return self._provider.health_detail

    def supports_language(self, code: str) -> bool:
        """Every language. An instruction-tuned model is prompted, not trained.

        Quality does fall off outside the languages the base model saw most of,
        but that shows up as low confidence — which the policy already escalates
        on — rather than as a refusal to try.
        """
        return True

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        # The compact prefix, not the cache-anchored one: a local model re-reads
        # the whole prompt on every call, so the long rubric costs ~9.9s here and
        # buys nothing. See prompts.loader.compact_system_prompt.
        system = compact_system_prompt()
        user = user_content(ctx)

        started = time.perf_counter()
        async with span(
            f"tier.{self.tier.value}",
            tier=self.tier.value,
            model=self._model,
            request_id=ctx.request_id,
            prompt_version=compact_system_prompt_version(),
        ):
            result = await self._classify_with_one_repair(ctx, system, user)
        latency_ms = (time.perf_counter() - started) * 1000.0

        parsed = result.parsed
        if not isinstance(parsed, LLMSentimentOutput):
            raise TierError(f"provider returned {type(parsed).__name__}, not a sentiment output")

        record_llm_usage(
            self._provider.name,
            result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
        )
        return parsed.to_verdict(
            self.tier,
            model_version=f"{result.provider.value}:{result.model}",
            latency_ms=latency_ms,
            usage=result.to_usage(),
        )

    # ── Internals ─────────────────────────────────────────────────────────

    async def _classify_with_one_repair(
        self, ctx: TierContext, system: str, user: str
    ) -> ProviderResult:
        """One call, then at most one more with the validation errors appended."""
        try:
            return await self._classify(ctx, system, user)
        except ProviderSchemaError as first:
            errors = str(first)

        logger.warning(
            "tier.t2.schema_repair",
            tier=self.tier.value,
            model=self._model,
            error=errors,
            **safe_text_fields(ctx.text, self._settings),
        )
        try:
            return await self._classify(ctx, system, user + repair_instruction(errors))
        except ProviderSchemaError as second:
            raise TierError(f"invalid output after one repair attempt: {second}") from second

    async def _classify(self, ctx: TierContext, system: str, user: str) -> ProviderResult:
        """Call the provider behind every guard, and speak the ladder's language.

        No retry policy: a transport failure here is not worth re-attempting
        inside the rung's own budget, because the ladder above *is* the retry and
        it retries with a better model. The repair pass above is the one
        exception, and it is a different request rather than the same one again.
        """
        timeout_s = self._budget_s(ctx)
        try:
            return await guarded(
                self._provider.classify,
                timeout_s=timeout_s,
                breaker=self._breaker,
                bulkhead=self._bulkhead,
                system_prompt=system,
                user_content=user,
                output_model=LLMSentimentOutput,
                model=self._model,
                max_tokens=_MAX_OUTPUT_TOKENS,
                cache_prefix=False,
                remaining_ms=timeout_s * 1000.0,
                request_id=ctx.request_id,
            )
        except ProviderSchemaError:
            raise
        except (BulkheadFullError, CircuitOpenError, ProviderUnavailableError) as exc:
            raise TierUnavailableError(str(exc)) from exc
        except TimeoutError as exc:
            raise TierError(f"local model exceeded its {timeout_s:.3f}s slice") from exc
        except ProviderError as exc:
            raise TierError(f"{type(exc).__name__}: {exc}") from exc

    def _budget_s(self, ctx: TierContext) -> float:
        """Seconds this call may take: the tier's slice, capped by what is left.

        The tier timeout is a ceiling, not an entitlement. Spending 4s here on a
        request with 300ms of budget remaining would leave Tier 3 nothing to work
        with and blow the caller's deadline for a verdict nobody waits for.
        """
        tier_timeout_ms = float(self._settings.tiers.for_tier(self.tier).timeout_ms)
        budget_ms = min(tier_timeout_ms, ctx.remaining_ms)
        if budget_ms <= 0.0:
            raise TierError("latency budget exhausted before the local model was called")
        return budget_ms / 1000.0
