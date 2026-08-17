"""Tier 3 tests.

Arbitration is the expensive behaviour, so most of this file pins exactly when it
fires: below the floor and only then, once and never twice, and with both calls'
tokens folded into the verdict. The rest pins the degradation contract — a
refusal, an outage, or a model that cannot follow the schema must all leave this
tier as a ``TierError`` so the ladder falls back to a cheaper rung.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from air_classifier.config import (
    AnthropicSettings,
    AppSettings,
    ProvidersSettings,
    Settings,
)
from air_classifier.constants import Label, Provider, ResourceKind
from air_classifier.observability.cost import ModelPrice, override_prices, reset_price_table
from air_classifier.pipeline.features import extract_features
from air_classifier.pipeline.tiers.base import TierContext, TierError, TierUnavailableError
from air_classifier.pipeline.tiers.t3_cloud_llm import CloudLLMTier
from air_classifier.providers.base import (
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier.providers.registry import ProviderRegistry
from air_classifier.resilience.breaker import CircuitBreaker
from air_classifier.resilience.bulkhead import Bulkhead
from air_classifier.schemas.common import LanguageInfo, Options, SafetyReport
from air_classifier.schemas.verdict import LLMSentimentOutput

TEXT = "Great battery but the camera is awful."
HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"


# ── Doubles ───────────────────────────────────────────────────────────────────


class _ScriptedProvider:
    """Replays a scripted list of results/exceptions and records every call."""

    def __init__(self, name: Provider, outcomes: list[Any], *, healthy: bool = True) -> None:
        self.name = name
        self._outcomes = list(outcomes)
        self.healthy = healthy
        self.calls: list[dict[str, Any]] = []

    @property
    def configured(self) -> bool:
        return True

    async def health(self) -> bool:
        return self.healthy

    async def classify(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[BaseModel],
        model: str,
        max_tokens: int,
        cache_prefix: bool = True,
    ) -> ProviderResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_content": user_content,
                "output_model": output_model,
                "model": model,
                "max_tokens": max_tokens,
                "cache_prefix": cache_prefix,
            }
        )
        if not self._outcomes:
            raise AssertionError(f"unscripted call to {model}")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None

    @property
    def models_called(self) -> list[str]:
        return [call["model"] for call in self.calls]


def _result(
    *,
    confidence: float,
    model: str = HAIKU,
    label: Label = Label.NEGATIVE,
    tokens_in: int = 100,
    tokens_out: int = 20,
    cache_read: int = 0,
    cost_usd: float = 0.0,
) -> ProviderResult:
    return ProviderResult(
        parsed=LLMSentimentOutput(
            label=label,
            polarity=-0.5,
            confidence=confidence,
            rationale="The camera spoils it.",
        ),
        provider=Provider.ANTHROPIC,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read,
        cost_usd=cost_usd,
    )


def _settings(**anthropic: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(
            preference=[Provider.ANTHROPIC],
            anthropic=AnthropicSettings(api_key=SecretStr("sk-ant"), **anthropic),
        ),
    )


def _ctx(
    settings: Settings | None = None,
    *,
    remaining_s: float = 5.0,
    options: Options | None = None,
    cost_spent_usd: float = 0.0,
) -> TierContext:
    return TierContext(
        request_id="req_t3",
        resource=ResourceKind.SENTIMENT,
        text=TEXT,
        original_length=len(TEXT),
        language=LanguageInfo(code="en", confidence=0.9),
        features=extract_features(TEXT),
        safety=SafetyReport(),
        options=options or Options(),
        deadline_at=time.monotonic() + remaining_s,
        cost_spent_usd=cost_spent_usd,
    )


def _tier(
    outcomes: list[Any],
    *,
    breakers: dict[Provider, CircuitBreaker] | None = None,
    bulkhead: Bulkhead | None = None,
    healthy: bool = True,
    **anthropic: Any,
) -> tuple[CloudLLMTier, _ScriptedProvider]:
    settings = _settings(**anthropic)
    provider = _ScriptedProvider(Provider.ANTHROPIC, outcomes, healthy=healthy)
    registry = ProviderRegistry(settings, {Provider.ANTHROPIC: provider}, breakers or {})
    return CloudLLMTier(settings, registry, bulkhead), provider


@pytest.fixture(autouse=True)
def _restore_prices() -> Any:
    """Cost assertions here override the price table; put it back afterwards."""
    yield
    reset_price_table()


# ── Happy path ────────────────────────────────────────────────────────────────


async def test_confident_primary_answer_becomes_the_verdict() -> None:
    tier, provider = _tier([_result(confidence=0.91)])

    verdict = await tier.analyse(_ctx())

    assert verdict.tier is tier.tier
    assert verdict.result.label is Label.NEGATIVE
    assert verdict.result.confidence == pytest.approx(0.91)
    assert verdict.model_version == f"anthropic:{HAIKU}"
    assert provider.models_called == [HAIKU]


async def test_the_cached_prefix_and_the_schema_are_both_requested() -> None:
    tier, provider = _tier([_result(confidence=0.9)])

    await tier.analyse(_ctx())

    call = provider.calls[0]
    assert call["cache_prefix"] is True
    assert call["output_model"] is LLMSentimentOutput
    assert call["max_tokens"] == 1024
    # The volatile half carries the text; the cached half must not.
    assert TEXT in call["user_content"]
    assert TEXT not in call["system_prompt"]


async def test_usage_from_the_call_reaches_the_verdict() -> None:
    tier, _ = _tier([_result(confidence=0.9, tokens_in=300, tokens_out=45, cache_read=4096)])

    verdict = await tier.analyse(_ctx())

    assert verdict.usage.tokens_in == 300
    assert verdict.usage.tokens_out == 45
    assert verdict.usage.cache_read_tokens == 4096
    assert verdict.usage.model == HAIKU


async def test_tier_reports_availability_from_the_registry() -> None:
    available, _ = _tier([])
    unavailable, _ = _tier([], healthy=False)

    assert await available.is_available() is True
    assert await unavailable.is_available() is False


async def test_every_language_is_supported_here() -> None:
    tier, _ = _tier([])

    assert tier.supports_language("ja") is True
    assert tier.model_version == f"anthropic:{HAIKU}"


# ── Arbitration ───────────────────────────────────────────────────────────────


async def test_low_confidence_escalates_to_the_arbiter() -> None:
    tier, provider = _tier(
        [
            _result(confidence=0.40, model=HAIKU),
            _result(confidence=0.88, model=SONNET, label=Label.MIXED),
        ]
    )

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU, SONNET]
    assert verdict.result.label is Label.MIXED
    assert verdict.result.confidence == pytest.approx(0.88)
    assert verdict.model_version == f"anthropic:{SONNET}"


async def test_the_arbiter_gets_its_own_token_ceiling() -> None:
    tier, provider = _tier(
        [_result(confidence=0.40), _result(confidence=0.9, model=SONNET)],
        arbiter_max_tokens=4096,
    )

    await tier.analyse(_ctx())

    assert provider.calls[1]["max_tokens"] == 4096


async def test_confidence_at_the_floor_does_not_arbitrate() -> None:
    # The floor is a "below" test, not "at or below": 0.65 is the configured
    # threshold and a result exactly on it is trusted.
    tier, provider = _tier([_result(confidence=0.65)])

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU]
    assert verdict.model_version == f"anthropic:{HAIKU}"


async def test_confidence_above_the_floor_does_not_arbitrate() -> None:
    tier, provider = _tier([_result(confidence=0.66)])

    await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU]


async def test_arbitration_happens_at_most_once() -> None:
    # The arbiter is also unconfident. That is an answer, not an invitation to
    # keep climbing — a loop here would spend the whole budget on one request.
    tier, provider = _tier(
        [
            _result(confidence=0.20, model=HAIKU),
            _result(confidence=0.21, model=SONNET),
        ]
    )

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU, SONNET]
    assert verdict.result.confidence == pytest.approx(0.21)


async def test_arbitration_can_be_switched_off() -> None:
    tier, provider = _tier([_result(confidence=0.10)], arbitration_enabled=False)

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU]
    assert verdict.result.confidence == pytest.approx(0.10)


async def test_both_calls_are_billed_to_the_request() -> None:
    override_prices(
        {
            HAIKU: ModelPrice.metered(input_per_mtok=1.0, output_per_mtok=5.0),
            SONNET: ModelPrice.metered(input_per_mtok=3.0, output_per_mtok=15.0),
        }
    )
    tier, _ = _tier(
        [
            _result(confidence=0.4, model=HAIKU, tokens_in=100, tokens_out=20, cost_usd=0.001),
            _result(confidence=0.9, model=SONNET, tokens_in=400, tokens_out=60, cost_usd=0.004),
        ]
    )

    verdict = await tier.analyse(_ctx())

    # A ceiling that only saw the winning call would understate spend by the
    # cost of the call that made the second one necessary.
    assert verdict.usage.tokens_in == 500
    assert verdict.usage.tokens_out == 80
    assert verdict.usage.cost_usd == pytest.approx(0.005)
    assert verdict.usage.model == SONNET


async def test_the_arbiter_is_shown_what_the_fast_model_concluded() -> None:
    tier, provider = _tier(
        [_result(confidence=0.30, model=HAIKU), _result(confidence=0.9, model=SONNET)]
    )

    await tier.analyse(_ctx())

    arbiter_turn = provider.calls[1]["user_content"]
    assert "t3_cloud_llm" in arbiter_turn
    assert "0.30" in arbiter_turn


async def test_a_failed_second_opinion_keeps_the_first() -> None:
    tier, provider = _tier([_result(confidence=0.30), ProviderUnavailableError("arbiter is down")])

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU, SONNET]
    assert verdict.succeeded is True
    assert verdict.result.confidence == pytest.approx(0.30)
    assert verdict.model_version == f"anthropic:{HAIKU}"


async def test_arbitration_is_skipped_once_the_cost_ceiling_is_spent() -> None:
    # The primary call alone exhausted the caller's budget. The policy's ceiling
    # check only runs between rungs, so this rung has to make it itself.
    tier, provider = _tier([_result(confidence=0.30, cost_usd=0.004)])

    verdict = await tier.analyse(_ctx(options=Options(max_cost_usd=0.005), cost_spent_usd=0.002))

    assert provider.models_called == [HAIKU]
    assert verdict.result.confidence == pytest.approx(0.30)


async def test_arbitration_proceeds_while_budget_remains() -> None:
    tier, provider = _tier(
        [_result(confidence=0.30, cost_usd=0.001), _result(confidence=0.9, model=SONNET)]
    )

    await tier.analyse(_ctx(options=Options(max_cost_usd=0.05)))

    assert provider.models_called == [HAIKU, SONNET]


# ── Schema repair ─────────────────────────────────────────────────────────────


async def test_one_repair_retry_is_attempted_after_a_bad_parse() -> None:
    tier, provider = _tier([ProviderSchemaError("polarity out of range"), _result(confidence=0.9)])

    verdict = await tier.analyse(_ctx())

    assert provider.models_called == [HAIKU, HAIKU]
    assert "Correction required" in provider.calls[1]["user_content"]
    assert verdict.succeeded is True


async def test_a_second_bad_parse_fails_the_tier() -> None:
    tier, provider = _tier([ProviderSchemaError("still wrong"), ProviderSchemaError("wrong again")])

    with pytest.raises(TierError):
        await tier.analyse(_ctx())
    assert len(provider.calls) == 2


async def test_the_repaired_prefix_is_unchanged() -> None:
    # The correction is appended to the *user* turn. Touching the system prefix
    # would invalidate the prompt cache for every request that follows.
    tier, provider = _tier([ProviderSchemaError("nope"), _result(confidence=0.9)])

    await tier.analyse(_ctx())

    assert provider.calls[0]["system_prompt"] == provider.calls[1]["system_prompt"]


# ── Degradation ───────────────────────────────────────────────────────────────


async def test_a_refusal_degrades_the_ladder_rather_than_failing_the_request() -> None:
    tier, _ = _tier([ProviderRefusalError("declined")])

    with pytest.raises(TierError):
        await tier.analyse(_ctx())


async def test_no_provider_available_degrades_the_ladder() -> None:
    tier, _ = _tier([], healthy=False)

    with pytest.raises(TierUnavailableError):
        await tier.analyse(_ctx())


async def test_an_open_breaker_degrades_the_ladder() -> None:
    breaker = CircuitBreaker("anthropic", threshold=1, cooldown_s=60.0)
    await breaker.record_failure()
    tier, provider = _tier([_result(confidence=0.9)], breakers={Provider.ANTHROPIC: breaker})

    with pytest.raises(TierUnavailableError):
        await tier.analyse(_ctx())
    assert provider.calls == []


async def test_a_transport_failure_degrades_the_ladder() -> None:
    tier, _ = _tier([ProviderUnavailableError("connection reset")])

    with pytest.raises(TierError):
        await tier.analyse(_ctx())


async def test_an_exhausted_budget_never_reaches_the_provider() -> None:
    tier, provider = _tier([_result(confidence=0.9)])

    with pytest.raises(TierUnavailableError):
        await tier.analyse(_ctx(remaining_s=-1.0))
    assert provider.calls == []


async def test_a_full_bulkhead_degrades_the_ladder() -> None:
    bulkhead = Bulkhead("llm", limit=1, acquire_timeout_s=0.01)
    tier, _ = _tier([_result(confidence=0.9)], bulkhead=bulkhead)

    async with bulkhead.acquire():
        with pytest.raises(TierUnavailableError):
            await tier.analyse(_ctx())
