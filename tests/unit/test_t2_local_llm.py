"""Tier 2 tests.

Driven entirely by a scripted stub provider: no HTTP client, no daemon, no
sockets. What is under test is the rung's *policy* — how many times it will talk
to a small model, what it does with each kind of failure, and whether it can be
made to hang — and none of that needs a real model to be exercised.

The two properties worth breaking a build over: the repair path runs at most
once, and every failure mode leaves the rung by way of ``TierError`` so the
ladder can climb past it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from air_classifier.config import (
    AirLlmSettings,
    AppSettings,
    Settings,
    TierSettings,
    TiersSettings,
)
from air_classifier.constants import Label, Provider, Tier
from air_classifier.observability import metrics
from air_classifier.pipeline.features import extract_features
from air_classifier.pipeline.tiers import base as tier_base
from air_classifier.pipeline.tiers.base import TierContext, TierError, TierUnavailableError
from air_classifier.pipeline.tiers.t2_local_llm import LocalLLMTier
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier.providers.prompts.loader import compact_system_prompt, system_prompt
from air_classifier.resilience.breaker import CircuitBreaker
from air_classifier.resilience.bulkhead import Bulkhead
from air_classifier.schemas.common import Emotion, LanguageInfo, Options, SafetyReport
from air_classifier.schemas.verdict import LLMSentimentOutput

TEXT = "Great battery but the camera is awful."
MODEL = "llama3.2:3b"


class StubProvider:
    """A scripted :class:`LLMProvider`. Records what it was asked, answers a queue.

    Outcomes are consumed in order; an exception in the queue is raised rather
    than returned, which is how the failure paths get exercised without a
    network able to misbehave in some other way.
    """

    name: Provider = Provider.OLLAMA

    def __init__(
        self,
        *outcomes: ProviderResult | Exception,
        healthy: bool = True,
        configured: bool = True,
        health_error: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._outcomes = list(outcomes)
        self._healthy = healthy
        self._configured = configured
        self._health_error = health_error
        self._delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def health_detail(self) -> str | None:
        return None if self._healthy else "stub unhealthy"

    async def health(self) -> bool:
        if self._health_error is not None:
            raise self._health_error
        return self._healthy

    async def classify(self, **kwargs: Any) -> ProviderResult:
        self.calls.append(kwargs)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        outcome = self._outcomes.pop(0) if self._outcomes else _result()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True


def _output(**overrides: Any) -> LLMSentimentOutput:
    fields: dict[str, Any] = {
        "label": Label.NEGATIVE,
        "polarity": -0.6,
        "confidence": 0.81,
        "rationale": "Praises the battery but calls the camera awful.",
        "emotions": [Emotion(name="annoyance", score=0.7)],
        "sarcasm_detected": False,
        "language_code": "en",
    }
    fields.update(overrides)
    return LLMSentimentOutput(**fields)


def _result(parsed: LLMSentimentOutput | None = None, **overrides: Any) -> ProviderResult:
    fields: dict[str, Any] = {
        "parsed": parsed or _output(),
        "provider": Provider.OLLAMA,
        "model": MODEL,
        "tokens_in": 412,
        "tokens_out": 96,
        "latency_ms": 240.0,
        "cost_usd": 0.0,
    }
    fields.update(overrides)
    return ProviderResult(**fields)


def _settings(*, timeout_ms: int = 4000, model: str = MODEL) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        tiers=TiersSettings(t2_local_llm=TierSettings(timeout_ms=timeout_ms)),
        air_llm=AirLlmSettings(local_model=model),
    )


def _ctx(*, remaining_s: float = 5.0) -> TierContext:
    return TierContext(
        request_id="req_test",
        text=TEXT,
        original_length=len(TEXT),
        language=LanguageInfo(code="en", confidence=0.9),
        features=extract_features(TEXT),
        safety=SafetyReport(),
        options=Options(),
        deadline_at=time.monotonic() + remaining_s,
    )


def _tier(
    provider: StubProvider,
    *,
    settings: Settings | None = None,
    bulkhead: Bulkhead | None = None,
    breaker: CircuitBreaker | None = None,
) -> LocalLLMTier:
    return LocalLLMTier(
        settings or _settings(),
        provider,
        bulkhead=bulkhead,
        breaker=breaker or CircuitBreaker("ollama-test", threshold=5, cooldown_s=30.0),
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_the_rung_satisfies_the_tier_protocol() -> None:
    assert isinstance(_tier(StubProvider()), tier_base.Tier)


def test_the_rung_identifies_itself_and_the_model_behind_it() -> None:
    tier = _tier(StubProvider())

    assert tier.tier is Tier.T2_LOCAL_LLM
    assert tier.model_version == MODEL


def test_every_language_is_attempted_because_the_model_is_prompted_not_trained() -> None:
    tier = _tier(StubProvider())

    assert tier.supports_language("en") is True
    assert tier.supports_language("ja") is True


# ── The happy path ────────────────────────────────────────────────────────────


async def test_analyse_converts_the_model_output_into_a_verdict() -> None:
    tier = _tier(StubProvider(_result()))

    verdict = await tier.analyse(_ctx())

    assert verdict.tier is Tier.T2_LOCAL_LLM
    assert verdict.result.label is Label.NEGATIVE
    assert verdict.result.confidence == pytest.approx(0.81)
    assert verdict.is_usable is True
    # The resolved provider:model, not the tier's configured alias — the same
    # shape Tier 3 reports, so a caller can always see who actually answered.
    assert verdict.model_version == "ollama:llama3.2:3b"
    assert verdict.emotions[0].name == "annoyance"
    assert verdict.language_code == "en"


async def test_the_verdict_carries_the_call_usage_at_no_cost() -> None:
    tier = _tier(StubProvider(_result()))

    verdict = await tier.analyse(_ctx())

    assert verdict.usage.tokens_in == 412
    assert verdict.usage.tokens_out == 96
    assert verdict.usage.cost_usd == 0.0
    assert verdict.usage.provider == Provider.OLLAMA.value
    assert verdict.usage.model == MODEL


async def test_the_verdict_times_the_whole_rung_not_the_provider_call() -> None:
    """Latency is what the caller waited for, guards and repairs included."""
    tier = _tier(StubProvider(_result(latency_ms=0.0)))

    verdict = await tier.analyse(_ctx())

    assert verdict.latency_ms > 0.0


async def test_token_volume_reaches_the_usage_metric() -> None:
    registry = metrics.get_registry()
    labels = {"provider": Provider.OLLAMA.value, "model": MODEL, "kind": "in"}
    before = registry.get_sample_value("air_classifier_llm_tokens_total", labels) or 0.0

    await _tier(StubProvider(_result())).analyse(_ctx())

    after = registry.get_sample_value("air_classifier_llm_tokens_total", labels) or 0.0
    assert after - before == pytest.approx(412.0)


# ── What the provider is asked for ────────────────────────────────────────────


async def test_the_rung_sends_the_compact_prefix_and_the_shared_schema() -> None:
    provider = StubProvider(_result())

    await _tier(provider).analyse(_ctx())

    call = provider.calls[0]
    # The compact prefix, never the cache-anchored one. The long prompt exists to
    # clear Anthropic's cache floor; a local model has no cache and re-reads every
    # token on every call, which cost ~9.9s against ~1.5s here.
    assert call["system_prompt"] == compact_system_prompt()
    assert call["system_prompt"] != system_prompt()
    assert call["output_model"] is LLMSentimentOutput
    assert call["model"] == MODEL
    assert call["max_tokens"] > 0


async def test_the_user_turn_carries_the_text_and_the_task_framing() -> None:
    provider = StubProvider(_result())

    await _tier(provider).analyse(_ctx())

    user = provider.calls[0]["user_content"]
    assert TEXT in user
    assert "distinct facets" in user


# ── The single repair retry ───────────────────────────────────────────────────


async def test_one_schema_failure_is_repaired_rather_than_escalated() -> None:
    provider = StubProvider(ProviderSchemaError("confidence: must be <= 1"), _result())

    verdict = await _tier(provider).analyse(_ctx())

    assert verdict.is_usable is True
    assert len(provider.calls) == 2


async def test_the_repair_appends_to_the_user_turn_and_leaves_the_prefix_alone() -> None:
    """Moving the prefix would cost the cache hit that makes retrying affordable."""
    provider = StubProvider(ProviderSchemaError("confidence: must be <= 1"), _result())

    await _tier(provider).analyse(_ctx())

    first, second = provider.calls
    assert second["system_prompt"] == first["system_prompt"]
    assert second["user_content"].startswith(first["user_content"])
    assert "Correction required" in second["user_content"]
    assert "confidence: must be <= 1" in second["user_content"]


async def test_a_second_schema_failure_escalates_instead_of_looping() -> None:
    """An unbounded repair loop against a 3B model is a real way to burn a budget."""
    provider = StubProvider(
        ProviderSchemaError("bad once"),
        ProviderSchemaError("bad twice"),
        _result(),
    )

    with pytest.raises(TierError) as caught:
        await _tier(provider).analyse(_ctx())

    assert len(provider.calls) == 2
    assert "bad twice" in str(caught.value)


# ── Failure mapping ───────────────────────────────────────────────────────────


async def test_an_unreachable_daemon_marks_the_rung_unavailable() -> None:
    provider = StubProvider(ProviderUnavailableError("connection refused"))

    with pytest.raises(TierUnavailableError):
        await _tier(provider).analyse(_ctx())


@pytest.mark.parametrize(
    "error",
    [ProviderRateLimitError("throttled"), ProviderRefusalError("declined")],
)
async def test_other_provider_errors_fail_the_rung(error: Exception) -> None:
    with pytest.raises(TierError):
        await _tier(StubProvider(error)).analyse(_ctx())


async def test_an_unexpected_parsed_type_fails_the_rung() -> None:
    """Defensive: a provider that returns the wrong model must not reach a verdict."""
    provider = StubProvider(_result(parsed=Options()))

    with pytest.raises(TierError):
        await _tier(provider).analyse(_ctx())


# ── Guards ────────────────────────────────────────────────────────────────────


async def test_a_slow_model_is_cut_off_at_the_tier_timeout() -> None:
    provider = StubProvider(delay_s=1.0)
    tier = _tier(provider, settings=_settings(timeout_ms=30))

    started = time.perf_counter()
    with pytest.raises(TierError):
        await tier.analyse(_ctx())

    assert time.perf_counter() - started < 0.5


async def test_the_remaining_request_budget_caps_the_tier_timeout() -> None:
    """A 4s slice is a ceiling, not an entitlement — Tier 3 still needs its turn."""
    provider = StubProvider(delay_s=1.0)
    tier = _tier(provider, settings=_settings(timeout_ms=4000))

    started = time.perf_counter()
    with pytest.raises(TierError):
        await tier.analyse(_ctx(remaining_s=0.05))

    assert time.perf_counter() - started < 0.5


async def test_an_exhausted_budget_fails_before_the_model_is_called() -> None:
    provider = StubProvider(_result())

    with pytest.raises(TierError):
        await _tier(provider).analyse(_ctx(remaining_s=-1.0))

    assert provider.calls == []


async def test_an_open_breaker_short_circuits_without_calling_the_model() -> None:
    breaker = CircuitBreaker("ollama-test", threshold=1, cooldown_s=30.0)
    await breaker.record_failure()
    provider = StubProvider(_result())

    with pytest.raises(TierUnavailableError):
        await _tier(provider, breaker=breaker).analyse(_ctx())

    assert provider.calls == []


async def test_a_saturated_bulkhead_rejects_rather_than_queues() -> None:
    """Local inference is CPU-bound: queueing behind it spreads the outage."""
    provider = StubProvider(_result(), _result(), delay_s=0.2)
    tier = _tier(provider, bulkhead=Bulkhead("ollama-test", limit=1, acquire_timeout_s=0.02))
    ctx = _ctx()

    outcomes = await asyncio.gather(tier.analyse(ctx), tier.analyse(ctx), return_exceptions=True)

    assert sum(isinstance(o, TierUnavailableError) for o in outcomes) == 1
    assert len(provider.calls) == 1


# ── Availability ──────────────────────────────────────────────────────────────


async def test_available_when_the_daemon_is_healthy_and_the_breaker_is_closed() -> None:
    assert await _tier(StubProvider(healthy=True)).is_available() is True


async def test_unavailable_when_the_daemon_is_unhealthy() -> None:
    assert await _tier(StubProvider(healthy=False)).is_available() is False


async def test_unavailable_when_the_provider_is_switched_off() -> None:
    assert await _tier(StubProvider(configured=False)).is_available() is False


async def test_unavailable_while_the_breaker_is_open_without_probing() -> None:
    breaker = CircuitBreaker("ollama-test", threshold=1, cooldown_s=30.0)
    await breaker.record_failure()
    provider = StubProvider(health_error=RuntimeError("would have raised"))

    assert await _tier(provider, breaker=breaker).is_available() is False


async def test_a_health_probe_that_raises_reads_as_unavailable() -> None:
    provider = StubProvider(health_error=RuntimeError("socket exploded"))

    assert await _tier(provider).is_available() is False


# ── unavailable_reason ───────────────────────────────────────────────────────


async def test_unavailable_reason_is_none_when_available() -> None:
    assert await _tier(StubProvider(healthy=True)).unavailable_reason() is None


async def test_unavailable_reason_names_air_llm_not_the_vendor_label() -> None:
    """`configured` is an air-llm-level check, not a per-vendor one — the
    message must not send an operator looking for a vendor-specific knob."""
    reason = await _tier(StubProvider(configured=False)).unavailable_reason()
    assert reason == "air-llm not configured"


async def test_unavailable_reason_names_the_open_breaker_before_probing_health() -> None:
    """Same precedence `is_available` checks in — an open breaker is reported
    even though the provider itself would answer `health_detail=None`."""
    breaker = CircuitBreaker("ollama-test", threshold=1, cooldown_s=30.0)
    await breaker.record_failure()
    provider = StubProvider(healthy=True)

    reason = await _tier(provider, breaker=breaker).unavailable_reason()

    assert reason is not None
    assert "circuit breaker open for ollama" in reason


async def test_unavailable_reason_falls_through_to_the_providers_own_detail() -> None:
    provider = StubProvider(healthy=False)

    assert await _tier(provider).unavailable_reason() == "stub unhealthy"
