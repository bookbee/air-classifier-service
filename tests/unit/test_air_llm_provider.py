"""AirLlmProvider — the thin client over air-llm's ``/v1/inference``.

Every test runs against a mocked transport; the base URL points at air-llm's
default local address and no real service is contacted.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel, SecretStr

from air_classifier.config import AirLlmSettings, Settings
from air_classifier.constants import Provider
from air_classifier.providers.air_llm_provider import AirLlmProvider
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderSchemaError,
    ProviderUnavailableError,
)

INFERENCE_URL = "http://localhost:8083/v1/inference"
READY_URL = "http://localhost:8083/v1/ready"


class _Out(BaseModel):
    label: str


def _settings(*, enabled: bool = True, key: str | None = "air-llm-token") -> Settings:
    return Settings(
        air_llm=AirLlmSettings(enabled=enabled, token=SecretStr(key) if key is not None else None)
    )


def _provider(name: Provider = Provider.ANTHROPIC, **kw: Any) -> AirLlmProvider:
    return AirLlmProvider(_settings(**kw), name)


def _body(**over: Any) -> dict[str, Any]:
    base = {
        "id": "abc",
        "task": "chat",
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "content": '{"label":"positive"}',
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cache_read_tokens": 2,
            "cache_write_tokens": 0,
        },
        "cost_usd": 0.006,
        "cached": False,
        "refusal": False,
        "finish_reason": "end_turn",
    }
    base.update(over)
    return base


async def _classify(provider: AirLlmProvider, *, model: str = "claude-haiku-4-5") -> Any:
    try:
        return await provider.classify(
            system_prompt="rubric",
            user_content="great product",
            output_model=_Out,
            model=model,
            max_tokens=256,
        )
    finally:
        await provider.aclose()


def test_configured_reflects_flag_and_token() -> None:
    assert _provider().configured is True
    assert _provider(enabled=False).configured is False
    assert _provider(key=None).configured is False


@respx.mock
async def test_classify_validates_and_maps_usage() -> None:
    route = respx.post(INFERENCE_URL).mock(return_value=httpx.Response(200, json=_body()))
    result = await _classify(_provider(Provider.ANTHROPIC))

    assert isinstance(result.parsed, _Out)
    assert result.parsed.label == "positive"
    assert result.provider is Provider.ANTHROPIC
    assert result.model == "claude-haiku-4-5"
    assert (result.tokens_in, result.tokens_out) == (10, 5)
    assert result.cache_read_tokens == 2
    assert result.cost_usd == 0.006

    sent = json.loads(route.calls.last.request.content)
    assert sent["task"] == "chat"
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["json_schema"]["properties"]["label"]["type"] == "string"
    assert sent["schema_name"] == "_Out"
    assert route.calls.last.request.headers["x-api-key"] == "air-llm-token"


@respx.mock
async def test_the_model_is_sent_verbatim_alias_or_bypass_or_bare() -> None:
    """Unlike the air-infra adapter this replaced, nothing here prefixes the
    model with a provider name — an alias like "classify-default" must reach
    air-llm exactly as configured, not mangled into "anthropic:classify-default"."""
    route = respx.post(INFERENCE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider(), model="classify-default")

    assert json.loads(route.calls.last.request.content)["model"] == "classify-default"


@respx.mock
async def test_result_provider_reflects_who_actually_answered() -> None:
    """air-llm's alias failover can serve a different vendor than the label this
    adapter was constructed with — cost/usage attribution must follow the real
    answer, not this instance's own ``name``."""
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(200, json=_body(provider="openai", model="gpt-4o-mini"))
    )

    result = await _classify(_provider(Provider.ANTHROPIC), model="classify-default")

    assert result.provider is Provider.OPENAI
    assert result.model == "gpt-4o-mini"


@respx.mock
async def test_refusal_becomes_refusal_error() -> None:
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(200, json=_body(refusal=True, content=""))
    )
    with pytest.raises(ProviderRefusalError):
        await _classify(_provider())


@respx.mock
async def test_length_finish_is_schema_error() -> None:
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(200, json=_body(finish_reason="length"))
    )
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_invalid_json_is_schema_error() -> None:
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(200, json=_body(content='{"not_label": 1}'))
    )
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_429_is_rate_limit_error() -> None:
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "3"}, json={"code": "rate_limited"}
        )
    )
    with pytest.raises(ProviderRateLimitError) as excinfo:
        await _classify(_provider())
    assert excinfo.value.retry_after_s == 3.0


@respx.mock
async def test_5xx_is_unavailable_error() -> None:
    respx.post(INFERENCE_URL).mock(
        return_value=httpx.Response(502, json={"code": "provider_unavailable"})
    )
    with pytest.raises(ProviderUnavailableError):
        await _classify(_provider())


async def test_classify_unconfigured_is_unavailable() -> None:
    provider = _provider(enabled=False)
    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


def test_air_llm_mode_wires_the_ladder() -> None:
    from air_classifier.main import build_ladder

    ladder = build_ladder(_settings(enabled=True))
    assert ladder.providers  # T2 local + T3 cloud, both air-llm-backed
    assert all(isinstance(p, AirLlmProvider) for p in ladder.providers)


def test_both_llm_tiers_self_build_a_bulkhead_when_none_is_passed() -> None:
    """main.py passes no explicit bulkhead to either tier's constructor — each
    must build its own from resilience settings, exactly as LocalLLMTier
    always has, or concurrency against that dependency goes unbounded. A
    missing bulkhead fails open (`guarded()` skips protection outright when
    `bulkhead is None`), so this has to be caught here, not from a symptom in
    production."""
    from air_classifier.constants import Tier as TierId
    from air_classifier.main import build_ladder

    settings = _settings(enabled=True)
    ladder = build_ladder(settings)
    by_tier = {tier.tier: tier for tier in ladder.tiers}

    t2_bulkhead = by_tier[TierId.T2_LOCAL_LLM]._bulkhead
    t3_bulkhead = by_tier[TierId.T3_CLOUD_LLM]._bulkhead
    assert t2_bulkhead is not None
    assert t3_bulkhead is not None
    # Distinct dependencies, not a shared or coincidentally-equal name — T3
    # guards the cloud call, not the one Ollama daemon T2 guards.
    assert t2_bulkhead.name != t3_bulkhead.name
    assert t2_bulkhead.limit == settings.resilience.llm_concurrency
    assert t3_bulkhead.limit == settings.resilience.llm_concurrency


def test_disabled_air_llm_builds_no_llm_tiers() -> None:
    from air_classifier.constants import Tier as TierId
    from air_classifier.main import build_ladder

    ladder = build_ladder(_settings(enabled=False))
    assert not any(p.configured for p in ladder.providers)
    assert {tier.tier for tier in ladder.tiers} == {TierId.T0_RULES, TierId.T1_CLASSIFIER}


@respx.mock
async def test_health_reads_overall_readiness_not_a_per_vendor_probe() -> None:
    """Not a per-vendor probe — see the adapter's own docstring on why."""
    respx.get(READY_URL).mock(
        return_value=httpx.Response(200, json={"status": "ready", "providers": {}})
    )
    provider = _provider()
    try:
        assert await provider.health() is True
        assert provider.health_detail is None
    finally:
        await provider.aclose()


@respx.mock
async def test_health_is_false_when_air_llm_reports_not_ready() -> None:
    respx.get(READY_URL).mock(
        return_value=httpx.Response(503, json={"status": "degraded", "providers": {}})
    )
    provider = _provider()
    try:
        assert await provider.health() is False
        assert provider.health_detail == "air-llm returned HTTP 503 on /v1/ready"
    finally:
        await provider.aclose()


@respx.mock
async def test_health_detail_names_the_unreachable_host_on_a_transport_error() -> None:
    """This is the message `/v1/ready` and `/v1/capabilities` show an operator
    — it has to say something more useful than "reported unavailable"."""
    respx.get(READY_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    provider = _provider()
    try:
        assert await provider.health() is False
        detail = provider.health_detail
        assert detail is not None
        assert "air-llm unreachable" in detail
        assert "http://localhost:8083" in detail
    finally:
        await provider.aclose()


@respx.mock
async def test_health_detail_clears_on_a_later_success() -> None:
    """Reflects only the most recent call — a recovered air-llm must not keep
    reporting a failure that already stopped happening."""
    route = respx.get(READY_URL).mock(
        side_effect=[
            httpx.Response(503, json={"status": "degraded", "providers": {}}),
            httpx.Response(200, json={"status": "ready", "providers": {}}),
        ]
    )
    provider = _provider()
    try:
        assert await provider.health() is False
        assert provider.health_detail is not None
        assert await provider.health() is True
        assert provider.health_detail is None
    finally:
        await provider.aclose()
    assert route.call_count == 2


async def test_health_is_false_when_unconfigured() -> None:
    provider = _provider(enabled=False)
    assert await provider.health() is False
    assert provider.health_detail == "not configured (AIR_LLM__ENABLED / AIR_LLM__TOKEN)"
