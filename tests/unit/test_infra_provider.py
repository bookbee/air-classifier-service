"""InfraGatewayProvider — the thin client over the air-infra model gateway.

Every test runs against a mocked transport; the base URL points at air-infra's
default local address and no real gateway is contacted.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel, SecretStr

from air_classifier.config import GatewaySettings, ProvidersSettings, Settings
from air_classifier.constants import Provider
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier.providers.infra_provider import InfraGatewayProvider

CHAT_URL = "http://localhost:8080/v1/models/chat"
CAPABILITIES_URL = "http://localhost:8080/v1/capabilities"


class _Out(BaseModel):
    label: str


def _settings(*, enabled: bool = True, key: str | None = "infra-token") -> Settings:
    return Settings(
        providers=ProvidersSettings(
            gateway=GatewaySettings(
                enabled=enabled, api_key=SecretStr(key) if key is not None else None
            )
        )
    )


def _provider(name: Provider = Provider.ANTHROPIC, **kw: Any) -> InfraGatewayProvider:
    return InfraGatewayProvider(_settings(**kw), name)


def _body(**over: Any) -> dict[str, Any]:
    base = {
        "id": "abc",
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


async def _classify(provider: InfraGatewayProvider) -> Any:
    try:
        return await provider.classify(
            system_prompt="rubric",
            user_content="great product",
            output_model=_Out,
            model="claude-haiku-4-5",
            max_tokens=256,
        )
    finally:
        await provider.aclose()


def test_configured_reflects_flag_and_key() -> None:
    assert _provider().configured is True
    assert _provider(enabled=False).configured is False
    assert _provider(key=None).configured is False


@respx.mock
async def test_classify_validates_and_maps_usage() -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_body()))
    result = await _classify(_provider(Provider.ANTHROPIC))

    assert isinstance(result.parsed, _Out)
    assert result.parsed.label == "positive"
    assert result.provider is Provider.ANTHROPIC
    assert result.model == "claude-haiku-4-5"
    assert (result.tokens_in, result.tokens_out) == (10, 5)
    assert result.cache_read_tokens == 2
    assert result.cost_usd == 0.006

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "anthropic:claude-haiku-4-5"  # provider-prefixed route
    assert sent["json_schema"]["properties"]["label"]["type"] == "string"
    assert sent["schema_name"] == "_Out"


@respx.mock
async def test_ollama_route_keeps_tagged_model_intact() -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_body(model="llama3.2:3b", cost_usd=0.0))
    )
    provider = InfraGatewayProvider(_settings(), Provider.OLLAMA)
    try:
        await provider.classify(
            system_prompt="s",
            user_content="u",
            output_model=_Out,
            model="llama3.2:3b",
            max_tokens=64,
        )
    finally:
        await provider.aclose()
    assert json.loads(route.calls.last.request.content)["model"] == "ollama:llama3.2:3b"


@respx.mock
async def test_refusal_becomes_refusal_error() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_body(refusal=True, content=""))
    )
    with pytest.raises(ProviderRefusalError):
        await _classify(_provider())


@respx.mock
async def test_length_finish_is_schema_error() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_body(finish_reason="length")))
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_invalid_json_is_schema_error() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_body(content='{"not_label": 1}'))
    )
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_429_is_rate_limit_error() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "3"}, json={"code": "rate_limited"}
        )
    )
    with pytest.raises(ProviderRateLimitError) as excinfo:
        await _classify(_provider())
    assert excinfo.value.retry_after_s == 3.0


@respx.mock
async def test_5xx_is_unavailable_error() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(502, json={"code": "provider_unavailable"})
    )
    with pytest.raises(ProviderUnavailableError):
        await _classify(_provider())


async def test_classify_unconfigured_is_unavailable() -> None:
    provider = _provider(enabled=False)
    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


def test_gateway_mode_wires_infra_providers() -> None:
    from air_classifier.main import build_ladder

    ladder = build_ladder(_settings(enabled=True))
    assert ladder.providers  # T2 local + both cloud, all gateway-backed
    assert all(isinstance(p, InfraGatewayProvider) for p in ladder.providers)
    assert ladder.registry is not None


def test_direct_mode_does_not_use_gateway() -> None:
    from air_classifier.main import build_ladder

    ladder = build_ladder(_settings(enabled=False))
    assert not any(isinstance(p, InfraGatewayProvider) for p in ladder.providers)


@respx.mock
async def test_health_reads_capabilities_per_provider() -> None:
    respx.get(CAPABILITIES_URL).mock(
        return_value=httpx.Response(200, json={"providers": {"anthropic": True, "openai": False}})
    )
    anthropic = _provider(Provider.ANTHROPIC)
    openai = _provider(Provider.OPENAI)
    try:
        assert await anthropic.health() is True
        assert await openai.health() is False
    finally:
        await anthropic.aclose()
        await openai.aclose()
