"""ModelGatewayProvider — the thin client over air-model-gateway's ``/v1/generate``.

Every test runs against a mocked transport; the base URL points at the
gateway's default local address and no real service is contacted.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx
import pytest
import respx
from pydantic import BaseModel, SecretStr

from air_classifier_service.config import ModelGatewaySettings, Settings
from air_classifier_service.constants import ModelClass, Provider
from air_classifier_service.providers.base import (
    ProviderBudgetExceededError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier_service.providers.model_gateway_provider import ModelGatewayProvider

GENERATE_URL = "http://localhost:8083/v1/generate"
READY_URL = "http://localhost:8083/v1/ready"

#: Every field air-model-gateway's ``GenerateInput`` declares
#: (src/air_model_gateway/schemas/generate.py). It is ``extra="forbid"``, so a
#: body key outside this set is a 422 on every call, which this service would
#: otherwise only see as every request quietly degrading to Tier 1.
GATEWAY_GENERATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "provider",
        "pin",
        "messages",
        "max_tokens",
        "temperature",
        "json_schema",
        "schema_name",
        "cache_prefix",
        "stream",
        "cache",
    }
)


class _Out(BaseModel):
    label: str


def _settings(*, enabled: bool = True, key: str | None = "gateway-token") -> Settings:
    return Settings(
        model_gateway=ModelGatewaySettings(
            enabled=enabled, token=SecretStr(key) if key is not None else None
        )
    )


def _provider(name: Provider = Provider.ANTHROPIC, **kw: Any) -> ModelGatewayProvider:
    return ModelGatewayProvider(_settings(**kw), name)


def _body(**over: Any) -> dict[str, Any]:
    """A `/v1/generate` 200 body, shaped like the gateway's `GenerateOutput`."""
    base = {
        "id": "gen_abc",
        "model_class": "auto",
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "content": '{"label":"positive"}',
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_read_tokens": 2,
            "cache_write_tokens": 0,
        },
        "cost_usd": 0.006,
        "cached": False,
        "refusal": False,
        "finish_reason": "stop",
        "provider_finish_reason": "end_turn",
    }
    base.update(over)
    return base


def _problem(status: int, code: str, **extra: Any) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/problem+json", **extra.pop("headers", {})},
        json={"type": "about:blank", "status": status, "code": code, "request_id": "r", **extra},
    )


async def _classify(
    provider: ModelGatewayProvider,
    *,
    model: ModelClass = ModelClass.AUTO,
    remaining_ms: float | None = None,
    request_id: str | None = None,
) -> Any:
    try:
        return await provider.classify(
            system_prompt="rubric",
            user_content="great product",
            output_model=_Out,
            model=model,
            max_tokens=256,
            remaining_ms=remaining_ms,
            request_id=request_id,
        )
    finally:
        await provider.aclose()


def test_configured_reflects_flag_and_token() -> None:
    assert _provider().configured is True
    assert _provider(enabled=False).configured is False
    assert _provider(key=None).configured is False


@respx.mock
async def test_classify_validates_and_maps_usage() -> None:
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))
    result = await _classify(_provider(Provider.ANTHROPIC))

    assert isinstance(result.parsed, _Out)
    assert result.parsed.label == "positive"
    assert result.provider is Provider.ANTHROPIC
    assert result.model == "claude-haiku-4-5"
    assert (result.tokens_in, result.tokens_out) == (10, 5)
    assert result.cache_read_tokens == 2
    assert result.cost_usd == 0.006

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "auto"
    assert sent["json_schema"]["properties"]["label"]["type"] == "string"
    assert sent["schema_name"] == "_Out"
    assert route.calls.last.request.headers["x-api-key"] == "gateway-token"


@respx.mock
async def test_the_body_sends_only_fields_the_gateway_declares() -> None:
    """The retired `/v1/inference` body carried `task: "chat"`; the gateway's
    `extra="forbid"` turns that, or any other leftover, into a 422 on every
    call. Pinned as a subset so a new optional field we start sending has to
    be one the gateway actually takes."""
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider())

    sent = json.loads(route.calls.last.request.content)
    assert "task" not in sent
    assert set(sent) <= GATEWAY_GENERATE_FIELDS
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


@pytest.mark.parametrize("model_class", list(ModelClass))
@respx.mock
async def test_the_model_class_is_sent_as_the_gateway_spells_it(
    model_class: ModelClass,
) -> None:
    """A class, never a model name or `provider:model`: the gateway resolves the
    chain, and a pin would need a grant this service doesn't have."""
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider(), model=model_class)

    assert json.loads(route.calls.last.request.content)["model"] == model_class.value


@respx.mock
async def test_request_id_is_forwarded_so_one_id_spans_both_services() -> None:
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider(), request_id="req_01ABC")

    assert route.calls.last.request.headers["x-request-id"] == "req_01ABC"


@respx.mock
async def test_no_request_id_header_when_none_is_given() -> None:
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider())

    assert "x-request-id" not in route.calls.last.request.headers


@respx.mock
async def test_remaining_budget_rides_as_an_advisory_header() -> None:
    """The gateway is free to ignore this — sending it costs nothing and lets a
    gateway that reads it stop early on a call whose answer would already be
    discarded by the time it arrived."""
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider(), remaining_ms=1234.6)

    assert route.calls.last.request.headers["x-remaining-budget-ms"] == "1235"


@respx.mock
async def test_no_remaining_budget_header_when_none_is_given() -> None:
    """The header is advisory and per-call — absent, not sent as a stale or
    zero value, when the caller did not compute one."""
    route = respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json=_body()))

    await _classify(_provider())

    assert "x-remaining-budget-ms" not in route.calls.last.request.headers


@respx.mock
async def test_result_provider_reflects_who_actually_answered() -> None:
    """A class's chain can fail over to a different vendor than the label this
    adapter was constructed with — cost/usage attribution must follow the real
    answer, not this instance's own ``name``."""
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(200, json=_body(provider="openai", model="gpt-4o-mini"))
    )

    result = await _classify(_provider(Provider.ANTHROPIC))

    assert result.provider is Provider.OPENAI
    assert result.model == "gpt-4o-mini"


@respx.mock
async def test_the_gateways_self_hosted_provider_is_attributed_as_itself() -> None:
    """The gateway's self-hosted slot reports `llama.cpp`; before the enum knew
    it, such an answer was booked under the calling tier's label instead."""
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(200, json=_body(provider="llama.cpp", model="qwen2.5-3b"))
    )

    result = await _classify(_provider(Provider.OLLAMA), model=ModelClass.LOCAL)

    assert result.provider is Provider.LLAMA_CPP


@respx.mock
async def test_an_unknown_provider_falls_back_to_the_adapters_label() -> None:
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(200, json=_body(provider="some-future-vendor"))
    )

    result = await _classify(_provider(Provider.ANTHROPIC))

    assert result.provider is Provider.ANTHROPIC


@respx.mock
async def test_refusal_becomes_refusal_error() -> None:
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(
            200, json=_body(refusal=True, content="", finish_reason="refusal")
        )
    )
    with pytest.raises(ProviderRefusalError):
        await _classify(_provider())


@respx.mock
async def test_length_finish_is_schema_error() -> None:
    """The gateway normalizes every provider's "cut off" spelling to `length`."""
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(
            200, json=_body(finish_reason="length", provider_finish_reason="MAX_TOKENS")
        )
    )
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_invalid_json_is_schema_error() -> None:
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(200, json=_body(content='{"not_label": 1}'))
    )
    with pytest.raises(ProviderSchemaError):
        await _classify(_provider())


@respx.mock
async def test_429_rate_limited_is_rate_limit_error() -> None:
    respx.post(GENERATE_URL).mock(
        return_value=_problem(429, "rate_limited", headers={"retry-after": "3"})
    )
    with pytest.raises(ProviderRateLimitError) as excinfo:
        await _classify(_provider())
    assert not isinstance(excinfo.value, ProviderBudgetExceededError)
    assert excinfo.value.retry_after_s == 3.0


@respx.mock
async def test_429_budget_exceeded_is_told_apart_from_a_throttle() -> None:
    """Same status, different remedy: a spent daily budget clears at the next
    UTC day. Still a `ProviderRateLimitError`, so every throttle handler keeps
    working, but the class name a failed tier records says which one it was."""
    respx.post(GENERATE_URL).mock(
        return_value=_problem(429, "budget_exceeded", headers={"retry-after": "3600"})
    )
    with pytest.raises(ProviderBudgetExceededError) as excinfo:
        await _classify(_provider())
    assert isinstance(excinfo.value, ProviderRateLimitError)
    assert excinfo.value.retry_after_s == 3600.0


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (422, "validation_error"),
        (422, "invalid_model_choice"),
        (422, "provider_rejected"),
        (401, "unauthorized"),
        (403, "policy_denied"),
        (404, "not_found"),
    ],
)
@respx.mock
async def test_a_4xx_is_a_request_error_not_an_outage(status: int, code: str) -> None:
    """The point of the distinction: "unavailable" reads to the ladder as a
    capacity gap and degrades quietly, so a contract or config break filed
    under it would hide behind 200s on every request."""
    respx.post(GENERATE_URL).mock(return_value=_problem(status, code, detail="quotes input"))
    with pytest.raises(ProviderRequestError) as excinfo:
        await _classify(_provider())
    assert not isinstance(excinfo.value, ProviderUnavailableError)
    assert (excinfo.value.status, excinfo.value.code) == (status, code)
    # `detail` can quote request content back; it must not reach the message.
    assert "quotes input" not in str(excinfo.value)


@respx.mock
async def test_a_4xx_without_a_problem_body_is_still_a_request_error() -> None:
    respx.post(GENERATE_URL).mock(return_value=httpx.Response(404, text="<html>nope</html>"))
    with pytest.raises(ProviderRequestError) as excinfo:
        await _classify(_provider())
    assert excinfo.value.code is None


@respx.mock
async def test_5xx_is_unavailable_error() -> None:
    respx.post(GENERATE_URL).mock(return_value=_problem(502, "provider_unavailable"))
    with pytest.raises(ProviderUnavailableError):
        await _classify(_provider())


@respx.mock
async def test_a_timeout_is_unavailable_error() -> None:
    respx.post(GENERATE_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderUnavailableError):
        await _classify(_provider())


async def test_classify_unconfigured_is_unavailable() -> None:
    provider = _provider(enabled=False)
    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


def test_model_gateway_mode_wires_the_ladder() -> None:
    from air_classifier_service.main import build_ladder

    ladder = build_ladder(_settings(enabled=True))
    assert ladder.providers  # T2 local + T3 cloud, both gateway-backed
    assert all(isinstance(p, ModelGatewayProvider) for p in ladder.providers)


def test_both_llm_tiers_self_build_a_bulkhead_when_none_is_passed() -> None:
    """main.py passes no explicit bulkhead to either tier's constructor — each
    must build its own from resilience settings, exactly as LocalLLMTier
    always has, or concurrency against that dependency goes unbounded. A
    missing bulkhead fails open (`guarded()` skips protection outright when
    `bulkhead is None`), so this has to be caught here, not from a symptom in
    production."""
    from air_classifier_service.constants import Tier as TierId
    from air_classifier_service.main import build_ladder

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


def test_disabled_model_gateway_builds_no_llm_tiers() -> None:
    from air_classifier_service.constants import Tier as TierId
    from air_classifier_service.main import build_ladder

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
async def test_health_is_false_when_model_gateway_reports_not_ready() -> None:
    respx.get(READY_URL).mock(
        return_value=httpx.Response(503, json={"status": "degraded", "providers": {}})
    )
    provider = _provider()
    try:
        assert await provider.health() is False
        assert provider.health_detail == "air-model-gateway returned HTTP 503 on /v1/ready"
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
        assert "air-model-gateway unreachable" in detail
        assert "http://localhost:8083" in detail
    finally:
        await provider.aclose()


@respx.mock
async def test_health_detail_clears_on_a_later_success() -> None:
    """Reflects only the most recent call — a recovered gateway must not keep
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
    assert (
        provider.health_detail == "not configured (MODEL_GATEWAY__ENABLED / MODEL_GATEWAY__TOKEN)"
    )
