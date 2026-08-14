"""Anthropic adapter tests.

Every test here drives a stubbed ``messages.parse``; nothing touches the network
and no API key is real. Three groups matter most: the request must carry a cache
breakpoint (the saving is silent when it is missing), the response's cache
counters must survive into ``ProviderResult`` (they are the only proof caching
happened), and the SDK's typed exceptions must be caught most-specific-first —
``RateLimitError`` subclasses ``APIStatusError``, so a wrong order collapses
"back off" into "provider is down".
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import SecretStr, ValidationError

from air_sentiment.config import AnthropicSettings, AppSettings, ProvidersSettings, Settings
from air_sentiment.constants import Label, Provider
from air_sentiment.providers.anthropic_provider import (
    ADAPTIVE_THINKING_MODELS,
    AnthropicProvider,
)
from air_sentiment.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_sentiment.schemas.verdict import LLMSentimentOutput

#: Distinguishes "use the default parsed output" from "the SDK returned None".
_UNSET: Any = object()

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"

#: Sampling parameters Sonnet 5 rejects outright. None of them may ever be sent.
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


# ── Doubles ───────────────────────────────────────────────────────────────────


class _FakeMessages:
    """Stand-in for ``client.messages``. Records calls, replays scripted outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = _FakeMessages(outcomes)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _output(confidence: float = 0.9) -> LLMSentimentOutput:
    return LLMSentimentOutput(
        label=Label.POSITIVE,
        polarity=0.8,
        confidence=confidence,
        rationale="Enthusiastic throughout.",
    )


def _response(
    *,
    parsed: Any = _UNSET,
    stop_reason: str = "end_turn",
    input_tokens: int = 120,
    output_tokens: int = 40,
    cache_read: int = 0,
    cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=_output() if parsed is _UNSET else parsed,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(
            anthropic=AnthropicSettings(api_key=SecretStr("sk-ant-test"), **overrides)
        ),
    )


def _provider(outcomes: list[Any], **overrides: Any) -> tuple[AnthropicProvider, _FakeClient]:
    provider = AnthropicProvider(_settings(**overrides))
    client = _FakeClient(outcomes)
    provider._client = client  # type: ignore[assignment]
    return provider, client


async def _classify(provider: AnthropicProvider, *, model: str = HAIKU) -> Any:
    return await provider.classify(
        system_prompt="RUBRIC",
        user_content="Loved it.",
        output_model=LLMSentimentOutput,
        model=model,
        max_tokens=1024,
    )


def _status_error(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status, request=request, headers=headers or {})


def _validation_error() -> ValidationError:
    try:
        LLMSentimentOutput.model_validate({"label": "positive"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected LLMSentimentOutput to reject an incomplete payload")


# ── Prompt caching ────────────────────────────────────────────────────────────


async def test_system_is_a_block_list_with_a_cache_breakpoint_on_the_last_block() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    system = client.messages.calls[0]["system"]
    assert isinstance(system, list)
    assert system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_cache_ttl_follows_configuration() -> None:
    provider, client = _provider([_response()], prompt_cache_ttl="5m")

    await _classify(provider)

    assert client.messages.calls[0]["system"][-1]["cache_control"]["ttl"] == "5m"


async def test_no_breakpoint_when_caching_is_disabled() -> None:
    provider, client = _provider([_response()], prompt_cache_enabled=False)

    await _classify(provider)

    assert "cache_control" not in client.messages.calls[0]["system"][-1]


async def test_no_breakpoint_when_the_caller_opts_out() -> None:
    provider, client = _provider([_response()])

    await provider.classify(
        system_prompt="RUBRIC",
        user_content="Loved it.",
        output_model=LLMSentimentOutput,
        model=HAIKU,
        max_tokens=1024,
        cache_prefix=False,
    )

    assert "cache_control" not in client.messages.calls[0]["system"][-1]


async def test_cache_counters_reach_the_result_and_the_price() -> None:
    provider, _ = _provider(
        [_response(input_tokens=10, output_tokens=20, cache_read=5000, cache_write=100)]
    )

    result = await _classify(provider)

    assert result.cache_read_tokens == 5000
    assert result.cache_write_tokens == 100
    assert result.tokens_in == 10
    # A cache read bills at a tenth of input, a 1h write at twice it, so both
    # classes must show up in the price rather than being rounded away.
    assert result.cost_usd == pytest.approx(
        (10 * 1.0 + 20 * 5.0 + 5000 * 0.1 + 100 * 2.0) / 1_000_000
    )


# ── Request shape ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [HAIKU, SONNET])
async def test_no_sampling_parameters_are_ever_sent(model: str) -> None:
    provider, client = _provider([_response()])

    await _classify(provider, model=model)

    call = client.messages.calls[0]
    assert not [p for p in SAMPLING_PARAMS if p in call]


async def test_haiku_gets_neither_effort_nor_thinking() -> None:
    provider, client = _provider([_response()])

    await _classify(provider, model=HAIKU)

    call = client.messages.calls[0]
    assert "effort" not in call
    assert "thinking" not in call


async def test_sonnet_is_told_not_to_think() -> None:
    provider, client = _provider([_response()])

    await _classify(provider, model=SONNET)

    call = client.messages.calls[0]
    assert call["thinking"] == {"type": "disabled"}
    assert "effort" not in call


async def test_haiku_is_not_in_the_adaptive_thinking_set() -> None:
    assert HAIKU not in ADAPTIVE_THINKING_MODELS
    assert SONNET in ADAPTIVE_THINKING_MODELS


async def test_structured_output_is_requested_by_model_not_by_prose() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    assert client.messages.calls[0]["output_format"] is LLMSentimentOutput


async def test_user_content_is_the_only_volatile_turn() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    call = client.messages.calls[0]
    assert call["messages"] == [{"role": "user", "content": "Loved it."}]
    assert call["system"][-1]["text"] == "RUBRIC"


# ── Refusals and unparseable output ───────────────────────────────────────────


async def test_refusal_stop_reason_raises_before_content_is_read() -> None:
    # `parsed_output` is deliberately None: a refusal carries nothing to read,
    # and an adapter that indexed content first would raise AttributeError here.
    provider, _ = _provider([_response(stop_reason="refusal", parsed=None)])

    with pytest.raises(ProviderRefusalError):
        await _classify(provider)


async def test_missing_parsed_output_is_a_schema_error() -> None:
    response = _response()
    response.parsed_output = None
    provider, _ = _provider([response])

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


async def test_validation_failure_is_a_schema_error() -> None:
    provider, _ = _provider([_validation_error()])

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


# ── Typed exception mapping ───────────────────────────────────────────────────


async def test_not_found_maps_to_unavailable() -> None:
    provider, _ = _provider(
        [anthropic.NotFoundError("no such model", response=_status_error(404), body=None)]
    )

    with pytest.raises(ProviderUnavailableError, match="not found"):
        await _classify(provider)


async def test_rate_limit_maps_to_rate_limit_and_carries_retry_after() -> None:
    # RateLimitError subclasses APIStatusError; catching the parent first would
    # turn a retryable throttle into an outage.
    provider, _ = _provider(
        [
            anthropic.RateLimitError(
                "slow down",
                response=_status_error(429, {"retry-after": "12"}),
                body=None,
            )
        ]
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _classify(provider)
    assert caught.value.retry_after_s == 12.0


async def test_generic_status_error_maps_to_unavailable() -> None:
    provider, _ = _provider(
        [anthropic.APIStatusError("boom", response=_status_error(500), body=None)]
    )

    with pytest.raises(ProviderUnavailableError, match="500"):
        await _classify(provider)


async def test_connection_error_maps_to_unavailable() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    provider, _ = _provider([anthropic.APIConnectionError(request=request)])

    with pytest.raises(ProviderUnavailableError, match="unreachable"):
        await _classify(provider)


# ── Configuration and lifecycle ───────────────────────────────────────────────


async def test_unconfigured_provider_reports_and_refuses() -> None:
    provider = AnthropicProvider(Settings(app=AppSettings(env="test")))

    assert provider.configured is False
    assert await provider.health() is False
    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


async def test_configured_provider_names_itself() -> None:
    provider, _ = _provider([])

    assert provider.name is Provider.ANTHROPIC
    assert provider.configured is True


async def test_health_probe_swallows_failures() -> None:
    provider, client = _provider([])
    client.models = SimpleNamespace(  # type: ignore[attr-defined]
        retrieve=_raiser(RuntimeError("dns is having a day"))
    )

    assert await provider.health() is False


async def test_health_probe_reads_model_metadata() -> None:
    provider, client = _provider([])
    seen: list[str] = []

    async def retrieve(model: str) -> object:
        seen.append(model)
        return object()

    client.models = SimpleNamespace(retrieve=retrieve)  # type: ignore[attr-defined]

    assert await provider.health() is True
    assert seen == [HAIKU]


async def test_aclose_releases_the_client() -> None:
    provider, client = _provider([])

    await provider.aclose()

    assert client.closed is True


def _raiser(exc: Exception) -> Any:
    async def _fn(*_: Any, **__: Any) -> Any:
        raise exc

    return _fn
