"""OpenAI adapter tests.

The accounting test is the one that earns its place: OpenAI reports cached
tokens *inside* ``prompt_tokens``, so an adapter that forwards both numbers
straight through bills the same tokens twice. Everything else mirrors the
Anthropic suite — typed exceptions most-specific-first, refusals raised rather
than parsed, and no sampling parameters on the wire.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr, ValidationError

from air_sentiment.config import AppSettings, OpenAISettings, ProvidersSettings, Settings
from air_sentiment.constants import Label, Provider
from air_sentiment.observability.cost import ModelPrice, override_prices, reset_price_table
from air_sentiment.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_sentiment.providers.openai_provider import PROMPT_CACHE_KEY, OpenAIProvider
from air_sentiment.schemas.verdict import LLMSentimentOutput

#: Distinguishes "use the default parsed output" from "the SDK returned None".
_UNSET: Any = object()

MODEL = "gpt-4.1-mini"

#: Sampling knobs the adapter deliberately never sends, so a model swap cannot
#: start rejecting requests over a parameter nobody asked for.
SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


# ── Doubles ───────────────────────────────────────────────────────────────────


class _FakeCompletions:
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
        self.completions = _FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _output(confidence: float = 0.9) -> LLMSentimentOutput:
    return LLMSentimentOutput(
        label=Label.NEGATIVE,
        polarity=-0.7,
        confidence=confidence,
        rationale="Complains throughout.",
    )


def _response(
    *,
    parsed: Any = _UNSET,
    refusal: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 1200,
    completion_tokens: int = 40,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    choices: list[Any] | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(
        refusal=refusal,
        parsed=_output() if parsed is _UNSET else parsed,
    )
    default_choices = [SimpleNamespace(message=message, finish_reason=finish_reason)]
    return SimpleNamespace(
        choices=default_choices if choices is None else choices,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
        ),
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(
            openai=OpenAISettings(api_key=SecretStr("sk-openai-test"), **overrides)
        ),
    )


def _provider(outcomes: list[Any], **overrides: Any) -> tuple[OpenAIProvider, _FakeClient]:
    provider = OpenAIProvider(_settings(**overrides))
    client = _FakeClient(outcomes)
    provider._client = client  # type: ignore[assignment]
    return provider, client


async def _classify(provider: OpenAIProvider, *, cache_prefix: bool = True) -> Any:
    return await provider.classify(
        system_prompt="RUBRIC",
        user_content="Hated it.",
        output_model=LLMSentimentOutput,
        model=MODEL,
        max_tokens=1024,
        cache_prefix=cache_prefix,
    )


def _status_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status, request=request, headers=headers or {})


def _validation_error() -> ValidationError:
    try:
        LLMSentimentOutput.model_validate({"label": "negative"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected LLMSentimentOutput to reject an incomplete payload")


@pytest.fixture
def priced_model() -> Any:
    """Price the test model so the cost assertions have something to assert on."""
    override_prices({MODEL: ModelPrice.metered(input_per_mtok=1.0, output_per_mtok=4.0)})
    yield
    reset_price_table()


# ── Request shape ─────────────────────────────────────────────────────────────


async def test_system_and_user_turns_are_sent_separately() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    assert client.completions.calls[0]["messages"] == [
        {"role": "system", "content": "RUBRIC"},
        {"role": "user", "content": "Hated it."},
    ]


async def test_structured_output_is_requested_by_model() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    assert client.completions.calls[0]["response_format"] is LLMSentimentOutput


async def test_output_ceiling_uses_the_non_deprecated_parameter() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    call = client.completions.calls[0]
    assert call["max_completion_tokens"] == 1024
    assert "max_tokens" not in call


async def test_no_sampling_parameters_are_sent() -> None:
    provider, client = _provider([_response()])

    await _classify(provider)

    call = client.completions.calls[0]
    assert not [p for p in SAMPLING_PARAMS if p in call]


async def test_cache_key_is_stable_and_omitted_when_opted_out() -> None:
    provider, client = _provider([_response(), _response()])

    await _classify(provider)
    await _classify(provider, cache_prefix=False)

    assert client.completions.calls[0]["prompt_cache_key"] == PROMPT_CACHE_KEY
    assert "prompt_cache_key" not in client.completions.calls[1]


# ── Usage accounting ──────────────────────────────────────────────────────────


async def test_cached_tokens_are_subtracted_from_prompt_tokens(priced_model: None) -> None:
    provider, _ = _provider(
        [
            _response(
                prompt_tokens=1200,
                completion_tokens=40,
                cached_tokens=1000,
                cache_write_tokens=50,
            )
        ]
    )

    result = await _classify(provider)

    # 1200 reported - 1000 cached - 50 written = 150 billed at the full rate.
    assert result.tokens_in == 150
    assert result.cache_read_tokens == 1000
    assert result.cache_write_tokens == 50
    assert result.cost_usd == pytest.approx(
        (150 * 1.0 + 40 * 4.0 + 1000 * 0.1 + 50 * 2.0) / 1_000_000
    )


async def test_uncached_call_bills_the_whole_prompt() -> None:
    provider, _ = _provider([_response(prompt_tokens=800, completion_tokens=30)])

    result = await _classify(provider)

    assert result.tokens_in == 800
    assert result.cache_read_tokens == 0


async def test_missing_usage_details_do_not_explode() -> None:
    response = _response()
    response.usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None)
    provider, _ = _provider([response])

    result = await _classify(provider)

    assert (result.tokens_in, result.tokens_out, result.cache_read_tokens) == (0, 0, 0)


# ── Refusals and unparseable output ───────────────────────────────────────────


async def test_in_band_refusal_raises() -> None:
    provider, _ = _provider([_response(refusal="I can't help with that.", parsed=None)])

    with pytest.raises(ProviderRefusalError):
        await _classify(provider)


async def test_content_filter_exception_maps_to_refusal() -> None:
    provider, _ = _provider([openai.ContentFilterFinishReasonError()])

    with pytest.raises(ProviderRefusalError):
        await _classify(provider)


async def test_truncated_completion_exception_is_a_schema_error() -> None:
    truncated = SimpleNamespace(usage=None)
    provider, _ = _provider([openai.LengthFinishReasonError(completion=truncated)])  # type: ignore[arg-type]

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


async def test_length_finish_reason_is_a_schema_error() -> None:
    provider, _ = _provider([_response(finish_reason="length")])

    with pytest.raises(ProviderSchemaError, match="output ceiling"):
        await _classify(provider)


async def test_missing_parsed_output_is_a_schema_error() -> None:
    provider, _ = _provider([_response(parsed=None)])

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


async def test_empty_choices_is_a_schema_error() -> None:
    provider, _ = _provider([_response(choices=[])])

    with pytest.raises(ProviderSchemaError, match="no choices"):
        await _classify(provider)


async def test_validation_failure_is_a_schema_error() -> None:
    provider, _ = _provider([_validation_error()])

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


# ── Typed exception mapping ───────────────────────────────────────────────────


async def test_not_found_maps_to_unavailable() -> None:
    provider, _ = _provider(
        [openai.NotFoundError("no model", response=_status_response(404), body=None)]
    )

    with pytest.raises(ProviderUnavailableError, match="not found"):
        await _classify(provider)


async def test_rate_limit_maps_to_rate_limit_and_carries_retry_after() -> None:
    # RateLimitError subclasses APIStatusError; order in the except chain is what
    # keeps a throttle from being reported as an outage.
    provider, _ = _provider(
        [
            openai.RateLimitError(
                "slow down",
                response=_status_response(429, {"retry-after": "7"}),
                body=None,
            )
        ]
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _classify(provider)
    assert caught.value.retry_after_s == 7.0


async def test_generic_status_error_maps_to_unavailable() -> None:
    provider, _ = _provider(
        [openai.APIStatusError("boom", response=_status_response(503), body=None)]
    )

    with pytest.raises(ProviderUnavailableError, match="503"):
        await _classify(provider)


async def test_connection_error_maps_to_unavailable() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    provider, _ = _provider([openai.APIConnectionError(request=request)])

    with pytest.raises(ProviderUnavailableError, match="unreachable"):
        await _classify(provider)


# ── Configuration and lifecycle ───────────────────────────────────────────────


async def test_unconfigured_provider_reports_and_refuses() -> None:
    provider = OpenAIProvider(Settings(app=AppSettings(env="test")))

    assert provider.configured is False
    assert await provider.health() is False
    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


async def test_configured_provider_names_itself() -> None:
    provider, _ = _provider([])

    assert provider.name is Provider.OPENAI
    assert provider.configured is True


async def test_health_probe_swallows_failures() -> None:
    provider, client = _provider([])

    async def retrieve(_: str) -> object:
        raise RuntimeError("connection reset")

    client.models = SimpleNamespace(retrieve=retrieve)  # type: ignore[attr-defined]

    assert await provider.health() is False


async def test_aclose_releases_the_client() -> None:
    provider, client = _provider([])

    await provider.aclose()

    assert client.closed is True
