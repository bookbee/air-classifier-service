"""Ollama adapter tests.

Every test here runs against a mocked transport, and the base URL points at a
host that does not exist. Ollama is genuinely running on developer machines, so
a test that merely *happened* to be intercepted would pass locally and hit a real
daemon the day the interception broke — an unreachable host makes that failure
loud instead of silent.

The mapping assertions are the substance of the file: which upstream failure
becomes which provider exception decides whether the tier retries, escalates, or
opens a breaker, and nothing downstream can tell the cases apart afterwards.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from air_sentiment.config import (
    AppSettings,
    OllamaSettings,
    ProvidersSettings,
    Settings,
)
from air_sentiment.constants import Label, Provider
from air_sentiment.providers import ollama_provider as module
from air_sentiment.providers.base import (
    LLMProvider,
    ProviderRateLimitError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_sentiment.providers.ollama_provider import OllamaProvider
from air_sentiment.schemas.verdict import LLMSentimentOutput

#: Deliberately unresolvable. See the module docstring.
BASE_URL = "http://ollama.invalid:11434"
CHAT_URL = f"{BASE_URL}/api/chat"
TAGS_URL = f"{BASE_URL}/api/tags"

GOOD_OUTPUT: dict[str, Any] = {
    "label": "negative",
    "polarity": -0.6,
    "confidence": 0.81,
    "rationale": "Praises the battery but calls the camera awful.",
    "emotions": [{"name": "annoyance", "score": 0.7}],
    "aspects": [{"name": "camera", "label": "negative", "confidence": 0.9}],
    "sarcasm_detected": False,
    "language_code": "en",
}


def _settings(*, enabled: bool = True, **overrides: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(
            ollama=OllamaSettings(enabled=enabled, base_url=BASE_URL, **overrides)
        ),
    )


def _envelope(
    content: str | None = None,
    *,
    prompt_eval_count: int | None = 412,
    eval_count: int | None = 96,
    model: str = "llama3.2:3b",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "done": True,
        "message": {
            "role": "assistant",
            "content": content if content is not None else json.dumps(GOOD_OUTPUT),
        },
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


@pytest.fixture
async def provider() -> AsyncIterator[OllamaProvider]:
    made = OllamaProvider(_settings())
    try:
        yield made
    finally:
        await made.aclose()


async def _classify(provider: OllamaProvider, **overrides: Any) -> ProviderResult:
    kwargs: dict[str, Any] = {
        "system_prompt": "You are a sentiment rubric.",
        "user_content": "Great battery but the camera is awful.",
        "output_model": LLMSentimentOutput,
        "model": "llama3.2:3b",
        "max_tokens": 768,
    }
    kwargs.update(overrides)
    return await provider.classify(**kwargs)


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_the_adapter_satisfies_the_provider_protocol() -> None:
    assert isinstance(OllamaProvider(_settings()), LLMProvider)


def test_name_identifies_the_local_backend() -> None:
    assert OllamaProvider(_settings()).name is Provider.OLLAMA


def test_configured_follows_the_enabled_flag() -> None:
    assert OllamaProvider(_settings()).configured is True
    assert OllamaProvider(_settings(enabled=False)).configured is False


# ── The request ───────────────────────────────────────────────────────────────


@respx.mock
async def test_classify_constrains_the_reply_to_the_output_schema(
    provider: OllamaProvider,
) -> None:
    """`format` is what stops a small model wrapping its JSON in prose."""
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    await _classify(provider)

    sent = json.loads(route.calls.last.request.content)
    assert sent["format"] == LLMSentimentOutput.model_json_schema()
    assert sent["format"]["additionalProperties"] is False


@respx.mock
async def test_classify_asks_for_a_single_deterministic_non_streamed_reply(
    provider: OllamaProvider,
) -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    await _classify(provider)

    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == 0
    # Read from settings rather than restating the literal: this assertion is
    # about the value being forwarded, not about what the default happens to be.
    assert sent["options"]["num_ctx"] == Settings().providers.ollama.num_ctx
    assert sent["options"]["num_predict"] == 768
    assert sent["keep_alive"] == "10m"


@respx.mock
async def test_classify_sends_the_two_turns_unmodified(provider: OllamaProvider) -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    await _classify(provider, system_prompt="RUBRIC", user_content="TEXT")

    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [
        {"role": "system", "content": "RUBRIC"},
        {"role": "user", "content": "TEXT"},
    ]


@respx.mock
async def test_generation_options_come_from_configuration() -> None:
    tuned = OllamaProvider(_settings(num_ctx=8192, keep_alive="30m"))
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    await _classify(tuned)
    await tuned.aclose()

    sent = json.loads(route.calls.last.request.content)
    assert sent["options"]["num_ctx"] == 8192
    assert sent["keep_alive"] == "30m"


# ── The response ──────────────────────────────────────────────────────────────


@respx.mock
async def test_classify_returns_a_validated_instance_of_the_output_model(
    provider: OllamaProvider,
) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    result = await _classify(provider)

    assert isinstance(result.parsed, LLMSentimentOutput)
    assert result.parsed.label is Label.NEGATIVE
    assert result.parsed.aspects[0].name == "camera"
    assert result.provider is Provider.OLLAMA


@respx.mock
async def test_classify_reports_token_volume_but_never_a_cost(
    provider: OllamaProvider,
) -> None:
    """Own hardware bills nothing per token, and still consumes them."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    result = await _classify(provider)

    assert result.tokens_in == 412
    assert result.tokens_out == 96
    assert result.cost_usd == 0.0
    assert result.latency_ms >= 0.0


@respx.mock
async def test_missing_token_counters_do_not_fail_a_good_generation(
    provider: OllamaProvider,
) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_envelope(prompt_eval_count=None, eval_count=None))
    )

    result = await _classify(provider)

    assert (result.tokens_in, result.tokens_out) == (0, 0)


@respx.mock
async def test_classify_reports_the_tag_the_daemon_actually_served(
    provider: OllamaProvider,
) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_envelope(model="llama3.2:3b-instruct-q4_K_M"))
    )

    result = await _classify(provider)

    assert result.model == "llama3.2:3b-instruct-q4_K_M"


# ── Schema failures ───────────────────────────────────────────────────────────


@respx.mock
async def test_prose_instead_of_json_is_a_schema_error(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_envelope("Sure! Here is the answer:"))
    )

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


@respx.mock
async def test_a_field_outside_its_bounds_is_a_schema_error(provider: OllamaProvider) -> None:
    """Bounds are unexpressible in the emitted schema, so pydantic is the gate."""
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=_envelope(json.dumps({**GOOD_OUTPUT, "confidence": 1.7}))
        )
    )

    with pytest.raises(ProviderSchemaError) as caught:
        await _classify(provider)

    assert "confidence" in str(caught.value)


@respx.mock
async def test_an_invented_label_is_a_schema_error(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=_envelope(json.dumps({**GOOD_OUTPUT, "label": "very positive"}))
        )
    )

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


@respx.mock
async def test_a_schema_error_names_the_field_without_quoting_the_value(
    provider: OllamaProvider,
) -> None:
    """The message is logged and replayed to the model; values are model output."""
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_envelope(json.dumps({**GOOD_OUTPUT, "rationale": "q" * 900, "confidence": 4.2})),
        )
    )

    with pytest.raises(ProviderSchemaError) as caught:
        await _classify(provider)

    message = str(caught.value)
    assert "rationale" in message
    assert "confidence" in message
    assert "qqqq" not in message


@respx.mock
async def test_an_empty_assistant_message_is_a_schema_error(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope("   ")))

    with pytest.raises(ProviderSchemaError):
        await _classify(provider)


# ── Transport failures ────────────────────────────────────────────────────────


@respx.mock
async def test_a_refused_connection_means_unavailable(provider: OllamaProvider) -> None:
    """The daemon being stopped is the common case, not an exceptional one."""
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


@respx.mock
async def test_a_read_timeout_means_unavailable(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


@respx.mock
async def test_http_429_means_rate_limited_and_carries_the_retry_delay(
    provider: OllamaProvider,
) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "2.5"}, json={"error": "busy"})
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _classify(provider)

    assert caught.value.retry_after_s == 2.5


@respx.mock
async def test_a_date_shaped_retry_after_is_dropped_rather_than_misread(
    provider: OllamaProvider,
) -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        await _classify(provider)

    assert caught.value.retry_after_s is None


@respx.mock
async def test_a_missing_model_is_unavailable_not_a_schema_error(
    provider: OllamaProvider,
) -> None:
    """A repair instruction cannot conjure a model that was never pulled."""
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})
    )

    with pytest.raises(ProviderUnavailableError) as caught:
        await _classify(provider)

    assert "not found" in str(caught.value)


@respx.mock
async def test_a_server_error_is_unavailable(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))

    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


@respx.mock
async def test_a_non_json_body_is_unavailable_not_a_schema_error(
    provider: OllamaProvider,
) -> None:
    """Something other than Ollama is answering on that port."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, text="<html>proxy</html>"))

    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


@respx.mock
async def test_an_envelope_without_a_message_is_unavailable(provider: OllamaProvider) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"model": "x", "done": True}))

    with pytest.raises(ProviderUnavailableError):
        await _classify(provider)


async def test_classify_refuses_before_any_io_when_disabled() -> None:
    disabled = OllamaProvider(_settings(enabled=False))

    with pytest.raises(ProviderUnavailableError):
        await _classify(disabled)


# ── Health ────────────────────────────────────────────────────────────────────


@respx.mock
async def test_health_is_true_when_the_daemon_lists_its_models(
    provider: OllamaProvider,
) -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))

    assert await provider.health() is True


@respx.mock
async def test_health_is_false_and_silent_when_the_daemon_is_down(
    provider: OllamaProvider,
) -> None:
    """A probe that raises is a probe that cannot be used by readiness."""
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    assert await provider.health() is False


@respx.mock
async def test_health_is_false_when_something_else_answers_the_port(
    provider: OllamaProvider,
) -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(503, text="upstream unavailable"))

    assert await provider.health() is False


async def test_health_is_false_when_disabled_without_touching_the_network() -> None:
    assert await OllamaProvider(_settings(enabled=False)).health() is False


# ── Client lifecycle ──────────────────────────────────────────────────────────


@respx.mock
async def test_the_client_is_built_once_and_only_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction must not perform I/O, and a burst must not open two pools."""
    built = 0
    real = httpx.AsyncClient

    def counting(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        nonlocal built
        built += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", counting)
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))

    made = OllamaProvider(_settings())
    assert built == 0

    await _classify(made)
    await _classify(made)
    await made.aclose()

    assert built == 1


async def test_aclose_is_safe_before_the_client_was_ever_built() -> None:
    await OllamaProvider(_settings()).aclose()


@respx.mock
async def test_a_closed_provider_refuses_further_calls() -> None:
    """Reopening on demand would hide a shutdown-ordering bug rather than surface it."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_envelope()))
    made = OllamaProvider(_settings())
    await _classify(made)

    await made.aclose()

    with pytest.raises(ProviderUnavailableError):
        await _classify(made)


# ── Live daemon ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("AIR_SENTIMENT_TEST_OLLAMA"),
    reason="set AIR_SENTIMENT_TEST_OLLAMA=1 with a local daemon and the model pulled",
)
async def test_a_real_daemon_returns_schema_valid_output() -> None:
    """Opt-in smoke test against a live Ollama. Never part of the default run."""
    local = Settings(app=AppSettings(env="test"))
    live = OllamaProvider(local)
    try:
        assert await live.health() is True
        result = await live.classify(
            system_prompt="Classify the sentiment of the text. Answer only with the JSON object.",
            user_content="The battery is great but the camera is awful.",
            output_model=LLMSentimentOutput,
            model=local.providers.ollama.model,
            max_tokens=768,
        )
    finally:
        await live.aclose()

    assert isinstance(result.parsed, LLMSentimentOutput)
    assert result.cost_usd == 0.0
