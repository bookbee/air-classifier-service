"""OpenAI adapter for Tier 3, used when the Anthropic breaker is open.

Same :class:`~air_classifier.providers.base.LLMProvider` contract as the Anthropic
adapter, but two provider differences are worth stating rather than discovering:

* **Caching is implicit.** There is no cache breakpoint to place; OpenAI caches
  long prefixes on its own and ``prompt_cache_key`` only steers requests that
  share a prefix onto the same cache. The one thing this adapter must get right
  is the accounting — ``prompt_tokens`` *includes* the cached tokens, unlike
  Anthropic's ``input_tokens``, so the cached portion is subtracted out before
  pricing or the same tokens get billed twice at the full rate.
* **Refusals are in-band.** The safety layer sets ``message.refusal`` and the
  parse helper raises ``ContentFilterFinishReasonError``; both mean the same
  thing to us and both become :class:`ProviderRefusalError`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ValidationError

from air_classifier.config import Settings
from air_classifier.constants import Provider
from air_classifier.observability.cost import estimate_cost
from air_classifier.observability.logging import get_logger
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:  # The SDK is an optional extra; nothing imports it at runtime here.
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam, ParsedChatCompletion

logger = get_logger(__name__)

#: Stable routing hint for the shared system prefix. Constant on purpose: a
#: per-request value would scatter identical prefixes across cache shards and
#: quietly cost the discount it exists to earn.
PROMPT_CACHE_KEY: Final[str] = "air-classifier-system-v1"

#: ``finish_reason`` that means the completion stopped short of the closing
#: brace, so nothing that validates could have come back.
_LENGTH_FINISH_REASON: Final[str] = "length"


@dataclass(frozen=True, slots=True)
class _SdkErrors:
    """The SDK exception classes this adapter maps, captured at construction."""

    not_found: type[Exception]
    rate_limit: type[Exception]
    status: type[Exception]
    connection: type[Exception]
    content_filter: type[Exception]
    length: type[Exception]


class OpenAIProvider:
    """GPT backend for the cloud tier."""

    name: Provider = Provider.OPENAI

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.providers.openai
        self._client: AsyncOpenAI | None = None
        self._errors: _SdkErrors | None = None

        try:
            import openai
        except ImportError:
            logger.info("openai.sdk_missing", hint="install air-classifier[openai]")
            return

        self._errors = _SdkErrors(
            not_found=openai.NotFoundError,
            rate_limit=openai.RateLimitError,
            status=openai.APIStatusError,
            connection=openai.APIConnectionError,
            content_filter=openai.ContentFilterFinishReasonError,
            length=openai.LengthFinishReasonError,
        )
        if self._config.api_key is None:
            return

        self._client = openai.AsyncOpenAI(
            api_key=self._config.api_key.get_secret_value(),
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            max_retries=self._config.max_retries,
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def health(self) -> bool:
        """Metadata read against the configured model. Never raises."""
        client = self._client
        if client is None:
            return False
        try:
            await client.models.retrieve(self._config.model)
        except Exception as exc:
            logger.warning("openai.health_failed", error=type(exc).__name__)
            return False
        return True

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
        client, errors = self._client, self._errors
        if client is None or errors is None:
            raise ProviderUnavailableError("openai provider is not configured")

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Only the conditional parameter is splatted; everything else stays a
        # named argument so the SDK's own types still check this call site.
        optional: dict[str, Any] = {}
        if cache_prefix:
            optional["prompt_cache_key"] = PROMPT_CACHE_KEY

        started = time.perf_counter()
        try:
            response: ParsedChatCompletion[BaseModel] = await client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=output_model,
                # `max_tokens` is deprecated in favour of this on every current
                # model, and only this one is honoured by the newer families.
                max_completion_tokens=max_tokens,
                **optional,
            )
        # Most-specific first: NotFoundError and RateLimitError both subclass
        # APIStatusError, so a broader clause above them would collapse "wrong
        # model id" and "back off" into a single indistinguishable failure.
        except errors.not_found as exc:
            raise ProviderUnavailableError(f"openai model {model!r} not found") from exc
        except errors.rate_limit as exc:
            raise ProviderRateLimitError(
                f"openai rate limited on {model!r}", retry_after_s=_retry_after(exc)
            ) from exc
        except errors.status as exc:
            raise ProviderUnavailableError(
                f"openai returned {getattr(exc, 'status_code', 'an error')} for {model!r}"
            ) from exc
        except errors.connection as exc:
            raise ProviderUnavailableError(f"openai unreachable: {type(exc).__name__}") from exc
        except errors.content_filter as exc:
            raise ProviderRefusalError(f"openai content filter declined {model!r}") from exc
        except errors.length as exc:
            raise ProviderSchemaError(
                f"openai truncated its reply on {model!r} before it could validate"
            ) from exc
        except ValidationError as exc:
            raise ProviderSchemaError(f"openai output failed validation: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if not response.choices:
            raise ProviderSchemaError(f"openai returned no choices on {model!r}")
        choice = response.choices[0]
        if choice.message.refusal is not None:
            raise ProviderRefusalError(f"openai declined the request on {model!r}")
        if choice.finish_reason == _LENGTH_FINISH_REASON:
            raise ProviderSchemaError(f"openai hit the output ceiling on {model!r}")

        parsed = choice.message.parsed
        if parsed is None:
            raise ProviderSchemaError(
                f"openai returned no parsed output on {model!r} "
                f"(finish_reason={choice.finish_reason})"
            )

        return self._to_result(parsed, response=response, model=model, latency_ms=latency_ms)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _to_result(
        self,
        parsed: BaseModel,
        *,
        response: ParsedChatCompletion[BaseModel],
        model: str,
        latency_ms: float,
    ) -> ProviderResult:
        usage = response.usage
        prompt_tokens = _count(usage, "prompt_tokens")
        tokens_out = _count(usage, "completion_tokens")
        cache_read = _count(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
        cache_write = _count(getattr(usage, "prompt_tokens_details", None), "cache_write_tokens")
        # Cached tokens are reported *inside* `prompt_tokens`. Pricing both would
        # bill the same tokens twice, once at full rate and once at the discount.
        tokens_in = max(0, prompt_tokens - cache_read - cache_write)

        return ProviderResult(
            parsed=parsed,
            provider=self.name,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(
                model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
        )


def _count(usage: object, field: str) -> int:
    """Read one token counter, tolerating a field the API left null."""
    value = getattr(usage, field, None)
    return int(value) if isinstance(value, int) else 0


def _retry_after(exc: Exception) -> float | None:
    """Seconds the server asked us to wait, when it said so.

    Read from the response headers rather than the message text: a string-matched
    delay is a parser that breaks the next time the wording changes.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
