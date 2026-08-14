"""Anthropic adapter for Tier 3.

Three things here are load-bearing and easy to get subtly wrong.

**Structured output.** The SDK's ``messages.parse`` returns a validated instance
of the requested Pydantic model, so there is no prose to scrape and no
retry-on-bad-JSON loop to maintain. A validation failure surfaces as
``pydantic.ValidationError`` and becomes a :class:`ProviderSchemaError`, which is
the one failure worth a single repair retry.

**Prompt caching.** The system prompt is sent as a block list with a
``cache_control`` breakpoint on the last block. The saving is large — a cache
read bills at roughly a tenth of the input rate — but it is also *silent* when it
does not happen: below a model's minimum cacheable prefix nothing errors, the
cache simply never fills. That is why ``cache_read_input_tokens`` and
``cache_creation_input_tokens`` are carried through into
:class:`~air_sentiment.providers.base.ProviderResult` rather than dropped: they
are the only evidence that caching is working at all.

**Thinking and sampling.** Labelling is not a reasoning task, so extended
thinking is switched off wherever a model would otherwise enable it adaptively,
and no sampling parameters are sent at all — several current Claude models reject
non-default ``temperature``/``top_p``/``top_k`` outright, so determinism is
steered through the rubric and the schema instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ValidationError

from air_sentiment.config import Settings
from air_sentiment.constants import Provider
from air_sentiment.observability.cost import estimate_cost
from air_sentiment.observability.logging import get_logger
from air_sentiment.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:  # The SDK is an optional extra; nothing imports it at runtime here.
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        MessageParam,
        ParsedMessage,
        TextBlockParam,
        ThinkingConfigParam,
    )

logger = get_logger(__name__)

#: Models whose extended thinking is adaptive-by-default. They must be told to
#: stop or a labelling call quietly bills reasoning tokens. Claude Haiku 4.5 is
#: deliberately absent: thinking is off by default there and it has no ``effort``
#: parameter at all, so passing one is an error rather than a no-op.
ADAPTIVE_THINKING_MODELS: Final[frozenset[str]] = frozenset({"claude-sonnet-5"})

#: ``stop_reason`` value that means the safety layer declined. Content must not
#: be read on a refusal — there is nothing in it to read.
_REFUSAL_STOP_REASON: Final[str] = "refusal"


@dataclass(frozen=True, slots=True)
class _SdkErrors:
    """The SDK exception classes this adapter maps, captured at construction.

    Held as values rather than looked up through the module on every call so the
    ``except`` chain below reads in the order it is evaluated, and so the lazy
    import stays confined to ``__init__``.
    """

    not_found: type[Exception]
    rate_limit: type[Exception]
    status: type[Exception]
    connection: type[Exception]


class AnthropicProvider:
    """Claude backend for the cloud tier."""

    name: Provider = Provider.ANTHROPIC

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.providers.anthropic
        self._client: AsyncAnthropic | None = None
        self._errors: _SdkErrors | None = None

        # Imported here, not at module scope: `air-sentiment[anthropic]` is an
        # optional extra, and an install without it must still be able to import
        # this module and report the provider as unconfigured.
        try:
            import anthropic
        except ImportError:
            logger.info("anthropic.sdk_missing", hint="install air-sentiment[anthropic]")
            return

        self._errors = _SdkErrors(
            not_found=anthropic.NotFoundError,
            rate_limit=anthropic.RateLimitError,
            status=anthropic.APIStatusError,
            connection=anthropic.APIConnectionError,
        )
        if self._config.api_key is None:
            return

        self._client = anthropic.AsyncAnthropic(
            api_key=self._config.api_key.get_secret_value(),
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            max_retries=self._config.max_retries,
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def health(self) -> bool:
        """Reachability probe against the configured model.

        A metadata read rather than a completion: it exercises credentials, DNS
        and the base URL without spending tokens. Never raises — an unreachable
        provider is a routing fact, not an error to propagate.
        """
        client = self._client
        if client is None:
            return False
        try:
            await client.models.retrieve(self._config.model)
        except Exception as exc:
            logger.warning("anthropic.health_failed", error=type(exc).__name__)
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
            raise ProviderUnavailableError("anthropic provider is not configured")

        messages: list[MessageParam] = [{"role": "user", "content": user_content}]
        # Only the conditional parameter is splatted; everything else stays a
        # named argument so the SDK's own types still check this call site.
        optional: dict[str, Any] = {}
        thinking = self._thinking_for(model)
        if thinking is not None:
            optional["thinking"] = thinking

        started = time.perf_counter()
        try:
            response: ParsedMessage[BaseModel] = await client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system_prompt, cache_prefix=cache_prefix),
                messages=messages,
                output_format=output_model,
                **optional,
            )
        # Most-specific first: NotFoundError and RateLimitError both subclass
        # APIStatusError, so a broader clause placed above them would swallow the
        # distinction between "wrong model id" and "back off and try again".
        except errors.not_found as exc:
            raise ProviderUnavailableError(f"anthropic model {model!r} not found") from exc
        except errors.rate_limit as exc:
            raise ProviderRateLimitError(
                f"anthropic rate limited on {model!r}", retry_after_s=_retry_after(exc)
            ) from exc
        except errors.status as exc:
            raise ProviderUnavailableError(
                f"anthropic returned {getattr(exc, 'status_code', 'an error')} for {model!r}"
            ) from exc
        except errors.connection as exc:
            raise ProviderUnavailableError(f"anthropic unreachable: {type(exc).__name__}") from exc
        except ValidationError as exc:
            raise ProviderSchemaError(f"anthropic output failed validation: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.stop_reason == _REFUSAL_STOP_REASON:
            raise ProviderRefusalError(f"anthropic declined the request on {model!r}")

        parsed = response.parsed_output
        if parsed is None:
            raise ProviderSchemaError(
                f"anthropic returned no parsed output on {model!r} "
                f"(stop_reason={response.stop_reason})"
            )

        return self._to_result(parsed, response=response, model=model, latency_ms=latency_ms)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    # ── Request shaping ───────────────────────────────────────────────────

    def _system_blocks(self, system_prompt: str, *, cache_prefix: bool) -> list[TextBlockParam]:
        """The stable prefix, with the cache breakpoint on the final block.

        The breakpoint marks the *end* of what may be cached, so it belongs on
        the last block and nowhere else.
        """
        block: TextBlockParam = {"type": "text", "text": system_prompt}
        if cache_prefix and self._config.prompt_cache_enabled:
            block["cache_control"] = {
                "type": "ephemeral",
                "ttl": self._config.prompt_cache_ttl,
            }
        return [block]

    def _thinking_for(self, model: str) -> ThinkingConfigParam | None:
        """Disable thinking only where it would otherwise turn itself on.

        Sending the field to a model that never thinks adaptively is harmless but
        untrue to the API surface; omitting it from one that does costs reasoning
        tokens on every classification.
        """
        if model not in ADAPTIVE_THINKING_MODELS:
            return None
        thinking: ThinkingConfigParam = {"type": "disabled"}
        return thinking

    # ── Response shaping ──────────────────────────────────────────────────

    def _to_result(
        self,
        parsed: BaseModel,
        *,
        response: ParsedMessage[BaseModel],
        model: str,
        latency_ms: float,
    ) -> ProviderResult:
        usage = response.usage
        # `input_tokens` excludes anything served from or written to the cache,
        # so the four counts are disjoint and can be priced independently.
        tokens_in = _count(usage, "input_tokens")
        tokens_out = _count(usage, "output_tokens")
        cache_read = _count(usage, "cache_read_input_tokens")
        cache_write = _count(usage, "cache_creation_input_tokens")

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

    Read from the response headers rather than from the message text: a
    string-matched delay is a parser that breaks the next time the wording
    changes.
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
