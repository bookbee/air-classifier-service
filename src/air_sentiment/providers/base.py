"""The LLM provider contract.

One protocol covers both the local and the cloud tiers. Ollama is not a special
case — it is a provider that happens to be free and nearby — so Tiers 2 and 3
share their prompt building, schema derivation, and validation entirely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from air_sentiment.constants import Provider
from air_sentiment.schemas.common import TierUsage


class ProviderError(Exception):
    """Base class for provider failures."""


class ProviderUnavailableError(ProviderError):
    """Not configured, unreachable, or circuit breaker open."""


class ProviderRateLimitError(ProviderError):
    """Throttled upstream. Retryable after a delay."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderSchemaError(ProviderError):
    """The model returned output that did not validate against the schema.

    Distinct from a transport failure because the remedy differs: this one is
    worth one bounded repair retry, a connection error is worth a backoff retry.
    """


class ProviderRefusalError(ProviderError):
    """The provider's safety layer declined the request.

    Not retryable on the same model. The registry treats it as a signal to try a
    different provider rather than to back off.
    """


class ProviderResult(BaseModel):
    """A validated structured response plus what it cost."""

    parsed: BaseModel
    provider: Provider
    model: str
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    def to_usage(self) -> TierUsage:
        return TierUsage(
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cost_usd=self.cost_usd,
            provider=self.provider.value,
            model=self.model,
        )


@runtime_checkable
class LLMProvider(Protocol):
    """A backend that can return schema-constrained structured output."""

    name: Provider

    async def health(self) -> bool:
        """Cheap reachability probe. Must not raise."""
        ...

    @property
    def configured(self) -> bool:
        """Whether credentials/endpoint are present. Cheap, synchronous, no I/O."""
        ...

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
        """Run one constrained completion.

        Args:
            system_prompt: Stable prefix. Must be byte-identical across requests
                for prompt caching to work — no timestamps, no request IDs, no
                per-caller interpolation above the cache breakpoint.
            user_content: The volatile part. Everything request-specific goes here.
            output_model: Pydantic model the response must validate against.
            model: Provider-specific model identifier.
            max_tokens: Output ceiling.
            cache_prefix: Request prompt caching of the system prefix where the
                provider supports it.

        Raises:
            ProviderUnavailableError: Not configured or unreachable.
            ProviderRateLimitError: Throttled.
            ProviderSchemaError: Output failed validation.
            ProviderRefusalError: Declined by the provider's safety layer.
        """
        ...

    async def aclose(self) -> None:
        """Release connections held by the underlying client."""
        ...
