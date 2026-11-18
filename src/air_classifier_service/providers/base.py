"""The LLM provider contract.

One protocol covers both the local and the cloud tiers. The local tier is not
a special case — it is a model that happens to be free and nearby, with which
concrete model that even is left entirely to air-model-gateway's routing — so Tiers
2 and 3 share their prompt building, schema derivation, and validation
entirely.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from air_classifier_service.constants import ModelClass, Provider
from air_classifier_service.schemas.common import TierUsage


class ProviderError(Exception):
    """Base class for provider failures."""


class ProviderUnavailableError(ProviderError):
    """Not configured, unreachable, or circuit breaker open."""


class ProviderRateLimitError(ProviderError):
    """Throttled upstream. Retryable after a delay."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderBudgetExceededError(ProviderRateLimitError):
    """This service's daily model budget at the gateway is spent.

    A rate limit only in transport (the gateway answers 429), not in meaning:
    it does not clear in seconds but at the next UTC day, so nothing here
    should retry it, and an operator reading the trace needs to tell "spend
    ceiling reached" apart from "briefly throttled". A subclass so every
    existing ``ProviderRateLimitError`` handler still treats it as a
    throttle, while the class name it carries into a failed tier's trace
    says which one it was.
    """


class ProviderRequestError(ProviderError):
    """The gateway refused the request itself, not the model's capacity.

    A 4xx other than 429: an invalid body, a model class or field the gateway
    doesn't accept, a missing grant, a bad token, a wrong base URL, or every
    model in the chain rejecting the input as invalid (``provider_rejected``).
    Retrying the identical request can't succeed, and nor is it an outage —
    usually it is a configuration or contract bug on *this* side. It is kept
    apart from :class:`ProviderUnavailableError` because the ladder treats
    "unavailable" as a capacity gap (``degraded: true``, climb on), which
    would let a contract break hide behind healthy-looking 200s.
    """

    def __init__(self, message: str, *, status: int, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


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

    @property
    def health_detail(self) -> str | None:
        """Why the most recent `health()` call failed, or `None` if it
        succeeded (or never ran). No I/O — reflects only the last call."""
        ...

    async def classify(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[BaseModel],
        model: ModelClass,
        max_tokens: int,
        cache_prefix: bool = True,
        remaining_ms: float | None = None,
        request_id: str | None = None,
    ) -> ProviderResult:
        """Run one constrained completion.

        Args:
            system_prompt: Stable prefix. Must be byte-identical across requests
                for prompt caching to work — no timestamps, no request IDs, no
                per-caller interpolation above the cache breakpoint.
            user_content: The volatile part. Everything request-specific goes here.
            output_model: Pydantic model the response must validate against.
            model: Which class of model to ask for. Never a concrete model:
                the gateway resolves the class to its chain.
            max_tokens: Output ceiling.
            cache_prefix: Request prompt caching of the system prefix where the
                provider supports it.
            remaining_ms: How long the caller will actually wait, computed the
                same way the timeout this call is wrapped in already is — not a
                new budget, a restatement of the one that already governs this
                call. Advisory: a provider under no obligation to read it must
                still behave correctly if it ignores it entirely. The point is
                to let a provider that *can* read it stop doing work for an
                answer that will already have been discarded by the time it
                would arrive, rather than to enforce anything on this side that
                the timeout this call is wrapped in does not already enforce.
            request_id: This service's request id, forwarded so one id follows
                a request through this service's logs and the gateway's
                audit line.

        Raises:
            ProviderUnavailableError: Not configured or unreachable.
            ProviderRateLimitError: Throttled.
            ProviderBudgetExceededError: The daily budget is spent (a
                ``ProviderRateLimitError`` subclass).
            ProviderRequestError: The request itself was refused (a 4xx).
            ProviderSchemaError: Output failed validation.
            ProviderRefusalError: Declined by the provider's safety layer.
        """
        ...

    async def aclose(self) -> None:
        """Release connections held by the underlying client."""
        ...
