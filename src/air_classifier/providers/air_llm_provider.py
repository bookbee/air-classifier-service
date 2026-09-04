"""air-llm adapter.

The same :class:`~air_classifier.providers.base.LLMProvider` contract this
service has always used, but every model call now goes through exactly one
place: air-llm's ``POST /v1/inference``. air-llm owns the provider keys,
cross-vendor routing/failover, cost accounting and caching (air-llm's own
docs/00-plan.md, docs/01-hld.md) — this adapter's whole job is to speak the
ladder's language on one side and air-llm's wire contract on the other.

Two instances exist, one per rung — ``Provider.OLLAMA`` for Tier 2,
``Provider.ANTHROPIC`` for Tier 3 — but ``name`` is a label now, not a
transport detail: it is not baked into the request the way the old per-vendor
SDK adapters (and the air-infra gateway adapter before this) prefixed it.
``model`` is sent to air-llm exactly as the caller supplied it — a bare
``"provider:model"`` bypass for Tier 2 (so a down Ollama fails this call
cleanly instead of air-llm silently reaching for a cloud fallback under the
Tier-2 label), or a routing alias carrying air-llm's own cross-vendor failover
chain for Tier 3. Which one a rung asks for is that rung's own decision
(`config.py`'s ``AirLlmSettings``), not this adapter's.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from air_classifier.config import Settings
from air_classifier.constants import Provider
from air_classifier.observability.logging import get_logger
from air_classifier.observability.tracing import span
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)

logger = get_logger(__name__)

_INFERENCE_PATH: Final[str] = "/v1/inference"
_READY_PATH: Final[str] = "/v1/ready"
_HEALTH_TIMEOUT_S: Final[float] = 2.0
_SCHEMA_ERROR_CHARS: Final[int] = 500
#: Advisory — see `LLMProvider.classify`'s `remaining_ms` docstring. air-llm is
#: free to ignore it; sending it costs nothing and lets it stop early if it chooses to.
_REMAINING_BUDGET_HEADER: Final[str] = "X-Remaining-Budget-Ms"
#: finish_reason air-llm reports when the model stopped before it could close
#: the object — nothing that validates could have come back.
_LENGTH_FINISH_REASON: Final[str] = "length"


class _InferenceMessage(BaseModel):
    role: str
    content: str


class _InferenceRequest(BaseModel):
    """The wire shape of air-llm's ``POST /v1/inference`` body."""

    task: str = "chat"
    model: str
    messages: list[_InferenceMessage]
    max_tokens: int
    json_schema: dict[str, Any]
    schema_name: str
    cache_prefix: bool = True


class _InferenceUsage(BaseModel):
    """Token/cache counters from air-llm's response.

    Every field defaults to zero and coerces a malformed or missing counter
    to zero rather than rejecting the response — a partial usage block is
    still worth the answer it came with.
    """

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @field_validator(
        "prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens",
        mode="before",
    )
    @classmethod
    def _zero_unless_a_nonnegative_int(cls, value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0


class _InferenceResponse(BaseModel):
    """air-llm's ``/v1/inference`` response.

    Every field is optional and defaulted: a missing or malformed field here
    should degrade the same way a bare ``dict.get()`` always did — to an
    absent/zeroed value — not fail the whole call. `classify()`'s own checks
    (refusal, finish_reason, schema validation of `content`) are what decide
    whether the *answer* was usable; this model only decides whether the
    envelope around it could be read at all.
    """

    model_config = ConfigDict(extra="ignore")

    refusal: bool = False
    finish_reason: str | None = None
    content: str = ""
    usage: _InferenceUsage = Field(default_factory=_InferenceUsage)
    provider: Provider | None = None
    model: str | None = None
    cost_usd: float = 0.0

    @field_validator("provider", mode="before")
    @classmethod
    def _none_if_unrecognized(cls, value: Any) -> Any:
        # A provider air-llm knows about but this service's enum does not
        # (yet) must fall back to the adapter's own label, not fail the
        # response outright — see `classify()`'s `body.provider or self.name`.
        if not isinstance(value, str):
            return None
        try:
            Provider(value)
        except ValueError:
            return None
        return value


class AirLlmProvider:
    """:class:`LLMProvider` backed by air-llm's ``/v1/inference``."""

    def __init__(self, settings: Settings, name: Provider) -> None:
        self.name = name
        self._config = settings.air_llm
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._closed = False
        #: Set by the most recent `health()` call; `None` on success. Not part
        #: of the hot path — `is_available()` only needs the bool `health()`
        #: itself returns — this exists so a caller that already knows the
        #: probe failed can ask what actually happened, for `/v1/ready` and
        #: `/v1/capabilities` to report instead of a bare "unavailable".
        self._last_health_detail: str | None = None

    @property
    def configured(self) -> bool:
        """Whether air-llm is switched on and a service token is present. No I/O."""
        return self._config.configured

    @property
    def health_detail(self) -> str | None:
        """Why the last `health()` call failed, or `None` if it last succeeded
        (or never ran). Never raises; reflects only the most recent call."""
        return self._last_health_detail

    async def health(self) -> bool:
        """Whether air-llm itself reports ready. Never raises.

        Deliberately not a per-vendor probe: which upstream actually answers a
        given call is air-llm's own routing decision, made per request from
        its failover chain. This asks the one question that is this adapter's
        to ask — is air-llm itself worth calling right now — and leaves the
        rest to the call itself.
        """
        if not self.configured:
            self._last_health_detail = "not configured (AIR_LLM__ENABLED / AIR_LLM__TOKEN)"
            return False
        try:
            client = await self._ready_client()
            response = await client.get(_READY_PATH, timeout=_HEALTH_TIMEOUT_S)
        except Exception as exc:
            logger.debug("provider.air_llm.health_failed", error=type(exc).__name__)
            self._last_health_detail = f"air-llm unreachable at {self._config.base_url}: " + (
                type(exc).__name__
            )
            return False
        if response.status_code != httpx.codes.OK:
            self._last_health_detail = f"air-llm returned HTTP {response.status_code} on /v1/ready"
            return False
        self._last_health_detail = None
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
        remaining_ms: float | None = None,
    ) -> ProviderResult:
        if not self.configured:
            raise ProviderUnavailableError("air-llm is not configured")

        client = await self._ready_client()
        request = _InferenceRequest(
            model=model,
            messages=[
                _InferenceMessage(role="system", content=system_prompt),
                _InferenceMessage(role="user", content=user_content),
            ],
            max_tokens=max_tokens,
            json_schema=output_model.model_json_schema(),
            schema_name=output_model.__name__,
            cache_prefix=cache_prefix,
        )

        started = time.perf_counter()
        async with span(
            "provider.air_llm", label=self.name.value, model=model, remaining_ms=remaining_ms
        ):
            body = await self._post_inference(
                client, request, model=model, remaining_ms=remaining_ms
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if body.refusal:
            raise ProviderRefusalError(f"air-llm declined {model!r}")
        if body.finish_reason == _LENGTH_FINISH_REASON:
            raise ProviderSchemaError(f"model {model!r} hit the output ceiling")

        try:
            parsed = output_model.model_validate_json(body.content)
        except ValidationError as exc:
            raise ProviderSchemaError(_summarise(exc)) from exc

        # The vendor that actually answered, honest even when air-llm's
        # failover chain served a different one than `model` named — this
        # matters for cost/usage attribution more than the label this adapter
        # was constructed with does.
        answered_by = body.provider or self.name
        return ProviderResult(
            parsed=parsed,
            provider=answered_by,
            model=body.model or model,
            tokens_in=body.usage.prompt_tokens,
            tokens_out=body.usage.completion_tokens,
            cache_read_tokens=body.usage.cache_read_tokens,
            cache_write_tokens=body.usage.cache_write_tokens,
            latency_ms=latency_ms,
            # air-llm's own cost book is authoritative — this service holds no
            # price table of its own to fall back to, and does not need one:
            # `TierContext.with_verdict` folds this straight into the
            # request's running spend against `options.max_cost_usd`.
            cost_usd=body.cost_usd,
        )

    async def aclose(self) -> None:
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ── Internals ─────────────────────────────────────────────────────────

    async def _ready_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise ProviderUnavailableError("air-llm provider is closed")
        client = self._client
        if client is not None:
            return client
        async with self._client_lock:
            if self._closed:
                raise ProviderUnavailableError("air-llm provider is closed")
            if self._client is None:
                token = self._config.token
                headers = {"X-API-Key": token.get_secret_value()} if token else {}
                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=httpx.Timeout(self._config.timeout_s),
                    headers=headers,
                )
            return self._client

    async def _post_inference(
        self,
        client: httpx.AsyncClient,
        request: _InferenceRequest,
        *,
        model: str,
        remaining_ms: float | None = None,
    ) -> _InferenceResponse:
        # Per-call header, not a client default: every call on this shared
        # client has a different remaining budget by the time it fires.
        headers = (
            {_REMAINING_BUDGET_HEADER: str(round(remaining_ms))}
            if remaining_ms is not None
            else None
        )
        try:
            response = await client.post(
                _INFERENCE_PATH, json=request.model_dump(mode="json"), headers=headers
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"air-llm timed out after {self._config.timeout_s:g}s"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"air-llm unreachable at {self._config.base_url}: {exc}"
            ) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                f"air-llm throttled {model!r}", retry_after_s=_retry_after(response)
            )
        if response.is_error:
            raise ProviderUnavailableError(
                f"air-llm returned HTTP {response.status_code} for {model!r}"
            )

        try:
            raw = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError("air-llm returned a body that is not JSON") from exc
        if not isinstance(raw, dict):
            raise ProviderUnavailableError("air-llm returned a JSON body that is not an object")
        try:
            return _InferenceResponse.model_validate(raw)
        except ValidationError as exc:
            raise ProviderUnavailableError(
                f"air-llm returned a response that does not fit the expected shape: {exc}"
            ) from exc


def _summarise(exc: ValidationError) -> str:
    """Field paths and reasons, without the offending values (which are model output)."""
    parts = [
        f"{'.'.join(str(item) for item in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts)[:_SCHEMA_ERROR_CHARS] or "output did not match the schema"


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
