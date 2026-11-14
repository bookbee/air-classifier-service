"""air-model-gateway adapter.

The same :class:`~air_classifier_service.providers.base.LLMProvider` contract
this service has always used, with every model call going through exactly one
place: air-model-gateway's ``POST /v1/generate``. The gateway owns provider
keys, cross-vendor routing and failover, cost accounting, budgets and caching
(its own docs/01-hld.md, docs/04-integration.md). This adapter's whole job is
to speak the ladder's language on one side and the gateway's wire contract on
the other.

Two instances exist, one per rung, labelled ``Provider.OLLAMA`` for Tier 2 and
``Provider.ANTHROPIC`` for Tier 3, but ``name`` is only a label: it names the
rung's breaker and bulkhead and is the fallback attribution if a response
doesn't say who answered. What is actually asked for is ``model``, a
:class:`~air_classifier_service.constants.ModelClass`, and which class a rung
asks for is that rung's decision (``config.py``'s ``ModelGatewaySettings``), not
this adapter's.

**Errors are sorted by what the ladder should do about them**, because the
ladder's only failure signals are the provider exceptions:

* 429 is a throttle (:class:`ProviderRateLimitError`), except the gateway's
  ``budget_exceeded``, which is :class:`ProviderBudgetExceededError`: it clears
  at the next UTC day, not in seconds.
* Any other 4xx is :class:`ProviderRequestError` and is logged at error
  level. The gateway rejected the request itself (an unknown field, a model
  class it won't take, a missing grant or token, a wrong base URL), so it is
  almost always a contract or configuration bug on this side. Reporting it as
  "unavailable" would read as a capacity gap, and the ladder would degrade
  every request to Tier 1 behind healthy-looking 200s.
* 5xx and transport failures are :class:`ProviderUnavailableError`: the
  gateway, or every model behind it, couldn't serve right now.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from air_classifier_service.config import Settings
from air_classifier_service.constants import HEADER_REQUEST_ID, ModelClass, Provider
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.observability.tracing import span
from air_classifier_service.providers.base import (
    ProviderBudgetExceededError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)

logger = get_logger(__name__)

_GENERATE_PATH: Final[str] = "/v1/generate"
_READY_PATH: Final[str] = "/v1/ready"
_HEALTH_TIMEOUT_S: Final[float] = 2.0
_SCHEMA_ERROR_CHARS: Final[int] = 500
#: Advisory — see `LLMProvider.classify`'s `remaining_ms` docstring. The gateway
#: doesn't read it today; sending it costs nothing, and headers (unlike body
#: fields) are not part of the gateway's `extra="forbid"` contract.
_REMAINING_BUDGET_HEADER: Final[str] = "X-Remaining-Budget-Ms"
#: The gateway's normalized finish_reason for "cut off at max_tokens", the
#: same for every provider: nothing that validates could have come back.
_LENGTH_FINISH_REASON: Final[str] = "length"
#: The gateway's problem `code` for a spent daily budget (its api/errors.py).
_BUDGET_EXCEEDED_CODE: Final[str] = "budget_exceeded"


class _GenerateMessage(BaseModel):
    role: str
    content: str


class _GenerateRequest(BaseModel):
    """The wire shape of the gateway's ``POST /v1/generate`` body.

    Only fields the gateway's ``GenerateInput`` declares: it is
    ``extra="forbid"``, so anything else is a 422 on every call. ``stream``,
    ``pin``, ``provider`` and ``temperature`` are left to the gateway's
    defaults on purpose: answers are machine-read whole, this service has no
    pin grant, and provider preference within a class is the gateway's
    routing to decide.
    """

    model: ModelClass
    messages: list[_GenerateMessage]
    max_tokens: int
    json_schema: dict[str, Any]
    schema_name: str
    cache_prefix: bool = True


class _GenerateUsage(BaseModel):
    """Token/cache counters from the gateway's response.

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
        "prompt_tokens",
        "completion_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        mode="before",
    )
    @classmethod
    def _zero_unless_a_nonnegative_int(cls, value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0


class _GenerateResponse(BaseModel):
    """The gateway's ``/v1/generate`` response (its ``GenerateOutput``).

    Every field is optional and defaulted: a missing or malformed field here
    should degrade to an absent/zeroed value, not fail the whole call.
    `classify()`'s own checks (refusal, finish_reason, schema validation of
    `content`) decide whether the *answer* was usable; this model only decides
    whether the envelope around it could be read at all. Fields this service
    has no use for (``id``, ``model_class``, ``cached``,
    ``provider_finish_reason``) are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    refusal: bool = False
    finish_reason: str | None = None
    content: str = ""
    usage: _GenerateUsage = Field(default_factory=_GenerateUsage)
    provider: Provider | None = None
    model: str | None = None
    cost_usd: float = 0.0

    @field_validator("provider", mode="before")
    @classmethod
    def _none_if_unrecognized(cls, value: Any) -> Any:
        # A provider the gateway knows about but this service's enum does not
        # (yet) must fall back to the adapter's own label, not fail the
        # response outright — see `classify()`'s `body.provider or self.name`.
        if not isinstance(value, str):
            return None
        try:
            Provider(value)
        except ValueError:
            return None
        return value


class ModelGatewayProvider:
    """:class:`LLMProvider` backed by air-model-gateway's ``/v1/generate``."""

    def __init__(self, settings: Settings, name: Provider) -> None:
        self.name = name
        self._config = settings.model_gateway
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
        """Whether the gateway is switched on and a service token is present. No I/O."""
        return self._config.configured

    @property
    def health_detail(self) -> str | None:
        """Why the last `health()` call failed, or `None` if it last succeeded
        (or never ran). Never raises; reflects only the most recent call."""
        return self._last_health_detail

    async def health(self) -> bool:
        """Whether the gateway itself reports ready. Never raises.

        Deliberately not a per-vendor probe: which upstream actually answers a
        given call is the gateway's routing decision, made per request from
        its chain. This asks the one question that is this adapter's to ask —
        is the gateway itself worth calling right now — and leaves the rest
        to the call itself. ``/v1/ready`` needs no token, so a revoked or
        wrong token shows up on the first call (as a
        :class:`ProviderRequestError`), not here.
        """
        if not self.configured:
            self._last_health_detail = (
                "not configured (MODEL_GATEWAY__ENABLED / MODEL_GATEWAY__TOKEN)"
            )
            return False
        try:
            client = await self._ready_client()
            response = await client.get(_READY_PATH, timeout=_HEALTH_TIMEOUT_S)
        except Exception as exc:
            logger.debug("provider.model_gateway.health_failed", error=type(exc).__name__)
            self._last_health_detail = (
                f"air-model-gateway unreachable at {self._config.base_url}: " + type(exc).__name__
            )
            return False
        if response.status_code != httpx.codes.OK:
            self._last_health_detail = (
                f"air-model-gateway returned HTTP {response.status_code} on /v1/ready"
            )
            return False
        self._last_health_detail = None
        return True

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
        if not self.configured:
            raise ProviderUnavailableError("air-model-gateway is not configured")

        client = await self._ready_client()
        request = _GenerateRequest(
            model=model,
            messages=[
                _GenerateMessage(role="system", content=system_prompt),
                _GenerateMessage(role="user", content=user_content),
            ],
            max_tokens=max_tokens,
            json_schema=output_model.model_json_schema(),
            schema_name=output_model.__name__,
            cache_prefix=cache_prefix,
        )

        started = time.perf_counter()
        async with span(
            "provider.model_gateway",
            label=self.name.value,
            model=model.value,
            remaining_ms=remaining_ms,
        ):
            body = await self._post_generate(
                client, request, remaining_ms=remaining_ms, request_id=request_id
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if body.refusal:
            raise ProviderRefusalError(f"the gateway's {model.value!r} model declined")
        if body.finish_reason == _LENGTH_FINISH_REASON:
            raise ProviderSchemaError(f"{model.value!r} model hit the output ceiling")

        try:
            parsed = output_model.model_validate_json(body.content)
        except ValidationError as exc:
            raise ProviderSchemaError(_summarise(exc)) from exc

        # The provider and concrete model that actually answered, honest even
        # when the class's chain failed over to a different vendor. This
        # matters for cost/usage attribution more than the label this adapter
        # was constructed with does.
        return ProviderResult(
            parsed=parsed,
            provider=body.provider or self.name,
            model=body.model or model.value,
            tokens_in=body.usage.prompt_tokens,
            tokens_out=body.usage.completion_tokens,
            cache_read_tokens=body.usage.cache_read_tokens,
            cache_write_tokens=body.usage.cache_write_tokens,
            latency_ms=latency_ms,
            # The gateway's cost book is authoritative — this service holds no
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
            raise ProviderUnavailableError("air-model-gateway provider is closed")
        client = self._client
        if client is not None:
            return client
        async with self._client_lock:
            if self._closed:
                raise ProviderUnavailableError("air-model-gateway provider is closed")
            if self._client is None:
                token = self._config.token
                headers = {"X-API-Key": token.get_secret_value()} if token else {}
                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=httpx.Timeout(self._config.timeout_s),
                    headers=headers,
                )
            return self._client

    async def _post_generate(
        self,
        client: httpx.AsyncClient,
        request: _GenerateRequest,
        *,
        remaining_ms: float | None,
        request_id: str | None,
    ) -> _GenerateResponse:
        # Per-call headers, not client defaults: every call on this shared
        # client has its own request id and remaining budget.
        headers: dict[str, str] = {}
        if remaining_ms is not None:
            headers[_REMAINING_BUDGET_HEADER] = str(round(remaining_ms))
        if request_id:
            headers[HEADER_REQUEST_ID] = request_id
        model = request.model.value
        try:
            response = await client.post(
                _GENERATE_PATH, json=request.model_dump(mode="json"), headers=headers or None
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"air-model-gateway timed out after {self._config.timeout_s:g}s"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"air-model-gateway unreachable at {self._config.base_url}: {exc}"
            ) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            if _problem_code(response) == _BUDGET_EXCEEDED_CODE:
                raise ProviderBudgetExceededError(
                    "this service's daily model budget at air-model-gateway is spent",
                    retry_after_s=_retry_after(response),
                )
            raise ProviderRateLimitError(
                f"air-model-gateway throttled {model!r}", retry_after_s=_retry_after(response)
            )
        if response.is_client_error:
            code = _problem_code(response)
            # Error level: this is a contract or configuration bug far more
            # often than anything about the request's text, and it fails every
            # call the same way until someone fixes it. Only the status and the
            # gateway's problem code are logged, never `detail`, which can
            # quote request content back.
            logger.error(
                "provider.model_gateway.request_rejected",
                status=response.status_code,
                code=code,
                model=model,
            )
            raise ProviderRequestError(
                f"air-model-gateway rejected the request for {model!r}: "
                f"HTTP {response.status_code} ({code or 'no problem code'})",
                status=response.status_code,
                code=code,
            )
        if response.is_error:
            raise ProviderUnavailableError(
                f"air-model-gateway returned HTTP {response.status_code} for {model!r}"
            )

        try:
            raw = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError(
                "air-model-gateway returned a body that is not JSON"
            ) from exc
        if not isinstance(raw, dict):
            raise ProviderUnavailableError(
                "air-model-gateway returned a JSON body that is not an object"
            )
        try:
            return _GenerateResponse.model_validate(raw)
        except ValidationError as exc:
            raise ProviderUnavailableError(
                f"air-model-gateway returned a response that does not fit the expected shape: {exc}"
            ) from exc


def _summarise(exc: ValidationError) -> str:
    """Field paths and reasons, without the offending values (which are model output)."""
    parts = [
        f"{'.'.join(str(item) for item in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts)[:_SCHEMA_ERROR_CHARS] or "output did not match the schema"


def _problem_code(response: httpx.Response) -> str | None:
    """The gateway's RFC 9457 problem ``code``, if the body carries one.

    Never raises: an error body that isn't problem+json (a proxy's HTML page,
    an empty body) just has no code, and the status alone decides.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    code = body.get("code") if isinstance(body, dict) else None
    return code if isinstance(code, str) else None


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
