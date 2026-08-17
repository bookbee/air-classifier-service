"""air-infra gateway adapter.

The same :class:`~air_classifier.providers.base.LLMProvider` contract as the SDK
adapters, but the transport is a single HTTP call to the air-infra model gateway
rather than a provider SDK. air-infra owns the provider keys, routing, cost and
cache; this adapter's whole job is to speak the ladder's language on one side and
the gateway's ``POST /v1/models/chat`` on the other.

One instance exists per logical provider (``anthropic``/``openai``/``ollama``),
so the Tier-3 registry can key and fail over between them exactly as it does the
direct adapters. The provider identity is carried to the gateway as a ``model``
prefix — ``anthropic:claude-haiku-4-5`` — which is how air-infra's router picks
the upstream. Structured output rides along as the caller's JSON Schema; the
gateway returns a JSON string and this adapter validates it, so a schema
violation is still the one failure worth a single repair retry.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

import httpx
from pydantic import BaseModel, ValidationError

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

_CHAT_PATH: Final[str] = "/v1/models/chat"
_CAPABILITIES_PATH: Final[str] = "/v1/capabilities"
_HEALTH_TIMEOUT_S: Final[float] = 2.0
_SCHEMA_ERROR_CHARS: Final[int] = 500
#: finish_reason the gateway reports when the model stopped before it could close
#: the object — nothing that validates could have come back.
_LENGTH_FINISH_REASON: Final[str] = "length"


class InfraGatewayProvider:
    """:class:`LLMProvider` backed by the air-infra model gateway."""

    def __init__(self, settings: Settings, name: Provider) -> None:
        self.name = name
        self._config = settings.providers.gateway
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    @property
    def configured(self) -> bool:
        """Whether the gateway is switched on and a token is present. No I/O."""
        return self._config.configured

    async def health(self) -> bool:
        """Whether the gateway reports this provider as reachable. Never raises.

        ``/v1/capabilities`` returns a per-provider availability map, which is the
        gateway's own view of whether the upstream key is present and usable —
        the honest signal for the registry to route on.
        """
        if not self.configured:
            return False
        try:
            client = await self._ready_client()
            response = await client.get(_CAPABILITIES_PATH, timeout=_HEALTH_TIMEOUT_S)
            if response.status_code != httpx.codes.OK:
                return False
            providers = response.json().get("providers", {})
        except Exception as exc:
            logger.debug(
                "provider.infra.health_failed",
                provider=self.name.value,
                error=type(exc).__name__,
            )
            return False
        return bool(providers.get(self.name.value, False))

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
        if not self.configured:
            raise ProviderUnavailableError("air-infra gateway is not configured")

        client = await self._ready_client()
        payload: dict[str, Any] = {
            # The provider prefix is how air-infra's router picks the upstream;
            # partition on the first colon leaves a tagged model (llama3.2:3b)
            # intact.
            "model": f"{self.name.value}:{model}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "json_schema": output_model.model_json_schema(),
            "schema_name": output_model.__name__,
            "cache_prefix": cache_prefix,
        }

        started = time.perf_counter()
        async with span("provider.infra", provider=self.name.value, model=model):
            body = await self._post_chat(client, payload, model=model)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if body.get("refusal"):
            raise ProviderRefusalError(f"air-infra gateway declined {model!r}")
        if body.get("finish_reason") == _LENGTH_FINISH_REASON:
            raise ProviderSchemaError(f"gateway model {model!r} hit the output ceiling")

        content = str(body.get("content") or "")
        try:
            parsed = output_model.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderSchemaError(_summarise(exc)) from exc

        usage = body.get("usage") or {}
        return ProviderResult(
            parsed=parsed,
            provider=self.name,
            model=str(body.get("model") or model),
            tokens_in=_count(usage.get("prompt_tokens")),
            tokens_out=_count(usage.get("completion_tokens")),
            cache_read_tokens=_count(usage.get("cache_read_tokens")),
            cache_write_tokens=_count(usage.get("cache_write_tokens")),
            latency_ms=latency_ms,
            # The gateway's cost is authoritative — it reflects air-infra's own
            # rates. CostMeter falls back to the local table only when it is zero.
            cost_usd=float(body.get("cost_usd") or 0.0),
        )

    async def aclose(self) -> None:
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ── Internals ─────────────────────────────────────────────────────────

    async def _ready_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise ProviderUnavailableError("air-infra provider is closed")
        client = self._client
        if client is not None:
            return client
        async with self._client_lock:
            if self._closed:
                raise ProviderUnavailableError("air-infra provider is closed")
            if self._client is None:
                api_key = self._config.api_key
                headers = {"X-API-Key": api_key.get_secret_value()} if api_key else {}
                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=httpx.Timeout(self._config.timeout_s),
                    headers=headers,
                )
            return self._client

    async def _post_chat(
        self, client: httpx.AsyncClient, payload: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        try:
            response = await client.post(_CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"air-infra timed out after {self._config.timeout_s:g}s"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"air-infra unreachable at {self._config.base_url}: {exc}"
            ) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                f"air-infra throttled {model!r}", retry_after_s=_retry_after(response)
            )
        if response.is_error:
            raise ProviderUnavailableError(
                f"air-infra returned HTTP {response.status_code} for {model!r}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError("air-infra returned a body that is not JSON") from exc
        if not isinstance(body, dict):
            raise ProviderUnavailableError("air-infra returned a JSON body that is not an object")
        return body


def _summarise(exc: ValidationError) -> str:
    """Field paths and reasons, without the offending values (which are model output)."""
    parts = [
        f"{'.'.join(str(item) for item in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts)[:_SCHEMA_ERROR_CHARS] or "output did not match the schema"


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
