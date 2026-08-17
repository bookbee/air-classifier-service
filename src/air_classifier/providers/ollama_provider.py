"""Ollama adapter for the local tier.

Ollama is reached over plain HTTP rather than through an SDK: the surface this
service needs is two endpoints, and a hand-written client keeps the failure
mapping explicit — which matters, because the whole point of Tier 2 is that its
failures are *cheap* and must escalate rather than propagate.

Two details carry most of the weight:

* ``format`` is set to the JSON Schema derived from the output model. Ollama
  turns that schema into a decoding grammar, so the model physically cannot emit
  prose around the object. Without it a 3B model wraps its JSON in an apology
  roughly as often as not, and every such reply costs a repair round trip.
* Cost is always ``0.0`` — the hardware is ours and there is no per-token bill —
  but token counts are still reported. Zeroing those too would make the usage
  roll-up claim the local tier consumed nothing at all, which hides the volume
  that sizes the box.
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Any, Final

import httpx
from pydantic import BaseModel, ValidationError

from air_classifier.config import Settings
from air_classifier.constants import Provider
from air_classifier.observability.logging import get_logger
from air_classifier.observability.tracing import span
from air_classifier.providers.base import (
    ProviderRateLimitError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)

logger = get_logger(__name__)

_CHAT_PATH: Final[str] = "/api/chat"
_TAGS_PATH: Final[str] = "/api/tags"

#: A readiness probe must answer inside a health check's own budget, so it does
#: not inherit the generation timeout — a loaded box would fail readiness by
#: being slow to *generate*, which is not what readiness asks.
_HEALTH_TIMEOUT_S: Final[float] = 2.0

#: Fixed seed alongside ``temperature: 0``. Temperature alone still leaves
#: sampling free to differ between runs on some backends, and an identical
#: request that returns two different labels makes every regression untraceable.
_SEED: Final[int] = 0

#: How much of an upstream error body is worth carrying into an exception
#: message. Enough to read "model not found", not enough to paste a stack trace.
_ERROR_EXCERPT_CHARS: Final[int] = 200

#: Validation summaries are fed back to the model as a repair instruction and
#: logged; both want the shape of the failure, not an essay.
_SCHEMA_ERROR_CHARS: Final[int] = 500


class OllamaProvider:
    """:class:`~air_classifier.providers.base.LLMProvider` over Ollama's HTTP API.

    The client is built on first use so that constructing a provider performs no
    I/O and never requires the daemon to be up at import or startup time — the
    service must boot, and report itself degraded, with Ollama stopped.
    """

    name: Provider = Provider.OLLAMA

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.providers.ollama
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    @property
    def configured(self) -> bool:
        """Whether the local backend is switched on. No I/O — see the protocol."""
        return self._config.enabled

    async def health(self) -> bool:
        """Whether the daemon answers. Never raises, by contract.

        ``/api/tags`` is the cheapest endpoint that proves the daemon is really
        serving: it needs no model loaded, and unlike a bare TCP connect it fails
        when the port is held by something that is not Ollama.
        """
        if not self.configured:
            return False
        try:
            client = await self._ready_client()
            response = await client.get(_TAGS_PATH, timeout=_HEALTH_TIMEOUT_S)
        except Exception as exc:
            logger.debug(
                "provider.ollama.health_failed",
                base_url=self._config.base_url,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False
        return response.status_code == httpx.codes.OK

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
        """One schema-constrained completion against the local daemon.

        ``cache_prefix`` is accepted and ignored: Ollama has no billable prompt
        cache to request. It keeps its own KV cache for a resident model, which
        ``keep_alive`` — not this flag — governs.
        """
        if not self.configured:
            raise ProviderUnavailableError("ollama is disabled in configuration")

        client = await self._ready_client()
        payload = self._payload(
            system_prompt=system_prompt,
            user_content=user_content,
            output_model=output_model,
            model=model,
            max_tokens=max_tokens,
        )

        started = time.perf_counter()
        async with span("provider.ollama", model=model, base_url=self._config.base_url):
            body = await self._post_chat(client, payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        content = _message_content(body)
        try:
            parsed = output_model.model_validate_json(content)
        except ValidationError as exc:
            # Carry field paths and reasons, never the offending values: those are
            # model output derived from the caller's text, and this message is
            # logged as well as replayed to the model.
            raise ProviderSchemaError(_summarise(exc)) from exc

        return ProviderResult(
            parsed=parsed,
            provider=Provider.OLLAMA,
            # The daemon echoes the tag it actually served, which may carry a
            # resolved digest the caller never spelled out.
            model=str(body.get("model") or model),
            tokens_in=_count(body.get("prompt_eval_count")),
            tokens_out=_count(body.get("eval_count")),
            latency_ms=latency_ms,
            cost_usd=0.0,
        )

    async def aclose(self) -> None:
        """Close the connection pool. Idempotent; a closed provider stays closed."""
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ── Internals ─────────────────────────────────────────────────────────

    async def _ready_client(self) -> httpx.AsyncClient:
        """The shared client, constructed on first use.

        The lock is held across construction so a burst of concurrent requests
        opens one pool rather than one each — the pool is what keeps a hot local
        model from paying a TCP handshake per classification.
        """
        if self._closed:
            raise ProviderUnavailableError("ollama provider is closed")
        client = self._client
        if client is not None:
            return client
        async with self._client_lock:
            if self._closed:
                raise ProviderUnavailableError("ollama provider is closed")
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=httpx.Timeout(self._config.timeout_s),
                )
            return self._client

    def _payload(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[BaseModel],
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # Streaming would hand back a sequence of fragments to reassemble for
            # no benefit: nothing downstream can act on a partial verdict.
            "stream": False,
            "format": _schema_for(output_model),
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": 0.0,
                "seed": _SEED,
                "num_ctx": self._config.num_ctx,
                "num_predict": max_tokens,
            },
        }

    async def _post_chat(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST one chat completion and return the decoded envelope.

        Everything that is not a well-formed Ollama envelope becomes
        ``ProviderUnavailableError`` rather than a schema error: a repair
        instruction only helps when the model was reached and replied badly.
        """
        try:
            response = await client.post(_CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"ollama timed out after {self._config.timeout_s:g}s"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                f"ollama unreachable at {self._config.base_url}: {exc}"
            ) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                "ollama throttled the request", retry_after_s=_retry_after(response)
            )
        if response.is_error:
            raise ProviderUnavailableError(
                f"ollama returned HTTP {response.status_code}: {_excerpt(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderUnavailableError("ollama returned a body that is not JSON") from exc
        if not isinstance(body, dict):
            raise ProviderUnavailableError("ollama returned a JSON body that is not an object")
        return body


# ── Response handling ─────────────────────────────────────────────────────────


def _message_content(body: dict[str, Any]) -> str:
    """The assistant turn's text, or the reason it cannot be used."""
    message = body.get("message")
    if not isinstance(message, dict):
        raise ProviderUnavailableError("ollama envelope carried no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        # Reached the model and got nothing usable back, which is precisely the
        # case one repair attempt exists for.
        raise ProviderSchemaError("ollama returned an empty assistant message")
    return content


def _summarise(exc: ValidationError) -> str:
    """Field paths and reasons from a validation failure, without the values."""
    parts = [
        f"{'.'.join(str(item) for item in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts)[:_SCHEMA_ERROR_CHARS] or "output did not match the schema"


def _count(value: Any) -> int:
    """A token count from the envelope, defaulting to zero.

    Ollama omits the counters on some paths (a cached or empty generation), and a
    missing counter must not fail a request that otherwise succeeded.
    """
    return value if isinstance(value, int) and value >= 0 else 0


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        # The header also permits an HTTP-date. Discarding it loses a hint;
        # misreading it as seconds would invent a delay of decades.
        return None


def _excerpt(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:_ERROR_EXCERPT_CHARS]
    reported = body.get("error") if isinstance(body, dict) else None
    if isinstance(reported, str):
        return reported[:_ERROR_EXCERPT_CHARS]
    return response.text[:_ERROR_EXCERPT_CHARS]


@lru_cache(maxsize=8)
def _schema_for(output_model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for an output model, derived once per class.

    Regenerating it per request is pure waste on the hot path: the schema is a
    property of the class, and the class is fixed at import.
    """
    return output_model.model_json_schema()
