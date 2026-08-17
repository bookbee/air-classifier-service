"""The ``/v1`` surface.

This module does two jobs. It aggregates the four v1 routers under one prefix,
and it owns the concerns shared by the three analysis resources but belonging to
none of them: the request-scoped :class:`AnalysisService`, the effective batch
ceiling, the batch envelope, and the error catalogue every analysis route
advertises in its OpenAPI entry.

Those helpers live here rather than in ``sentiment.py`` because none of the three
resources is the senior one, and a limit check duplicated across three files is
how two of them end up disagreeing about it after a refactor. The resource
modules import them from here; this module imports the resource modules from
inside :func:`build_v1_router`, which is what keeps that relationship from
closing into an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Request, Response, status

from air_classifier.api.deps import AppStateDep, apply_key_policy
from air_classifier.api.errors import ValidationError
from air_classifier.api.middleware import new_request_id
from air_classifier.config import Settings
from air_classifier.pipeline.service import AnalysisService
from air_classifier.schemas.common import Options, RequestUsage
from air_classifier.schemas.errors import FieldError, ProblemDetail
from air_classifier.schemas.responses import (
    BaseAnalysisResponse,
    BatchItemResult,
    BatchResponse,
)
from air_classifier.security.api_keys import ApiKeyRecord

__all__ = [
    "ANALYSIS_ERROR_RESPONSES",
    "V1_PREFIX",
    "AnalysisServiceDep",
    "batch_response",
    "build_v1_router",
    "ensure_batch_size",
    "get_analysis_service",
    "key_clamp",
    "replay_response",
    "request_id_of",
]

#: Mounted by ``main.py``. Everything under it is versioned together.
V1_PREFIX: Final[str] = "/v1"

#: Media type of a stored idempotent response. Replays are served as bytes, so
#: nothing else re-derives it.
_JSON_MEDIA_TYPE: Final[str] = "application/json"


# ── Shared route dependencies ─────────────────────────────────────────────────


def get_analysis_service(state: AppStateDep) -> AnalysisService:
    """The ladder, wrapped for one request.

    Built per request rather than at startup because it is three bindings and no
    state of its own — and because building it here means a test can swap the
    orchestrator on ``AppState`` without something else having already captured
    the old one.
    """
    return AnalysisService(state.orchestrator, state.settings, state.cache)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


def request_id_of(request: Request) -> str:
    """The id stamped by ``RequestContextMiddleware``.

    The fallback exists for an app assembled without that middleware — a test
    harness, usually. Minting one here keeps the response's ``request_id``
    populated; it will simply not appear in any log line, which is the correct
    trade for a configuration nobody deploys.
    """
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else new_request_id()


def ensure_batch_size(count: int, settings: Settings) -> None:
    """Enforce ``app.max_batch_items``, the ceiling that actually applies.

    ``BatchRequest`` caps items at 1000 so the schema has a bound at all; this is
    the operator's number, and it is lower on every real deployment. Checked in
    the route rather than in the model because it is configuration — a deployment
    may raise or lower it without a schema change — and the caller is told the
    limit they hit rather than the one in the schema.
    """
    limit = settings.app.max_batch_items
    if count > limit:
        raise ValidationError(
            f"items: {count} items were sent; this deployment accepts at most {limit}",
            errors=[
                FieldError(
                    field="items",
                    code="too_long",
                    message=f"at most {limit} items per batch",
                )
            ],
        )


def key_clamp(key: ApiKeyRecord) -> Callable[[Options], Options]:
    """The key's policy as a plain function, for per-item batch options.

    An item may carry its own ``options``, and without this it could use them to
    climb past the ceiling its key was issued with — asking for less is the point
    of the override, asking for more is not.
    """

    def clamp(options: Options) -> Options:
        return apply_key_policy(options, key)

    return clamp


def replay_response(stored: str) -> Response:
    """Serve a stored idempotent response verbatim.

    Bytes rather than a re-rendered model: a replay that differs from the
    original — even only in field order, or in a field a later deploy renders
    differently — is not a replay.
    """
    return Response(content=stored, media_type=_JSON_MEDIA_TYPE)


def batch_response[ResponseT: BaseAnalysisResponse](
    model: type[BatchResponse[ResponseT]],
    *,
    request_id: str,
    items: Sequence[BatchItemResult[ResponseT]],
    latency_ms: float,
) -> BatchResponse[ResponseT]:
    """Wrap per-item outcomes in the batch envelope.

    ``usage`` rolls up the items that succeeded only. A failed item consumed no
    tier, and counting a slot that produced nothing would overstate what the
    caller is billed for.
    """
    ordered = list(items)
    succeeded = sum(1 for item in ordered if item.ok)
    results = [item.result for item in ordered if item.result is not None]

    return model(
        request_id=request_id,
        items=ordered,
        succeeded=succeeded,
        failed=len(ordered) - succeeded,
        usage=RequestUsage(
            tiers_run=sum(result.usage.tiers_run for result in results),
            tokens_in=sum(result.usage.tokens_in for result in results),
            tokens_out=sum(result.usage.tokens_out for result in results),
            cache_read_tokens=sum(result.usage.cache_read_tokens for result in results),
            est_cost_usd=sum(result.usage.est_cost_usd for result in results),
        ),
        latency_ms=max(0.0, round(latency_ms, 3)),
    )


# ── OpenAPI error catalogue ───────────────────────────────────────────────────

#: Every analysis route can fail in exactly these ways — docs/02-lld.md §9.
#: Declared once and attached to all six, so a generated client gets one error
#: model everywhere — and so the *absence* of a "provider unavailable" entry is
#: visible: a cloud outage is a 200 with ``degraded: true``, not an error.
ANALYSIS_ERROR_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ProblemDetail,
        "description": "Unparseable body, or a payload above the configured size cap.",
    },
    status.HTTP_401_UNAUTHORIZED: {
        "model": ProblemDetail,
        "description": "No `X-API-Key` header was supplied.",
    },
    status.HTTP_403_FORBIDDEN: {
        "model": ProblemDetail,
        "description": "The key is unknown, lacks the scope, or forbids the requested options.",
    },
    status.HTTP_409_CONFLICT: {
        "model": ProblemDetail,
        "description": "`Idempotency-Key` was reused with a different request body.",
    },
    # `HTTP_422_UNPROCESSABLE_CONTENT`, not `..._ENTITY`: starlette deprecated
    # the older spelling, and reading it here emitted a warning at import time.
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ProblemDetail,
        "description": "Schema validation failed, or the batch exceeded `max_batch_items`.",
    },
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": ProblemDetail,
        "description": "Quota exhausted for this key. Carries `Retry-After`.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ProblemDetail,
        "description": "Unhandled defect.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ProblemDetail,
        "description": "Every enabled tier failed to produce a usable verdict.",
    },
}


# ── Aggregation ───────────────────────────────────────────────────────────────


def build_v1_router() -> APIRouter:
    """The whole ``/v1`` surface, ready to mount.

    A function rather than a module-level router so the app factory controls when
    the sub-routers are imported, and so a test can build a second, independent
    instance without the first one's registrations leaking into it.
    """
    # Imported here, not at module scope: the resource modules import the shared
    # helpers above, and a module-level import in this direction would close the
    # loop. Deferring it to construction time breaks the cycle without either
    # side having to care about import order.
    from air_classifier.api.v1 import feedback, reviews, sentiment, system

    v1 = APIRouter(prefix=V1_PREFIX)
    # System first so `/v1/health` keeps resolving even if a resource router ever
    # grows a path that could shadow it.
    v1.include_router(system.router)
    v1.include_router(sentiment.router)
    v1.include_router(feedback.router)
    v1.include_router(reviews.router)
    return v1
