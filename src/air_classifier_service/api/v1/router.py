"""The ``/v1`` surface.

Aggregates the ``/v1`` sub-routers under one prefix, and owns the concerns
that belong to the HTTP layer generally rather than to ``classify.py``
specifically: the request-scoped :class:`AnalysisService`, the batch size
ceiling, idempotency replay, and the error catalogue the analysis route
advertises in its OpenAPI entry. Kept separate from ``classify.py`` so that
file stays a route handler and this one stays generic infrastructure — a
distinction worth keeping even with one route today, because the alternative
is route logic and request-plumbing tangled in the same file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, Request, Response, status

from air_classifier_service.api.deps import AppStateDep, apply_key_policy
from air_classifier_service.api.errors import ValidationError
from air_classifier_service.api.middleware import new_request_id
from air_classifier_service.config import Settings
from air_classifier_service.pipeline.service import AnalysisService
from air_classifier_service.pipeline.summary_service import SummaryService
from air_classifier_service.schemas.common import Options
from air_classifier_service.schemas.errors import FieldError, ProblemDetail
from air_classifier_service.security.api_keys import ApiKeyRecord

__all__ = [
    "ANALYSIS_ERROR_RESPONSES",
    "V1_PREFIX",
    "AnalysisServiceDep",
    "SummaryServiceDep",
    "build_v1_router",
    "ensure_batch_size",
    "get_analysis_service",
    "get_summary_service",
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
    return AnalysisService(state.orchestrator, state.settings, state.cache, state.audit)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


def get_summary_service(state: AppStateDep) -> SummaryService:
    """The customer-summary refresh service, built once at startup.

    Unlike :func:`get_analysis_service` this is not rebuilt per request: it
    owns its own bulkhead/breaker (`pipeline/summary_service.py`), which must
    be shared across calls to mean anything. `main.py`'s lifespan always
    constructs one — the narrative-refresh call inside it degrades on its own
    when air-model-gateway is not configured, so there is no "not configured" 503 here.
    """
    assert state.summary_service is not None, (
        "AppState.summary_service is None; the application lifespan did not run"
    )
    return state.summary_service


SummaryServiceDep = Annotated[SummaryService, Depends(get_summary_service)]


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

    The schema caps items at 1000 so it has a bound at all; this is the
    operator's number, and it is lower on every real deployment. Checked in
    the route rather than in the model because it is configuration — a
    deployment may raise or lower it without a schema change — and the caller
    is told the limit they hit rather than the one in the schema.
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


# ── OpenAPI error catalogue ───────────────────────────────────────────────────

#: Every way the analysis route can fail — docs/02-lld.md §9. Declared once and
#: attached to both its methods, so the *absence* of a "provider unavailable"
#: entry is visible: a cloud outage is a 200 with ``degraded: true``, not an
#: error.
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
    # Imported here, not at module scope: classify.py imports the shared
    # helpers above, and a module-level import in this direction would close
    # the loop. Deferring it to construction time breaks the cycle without
    # either side having to care about import order.
    from air_classifier_service.api.v1 import classify, summary, system

    v1 = APIRouter(prefix=V1_PREFIX)
    # System first so `/v1/health` keeps resolving even if the analysis router
    # ever grows a path that could shadow it.
    v1.include_router(system.router)
    v1.include_router(classify.router)
    v1.include_router(summary.router)
    return v1
