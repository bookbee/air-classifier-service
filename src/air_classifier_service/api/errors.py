"""The error model — docs/02-lld.md §9.

Two rules hold everywhere in this module, and everything else is plumbing:

* **Every** failure leaves the service as an RFC 9457 ``application/problem+json``
  body. There is no second error shape, no bare ``{"detail": ...}`` from
  FastAPI's defaults, and no HTML from Starlette's. A caller writes one parser.
* **Nothing internal escapes.** The catch-all logs the traceback and returns a
  fixed 500 body. An exception message routinely carries a connection string, a
  file path, or a fragment of user text, and an error response is the least
  guarded surface the service has.

Note what the catalogue does *not* contain: there is no "provider unavailable".
A cloud outage is a lower rung's verdict with ``degraded=true`` and a 200. The
only path from the ladder to a 5xx is :class:`NoVerdictError` — every enabled
rung failed — which lands on 503 ``no-verdict``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any, ClassVar, Final

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from air_classifier_service.constants import HEADER_REQUEST_ID
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.pipeline.orchestrator import NoVerdictError
from air_classifier_service.schemas.errors import FieldError, ProblemDetail

__all__ = [
    "AppError",
    "IdempotencyConflictError",
    "InsufficientScopeError",
    "InternalError",
    "MalformedRequestError",
    "MissingApiKeyError",
    "NoVerdictAvailableError",
    "NoVerdictWithinLimitsError",
    "NotFoundError",
    "ProblemResponse",
    "RateLimitedError",
    "ValidationError",
    "install_error_handlers",
    "problem_response",
]

_log = get_logger(__name__)

#: RFC 9457 media type. Distinct from ``application/json`` so a client can route
#: on the content type alone rather than on the status code.
PROBLEM_CONTENT_TYPE: Final[str] = "application/problem+json"

#: Loc prefixes pydantic prepends to a field path. ``body`` is stripped so the
#: reported field matches what the caller wrote in their JSON; the others are
#: kept, because ``limit`` alone would not say whether it was a query parameter
#: or a header that was rejected.
_STRIPPED_LOC_PREFIXES: Final[frozenset[str]] = frozenset({"body"})


# ── Exception hierarchy ───────────────────────────────────────────────────────


class AppError(Exception):
    """Base for every failure this service reports deliberately.

    Subclasses fix ``slug``/``title``/``status`` as class attributes rather than
    constructor arguments: the catalogue in §9 is a contract, and a per-raise
    status is how two call sites end up disagreeing about what a 403 means.
    """

    slug: ClassVar[str] = "internal-error"
    title: ClassVar[str] = "Internal server error"
    status: ClassVar[int] = 500
    #: Whether the same request, unmodified, might succeed if retried. False by
    #: default — most of the catalogue is a client error or a defect, and
    #: neither is fixed by trying again. Only the handful of genuinely
    #: transient-dependency failures below override it.
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        detail: str | None = None,
        *,
        errors: Sequence[FieldError] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail or self.title)
        self.detail = detail
        self.errors: list[FieldError] = list(errors or ())
        self.headers: dict[str, str] = dict(headers or {})

    def to_problem(
        self, *, instance: str | None = None, request_id: str | None = None
    ) -> ProblemDetail:
        """Render as the body the caller receives."""
        return ProblemDetail.build(
            slug=self.slug,
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=instance,
            request_id=request_id,
            errors=self.errors,
            retryable=self.retryable,
        )


class MalformedRequestError(AppError):
    """Unparseable body, or a payload larger than the configured ceiling."""

    slug = "malformed-request"
    title = "Malformed request"
    status = 400


class MissingApiKeyError(AppError):
    """No ``X-API-Key`` header, or one that matches no known key."""

    slug = "missing-api-key"
    title = "Missing or invalid API key"
    status = 401


class InsufficientScopeError(AppError):
    """The key lacks the scope, or asked for a tier above its ceiling."""

    slug = "insufficient-scope"
    title = "Insufficient scope"
    status = 403


class NotFoundError(AppError):
    """Unknown route or unknown resource."""

    slug = "not-found"
    title = "Not found"
    status = 404


class IdempotencyConflictError(AppError):
    """The same ``Idempotency-Key`` arrived with a different body."""

    slug = "idempotency-conflict"
    title = "Idempotency key conflict"
    status = 409


class ValidationError(AppError):
    """A schema violation. Shadows pydantic's name inside this module only.

    Carries the per-field breakdown in ``errors`` so a client can highlight the
    offending inputs without parsing prose out of ``detail``.
    """

    slug = "validation-error"
    title = "Request validation failed"
    status = 422


class RateLimitedError(AppError):
    """Quota exhausted for this key.

    ``retry_after`` is seconds and is floored at 1 in the header: RFC 9110 wants
    an integer, and a sub-second remainder rounded down to 0 invites a client to
    retry immediately and get denied again.
    """

    slug = "rate-limited"
    title = "Rate limit exceeded"
    status = 429
    retryable = True

    def __init__(
        self,
        detail: str | None = None,
        *,
        retry_after: float = 1.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        # Caller-supplied headers win: `RateLimitResult.headers()` already
        # carries a Retry-After computed from the same window this error
        # describes, alongside the RateLimit-* family.
        merged = {"Retry-After": str(max(1, math.ceil(retry_after)))}
        merged.update(headers or {})
        super().__init__(detail, headers=merged)
        self.retry_after = retry_after


class InternalError(AppError):
    """An unhandled defect. Should be alert-worthy and rare."""

    slug = "internal-error"
    title = "Internal server error"
    status = 500


class NoVerdictAvailableError(AppError):
    """Every enabled rung failed or abstained. The only 5xx from the ladder."""

    slug = "no-verdict"
    title = "No verdict available"
    status = 503
    # By construction (see the module docstring) this is never a defect in this
    # service — every enabled tier failed or abstained, almost always because a
    # dependency (air-model-gateway, a cloud provider) was down or slow. That can clear on
    # its own, which is exactly what "retryable" is for.
    retryable = True


class NoVerdictWithinLimitsError(AppError):
    """The caller's own options stopped the ladder before any rung answered.

    Same root as :class:`NoVerdictAvailableError` — no usable verdict — but
    nothing failed: `max_tier`, `min_tier`, `max_cost_usd` or
    `latency_budget_ms` ruled out every rung that could have answered. The
    identical request fails identically, so it is a 422 the caller fixes by
    relaxing the named limit, never a retryable 5xx (SIT CL-003). Not in
    ``_BY_STATUS``: 422 already maps to ``validation-error`` there.
    """

    slug = "no-verdict-within-limits"
    title = "No verdict within the request's limits"
    status = 422


#: Status → catalogue entry, used to give Starlette's own ``HTTPException``s
#: (404 from the router, 405 from a method mismatch) the same body shape as ours.
_BY_STATUS: Final[dict[int, type[AppError]]] = {
    cls.status: cls
    for cls in (
        MalformedRequestError,
        MissingApiKeyError,
        InsufficientScopeError,
        NotFoundError,
        IdempotencyConflictError,
        ValidationError,
        RateLimitedError,
        InternalError,
        NoVerdictAvailableError,
    )
}


# ── Response construction ─────────────────────────────────────────────────────


class ProblemResponse(JSONResponse):
    """A JSON response typed as ``application/problem+json``."""

    media_type = PROBLEM_CONTENT_TYPE


def problem_response(
    problem: ProblemDetail, *, headers: Mapping[str, str] | None = None
) -> ProblemResponse:
    """Wrap a problem in its response.

    ``exclude_none`` keeps optional RFC 9457 members off the wire rather than
    sending them as nulls — a member that is absent and one that is present but
    null mean the same thing to the spec, and the former is smaller.

    The request id is echoed as a header as well as carried in the body so that
    a proxy, a log scraper, or a client that only reads headers on failures can
    still correlate the response with a trace.
    """
    merged = dict(headers or {})
    if problem.request_id:
        merged.setdefault(HEADER_REQUEST_ID, problem.request_id)
    return ProblemResponse(
        status_code=problem.status,
        content=problem.model_dump(mode="json", exclude_none=True),
        headers=merged,
    )


def _instance(request: Request) -> str:
    """The occurrence URI: the path, without the query string.

    Query strings carry caller-supplied values, and an error body is copied into
    tickets and log aggregators far more often than a success body is.
    """
    return request.url.path


def _request_id(request: Request) -> str | None:
    """The id stamped by ``RequestContextMiddleware``, if it is installed.

    This module deliberately does not mint one of its own: a second generator
    would let a response carry an id that appears in no log line.
    """
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def _slug_for_status(status: int) -> tuple[str, str, bool]:
    """Catalogue entry for a status, or one derived from the HTTP phrase.

    The §9 table does not cover every status Starlette can raise (405, 415, …).
    Deriving the slug from the reason phrase keeps those responses in the same
    shape and gives them a stable, guessable ``type`` URI instead of collapsing
    them onto a status the caller did not actually hit.

    The third element is ``retryable``. A catalogued status uses its class's own
    verdict; an uncatalogued one falls back to a status-family heuristic — 5xx
    defaults to retryable (a server-side problem, plausibly transient), 4xx does
    not (a client error, unchanged by trying again) — 501 is the one deliberate
    exception, since "not implemented" cannot become true by retrying.
    """
    known = _BY_STATUS.get(status)
    if known is not None:
        return known.slug, known.title, known.retryable
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "HTTP error"
    # Slugs land in a URI, so collapse everything outside [a-z0-9] rather than
    # only spaces — "I'm a Teapot" must not put an apostrophe in a `type`.
    slug = re.sub(r"[^a-z0-9]+", "-", phrase.lower()).strip("-")
    retryable = status >= 500 and status != HTTPStatus.NOT_IMPLEMENTED
    return slug, phrase, retryable


# ── Handlers ──────────────────────────────────────────────────────────────────


async def _app_error_handler(request: Request, exc: Exception) -> Response:
    """Anything raised deliberately by our own code."""
    assert isinstance(exc, AppError)  # registered for exactly this type
    problem = exc.to_problem(instance=_instance(request), request_id=_request_id(request))
    return problem_response(problem, headers=exc.headers)


async def _validation_error_handler(request: Request, exc: Exception) -> Response:
    """Pydantic's per-field failures, flattened into ``errors``.

    FastAPI parses and validates the body *before* it runs a route's
    dependencies, so without the authentication step below an anonymous caller
    gets a 422 describing the body — learning that the endpoint parses it, and
    its schema — instead of the 401 every other unauthenticated request gets
    (SIT CL-006b). Every route that takes a body or parameters is authenticated,
    so authenticating here changes the answer only for callers who would have
    been refused anyway.
    """
    assert isinstance(exc, RequestValidationError)
    unauthenticated = await _authentication_failure(request)
    if unauthenticated is not None:
        return unauthenticated
    raw_errors = exc.errors()
    if any(raw.get("type") == "json_invalid" for raw in raw_errors):
        # Unparseable is the catalogue's 400, and "1: JSON decode error" (the
        # character offset, then pydantic's message) tells the caller nothing.
        return await _app_error_handler(
            request, MalformedRequestError("The request body is not valid JSON.")
        )
    field_errors = [_as_field_error(raw) for raw in raw_errors]
    first = field_errors[0] if field_errors else None
    detail = f"{first.field}: {first.message}" if first and first.message else None
    return await _app_error_handler(request, ValidationError(detail, errors=field_errors))


async def _authentication_failure(request: Request) -> Response | None:
    """The 401 this request would have got had its dependencies run, if any.

    `None` when the caller authenticates, or when the app has no state to
    authenticate against (a bare app built by hand in tests). Imported lazily:
    ``api.deps`` imports this module.
    """
    from air_classifier_service.api.deps import STATE_ATTR, require_api_key

    if getattr(request.app.state, STATE_ATTR, None) is None:
        return None
    try:
        await require_api_key(request)
    except AppError as auth_error:
        return await _app_error_handler(request, auth_error)
    return None


async def _no_verdict_handler(request: Request, exc: Exception) -> Response:
    """The ladder's only route to a 5xx — or, when the caller's own limits
    caused it, a non-retryable 422 that names the limit."""
    assert isinstance(exc, NoVerdictError)
    if not exc.retryable:
        _log.info("request.no_verdict_within_limits", path=_instance(request), limit=exc.limit)
        return await _app_error_handler(request, NoVerdictWithinLimitsError(exc.limit_detail()))
    _log.error("request.no_verdict", path=_instance(request))
    return await _app_error_handler(
        request, NoVerdictAvailableError("No tier produced a usable verdict.")
    )


async def _http_exception_handler(request: Request, exc: Exception) -> Response:
    """Starlette's and FastAPI's ``HTTPException``, re-shaped as a problem.

    Headers are preserved because some are load-bearing on the status they
    accompany — ``Allow`` on a 405, ``WWW-Authenticate`` on a 401.
    """
    assert isinstance(exc, StarletteHTTPException)
    slug, title, retryable = _slug_for_status(exc.status_code)
    problem = ProblemDetail.build(
        slug=slug,
        title=title,
        status=exc.status_code,
        detail=str(exc.detail) if exc.detail else None,
        instance=_instance(request),
        request_id=_request_id(request),
        retryable=retryable,
    )
    return problem_response(problem, headers=exc.headers or {})


async def _unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """The last line of defence.

    The traceback goes to the log, where it is useful and access-controlled. The
    caller gets a fixed sentence and the request id — enough to open a ticket,
    not enough to learn anything about the inside of the process.
    """
    _log.error(
        "request.unhandled_exception",
        exc_info=exc,
        method=request.method,
        path=_instance(request),
        error_type=type(exc).__name__,
    )
    return await _app_error_handler(
        request, InternalError("An unexpected error occurred. Quote the request id when reporting.")
    )


def _as_field_error(raw: Mapping[str, Any]) -> FieldError:
    """One pydantic error dict → one :class:`FieldError`.

    Only ``loc``/``msg``/``type`` are read. ``ctx`` and ``input`` are ignored on
    purpose: ``input`` is the caller's rejected value, and echoing it into an
    error body is how a redaction pipeline gets bypassed by its own 422.
    """
    loc: tuple[Any, ...] = tuple(raw.get("loc", ()))
    if loc and loc[0] in _STRIPPED_LOC_PREFIXES:
        loc = loc[1:]
    return FieldError(
        field=".".join(str(part) for part in loc) or "__root__",
        code=str(raw.get("type", "invalid")),
        message=str(raw["msg"]) if raw.get("msg") else None,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register every handler on ``app``.

    Registration order does not matter — Starlette dispatches on the most
    specific registered class — but the set does: without the ``Exception``
    entry the catch-all is uvicorn's, which returns a bare
    ``Internal Server Error`` in ``text/plain``.
    """
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(NoVerdictError, _no_verdict_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
