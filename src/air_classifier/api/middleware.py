"""Request-scoped middleware: identity, access logging, and a body ceiling.

All three are written as raw ASGI middleware rather than against
``BaseHTTPMiddleware``. That is not a micro-optimisation:

* ``BaseHTTPMiddleware`` runs the downstream app in a child task, which gets a
  *copy* of the contextvar context. Binding before the call still propagates
  down, but anything bound downstream — the api key id, say — is invisible on
  the way back out, and ``clear_request_context()`` in the parent would not be
  clearing what the request actually used.
* Reading the raw ``http.response.start`` message gives the access log the real
  status, including for responses this middleware never constructed.
* The body ceiling has to wrap ``receive``, which ``BaseHTTPMiddleware`` does
  not expose.

Cross-request state is the recurring hazard here, so it has one rule: anything
bound onto the logging context is cleared in a ``finally``. Worker tasks are
reused, and a missed clear does not lose a field — it attributes one caller's
request id to the next caller's logs.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Final

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from air_classifier.api.errors import MalformedRequestError, problem_response
from air_classifier.config import Settings
from air_classifier.constants import HEADER_REQUEST_ID
from air_classifier.observability import metrics
from air_classifier.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from air_classifier.observability.tracing import current_trace_id

__all__ = [
    "API_KEY_ID_ATTR",
    "AccessLogMiddleware",
    "BodySizeLimitMiddleware",
    "RequestContextMiddleware",
    "install_middleware",
    "new_request_id",
    "resolve_request_id",
]

_log = get_logger(__name__)

#: What an inbound `X-Request-ID` may contain. An id supplied by the caller ends
#: up in every log line for the request and in ticket titles downstream, so the
#: charset excludes anything that could forge a field in a log record, terminate
#: a line, or inject an escape sequence into a terminal reading it.
_REQUEST_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9._:-]{1,128}\Z")

#: Prefix on generated ids, so a grep tells an id we minted from one we accepted.
_REQUEST_ID_PREFIX: Final[str] = "req_"

#: Metric label for a request that matched no route. The raw path would be a
#: caller-controlled label value, and a 404 flood would then multiply the
#: request-count time series without bound.
_UNMATCHED_ENDPOINT: Final[str] = "unmatched"

#: Access-log label for a request that carried no usable key.
_ANONYMOUS_KEY_ID: Final[str] = "anonymous"

#: Attribute on ``request.state`` carrying the resolved API key *id*.
#:
#: Exported rather than a literal because the writer and the reader sit on
#: opposite sides of the ASGI boundary — ``api/deps.require_api_key`` stamps it,
#: this module reads it out of the raw scope — and a typo in either place would
#: not fail anything, it would silently attribute every request to `anonymous`.
API_KEY_ID_ATTR: Final[str] = "api_key_id"


def new_request_id() -> str:
    """Mint an id for a request that arrived without one."""
    return f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex}"


def resolve_request_id(supplied: str | None) -> str:
    """Accept the caller's id if it is safe, otherwise mint one.

    A rejected value is replaced rather than sanitised. Trimming a hostile id to
    its safe characters yields something that still *looks* like the caller's
    id, which is exactly the confusion an attacker wants when the two appear
    side by side in a log; a fresh id is unambiguous.
    """
    if supplied is not None and _REQUEST_ID_PATTERN.match(supplied):
        return supplied
    return new_request_id()


class RequestContextMiddleware:
    """Establish the request's identity, and take it down again afterwards."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(HEADER_REQUEST_ID))
        # `Request.state` is a view onto `scope["state"]`, so writing it here
        # makes the id readable from every handler and every other middleware,
        # including the error handlers that run outside this one.
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        context: dict[str, str] = {"request_id": request_id}
        trace_id = current_trace_id()
        if trace_id is not None:
            context["trace_id"] = trace_id
        bind_request_context(**context)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # setdefault, not set: a handler that already chose an id — the
                # error handlers do — keeps it, and the header never doubles up.
                MutableHeaders(scope=message).setdefault(HEADER_REQUEST_ID, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()


class AccessLogMiddleware:
    """One structured line and one metric observation per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        # A request whose app raised before sending anything is a 500 to the
        # client, so that is what the log and the metric must say.
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._record(scope, status=status, duration_s=time.perf_counter() - started)

    def _record(self, scope: Scope, *, status: int, duration_s: float) -> None:
        state = scope.get("state") or {}
        api_key_id = state.get(API_KEY_ID_ATTR) or _ANONYMOUS_KEY_ID
        request_id = state.get("request_id")
        path = scope.get("path", "")

        _log.info(
            "http.access",
            method=scope.get("method", ""),
            path=path,
            status=status,
            duration_ms=round(duration_s * 1000, 3),
            api_key_id=api_key_id,
            request_id=request_id,
        )
        metrics.record_request(
            endpoint=self._endpoint(scope),
            status=status,
            api_key_id=api_key_id,
            duration_s=duration_s,
        )

    @staticmethod
    def _endpoint(scope: Scope) -> str:
        """The route *template*, never the resolved path.

        `/v1/reviews/{review_id}` is one time series; the concrete paths are
        unbounded. The router writes the matched route onto the scope before the
        response is sent, so it is available by the time this runs.

        The matched route's own ``path`` is *not* usable directly. FastAPI 0.141
        resolves an included router lazily, so a route declared on a sub-router
        reports the path relative to the router that declared it — `/sentiment`,
        never `/v1/sentiment`. Labelling on that would drop the version prefix
        every dashboard filters on, and would silently merge `/v1/sentiment`
        with a future `/v2/sentiment` into one series.

        So the template is rebuilt from the request path by putting the matched
        path parameters back. If any parameter cannot be substituted the
        relative template is used instead: it is wrong in the same small way the
        old behaviour was, whereas returning a path with a caller-controlled
        segment in it would put unbounded cardinality into Prometheus.
        """
        route = scope.get("route")
        relative = getattr(route, "path", None)
        if not isinstance(relative, str):
            return _UNMATCHED_ENDPOINT

        template = scope.get("path")
        if not isinstance(template, str) or not template:
            return relative

        for name, value in (scope.get("path_params") or {}).items():
            rendered = str(value)
            if not rendered or rendered not in template:
                return relative
            template = template.replace(rendered, f"{{{name}}}", 1)
        return template


class _BodyTooLargeError(Exception):
    """Internal signal from the wrapped ``receive``.

    Deliberately not an :class:`~air_classifier.api.errors.AppError`: it must
    travel back to this middleware rather than being converted into a response
    by Starlette's exception middleware, so that the ceiling produces the same
    response wherever in the stack it is installed.
    """


class BodySizeLimitMiddleware:
    """Refuse request bodies over ``app.max_body_bytes``.

    ``Content-Length`` is checked first because it lets an oversized upload be
    refused before a single byte is read. It is not sufficient on its own: it is
    absent under chunked transfer encoding, and it is a client-supplied number
    that a client is free to understate. So the stream is counted as well, and
    the ceiling is enforced against whichever of the two trips first.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                await self._reject(scope, receive, send, "Content-Length is not an integer.")
                return
            if length > self.max_bytes:
                await self._reject(scope, receive, send, self._too_large_detail(length))
                return

        received = 0
        started = False

        async def receive_wrapper() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLargeError:
            if started:
                # Headers are already on the wire; there is no status left to
                # change. Dropping the connection is the only honest signal.
                raise
            await self._reject(scope, receive, send, self._too_large_detail(received))

    def _too_large_detail(self, size: int) -> str:
        return f"Request body of {size} bytes exceeds the {self.max_bytes} byte limit."

    async def _reject(self, scope: Scope, receive: Receive, send: Send, detail: str) -> None:
        """Emit the 400 directly.

        §9 files an oversized payload under `malformed-request`, not 413: the
        ceiling is a parsing limit rather than a quota, and the catalogue is the
        contract the client codes against.
        """
        request = Request(scope)
        error = MalformedRequestError(detail)
        problem = error.to_problem(
            instance=request.url.path,
            request_id=getattr(request.state, "request_id", None),
        )
        await problem_response(problem)(scope, receive, send)


def install_middleware(app: FastAPI, settings: Settings) -> None:
    """Install the stack in the order it has to run in.

    ``add_middleware`` prepends, so these are added innermost-first and end up
    ordered outermost → innermost as:

    1. :class:`RequestContextMiddleware` — everything below it, including the
       body-limit rejection, needs a request id to report and to echo.
    2. :class:`AccessLogMiddleware` — outside the body limit so that a refused
       upload is still one logged, counted request.
    3. :class:`BodySizeLimitMiddleware` — innermost, closest to the app that
       reads the body.
    """
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.app.max_body_bytes)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)
