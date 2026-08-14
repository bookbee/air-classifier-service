"""Unit tests for the request middleware stack."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import structlog
from fastapi import APIRouter, FastAPI, Request
from prometheus_client import CollectorRegistry
from structlog.testing import capture_logs

from air_sentiment.api.errors import (
    PROBLEM_CONTENT_TYPE,
    MissingApiKeyError,
    install_error_handlers,
)
from air_sentiment.api.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    install_middleware,
    new_request_id,
    resolve_request_id,
)
from air_sentiment.config import AppSettings, Settings
from air_sentiment.constants import ERROR_TYPE_BASE, HEADER_REQUEST_ID
from air_sentiment.observability import metrics

#: Small enough that a test payload can exceed it without shipping a megabyte,
#: but above the 1024 floor `AppSettings` enforces.
SMALL_BODY_LIMIT = 1024


@pytest.fixture
def app_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"app": AppSettings(env="test", max_body_bytes=SMALL_BODY_LIMIT)}
    )


def build_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    install_middleware(app, settings)

    @app.get("/ping")
    async def ping(request: Request) -> dict[str, Any]:
        # Echoes what downstream code can actually see, which is the point of
        # the middleware: identity on `state`, identity on the log context.
        return {
            "state_request_id": getattr(request.state, "request_id", None),
            "bound": dict(structlog.contextvars.get_contextvars()),
        }

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.post("/ignores-body")
    async def ignores_body() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/keyed")
    async def keyed(request: Request) -> dict[str, bool]:
        # Stands in for the auth dependency, which stamps the key id on state.
        request.state.api_key_id = "key_abc"
        return {"ok": True}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    @app.get("/denied")
    async def denied() -> None:
        raise MissingApiKeyError()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    return app


@pytest.fixture
async def client(app_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app(app_settings), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture(autouse=True)
def _isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test its own counters.

    The metric objects are module-level singletons, so without this the request
    counts assert against whatever the rest of the suite already recorded.
    """
    registry = CollectorRegistry()
    requests_total = metrics.Counter(
        "air_sentiment_requests_total",
        "test copy",
        ("endpoint", "status", "api_key_id"),
        registry=registry,
    )
    duration = metrics.Histogram(
        "air_sentiment_request_duration_seconds",
        "test copy",
        ("endpoint",),
        registry=registry,
    )
    monkeypatch.setattr(metrics, "REQUESTS_TOTAL", requests_total)
    monkeypatch.setattr(metrics, "REQUEST_DURATION_SECONDS", duration)
    yield


def request_count(endpoint: str, status: int, api_key_id: str) -> float:
    value = metrics.REQUESTS_TOTAL.labels(
        endpoint=endpoint, status=str(status), api_key_id=api_key_id
    )._value.get()
    return float(value)


# ── Request id resolution ─────────────────────────────────────────────────────


def test_new_request_id_is_prefixed_and_unique() -> None:
    first, second = new_request_id(), new_request_id()
    assert first.startswith("req_")
    assert first != second


@pytest.mark.parametrize(
    "supplied",
    ["abc123", "req_01J8XY", "trace.id:1-2-3", "a" * 128],
)
def test_safe_request_ids_are_accepted(supplied: str) -> None:
    assert resolve_request_id(supplied) == supplied


@pytest.mark.parametrize(
    "hostile",
    [
        "",
        "a" * 129,
        'evil" injected="1',
        "line\nbreak",
        "carriage\rreturn",
        "\x1b[31mred",
        "null\x00byte",
        "spaces are out",
        "unicode\u2013dash",
        "<script>alert(1)</script>",
        '{"json":"forgery"}',
    ],
)
def test_hostile_request_ids_are_replaced_not_sanitised(hostile: str) -> None:
    """A rejected id is replaced outright, never trimmed into a lookalike."""
    resolved = resolve_request_id(hostile)
    assert resolved != hostile
    assert resolved.startswith("req_")


# ── RequestContextMiddleware ──────────────────────────────────────────────────


async def test_request_id_is_generated_when_absent(client: httpx.AsyncClient) -> None:
    response = await client.get("/ping")

    request_id = response.headers[HEADER_REQUEST_ID]
    assert request_id.startswith("req_")
    body = response.json()
    assert body["state_request_id"] == request_id
    assert body["bound"]["request_id"] == request_id


async def test_request_id_is_echoed_when_supplied(client: httpx.AsyncClient) -> None:
    response = await client.get("/ping", headers={HEADER_REQUEST_ID: "caller-supplied-42"})

    assert response.headers[HEADER_REQUEST_ID] == "caller-supplied-42"
    assert response.json()["state_request_id"] == "caller-supplied-42"


async def test_hostile_request_id_never_reaches_state_or_logs(client: httpx.AsyncClient) -> None:
    """The header is caller-controlled; nothing downstream may trust it."""
    hostile = 'evil"\n injected="1'
    with capture_logs() as entries:
        response = await client.get("/ping", headers={HEADER_REQUEST_ID: hostile})

    assert response.headers[HEADER_REQUEST_ID].startswith("req_")
    assert response.json()["state_request_id"] != hostile
    assert all(entry.get("request_id") != hostile for entry in entries)


async def test_context_is_cleared_after_each_request(client: httpx.AsyncClient) -> None:
    """Contextvars outlive the task, so a leak would mislabel the next caller."""
    await client.get("/ping", headers={HEADER_REQUEST_ID: "first-request"})

    assert structlog.contextvars.get_contextvars() == {}


async def test_context_is_cleared_even_when_the_app_raises(client: httpx.AsyncClient) -> None:
    response = await client.get("/boom")

    assert response.status_code == 500
    assert structlog.contextvars.get_contextvars() == {}


async def test_request_ids_do_not_bleed_between_requests(client: httpx.AsyncClient) -> None:
    first = await client.get("/ping", headers={HEADER_REQUEST_ID: "first-request"})
    second = await client.get("/ping")

    assert first.json()["bound"]["request_id"] == "first-request"
    assert second.json()["bound"]["request_id"] != "first-request"


async def test_request_id_is_echoed_on_error_responses(client: httpx.AsyncClient) -> None:
    for path in ("/denied", "/boom", "/nope"):
        response = await client.get(path, headers={HEADER_REQUEST_ID: "err-trace-1"})

        assert response.headers[HEADER_REQUEST_ID] == "err-trace-1"
        assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
        assert response.json()["request_id"] == "err-trace-1"


# ── AccessLogMiddleware ───────────────────────────────────────────────────────


async def test_access_log_line_has_the_required_fields(client: httpx.AsyncClient) -> None:
    with capture_logs() as entries:
        await client.get("/keyed", headers={HEADER_REQUEST_ID: "log-me"})

    access = [entry for entry in entries if entry.get("event") == "http.access"]
    assert len(access) == 1
    line = access[0]
    assert line["method"] == "GET"
    assert line["path"] == "/keyed"
    assert line["status"] == 200
    assert line["api_key_id"] == "key_abc"
    assert line["request_id"] == "log-me"
    assert line["duration_ms"] >= 0.0


async def test_access_log_never_contains_the_api_key(client: httpx.AsyncClient) -> None:
    with capture_logs() as entries:
        await client.get("/keyed", headers={"X-API-Key": "sk_live_supersecret"})

    assert "sk_live_supersecret" not in json.dumps(entries, default=str)


async def test_unauthenticated_requests_log_as_anonymous(client: httpx.AsyncClient) -> None:
    with capture_logs() as entries:
        await client.get("/ping")

    access = next(entry for entry in entries if entry.get("event") == "http.access")
    assert access["api_key_id"] == "anonymous"


async def test_metrics_use_the_route_template_not_the_path(client: httpx.AsyncClient) -> None:
    """`/items/{item_id}` is one series; the concrete ids are unbounded."""
    await client.get("/items/abc")
    await client.get("/items/def")

    assert request_count("/items/{item_id}", 200, "anonymous") == 2.0


async def test_metrics_label_keeps_the_router_prefix(app_settings: Settings) -> None:
    """A route on an included router is labelled with its mounted path.

    FastAPI 0.141 resolves included routers lazily, so the matched route reports
    the path relative to the router that declared it. Labelling on that would
    publish `/sentiment` for `/v1/sentiment` — dropping the version prefix every
    dashboard filters on, and merging a future `/v2` into the same series.
    """
    inner = APIRouter()

    @inner.get("/sentiment")
    async def sentiment() -> dict[str, bool]:
        return {"ok": True}

    @inner.get("/reviews/{review_id}")
    async def review(review_id: str) -> dict[str, str]:
        return {"review_id": review_id}

    v1 = APIRouter(prefix="/v1")
    v1.include_router(inner)

    app = FastAPI()
    install_middleware(app, app_settings)
    app.include_router(v1)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        await http.get("/v1/sentiment")
        await http.get("/v1/reviews/abc")
        await http.get("/v1/reviews/def")

    assert request_count("/v1/sentiment", 200, "anonymous") == 1.0
    # Prefix kept *and* the path parameter still collapsed to its template.
    assert request_count("/v1/reviews/{review_id}", 200, "anonymous") == 2.0


async def test_unmatched_routes_do_not_grow_metric_cardinality(
    client: httpx.AsyncClient,
) -> None:
    await client.get("/no-such-route")
    await client.get("/another-missing-route")

    assert request_count("unmatched", 404, "anonymous") == 2.0


async def test_metrics_record_the_key_and_the_status(client: httpx.AsyncClient) -> None:
    await client.get("/keyed")
    await client.get("/denied")

    assert request_count("/keyed", 200, "key_abc") == 1.0
    assert request_count("/denied", 401, "anonymous") == 1.0


async def test_a_crashed_request_is_still_logged_and_counted(client: httpx.AsyncClient) -> None:
    """The client saw a 500, so the log and the metric must say 500 too."""
    with capture_logs() as entries:
        response = await client.get("/boom")

    assert response.status_code == 500
    access = next(entry for entry in entries if entry.get("event") == "http.access")
    assert access["status"] == 500
    assert request_count("/boom", 500, "anonymous") == 1.0


# ── BodySizeLimitMiddleware ───────────────────────────────────────────────────


async def test_body_within_the_limit_is_accepted(client: httpx.AsyncClient) -> None:
    response = await client.post("/echo", content=b"x" * SMALL_BODY_LIMIT)

    assert response.status_code == 200
    assert response.json()["size"] == SMALL_BODY_LIMIT


async def test_oversized_body_is_refused_on_content_length(client: httpx.AsyncClient) -> None:
    response = await client.post("/echo", content=b"x" * (SMALL_BODY_LIMIT + 1))

    assert response.status_code == 400
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/malformed-request"
    assert body["instance"] == "/echo"
    assert body["request_id"] == response.headers[HEADER_REQUEST_ID]


async def test_a_lying_content_length_is_caught_while_streaming(
    app_settings: Settings,
) -> None:
    """The header is a client-supplied number; the stream is the truth."""

    async def oversized() -> AsyncIterator[bytes]:
        for _ in range(4):
            yield b"x" * SMALL_BODY_LIMIT

    transport = httpx.ASGITransport(app=build_app(app_settings), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        # No Content-Length: httpx sends a chunked body for an async iterator.
        response = await http.post("/echo", content=oversized())

    assert response.status_code == 400
    assert response.json()["type"] == f"{ERROR_TYPE_BASE}/malformed-request"


async def test_a_non_integer_content_length_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/echo", content=b"hello", headers={"content-length": "not-a-number"}
    )

    assert response.status_code == 400
    assert response.json()["type"] == f"{ERROR_TYPE_BASE}/malformed-request"


async def test_a_refused_body_is_still_logged_and_counted(client: httpx.AsyncClient) -> None:
    """The access log sits outside the ceiling, so a refusal is observable."""
    with capture_logs() as entries:
        await client.post("/echo", content=b"x" * (SMALL_BODY_LIMIT * 2))

    access = next(entry for entry in entries if entry.get("event") == "http.access")
    assert access["status"] == 400
    assert access["path"] == "/echo"


async def test_a_route_that_ignores_the_body_still_honours_content_length(
    client: httpx.AsyncClient,
) -> None:
    """The cheap check runs before the app, so it does not depend on a read."""
    response = await client.post("/ignores-body", content=b"x" * (SMALL_BODY_LIMIT + 1))

    assert response.status_code == 400


# ── Ordering ──────────────────────────────────────────────────────────────────


def test_install_middleware_orders_the_stack(app_settings: Settings) -> None:
    """Context outermost, then the access log, then the body ceiling."""
    app = build_app(app_settings)

    # `user_middleware` is stored outermost-first.
    assert [entry.cls for entry in app.user_middleware] == [
        RequestContextMiddleware,
        AccessLogMiddleware,
        BodySizeLimitMiddleware,
    ]
