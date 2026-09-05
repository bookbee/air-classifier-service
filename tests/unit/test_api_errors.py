"""Unit tests for the RFC 9457 error model."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from structlog.testing import capture_logs

from air_classifier_service.api.errors import (
    PROBLEM_CONTENT_TYPE,
    AppError,
    IdempotencyConflictError,
    InsufficientScopeError,
    InternalError,
    MalformedRequestError,
    MissingApiKeyError,
    NotFoundError,
    NoVerdictAvailableError,
    RateLimitedError,
    ValidationError,
    install_error_handlers,
)
from air_classifier_service.api.middleware import install_middleware
from air_classifier_service.config import Settings
from air_classifier_service.constants import ERROR_TYPE_BASE, HEADER_REQUEST_ID
from air_classifier_service.pipeline.orchestrator import NoVerdictError


class Payload(BaseModel):
    text: str = Field(min_length=1)
    score: int = Field(ge=0)


def build_app(settings: Settings) -> FastAPI:
    """A minimal app carrying one route per failure mode under test."""
    app = FastAPI()
    install_error_handlers(app)
    install_middleware(app, settings)

    @app.post("/echo")
    async def echo(payload: Payload) -> dict[str, str]:
        return {"text": payload.text}

    @app.get("/raise/{slug}")
    async def raise_app_error(slug: str) -> None:
        errors: dict[str, AppError] = {
            "malformed": MalformedRequestError("bad body"),
            "missing-key": MissingApiKeyError(),
            "scope": InsufficientScopeError("needs sentiment:write"),
            "not-found": NotFoundError(),
            "conflict": IdempotencyConflictError(),
            "validation": ValidationError("nope"),
            "rate-limited": RateLimitedError("slow down", retry_after=2.2),
            "internal": InternalError(),
            "no-verdict": NoVerdictAvailableError(),
        }
        raise errors[slug]

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("connection string postgres://user:hunter2@db/prod")

    @app.get("/no-verdict")
    async def no_verdict() -> None:
        raise NoVerdictError("every rung failed")

    @app.get("/http-exception")
    async def http_exception() -> None:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    @app.get("/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=418)

    @app.get("/uncatalogued-5xx")
    async def uncatalogued_5xx() -> None:
        raise HTTPException(status_code=502, detail="upstream said no")

    @app.get("/not-implemented")
    async def not_implemented() -> None:
        raise HTTPException(status_code=501)

    return app


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """Exercises the real ASGI stack — middleware, handlers, and all."""
    transport = httpx.ASGITransport(app=build_app(settings), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


# ── The hierarchy itself ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "slug", "status", "retryable"),
    [
        (MalformedRequestError(), "malformed-request", 400, False),
        (MissingApiKeyError(), "missing-api-key", 401, False),
        (InsufficientScopeError(), "insufficient-scope", 403, False),
        (NotFoundError(), "not-found", 404, False),
        (IdempotencyConflictError(), "idempotency-conflict", 409, False),
        (ValidationError(), "validation-error", 422, False),
        (RateLimitedError(), "rate-limited", 429, True),
        (InternalError(), "internal-error", 500, False),
        (NoVerdictAvailableError(), "no-verdict", 503, True),
    ],
)
def test_catalogue_matches_the_lld(
    error: AppError, slug: str, status: int, retryable: bool
) -> None:
    """Every entry in docs/02-lld.md §9, with its exact slug, status and
    retryable verdict — a caller must be able to trust this without a
    status-code lookup table of its own."""
    assert error.slug == slug
    assert error.status == status
    assert error.retryable is retryable
    problem = error.to_problem(instance="/v1/classify", request_id="req_1")
    assert problem.type == f"{ERROR_TYPE_BASE}/{slug}"
    assert problem.status == status
    assert problem.title == error.title
    assert problem.instance == "/v1/classify"
    assert problem.request_id == "req_1"
    assert problem.retryable is retryable


def test_rate_limited_floors_retry_after_at_one_second() -> None:
    """A sub-second remainder must not round down to an immediate retry."""
    assert RateLimitedError(retry_after=0.2).headers["Retry-After"] == "1"
    assert RateLimitedError(retry_after=2.2).headers["Retry-After"] == "3"


def test_rate_limited_lets_supplied_headers_win() -> None:
    """`RateLimitResult.headers()` already computed the authoritative values."""
    error = RateLimitedError(retry_after=1.0, headers={"Retry-After": "30", "RateLimit-Limit": "5"})
    assert error.headers == {"Retry-After": "30", "RateLimit-Limit": "5"}


# ── Handler behaviour over HTTP ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("slug", "status"),
    [
        ("malformed", 400),
        ("missing-key", 401),
        ("scope", 403),
        ("not-found", 404),
        ("conflict", 409),
        ("validation", 422),
        ("rate-limited", 429),
        ("internal", 500),
        ("no-verdict", 503),
    ],
)
async def test_app_errors_render_as_problem_json(
    client: httpx.AsyncClient, slug: str, status: int
) -> None:
    response = await client.get(f"/raise/{slug}")

    assert response.status_code == status
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    body = response.json()
    assert body["status"] == status
    assert body["type"].startswith(f"{ERROR_TYPE_BASE}/")
    assert body["instance"] == f"/raise/{slug}"
    assert body["request_id"] == response.headers[HEADER_REQUEST_ID]


async def test_rate_limited_response_carries_retry_after(client: httpx.AsyncClient) -> None:
    response = await client.get("/raise/rate-limited")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"


async def test_validation_error_maps_pydantic_fields(client: httpx.AsyncClient) -> None:
    """Dotted paths, machine codes, and the `body` prefix stripped."""
    response = await client.post("/echo", json={"text": "", "score": -1})

    assert response.status_code == 422
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/validation-error"
    assert body["title"] == "Request validation failed"
    fields = {entry["field"]: entry for entry in body["errors"]}
    assert set(fields) == {"text", "score"}
    assert fields["text"]["code"] == "string_too_short"
    assert fields["score"]["code"] == "greater_than_equal"
    assert body["detail"].startswith("text: ")


async def test_validation_error_never_echoes_the_rejected_value(
    client: httpx.AsyncClient,
) -> None:
    """`input` is caller text; echoing it would route around PII redaction."""
    secret = "alice@example.com is my email"
    response = await client.post("/echo", json={"text": secret, "score": "not-a-number"})

    assert response.status_code == 422
    assert secret not in response.text


async def test_missing_body_is_a_validation_error(client: httpx.AsyncClient) -> None:
    response = await client.post("/echo", json={})

    assert response.status_code == 422
    fields = {entry["field"] for entry in response.json()["errors"]}
    assert fields == {"text", "score"}


async def test_unparseable_json_is_a_problem(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/echo", content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    assert response.json()["request_id"]


async def test_unknown_route_is_a_not_found_problem(client: httpx.AsyncClient) -> None:
    """Starlette's own 404 is re-shaped, not passed through as `{"detail":...}`."""
    response = await client.get("/nope")

    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    assert response.json()["type"] == f"{ERROR_TYPE_BASE}/not-found"


async def test_method_not_allowed_keeps_its_status_and_gets_a_slug(
    client: httpx.AsyncClient,
) -> None:
    """A status outside the §9 table still gets a stable, non-colliding type."""
    response = await client.delete("/echo")

    assert response.status_code == 405
    assert response.json()["type"] == f"{ERROR_TYPE_BASE}/method-not-allowed"


async def test_http_exception_detail_is_preserved(client: httpx.AsyncClient) -> None:
    response = await client.get("/http-exception")

    assert response.status_code == 415
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/unsupported-media-type"
    assert body["detail"] == "Unsupported media type"
    assert body["request_id"]


async def test_unknown_status_falls_back_to_its_phrase(client: httpx.AsyncClient) -> None:
    response = await client.get("/teapot")

    assert response.status_code == 418
    # "I'm a Teapot" — the apostrophe is collapsed, so the type stays URI-safe.
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/i-m-a-teapot"
    # Uncatalogued 4xx: a client error, unchanged by retrying.
    assert body["retryable"] is False


async def test_uncatalogued_5xx_defaults_to_retryable(client: httpx.AsyncClient) -> None:
    """A status outside the §9 catalogue still gets a retryable verdict — a
    caller should never have to special-case "status not in the docs" into
    "no signal at all"."""
    response = await client.get("/uncatalogued-5xx")

    assert response.status_code == 502
    assert response.json()["retryable"] is True


async def test_not_implemented_is_the_deliberate_5xx_exception(
    client: httpx.AsyncClient,
) -> None:
    """501 is a 5xx that retrying can never fix — the endpoint still will not
    be implemented — so it is carved out of the otherwise-retryable default."""
    response = await client.get("/not-implemented")

    assert response.status_code == 501
    assert response.json()["retryable"] is False


# ── The ladder's only 5xx ─────────────────────────────────────────────────────


async def test_no_verdict_error_maps_to_503(client: httpx.AsyncClient) -> None:
    """`NoVerdictError` is the single path from the ladder to a 5xx."""
    response = await client.get("/no-verdict")

    assert response.status_code == 503
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/no-verdict"
    assert body["status"] == 503
    # The orchestrator's own message is internal detail; ours is the contract.
    assert "every rung failed" not in response.text


# ── The catch-all ─────────────────────────────────────────────────────────────


async def test_catch_all_never_leaks_the_internal_message(client: httpx.AsyncClient) -> None:
    response = await client.get("/boom")

    assert response.status_code == 500
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    body = response.json()
    assert body["type"] == f"{ERROR_TYPE_BASE}/internal-error"
    assert body["title"] == "Internal server error"
    assert body["request_id"]
    # Neither the exception text nor a frame of the traceback.
    assert "hunter2" not in response.text
    assert "postgres://" not in response.text
    assert "Traceback" not in response.text
    assert "test_api_errors" not in response.text


async def test_catch_all_logs_the_traceback(settings: Settings) -> None:
    """The stack the caller does not get must still reach the operator."""
    transport = httpx.ASGITransport(app=build_app(settings), raise_app_exceptions=False)
    with capture_logs() as entries:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            await http.get("/boom")

    unhandled = [e for e in entries if e.get("event") == "request.unhandled_exception"]
    assert len(unhandled) == 1
    assert unhandled[0]["log_level"] == "error"
    assert unhandled[0]["error_type"] == "RuntimeError"
    assert unhandled[0]["path"] == "/boom"
    # `exc_info` is what the configured renderer turns into a traceback.
    assert isinstance(unhandled[0]["exc_info"], RuntimeError)
