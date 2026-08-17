"""Unit tests for the per-request dependencies.

The dependencies are exercised directly against a hand-built ``Request`` rather
than through a client for most of these: what is under test is the decision
(which exception, which clamped option), and routing a decision through an HTTP
round trip would only add the error handlers' translation table to the things
that can make the test fail. The HTTP-level behaviour that *is* specific to the
transport — quota headers landing on a successful response — gets a real app.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.responses import Response

from air_classifier.api.deps import (
    RATE_LIMIT_HEADERS_ATTR,
    STATE_ATTR,
    AppState,
    IdempotencyRecord,
    RateLimitedKeyDep,
    apply_key_policy,
    enforce_rate_limit,
    get_app_state,
    lookup_idempotent,
    require_api_key,
    require_scope,
    store_idempotent,
)
from air_classifier.api.errors import (
    IdempotencyConflictError,
    InsufficientScopeError,
    MissingApiKeyError,
    RateLimitedError,
    install_error_handlers,
)
from air_classifier.api.middleware import API_KEY_ID_ATTR
from air_classifier.cache.base import idempotency_cache_key
from air_classifier.cache.memory import MemoryCache
from air_classifier.config import AppSettings, CacheSettings, SecuritySettings, Settings
from air_classifier.constants import HEADER_API_KEY, HEADER_IDEMPOTENCY_KEY, Tier
from air_classifier.pipeline.orchestrator import Orchestrator
from air_classifier.pipeline.policy import EscalationPolicy
from air_classifier.schemas.common import Options
from air_classifier.security.api_keys import (
    SCOPE_ADMIN_READ,
    SCOPE_SENTIMENT_READ,
    ApiKeyRecord,
    ApiKeyStore,
    hash_key,
)
from air_classifier.security.quotas import RateLimiter

SALT = "test-salt"
RAW_KEY = "airc_test_key"
BODY = b'{"text":"hello"}'


# ── Fixtures & builders ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_log_context() -> Iterator[None]:
    """`require_api_key` binds `api_key_id` onto contextvars, which outlive a test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


def make_settings(**security: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        cache=CacheSettings(key_prefix="test:"),
        security=SecuritySettings(hash_salt=SecretStr(SALT), **security),
    )


def make_key(
    *,
    key_id: str = "key-1",
    raw: str = RAW_KEY,
    scopes: set[str] | None = None,
    max_tier: Tier = Tier.T3_CLOUD_LLM,
    force_pii_redaction: bool = False,
    rate_limit_rpm: int | None = None,
    burst: int | None = None,
    enabled: bool = True,
) -> ApiKeyRecord:
    return ApiKeyRecord(
        id=key_id,
        name=f"name of {key_id}",
        key_hash=hash_key(raw, SALT),
        scopes=scopes if scopes is not None else {SCOPE_SENTIMENT_READ},
        max_tier=max_tier,
        force_pii_redaction=force_pii_redaction,
        rate_limit_rpm=rate_limit_rpm,
        burst=burst,
        enabled=enabled,
    )


def make_state(
    cache: MemoryCache,
    *,
    settings: Settings | None = None,
    records: list[ApiKeyRecord] | None = None,
) -> AppState:
    resolved = settings or make_settings()
    store = ApiKeyStore(
        records or [],
        salt=SALT,
        allow_unauthenticated=resolved.security.allow_unauthenticated,
    )
    return AppState(
        settings=resolved,
        cache=cache,
        api_keys=store,
        rate_limiter=RateLimiter(cache, resolved),
        orchestrator=Orchestrator([], EscalationPolicy(resolved), resolved),
    )


def make_request(state: AppState | None, headers: dict[str, str] | None = None) -> Request:
    """A minimal ASGI scope — enough for `.app`, `.headers`, `.state` and `.url`."""
    app = FastAPI()
    if state is not None:
        state.attach(app)
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/sentiment",
        "raw_path": b"/v1/sentiment",
        "root_path": "",
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "headers": [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or {}).items()
        ],
        "app": app,
    }
    return Request(scope)


# ── AppState ──────────────────────────────────────────────────────────────────


async def test_app_state_round_trips_through_the_app(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache)
    request = make_request(state)

    assert get_app_state(request) is state


async def test_missing_app_state_is_a_wiring_error() -> None:
    request = make_request(None)

    # Not a 4xx/5xx problem detail: a route can only see this if the lifespan
    # never ran, which no caller can cause.
    with pytest.raises(RuntimeError, match=STATE_ATTR):
        get_app_state(request)


# ── Authentication ────────────────────────────────────────────────────────────


async def test_missing_header_is_unauthenticated(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key()])
    request = make_request(state)

    with pytest.raises(MissingApiKeyError) as exc:
        await require_api_key(request)
    assert exc.value.status == 401


async def test_blank_header_is_treated_as_missing(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key()])
    request = make_request(state, {HEADER_API_KEY: "   "})

    with pytest.raises(MissingApiKeyError):
        await require_api_key(request)


async def test_unknown_key_is_forbidden(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key()])
    request = make_request(state, {HEADER_API_KEY: "airc_not_a_real_key"})

    with pytest.raises(InsufficientScopeError) as exc:
        await require_api_key(request)
    assert exc.value.status == 403


async def test_disabled_key_is_forbidden(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key(enabled=False)])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    with pytest.raises(InsufficientScopeError):
        await require_api_key(request)


async def test_error_messages_never_carry_the_key(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key()])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY + "-wrong"})

    with pytest.raises(InsufficientScopeError) as exc:
        await require_api_key(request)
    assert RAW_KEY not in str(exc.value)
    assert RAW_KEY not in repr(exc.value)


async def test_valid_key_resolves_to_its_record(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key(key_id="key-7")])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    record = await require_api_key(request)

    assert record.id == "key-7"
    # The id — never the key — is bound for downstream log lines.
    assert structlog.contextvars.get_contextvars()["api_key_id"] == "key-7"
    # And stamped on the scope as well, which is the only copy the access-log
    # middleware can see: it is raw ASGI outside this dependency's task, so a
    # contextvar alone would leave every request attributed to `anonymous`.
    assert getattr(request.state, API_KEY_ID_ATTR) == "key-7"


async def test_allow_unauthenticated_mints_the_anonymous_identity(
    memory_cache: MemoryCache,
) -> None:
    settings = make_settings(allow_unauthenticated=True)
    state = make_state(memory_cache, settings=settings)
    request = make_request(state)

    record = await require_api_key(request)

    assert record.id == "anonymous"


async def test_scope_is_enforced(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key(scopes={SCOPE_SENTIMENT_READ})])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    assert (await require_scope(SCOPE_SENTIMENT_READ)(request)).id == "key-1"

    with pytest.raises(InsufficientScopeError) as exc:
        await require_scope(SCOPE_ADMIN_READ)(request)
    assert exc.value.status == 403


async def test_scope_check_still_reports_a_missing_key_as_401(memory_cache: MemoryCache) -> None:
    state = make_state(memory_cache, records=[make_key()])
    request = make_request(state)

    with pytest.raises(MissingApiKeyError):
        await require_scope(SCOPE_SENTIMENT_READ)(request)


# ── Rate limiting ─────────────────────────────────────────────────────────────


async def test_rate_limiter_allows_then_denies(memory_cache: MemoryCache) -> None:
    key = make_key(rate_limit_rpm=1, burst=0)
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    first = await enforce_rate_limit(request, key)
    assert first.allowed
    assert first.limit == 1
    assert first.remaining == 0

    with pytest.raises(RateLimitedError) as exc:
        await enforce_rate_limit(request, key)
    assert exc.value.status == 429
    assert exc.value.retry_after > 0


async def test_quota_headers_are_published_on_success(memory_cache: MemoryCache) -> None:
    key = make_key(rate_limit_rpm=10, burst=0)
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})
    response = Response()

    await enforce_rate_limit(request, key, response=response)

    # Both places: the response the route is building, and request.state for
    # whatever the middleware ends up sending.
    assert response.headers["RateLimit-Limit"] == "10"
    assert response.headers["RateLimit-Remaining"] == "9"
    assert int(response.headers["RateLimit-Reset"]) >= 0
    stashed = getattr(request.state, RATE_LIMIT_HEADERS_ATTR)
    assert stashed["RateLimit-Remaining"] == "9"


async def test_quota_headers_are_published_on_rejection(memory_cache: MemoryCache) -> None:
    key = make_key(rate_limit_rpm=1, burst=0)
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    await enforce_rate_limit(request, key)
    with pytest.raises(RateLimitedError):
        await enforce_rate_limit(request, key)

    stashed = getattr(request.state, RATE_LIMIT_HEADERS_ATTR)
    assert stashed["RateLimit-Remaining"] == "0"
    assert int(stashed["Retry-After"]) >= 1


async def test_quotas_are_charged_per_key(memory_cache: MemoryCache) -> None:
    noisy = make_key(key_id="noisy", raw="airc_noisy", rate_limit_rpm=1, burst=0)
    quiet = make_key(key_id="quiet", raw="airc_quiet", rate_limit_rpm=1, burst=0)
    state = make_state(memory_cache, records=[noisy, quiet])
    request = make_request(state)

    await enforce_rate_limit(request, noisy)
    with pytest.raises(RateLimitedError):
        await enforce_rate_limit(request, noisy)

    assert (await enforce_rate_limit(request, quiet)).allowed


# ── Key policy ────────────────────────────────────────────────────────────────


async def test_tier_ceiling_is_clamped_silently() -> None:
    key = make_key(max_tier=Tier.T1_CLASSIFIER)
    options = Options(max_tier=Tier.T3_CLOUD_LLM)

    applied = apply_key_policy(options, key)

    assert applied.max_tier is Tier.T1_CLASSIFIER
    # The caller's object is untouched, so what they asked for is still what the
    # request body says they asked for.
    assert options.max_tier is Tier.T3_CLOUD_LLM
    assert applied is not options


async def test_clamping_lowers_a_floor_that_would_outrank_the_ceiling() -> None:
    key = make_key(max_tier=Tier.T1_CLASSIFIER)
    options = Options(min_tier=Tier.T2_LOCAL_LLM, max_tier=Tier.T3_CLOUD_LLM)

    applied = apply_key_policy(options, key)

    # min_tier <= max_tier is an Options invariant and model_copy does not
    # re-validate it, so the floor has to come down with the ceiling.
    assert applied.min_tier is Tier.T1_CLASSIFIER
    assert applied.max_tier is Tier.T1_CLASSIFIER


async def test_a_request_below_the_ceiling_is_left_alone() -> None:
    key = make_key(max_tier=Tier.T3_CLOUD_LLM)
    options = Options(max_tier=Tier.T1_CLASSIFIER, min_tier=Tier.T0_RULES)

    applied = apply_key_policy(options, key)

    assert applied.max_tier is Tier.T1_CLASSIFIER
    assert applied.min_tier is Tier.T0_RULES


async def test_explicit_redaction_opt_out_is_refused() -> None:
    key = make_key(force_pii_redaction=True)
    options = Options(redact_pii=False)

    with pytest.raises(InsufficientScopeError) as exc:
        apply_key_policy(options, key)
    assert exc.value.status == 403


async def test_implicit_default_against_a_forcing_key_is_allowed() -> None:
    key = make_key(force_pii_redaction=True)
    options = Options()

    applied = apply_key_policy(options, key)

    assert applied.redact_pii is True


async def test_explicit_redaction_opt_in_against_a_forcing_key_is_allowed() -> None:
    key = make_key(force_pii_redaction=True)
    options = Options(redact_pii=True)

    assert apply_key_policy(options, key).redact_pii is True


async def test_opting_out_is_fine_when_the_key_does_not_force_redaction() -> None:
    key = make_key(force_pii_redaction=False)
    options = Options(redact_pii=False)

    assert apply_key_policy(options, key).redact_pii is False


# ── Idempotency ───────────────────────────────────────────────────────────────


async def test_no_header_means_no_idempotency(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_API_KEY: RAW_KEY})

    assert await lookup_idempotent(request, key, body=BODY) is None
    await store_idempotent(request, key, body=BODY, response='{"ok":true}')
    assert memory_cache.size == 0


async def test_replay_returns_the_stored_response(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    headers = {HEADER_API_KEY: RAW_KEY, HEADER_IDEMPOTENCY_KEY: "idem-1"}
    first = make_request(state, headers)

    assert await lookup_idempotent(first, key, body=BODY) is None
    await store_idempotent(first, key, body=BODY, response='{"label":"positive"}')

    replay = make_request(state, headers)
    assert await lookup_idempotent(replay, key, body=BODY) == '{"label":"positive"}'


async def test_same_key_with_a_different_body_conflicts(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    headers = {HEADER_API_KEY: RAW_KEY, HEADER_IDEMPOTENCY_KEY: "idem-1"}
    request = make_request(state, headers)
    await store_idempotent(request, key, body=BODY, response='{"label":"positive"}')

    with pytest.raises(IdempotencyConflictError) as exc:
        await lookup_idempotent(request, key, body=b'{"text":"something else"}')
    assert exc.value.status == 409


async def test_idempotency_is_scoped_per_key(memory_cache: MemoryCache) -> None:
    mine = make_key(key_id="mine", raw="airc_mine")
    yours = make_key(key_id="yours", raw="airc_yours")
    state = make_state(memory_cache, records=[mine, yours])
    headers = {HEADER_IDEMPOTENCY_KEY: "shared-key"}
    request = make_request(state, headers)

    await store_idempotent(request, mine, body=BODY, response='{"mine":true}')

    # Same header value, different caller: a miss rather than a conflict or,
    # far worse, someone else's answer.
    assert await lookup_idempotent(request, yours, body=b'{"text":"other"}') is None


async def test_unreadable_entry_degrades_to_a_miss(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_IDEMPOTENCY_KEY: "idem-1"})
    await memory_cache.set(
        idempotency_cache_key(prefix="test:", api_key_id=key.id, idempotency_key="idem-1"),
        "not json at all",
        60,
    )

    assert await lookup_idempotent(request, key, body=BODY) is None


async def test_string_and_bytes_bodies_hash_alike(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_IDEMPOTENCY_KEY: "idem-1"})

    await store_idempotent(request, key, body=BODY.decode(), response="{}")

    assert await lookup_idempotent(request, key, body=BODY) == "{}"


async def test_zero_ttl_disables_idempotency_storage(memory_cache: MemoryCache) -> None:
    settings = make_settings()
    settings = settings.model_copy(
        update={"cache": CacheSettings(key_prefix="test:", idempotency_ttl_seconds=0)}
    )
    key = make_key()
    state = make_state(memory_cache, settings=settings, records=[key])
    request = make_request(state, {HEADER_IDEMPOTENCY_KEY: "idem-1"})

    await store_idempotent(request, key, body=BODY, response="{}")

    assert memory_cache.size == 0


async def test_stored_entry_holds_a_digest_not_the_body(memory_cache: MemoryCache) -> None:
    key = make_key()
    state = make_state(memory_cache, records=[key])
    request = make_request(state, {HEADER_IDEMPOTENCY_KEY: "idem-1"})

    await store_idempotent(request, key, body=BODY, response="{}")

    raw = await memory_cache.get(
        idempotency_cache_key(prefix="test:", api_key_id=key.id, idempotency_key="idem-1")
    )
    assert raw is not None
    entry = IdempotencyRecord.model_validate_json(raw)
    assert "hello" not in entry.body_sha256
    assert len(entry.body_sha256) == 64


# ── Through a real app ────────────────────────────────────────────────────────
#
# The dependencies are wired the way a router will wire them, with the real
# error handlers installed, so these assert what a caller actually receives:
# the status from the §9 catalogue, a problem+json body, and the quota headers.


def build_app(state: AppState) -> FastAPI:
    """Two routes: one plain, one idempotent. Both behind key + quota."""
    app = FastAPI()
    install_error_handlers(app)
    state.attach(app)
    calls = {"n": 0}

    @app.post("/v1/ping")
    async def ping(key: RateLimitedKeyDep) -> dict[str, str]:
        return {"key_id": key.id}

    @app.post("/v1/echo")
    async def echo(request: Request, key: RateLimitedKeyDep) -> Response:
        body = await request.body()
        replay = await lookup_idempotent(request, key, body=body)
        if replay is not None:
            return Response(replay, media_type="application/json")
        calls["n"] += 1
        payload = json.dumps({"calls": calls["n"]})
        await store_idempotent(request, key, body=body, response=payload)
        return Response(payload, media_type="application/json")

    return app


@pytest.fixture
async def client(memory_cache: MemoryCache) -> AsyncIterator[TestClient]:
    state = make_state(memory_cache, records=[make_key(rate_limit_rpm=2, burst=0)])
    with TestClient(build_app(state)) as test_client:
        yield test_client


async def test_http_missing_key_is_401_problem_json(client: TestClient) -> None:
    response = client.post("/v1/ping")

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"].endswith("/missing-api-key")


async def test_http_unknown_key_is_403(client: TestClient) -> None:
    response = client.post("/v1/ping", headers={HEADER_API_KEY: "airc_wrong"})

    assert response.status_code == 403
    assert response.json()["type"].endswith("/insufficient-scope")


async def test_http_success_carries_quota_headers(client: TestClient) -> None:
    response = client.post("/v1/ping", headers={HEADER_API_KEY: RAW_KEY})

    assert response.status_code == 200
    assert response.json() == {"key_id": "key-1"}
    assert response.headers["RateLimit-Limit"] == "2"
    assert response.headers["RateLimit-Remaining"] == "1"


async def test_http_quota_exhaustion_is_429_with_retry_after(client: TestClient) -> None:
    headers = {HEADER_API_KEY: RAW_KEY}
    assert client.post("/v1/ping", headers=headers).status_code == 200
    assert client.post("/v1/ping", headers=headers).status_code == 200

    denied = client.post("/v1/ping", headers=headers)

    assert denied.status_code == 429
    assert denied.json()["type"].endswith("/rate-limited")
    assert int(denied.headers["Retry-After"]) >= 1
    assert denied.headers["RateLimit-Remaining"] == "0"


async def test_http_replay_returns_the_first_answer(client: TestClient) -> None:
    headers = {HEADER_API_KEY: RAW_KEY, HEADER_IDEMPOTENCY_KEY: "idem-1"}

    first = client.post("/v1/echo", headers=headers, json={"text": "hello"})
    replay = client.post("/v1/echo", headers=headers, json={"text": "hello"})

    assert first.json() == {"calls": 1}
    # Byte-identical, and the handler body never ran a second time.
    assert replay.content == first.content


async def test_http_same_key_different_body_is_409(client: TestClient) -> None:
    headers = {HEADER_API_KEY: RAW_KEY, HEADER_IDEMPOTENCY_KEY: "idem-1"}
    client.post("/v1/echo", headers=headers, json={"text": "hello"})

    conflict = client.post("/v1/echo", headers=headers, json={"text": "different"})

    assert conflict.status_code == 409
    assert conflict.json()["type"].endswith("/idempotency-conflict")
