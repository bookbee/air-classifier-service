"""The assembled service, booted.

Every other suite exercises one module behind fakes. This one starts the real
application through :func:`air_classifier.main.create_app` — real lifespan, real
middleware stack, real error handlers, real Tier 0 and Tier 1 — and asks the
question none of the unit suites can: *does it come up at all, and does it come
up honest about what it can do?*

The deployment under test is the awkward one on purpose: no Redis, no air-llm.
That configuration is not a degenerate case to be tolerated, it is the one a
developer runs on a laptop and the one a fresh cluster runs before its secrets
land, and the service is specified to boot in it with a two-rung ladder. A
regression here — a tier constructor that raises on a missing dependency
instead of standing itself down — turns every one of those into a crash loop.

Nothing here opens a socket. The only component that would try is Tier 2's
availability probe, and the one test that wants it intercepts the call with
``respx`` rather than hoping the developer's machine has no daemon listening:
"degraded" has to be a property of the assertion, not of the host.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Final

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

# Private, but it is the type structlog's own `get_logger` returns, and the
# alternative — duck-typing on an attribute name — would be no more stable and
# considerably less obvious.
from air_classifier import __version__
from air_classifier.api.deps import STATE_ATTR, AppState
from air_classifier.api.errors import PROBLEM_CONTENT_TYPE
from air_classifier.cache.memory import MemoryCache
from air_classifier.config import (
    AirLlmSettings,
    AppSettings,
    CacheSettings,
    SecuritySettings,
    Settings,
)
from air_classifier.constants import HEADER_API_KEY, HEADER_REQUEST_ID
from air_classifier.constants import Tier as TierId
from air_classifier.main import STATE_ANALYSIS_SERVICE, create_app
from air_classifier.pipeline.service import AnalysisService
from air_classifier.pipeline.tiers.scorers import LEXICON_SCORER_NAME
from air_classifier.schemas.classify import ClassificationResponse
from air_classifier.security.api_keys import ALL_SCOPES, hash_key

#: Presented as ``X-API-Key``. Real auth rather than
#: ``security.allow_unauthenticated``: the anonymous identity takes a different
#: branch through the key store, and this suite is about the wiring a deployment
#: actually runs.
RAW_API_KEY: Final[str] = "air-classifier-boot-test-key"
_SALT: Final[str] = "app-boot-test-salt"

AUTH: Final[dict[str, str]] = {HEADER_API_KEY: RAW_API_KEY}

pytestmark = pytest.mark.integration

#: air-llm's readiness endpoint — what every tier's own probe calls.
_AIR_LLM_READY_URL: Final[str] = "http://localhost:8083/v1/ready"

#: Unambiguous polarity, so the verdict is decided low on the ladder rather than
#: by whichever rung happens to be least uncertain about a borderline sentence.
_CLEARLY_POSITIVE: Final[str] = "This is absolutely wonderful and I love it."


def _api_keys_inline() -> str:
    """One key record, in the JSON shape ``ApiKeyStore.load`` parses."""
    return json.dumps(
        [
            {
                "id": "boot-test",
                "name": "app boot test key",
                "key_hash": hash_key(RAW_API_KEY, _SALT),
                "scopes": sorted(ALL_SCOPES),
            }
        ]
    )


def build_settings(*, air_llm_enabled: bool = False, docs_enabled: bool = True) -> Settings:
    """A deployment with nothing external configured.

    ``air_llm_enabled`` defaults to *off* so the ladder is exactly two rungs and
    no test can accidentally depend on air-llm being reachable from the machine
    running it. Tiers 2 and 3 share this one gate now — the one test that wants
    Tier 2 present without Tier 3 turns air-llm on here and disables Tier 3 on
    its own (``settings.tiers.t3_cloud_llm.enabled = False``), then mocks the
    transport.
    """
    return Settings(
        app=AppSettings(env="test", docs_enabled=docs_enabled),
        air_llm=AirLlmSettings(
            enabled=air_llm_enabled,
            token=SecretStr("test-token") if air_llm_enabled else None,
        ),
        cache=CacheSettings(backend="memory"),
        security=SecuritySettings(
            api_keys_inline=_api_keys_inline(),
            hash_salt=SecretStr(_SALT),
        ),
    )


def app_state_of(app: FastAPI) -> AppState:
    """The typed singleton container the lifespan published."""
    state = getattr(app.state, STATE_ATTR)
    assert isinstance(state, AppState)
    return state


def tier_ids(app: FastAPI) -> tuple[TierId, ...]:
    return tuple(tier.tier for tier in app_state_of(app).tiers)


# NOTE: the process-wide logging state this suite disturbs by booting the real
# service is saved and restored by `_restore_global_logging` in the root
# conftest, which covers every suite that calls `create_app` rather than only
# this one.


@pytest.fixture
def app() -> FastAPI:
    """A local-only application, not yet started."""
    return create_app(build_settings())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """The started application.

    Entering ``TestClient`` runs the lifespan, so every test using this fixture
    already asserts that startup did not raise; leaving it runs the shutdown.
    """
    with TestClient(app) as started:
        yield started


# ── Boot ──────────────────────────────────────────────────────────────────────


def test_boots_with_no_external_services_and_serves_two_rungs(app: FastAPI) -> None:
    """The headline property: nothing configured, and the service still comes up.

    Tiers 2 and 3 both stand down because air-llm is switched off. Both are
    *skips*, and the difference between a skip and a crash is the difference
    between a degraded deployment and no deployment.
    """
    with TestClient(app):
        assert tier_ids(app) == (TierId.T0_RULES, TierId.T1_CLASSIFIER)


def test_startup_publishes_everything_the_routes_read(app: FastAPI) -> None:
    """The two views of ``app.state`` — typed container and named attributes.

    ``api/deps.py`` reads the container; ``api/v1/system.py`` reads the three
    named attributes so its routes still answer without a lifespan. Both are the
    factory's contract, and a rename that satisfied only one of them would show
    up as a 500 from a readiness probe rather than as an import error.
    """
    with TestClient(app):
        state = app_state_of(app)
        assert isinstance(state.cache, MemoryCache)
        assert len(state.api_keys) == 1

        assert app.state.settings is state.settings
        assert app.state.tiers == state.tiers
        assert app.state.cache is state.cache
        assert isinstance(getattr(app.state, STATE_ANALYSIS_SERVICE), AnalysisService)


def test_disabled_rungs_are_never_constructed() -> None:
    """A rung switched off in settings is absent, not present-and-unavailable.

    That equivalence is what lets ``/v1/capabilities`` report `enabled: false`
    for a rung nobody built without having to model a third state.
    """
    settings = build_settings()
    settings.tiers.t1_classifier.enabled = False
    with TestClient(create_app(settings)) as started:
        assert tier_ids(started.app) == (TierId.T0_RULES,)


def test_shutdown_releases_the_cache(app: FastAPI) -> None:
    """Shutdown actually runs, and closes what startup opened.

    Asserted through the backend's own state rather than by "no exception was
    raised": a ``finally`` that is never reached fails silently, and this is the
    only place the cache's lifetime is owned.
    """
    with TestClient(app) as started:
        started.post("/v1/classify", json={"text": _CLEARLY_POSITIVE}, headers=AUTH)
        cache = app_state_of(app).cache
        assert isinstance(cache, MemoryCache)
        # The request charged a rate-limit bucket and cached a verdict.
        assert cache.size > 0

    assert cache.size == 0


# ── Operational endpoints ─────────────────────────────────────────────────────


def test_health_is_up(client: TestClient) -> None:
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "air-classifier", "version": __version__}


def test_ready_is_ok_when_every_registered_rung_answers(client: TestClient) -> None:
    """A two-rung ladder is a complete ladder, not a degraded one.

    The rungs that were never built are not components, so they cannot be
    "failing" — which is why the degraded signal in the next test comes from a
    rung that exists and cannot serve, not from one that is absent.
    """
    response = client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {component["name"] for component in body["components"]} == {
        TierId.T0_RULES.value,
        TierId.T1_CLASSIFIER.value,
        "cache",
    }


@respx.mock
def test_ready_is_degraded_not_unready_when_an_upper_rung_is_down() -> None:
    """An unreachable dependency costs quality, not rotation.

    Tier 2 is built (air-llm is switched on, Tier 3 disabled on its own) but
    air-llm refuses the connection. Returning 503 here would drain every pod in
    the fleet at once for a failure the ladder is designed to absorb, so the
    contract is 200 plus ``degraded`` — and this is the test that pins it end
    to end.
    """
    respx.get(_AIR_LLM_READY_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    settings = build_settings(air_llm_enabled=True)
    settings.tiers.t3_cloud_llm.enabled = False
    with TestClient(create_app(settings)) as started:
        assert tier_ids(started.app) == (
            TierId.T0_RULES,
            TierId.T1_CLASSIFIER,
            TierId.T2_LOCAL_LLM,
        )
        response = started.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    down = {component["name"] for component in body["components"] if not component["ready"]}
    assert down == {TierId.T2_LOCAL_LLM.value}


def test_capabilities_reports_the_lexicon_fallback(client: TestClient) -> None:
    """The silent degradation this endpoint exists to make loud.

    No ONNX artifact is configured, so Tier 1 runs on the lexicon scorer: the
    service still answers and the dashboards still move, but every confidence
    number — and therefore every escalation threshold — is worse. The only way
    an operator learns that from outside the process is here.
    """
    response = client.get("/v1/capabilities", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    by_tier: dict[str, dict[str, Any]] = {entry["tier"]: entry for entry in body["tiers"]}
    # Reported over the whole ladder, so a rung that was never built is visible
    # as absent-by-choice rather than missing from the list.
    assert set(by_tier) == {tier.value for tier in TierId.ordered()}

    classifier = by_tier[TierId.T1_CLASSIFIER.value]
    assert classifier["available"] is True
    assert classifier["model_version"] == LEXICON_SCORER_NAME
    assert classifier["detail"]["scorer_kind"] == "lexicon"
    assert classifier["detail"]["onnx_loaded"] is False

    cloud = by_tier[TierId.T3_CLOUD_LLM.value]
    assert cloud["available"] is False

    assert body["max_batch_items"] == 100


def test_capabilities_requires_a_key_unlike_health_and_ready(client: TestClient) -> None:
    """Deployment detail, not a load-balancer boolean.

    `/v1/health` and `/v1/ready` stay open for a kubelet that carries no
    credential; `/v1/capabilities` discloses configured model/alias names and
    which Tier-1 backend actually loaded, so it sits behind the same
    auth/rate-limit dependency `/v1/classify` does instead.
    """
    unauthenticated = client.get("/v1/capabilities")
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/ready").status_code == 200


def test_metrics_are_exposed_outside_v1_and_outside_auth(client: TestClient) -> None:
    """A scraper carries no credentials, so the exposition cannot demand any."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "air_classifier" in response.text


def test_openapi_is_served_when_docs_are_enabled(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == __version__
    assert "/v1/classify" in schema["paths"]
    # The exposition is line-oriented text; describing it as JSON would mislead
    # every generated client.
    assert "/metrics" not in schema["paths"]


def test_docs_can_be_closed() -> None:
    """`docs_enabled=false` closes the schema too, not only the rendered page."""
    with TestClient(create_app(build_settings(docs_enabled=False))) as started:
        assert started.get("/openapi.json").status_code == 404
        assert started.get("/docs").status_code == 404


# ── A real request through the assembled stack ────────────────────────────────


# ── The unified entry point ───────────────────────────────────────────────────


def test_classify_is_analysed_by_a_low_rung_end_to_end(client: TestClient) -> None:
    """One authenticated request, all the way down and back.

    With only Tiers 0 and 1 built, the ladder cannot escalate past the
    classifier however uncertain it is — so a response that validates against
    ``ClassificationResponse`` and names a low rung is the proof that
    preprocessing, the policy, the orchestrator and the response shaping are
    all connected to each other and not merely importable.
    """
    response = client.post("/v1/classify", json={"text": _CLEARLY_POSITIVE}, headers=AUTH)

    assert response.status_code == 200
    body = ClassificationResponse.model_validate(response.json())
    assert body.decided_by in (TierId.T0_RULES, TierId.T1_CLASSIFIER)
    assert body.escalation_trace
    assert body.request_id == response.headers[HEADER_REQUEST_ID]
    # The quota headers ride on the success path, not only on a 429.
    assert "RateLimit-Remaining" in response.headers


def test_classify_returns_the_detailed_contract(client: TestClient) -> None:
    """One call, the full shape — with the fields only an LLM tier can supply
    honestly absent, since this deployment has no LLM tier built (module
    docstring: two rungs, on purpose).
    """
    response = client.post("/v1/classify", json={"text": _CLEARLY_POSITIVE}, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"]["label"] == "positive"
    assert body["decided_by"] in ("t0_rules", "t1_classifier")
    # T0/T1 cannot produce these — see schemas/classify.py.
    assert body["tone"] is None
    assert body["intent"] is None
    assert body["topics"] == []
    # The heuristic fields are always present regardless of which rung decided.
    assert body["urgency"]["label"] in ("low", "medium", "high", "critical")
    assert body["routing"]["priority"] in ("P1", "P2", "P3", "P4")
    assert isinstance(body["requires_human"], bool)
    assert body["model"]["name"]
    assert body["model"]["version"] == __version__


def test_classify_urgency_and_requires_human_track_the_text(client: TestClient) -> None:
    """A message an operator would call an emergency reads as one here too."""
    response = client.post(
        "/v1/classify",
        json={"text": "URGENT: I was charged twice and locked out of my account."},
        headers=AUTH,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["urgency"]["label"] == "critical"
    assert body["routing"]["priority"] == "P1"
    assert body["requires_human"] is True


def test_classify_batch_analyses_every_item(client: TestClient) -> None:
    """Batch variant, same per-item isolation the other three routes already have."""
    response = client.post(
        "/v1/classify/batch",
        json={"items": [{"text": _CLEARLY_POSITIVE}, {"text": "not acceptable"}]},
        headers=AUTH,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 2
    labels = [item["result"]["sentiment"]["label"] for item in body["items"]]
    assert "positive" in labels


def test_an_unauthenticated_request_is_a_problem_document(client: TestClient) -> None:
    """The error handlers are installed on the app the factory returns.

    FastAPI's own 401 body is ``{"detail": ...}`` in ``application/json``; §9
    says every failure leaves this service as RFC 9457. Asserted here rather
    than only in the unit suite because installation order is the factory's job.
    """
    response = client.post("/v1/classify", json={"text": _CLEARLY_POSITIVE})

    assert response.status_code == 401
    assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
    body = response.json()
    assert body["type"].endswith("/missing-api-key")
    assert body["instance"] == "/v1/classify"
    assert body["request_id"] == response.headers[HEADER_REQUEST_ID]


def test_an_oversized_body_is_refused_by_the_middleware() -> None:
    """The body ceiling is installed, and its rejection is still a problem doc.

    Exercised through the factory because the ceiling is one of the three
    middlewares whose *order* only exists here: it sits inside the request-context
    middleware, which is what lets a refused upload still carry a request id.
    """
    settings = build_settings()
    settings.app.max_body_bytes = 2048
    with TestClient(create_app(settings)) as started:
        response = started.post("/v1/classify", json={"text": "x" * 4096}, headers=AUTH)

    assert response.status_code == 400
    assert response.json()["type"].endswith("/malformed-request")
    assert HEADER_REQUEST_ID in response.headers
