"""Tests for the operational endpoints.

Driven through ``httpx.ASGITransport`` rather than by calling the handlers
directly: the interesting behaviour of these routes lives in status codes,
content types and response shapes, none of which a direct call exercises. No
socket is opened — the transport speaks ASGI to the app object in-process.

The tiers and cache are stubs, and deliberately hostile ones: probes that hang,
probes that raise, and a tier that fails the test outright if a liveness check
so much as looks at it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import CONTENT_TYPE_LATEST

from air_classifier_service import __version__
from air_classifier_service.api.deps import AppState
from air_classifier_service.api.v1 import system
from air_classifier_service.cache.base import CacheBackend
from air_classifier_service.cache.memory import MemoryCache
from air_classifier_service.config import AppSettings, ClassifierSettings, Settings, TierSettings
from air_classifier_service.config import TiersSettings as TiersConfig
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.observability import metrics
from air_classifier_service.pipeline.orchestrator import Orchestrator
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.tiers.base import Tier as TierProtocol
from air_classifier_service.pipeline.tiers.base import TierContext
from air_classifier_service.pipeline.tiers.scorers import LEXICON_SCORER_NAME
from air_classifier_service.providers.prompts.loader import compact_system_prompt_version
from air_classifier_service.schemas.verdict import TierVerdict
from air_classifier_service.security.api_keys import ApiKeyStore
from air_classifier_service.security.quotas import RateLimiter


class ProbeExplodedError(AssertionError):
    """Raised by a stub whose probe should never have been called."""


class StubTier:
    """A rung whose availability answer is scripted.

    ``probe_calls`` is what makes the liveness assertion possible: proving
    ``/v1/health`` does not probe requires observing that nothing probed.
    """

    tier: TierId
    model_version: str

    def __init__(
        self,
        tier: TierId,
        *,
        available: bool = True,
        model_version: str = "stub-1.0",
        delay_s: float = 0.0,
        raises: type[BaseException] | None = None,
    ) -> None:
        self.tier = tier
        self.model_version = model_version
        self._available = available
        self._delay_s = delay_s
        self._raises = raises
        self.probe_calls = 0

    async def is_available(self) -> bool:
        self.probe_calls += 1
        if self._raises is not None:
            raise self._raises("stub tier probe failure: redis://user:hunter2@cache:6379")
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._available

    async def unavailable_reason(self) -> str | None:
        return None if self._available else "stub unavailable"

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        raise NotImplementedError("system routes never analyse")

    def supports_language(self, code: str) -> bool:
        return True


class StubCache:
    """Cache backend that only implements what readiness uses."""

    def __init__(self, *, reachable: bool = True) -> None:
        self._reachable = reachable
        self.ping_calls = 0

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def incr(self, key: str, ttl_seconds: int) -> int:
        return 0

    async def ping(self) -> bool:
        self.ping_calls += 1
        return self._reachable

    async def aclose(self) -> None:
        return None


#: Structural conformance checked by mypy, the same way the real tiers do it —
#: a stub that has drifted from the protocol tests nothing worth knowing.
_TIER_CHECK: Final[type[TierProtocol]] = StubTier
_CACHE_CHECK: Final[type[CacheBackend]] = StubCache


def build_app(
    settings: Settings,
    *,
    tiers: Sequence[TierProtocol] = (),
    cache: CacheBackend | None = None,
) -> FastAPI:
    """Assemble a minimal app carrying only the system routes.

    Mounted at ``/v1`` here because the router is prefix-less; in the real app
    the v1 aggregator supplies the same prefix.

    ``/v1/capabilities`` now sits behind the same auth/rate-limit dependency
    ``/v1/classify`` does, which reads a full :class:`AppState` off
    ``app.state`` — a different, parallel convention from the three named
    attributes set directly below, which ``/v1/health`` and ``/v1/ready``
    still read without one (see ``main.py``'s lifespan comment on why both
    exist). A minimal, always-open one is attached here so every test in this
    file keeps working unauthenticated, the way ``/v1/health``/``/v1/ready``
    still answer for real.
    """
    app = FastAPI()
    app.include_router(system.router, prefix="/v1")
    app.include_router(system.build_metrics_router(settings))
    app.state.settings = settings
    app.state.tiers = tuple(tiers)
    app.state.cache = cache
    AppState(
        settings=settings,
        cache=MemoryCache(),
        api_keys=ApiKeyStore([], salt="test-salt", allow_unauthenticated=True),
        rate_limiter=RateLimiter(MemoryCache(), settings),
        orchestrator=Orchestrator((), EscalationPolicy(settings), settings),
        tiers=tuple(tiers),
    ).attach(app)
    return app


async def client_for(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://sentiment.test")


@pytest.fixture
def app_settings() -> Settings:
    return Settings(app=AppSettings(env="test"))


@pytest.fixture
async def all_tiers() -> AsyncIterator[list[StubTier]]:
    yield [StubTier(tier_id) for tier_id in TierId.ordered()]


# ── Liveness ──────────────────────────────────────────────────────────────────


async def test_health_reports_ok_without_probing_any_tier(app_settings: Settings) -> None:
    """The liveness contract: alive is a statement about this process only.

    Every stub here raises on probe, so any dependency check at all turns this
    into a 500. That is the point — a liveness probe that fails when a provider
    is down gets the pod killed during someone else's incident.
    """
    tiers = [StubTier(tier_id, raises=ProbeExplodedError) for tier_id in TierId.ordered()]
    app = build_app(app_settings, tiers=tiers, cache=StubCache(reachable=False))

    async with await client_for(app) as client:
        response = await client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["service"] == app_settings.obs.service_name
    assert all(tier.probe_calls == 0 for tier in tiers)


async def test_health_works_with_nothing_attached_to_app_state() -> None:
    """A process that has not finished wiring itself is still alive."""
    app = FastAPI()
    app.include_router(system.router, prefix="/v1")

    async with await client_for(app) as client:
        response = await client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Readiness ─────────────────────────────────────────────────────────────────


async def test_ready_is_ok_when_every_component_answers(
    app_settings: Settings, all_tiers: list[StubTier]
) -> None:
    cache = StubCache()
    app = build_app(app_settings, tiers=all_tiers, cache=cache)

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert {component["name"] for component in body["components"]} == {
        *(tier_id.value for tier_id in TierId.ordered()),
        "cache",
    }
    assert all(component["ready"] for component in body["components"])
    assert cache.ping_calls == 1


async def test_ready_is_degraded_not_unready_when_only_the_cloud_tier_is_down(
    app_settings: Settings,
) -> None:
    """A cloud outage must not drain the fleet.

    Tiers 0-2 can still produce a verdict, so the service is genuinely serving.
    Reporting `not_ready` here would remove every pod from rotation at once and
    turn a quality degradation into a total outage.
    """
    tiers = [
        StubTier(TierId.T0_RULES),
        StubTier(TierId.T1_CLASSIFIER),
        StubTier(TierId.T2_LOCAL_LLM),
        StubTier(TierId.T3_CLOUD_LLM, available=False),
    ]
    app = build_app(app_settings, tiers=tiers, cache=StubCache())

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"

    failed = {c["name"]: c for c in body["components"] if not c["ready"]}
    assert set(failed) == {TierId.T3_CLOUD_LLM.value}
    # StubTier.unavailable_reason's own text, not the generic fallback — proof
    # `/v1/ready` actually asks the tier why, rather than settling for "no".
    assert failed[TierId.T3_CLOUD_LLM.value]["detail"] == "stub unavailable"


async def test_ready_is_503_when_no_tier_can_serve(app_settings: Settings) -> None:
    tiers = [StubTier(tier_id, available=False) for tier_id in TierId.ordered()]
    app = build_app(app_settings, tiers=tiers, cache=StubCache())

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert not any(c["ready"] for c in body["components"] if c["name"] != "cache")


async def test_ready_is_503_when_no_tiers_are_registered_at_all(
    app_settings: Settings,
) -> None:
    """An app whose orchestrator has no rungs cannot answer anything."""
    app = build_app(app_settings, tiers=(), cache=StubCache())

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


async def test_ready_degrades_but_stays_in_rotation_when_the_cache_is_unreachable(
    app_settings: Settings, all_tiers: list[StubTier]
) -> None:
    """The cache is an optimisation, never a readiness gate.

    `CacheBackend` is contractually allowed to miss, so an unreachable Redis
    costs latency and money but not correctness. Draining the pod for it would
    make a cache blip a service outage.
    """
    app = build_app(app_settings, tiers=all_tiers, cache=StubCache(reachable=False))

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert [c["name"] for c in body["components"] if not c["ready"]] == ["cache"]


async def test_ready_treats_a_hanging_probe_as_unavailable(
    app_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that never returns is a failed probe, not a hung endpoint."""
    monkeypatch.setattr(system, "_PROBE_TIMEOUT_S", 0.02)
    tiers = [
        StubTier(TierId.T0_RULES),
        StubTier(TierId.T3_CLOUD_LLM, delay_s=5.0),
    ]
    app = build_app(app_settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    stalled = next(c for c in body["components"] if c["name"] == TierId.T3_CLOUD_LLM.value)
    assert stalled["ready"] is False
    assert "timed out" in stalled["detail"]


async def test_ready_reports_a_raising_probe_without_leaking_its_message(
    app_settings: Settings,
) -> None:
    """Probe details name the exception type, never its message.

    A connection error renders the target it failed to reach, and a Redis URL
    can carry a password. `/v1/ready` is one of the least-guarded surfaces on
    the service, so nothing but the class name goes into the body.
    """
    tiers = [
        StubTier(TierId.T0_RULES),
        StubTier(TierId.T2_LOCAL_LLM, raises=ConnectionError),
    ]
    app = build_app(app_settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/ready")

    assert response.status_code == 200
    component = next(c for c in response.json()["components"] if c["name"] == "t2_local_llm")
    assert component["ready"] is False
    assert component["detail"] == "probe raised ConnectionError"
    assert "hunter2" not in response.text


async def test_ready_probes_components_concurrently(app_settings: Settings) -> None:
    """Four slow rungs must cost one probe's latency, not four.

    Sequential probing makes readiness latency the sum of the slowest
    dependencies, which is how a readiness check starts timing out for reasons
    unrelated to readiness.
    """
    delay = 0.1
    tiers = [StubTier(tier_id, delay_s=delay) for tier_id in TierId.ordered()]
    app = build_app(app_settings, tiers=tiers)

    started = asyncio.get_running_loop().time()
    async with await client_for(app) as client:
        response = await client.get("/v1/ready")
    elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 200
    # Generous: sequential would be >= 4 * delay, concurrent ~1 * delay.
    assert elapsed < len(tiers) * delay


# ── Capabilities ──────────────────────────────────────────────────────────────


async def test_capabilities_reports_the_live_lexicon_fallback_backend(
    app_settings: Settings,
) -> None:
    """A missing ONNX artifact must be visible from outside the process.

    The fallback keeps the service answering while quietly degrading every
    confidence number the escalation thresholds gate on, so `scorer_kind` is the
    only external evidence that it happened.
    """
    tiers = [
        StubTier(TierId.T0_RULES, model_version="rules-1.0"),
        StubTier(TierId.T1_CLASSIFIER, model_version=LEXICON_SCORER_NAME),
    ]
    app = build_app(app_settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/capabilities")

    assert response.status_code == 200
    entry = next(t for t in response.json()["tiers"] if t["tier"] == TierId.T1_CLASSIFIER.value)
    assert entry["model_version"] == LEXICON_SCORER_NAME
    assert entry["detail"]["scorer_kind"] == "lexicon"
    assert entry["detail"]["onnx_loaded"] is False
    assert entry["detail"]["scorer_backend"] == LEXICON_SCORER_NAME


async def test_capabilities_reports_a_loaded_onnx_backend(app_settings: Settings) -> None:
    tiers = [StubTier(TierId.T1_CLASSIFIER, model_version="distilbert-sst2-onnx-1.0")]
    app = build_app(app_settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/capabilities")

    entry = next(t for t in response.json()["tiers"] if t["tier"] == TierId.T1_CLASSIFIER.value)
    assert entry["detail"]["scorer_kind"] == "onnx"
    assert entry["detail"]["onnx_loaded"] is True


async def test_capabilities_lists_every_rung_including_disabled_ones(
    app_settings: Settings,
) -> None:
    """A rung switched off in config is absent by choice, not missing.

    Omitting it would make "operator disabled the cloud tier" and "the cloud
    tier failed to construct" look identical from the outside.
    """
    settings = app_settings.model_copy(
        update={
            "tiers": TiersConfig(t3_cloud_llm=TierSettings(enabled=False)),
        }
    )
    tiers = [
        StubTier(tier_id) for tier_id in TierId.ordered() if tier_id is not TierId.T3_CLOUD_LLM
    ]
    app = build_app(settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/capabilities")

    inventory = {t["tier"]: t for t in response.json()["tiers"]}
    assert list(inventory) == [tier_id.value for tier_id in TierId.ordered()]

    cloud = inventory[TierId.T3_CLOUD_LLM.value]
    assert cloud["enabled"] is False
    assert cloud["available"] is False
    assert cloud["model_version"] == "unavailable"


async def test_capabilities_marks_an_enabled_but_unreachable_rung(
    app_settings: Settings,
) -> None:
    tiers = [
        StubTier(TierId.T0_RULES),
        StubTier(TierId.T2_LOCAL_LLM, available=False, model_version="llama3.2:3b"),
    ]
    app = build_app(app_settings, tiers=tiers)

    async with await client_for(app) as client:
        response = await client.get("/v1/capabilities")

    entry = next(t for t in response.json()["tiers"] if t["tier"] == TierId.T2_LOCAL_LLM.value)
    assert entry["enabled"] is True
    assert entry["available"] is False
    assert entry["model_version"] == "llama3.2:3b"
    assert entry["detail"]["languages"] == "any"
    # Present regardless of tier health — an operator diagnosing a regression
    # needs to know which prompt was live even while the rung is down.
    assert entry["detail"]["prompt_version"] == compact_system_prompt_version()


async def test_capabilities_reports_the_limits_a_caller_must_respect(
    app_settings: Settings,
) -> None:
    settings = app_settings.model_copy(
        update={
            "app": AppSettings(env="test", max_batch_items=42),
            "classifier": ClassifierSettings(supported_languages=["en", "de"]),
        }
    )
    app = build_app(settings, tiers=[StubTier(TierId.T1_CLASSIFIER)])

    async with await client_for(app) as client:
        response = await client.get("/v1/capabilities")

    body = response.json()
    assert body["service"] == settings.obs.service_name
    assert body["version"] == __version__
    assert body["max_batch_items"] == 42
    assert body["max_text_chars"] == 20_000
    assert body["supported_languages"] == ["en", "de"]


# ── Metrics ───────────────────────────────────────────────────────────────────


async def test_metrics_returns_the_prometheus_exposition(app_settings: Settings) -> None:
    """Content type matters as much as the body: a scraper parses on it."""
    metrics.record_request(endpoint="/v1/classify", status=200, api_key_id="test-key")
    app = build_app(app_settings)

    async with await client_for(app) as client:
        response = await client.get(app_settings.obs.metrics_path)

    assert response.status_code == 200
    # Compared against the client library's own constant rather than a literal:
    # the exposition version travels in the content type and a hard-coded copy
    # would fail on a `prometheus-client` upgrade that changed nothing here.
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert response.headers["content-type"].startswith("text/plain")
    assert "air_classifier_requests_total" in response.text


async def test_metrics_is_not_namespaced_under_v1(app_settings: Settings) -> None:
    """Scrape config points at a fixed path; API versioning must not move it."""
    app = build_app(app_settings)

    async with await client_for(app) as client:
        assert (await client.get("/metrics")).status_code == 200
        assert (await client.get("/v1/metrics")).status_code == 404


async def test_metrics_honours_a_configured_path(app_settings: Settings) -> None:
    settings = app_settings.model_copy(
        update={"obs": app_settings.obs.model_copy(update={"metrics_path": "/internal/metrics"})}
    )
    app = build_app(settings)

    async with await client_for(app) as client:
        assert (await client.get("/internal/metrics")).status_code == 200
        assert (await client.get("/metrics")).status_code == 404


async def test_metrics_route_is_absent_when_disabled(app_settings: Settings) -> None:
    """Absent rather than empty: a flat line reads as "no traffic", not "off"."""
    settings = app_settings.model_copy(
        update={"obs": app_settings.obs.model_copy(update={"metrics_enabled": False})}
    )
    app = build_app(settings)

    async with await client_for(app) as client:
        assert (await client.get("/metrics")).status_code == 404


async def test_metrics_is_absent_from_the_openapi_document(app_settings: Settings) -> None:
    """The exposition is line-oriented text; OpenAPI would describe it wrongly."""
    app = build_app(app_settings)

    async with await client_for(app) as client:
        schema = (await client.get("/openapi.json")).json()

    assert app_settings.obs.metrics_path not in schema["paths"]
    assert "/v1/health" in schema["paths"]
