"""Operational endpoints: liveness, readiness, capabilities, and the metrics scrape.

These four are the routes an operator, a kubelet or a Prometheus scraper
touches, and each answers a different question:

* ``/v1/health`` — *is this process alive?* Nothing else. It deliberately holds
  no dependency on tiers, cache, or providers, because a liveness probe that
  fails on a downstream outage gets the pod restarted during somebody else's
  incident — turning a partial degradation into a rolling restart storm.
  Unauthenticated: a kubelet carries no credential.
* ``/v1/ready`` — *should this pod receive traffic?* It probes what it depends
  on, but grades the result on the ladder's own terms: an upper rung being down
  is ``degraded``, not ``not_ready``, because the whole point of the escalation
  ladder is that a lower rung still answers. Only "no tier at all can serve"
  justifies pulling the pod out of rotation. Unauthenticated, for the same
  reason as ``/v1/health``.
* ``/v1/capabilities`` — *what is this deployment actually capable of right now?*
  Notably which Tier-1 scorer backend is live, since a missing ONNX artifact
  degrades answer quality silently and is otherwise invisible from outside.
  Unlike its two siblings above, this is operator detail rather than a
  load-balancer boolean — it discloses configured model classes and
  per-tier availability — so it is authenticated and rate-limited exactly like
  ``/v1/classify``, not open the way a liveness probe has to be.
* ``/metrics`` — the Prometheus exposition, mounted outside ``/v1`` and outside
  auth.

The three ``/v1`` routes are declared on a prefix-less router; the v1 aggregator
supplies ``/v1``. The metrics route is built separately by
:func:`build_metrics_router` because its path is configuration
(``obs.metrics_path``), not a literal.

Dependencies are read off ``app.state`` rather than imported as module globals so
that the app factory owns object lifetimes and tests can assemble a router with
stubs and no lifespan at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, cast

from fastapi import APIRouter, Depends, Request, Response, status

from air_classifier_service import __version__
from air_classifier_service.api.deps import RateLimitedKeyDep
from air_classifier_service.cache.base import CacheBackend
from air_classifier_service.config import Settings, get_settings
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.observability import metrics
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.pipeline.tiers.base import Tier as TierProtocol
from air_classifier_service.pipeline.tiers.scorers import LEXICON_SCORER_NAME
from air_classifier_service.providers.prompts.loader import (
    compact_system_prompt_version,
    system_prompt_version,
)
from air_classifier_service.schemas.classify import MAX_TEXT_CHARS
from air_classifier_service.schemas.common import CapabilitiesResponse, TierCapability
from air_classifier_service.schemas.errors import ProblemDetail
from air_classifier_service.schemas.responses import HealthResponse, ReadyComponent, ReadyResponse

__all__ = [
    "STATE_CACHE",
    "STATE_SETTINGS",
    "STATE_TIERS",
    "build_metrics_router",
    "get_app_settings",
    "get_cache_backend",
    "get_registered_tiers",
    "router",
]

_log = get_logger(__name__)

# ── app.state contract ────────────────────────────────────────────────────────
#
# Named constants rather than string literals scattered through the handlers, so
# the app factory and this module cannot drift apart without a rename that hits
# both. Every one of them is optional: the routes degrade to a sensible answer
# rather than raising, because a 500 from the readiness endpoint tells an
# operator nothing about the service it was asked about.

STATE_SETTINGS: Final[str] = "settings"
STATE_TIERS: Final[str] = "tiers"
STATE_CACHE: Final[str] = "cache"

# ── Readiness vocabulary ──────────────────────────────────────────────────────

#: Everything the pod depends on answered.
STATUS_OK: Final[str] = "ok"
#: Something is down, but a rung of the ladder can still produce a verdict.
STATUS_DEGRADED: Final[str] = "degraded"
#: No tier can serve. This is the only state worth removing the pod for.
STATUS_NOT_READY: Final[str] = "not_ready"

#: Ceiling on a *single* probe. Probes run concurrently, so the endpoint's worst
#: case is one probe rather than their sum — but a readiness handler that can
#: outlive the kubelet's own `timeoutSeconds` reports "unhealthy" by timing out
#: instead of by answering, which loses the degraded/not-ready distinction this
#: endpoint exists to draw. Set the kubelet timeout comfortably above this.
#:
#: 1.0s until the air-model-gateway cutover, when it was still sized for T2/T3's own
#: direct-SDK or same-host provider calls. Reaching air-model-gateway over
#: `host.docker.internal` (docker-compose.yml) is a real extra hop, not a
#: same-process call, and measured 1.2-1.5s round trip on Docker Desktop —
#: the *network*, not air-model-gateway being slow or unhealthy. The old ceiling made
#: T2/T3 report `available: false` on a perfectly healthy deployment; this one
#: clears the measured max with the same ~2x-headroom margin every other
#: measured timeout in this service uses (docs/02-lld.md's changelog), and
#: comfortably exceeds ModelGatewayProvider's own `_HEALTH_TIMEOUT_S` (2.0s) so that
#: timeout gets to be the one that actually fires.
_PROBE_TIMEOUT_S: Final[float] = 3.0

#: Component name for the cache probe. Tier components use `Tier.value`.
_CACHE_COMPONENT: Final[str] = "cache"

#: Reported for a rung that is switched off and therefore never constructed:
#: there is no loaded artifact to name a version for.
_NO_MODEL_VERSION: Final[str] = "unavailable"

router = APIRouter(tags=["system"])


# ── Dependencies ──────────────────────────────────────────────────────────────


def get_app_settings(request: Request) -> Settings:
    """Settings the app was built with.

    Falls back to the process singleton so the router is usable in a bare test
    app that never ran a lifespan.
    """
    settings = cast("Settings | None", getattr(request.app.state, STATE_SETTINGS, None))
    return settings if settings is not None else get_settings()


def get_registered_tiers(request: Request) -> Sequence[TierProtocol]:
    """The tier instances the orchestrator was built with, cheapest first.

    Only *enabled* rungs are constructed, so absence from this sequence and
    ``enabled=False`` in settings are the same fact seen from two sides.
    """
    tiers = cast("Sequence[TierProtocol] | None", getattr(request.app.state, STATE_TIERS, None))
    return tiers if tiers is not None else ()


def get_cache_backend(request: Request) -> CacheBackend | None:
    """The live cache backend, or None when the app has not attached one."""
    return cast("CacheBackend | None", getattr(request.app.state, STATE_CACHE, None))


# ── Probing ───────────────────────────────────────────────────────────────────


async def _probe(
    name: str,
    probe: Callable[[], Awaitable[bool]],
    timeout_s: float,
    *,
    reason: Callable[[], Awaitable[str | None]] | None = None,
) -> ReadyComponent:
    """Run one availability probe, converting every failure mode into a verdict.

    Both protocols involved promise not to raise (`Tier.is_available`,
    `CacheBackend.ping`), and readiness is precisely the wrong place to take
    that promise on trust — a probe that throws is exactly the symptom this
    endpoint exists to report.

    The `detail` on a *raised* exception carries the exception type, never its
    message: a connection error from Redis renders the URL it failed to reach,
    and that URL may carry a password. The full exception goes to the log,
    which is already an authenticated surface. `reason` is different — it is
    `Tier.unavailable_reason`, which is written to return an operator-safe
    string already (no secrets, no raw exception text) — see its own
    docstring — so it is used verbatim when a caller supplies one; a bare
    "reported unavailable" only remains for the callers that do not (the
    cache probe has no equivalent to ask).
    """
    try:
        available = await asyncio.wait_for(probe(), timeout_s)
    except TimeoutError:
        return ReadyComponent(name=name, ready=False, detail=f"probe timed out after {timeout_s}s")
    except Exception as exc:
        _log.warning("readiness.probe_raised", component=name, error=type(exc).__name__)
        return ReadyComponent(name=name, ready=False, detail=f"probe raised {type(exc).__name__}")

    if available:
        return ReadyComponent(name=name, ready=True, detail=None)

    detail = "reported unavailable"
    if reason is not None:
        try:
            specific = await asyncio.wait_for(reason(), timeout_s)
        except Exception as exc:
            _log.debug("readiness.reason_probe_raised", component=name, error=type(exc).__name__)
            specific = None
        if specific:
            detail = specific
    return ReadyComponent(name=name, ready=False, detail=detail)


async def _probe_tiers(tiers: Sequence[TierProtocol]) -> list[ReadyComponent]:
    """Probe every rung at once.

    Concurrently because these are independent network round trips: probing four
    rungs in sequence makes the readiness endpoint's latency the *sum* of the
    slowest dependencies, which is how a readiness check starts timing out for
    reasons that have nothing to do with readiness.
    """
    return list(
        await asyncio.gather(
            *(
                _probe(
                    tier.tier.value,
                    tier.is_available,
                    _PROBE_TIMEOUT_S,
                    reason=tier.unavailable_reason,
                )
                for tier in tiers
            )
        )
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Reports that the process is up. Never probes tiers or dependencies.",
)
async def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Liveness.

    The absence of any dependency in this signature is the feature. Reading the
    service name off settings is an attribute access, not a probe; nothing here
    can fail because something downstream did.
    """
    return HealthResponse(status=STATUS_OK, service=settings.obs.service_name, version=__version__)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadyResponse,
            "description": "No tier can serve; the pod should leave rotation.",
        }
    },
    description=(
        "Probes every enabled tier and the cache backend. Returns 200 for both "
        "`ok` and `degraded`; 503 only when no tier can produce a verdict."
    ),
)
async def ready(
    response: Response,
    tiers: Sequence[TierProtocol] = Depends(get_registered_tiers),
    cache: CacheBackend | None = Depends(get_cache_backend),
) -> ReadyResponse:
    """Readiness, graded on the ladder's terms.

    The grading is the whole content of this endpoint. A cloud provider outage
    takes Tier 3 down while Tiers 0-2 keep answering; reporting `not_ready` for
    that would drain every pod in the fleet simultaneously and convert a quality
    degradation into a total outage. So:

    * any tier available → at worst `degraded`, HTTP 200;
    * no tier available → `not_ready`, HTTP 503.

    The cache is never a readiness gate for the same reason it never fails a
    request: it is an optimisation, and `CacheBackend` is contractually allowed
    to miss. An unreachable cache costs latency and money, so it shows up as
    `degraded` — visible, alertable, and not traffic-affecting.
    """
    tier_components = await _probe_tiers(tiers)
    components = list(tier_components)
    if cache is not None:
        components.append(await _probe(_CACHE_COMPONENT, cache.ping, _PROBE_TIMEOUT_S))

    can_serve = any(component.ready for component in tier_components)
    if not can_serve:
        ready_status = STATUS_NOT_READY
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif all(component.ready for component in components):
        ready_status = STATUS_OK
    else:
        ready_status = STATUS_DEGRADED

    if ready_status != STATUS_OK:
        # The body is the operator-facing detail; this line is what a log-based
        # alert fires on, since a probe response is not retained anywhere.
        _log.warning(
            "readiness.not_ok",
            status=ready_status,
            failed=[component.name for component in components if not component.ready],
        )

    # Note this 503 is a ReadyResponse, not an RFC 9457 problem: the catalogue's
    # 503 (`no-verdict`) describes a failed analysis call, and a probe endpoint's
    # consumers — kubelet and operators — want the component list, not a problem
    # document. The status code is the part Kubernetes reads either way.
    return ReadyResponse(status=ready_status, components=components)


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Tier inventory and service limits",
    description=(
        "What this deployment can do right now: which rungs are enabled and "
        "available, their model versions, and the request limits in force. "
        "Authenticated and rate-limited like `/v1/classify` — unlike liveness "
        "and readiness, this is deployment detail for an operator, not a "
        "boolean a load balancer or kubelet needs to probe."
    ),
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ProblemDetail,
            "description": "No `X-API-Key` header was supplied.",
        },
        status.HTTP_403_FORBIDDEN: {
            "model": ProblemDetail,
            "description": "The key is unknown, malformed or revoked.",
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "model": ProblemDetail,
            "description": "Quota exhausted for this key. Carries `Retry-After`.",
        },
    },
)
async def capabilities(
    key: RateLimitedKeyDep,
    settings: Settings = Depends(get_app_settings),
    tiers: Sequence[TierProtocol] = Depends(get_registered_tiers),
) -> CapabilitiesResponse:
    """Capability inventory.

    Reported over the full ladder rather than only the constructed rungs, so a
    tier switched off in config is visibly *absent by choice* rather than
    missing from the list and indistinguishable from one that failed to load.

    Authenticated and rate-limited, unlike its two siblings above: it discloses
    configured model classes, which Tier-1 scorer backend actually loaded,
    and per-tier availability — real deployment detail, not the liveness/
    readiness boolean a load balancer or kubelet needs to probe without
    credentials. `key` is otherwise unused here; the dependency's job is the
    401/403/429 it can raise before the handler body runs.
    """
    instances = {tier.tier: tier for tier in tiers}
    probes = {component.name: component for component in await _probe_tiers(tiers)}

    inventory = [
        TierCapability(
            tier=tier_id,
            enabled=settings.tiers.for_tier(tier_id).enabled,
            available=probes[tier_id.value].ready if tier_id.value in probes else False,
            model_version=instance.model_version if instance else _NO_MODEL_VERSION,
            detail=_tier_detail(tier_id, instance, settings),
        )
        for tier_id, instance in (
            (candidate, instances.get(candidate)) for candidate in TierId.ordered()
        )
    ]

    return CapabilitiesResponse(
        service=settings.obs.service_name,
        version=__version__,
        tiers=inventory,
        # The languages the service is *tuned* for, which is the classifier's
        # set. The LLM rungs accept anything (they are prompted, not trained),
        # but advertising "every language" on the strength of a rung that may be
        # disabled or unreachable would promise quality nothing here assures.
        # Per-rung reach is in each entry's `detail.languages`.
        supported_languages=list(settings.classifier.supported_languages),
        max_batch_items=settings.app.max_batch_items,
        max_text_chars=MAX_TEXT_CHARS,
    )


def _tier_detail(
    tier_id: TierId, instance: TierProtocol | None, settings: Settings
) -> dict[str, Any]:
    """Per-rung extras, most importantly Tier 1's live scorer backend.

    The ONNX artifact is optional and `build_scorer` falls back to the lexicon
    when it fails to load. That fallback is a permanent, silent accuracy loss:
    the service keeps answering and the dashboards keep moving, only the
    confidence numbers get worse — and confidence is what every escalation
    threshold gates on. Naming the live backend here is what lets an operator
    detect it from outside the process instead of from a boot log nobody reads.

    `ClassifierTier` sets its `model_version` to the scorer's own name, so the
    version string *is* the backend identity; `LEXICON_SCORER_NAME` is the only
    value that means "the model artifact did not load".
    """
    if instance is None:
        return {}

    if tier_id is TierId.T1_CLASSIFIER:
        backend = instance.model_version
        return {
            "scorer_backend": backend,
            "scorer_kind": "lexicon" if backend == LEXICON_SCORER_NAME else "onnx",
            "onnx_loaded": backend != LEXICON_SCORER_NAME,
            "languages": list(settings.classifier.supported_languages),
        }

    if tier_id is TierId.T2_LOCAL_LLM:
        return {"languages": "any", "prompt_version": compact_system_prompt_version()}

    if tier_id is TierId.T3_CLOUD_LLM:
        return {"languages": "any", "prompt_version": system_prompt_version()}

    return {}


# ── Metrics ───────────────────────────────────────────────────────────────────


def build_metrics_router(settings: Settings) -> APIRouter:
    """Router carrying the Prometheus scrape endpoint at ``obs.metrics_path``.

    Built rather than declared because the path is configuration. It is
    deliberately *not* under ``/v1`` and deliberately *not* behind API-key auth:
    scrapers do not carry credentials, and a metrics endpoint that 401s is a
    metrics endpoint nobody has.

    That makes network placement the only access control it has. **In production
    this must be bound to an internal port or an internal network** — the
    exposition carries per-key request counts and cost totals, which is a map of
    who your customers are and how much they spend.
    """
    metrics_router = APIRouter()

    if not settings.obs.metrics_enabled:
        # An endpoint that exists and returns nothing looks to a scraper like a
        # service with no traffic. Absent is honest; the scrape target fails
        # loudly instead of reporting a flat line.
        return metrics_router

    @metrics_router.get(
        settings.obs.metrics_path,
        # The exposition format is line-oriented text, not JSON. Advertising it
        # in the OpenAPI document would describe it wrongly to every generated
        # client that read it.
        include_in_schema=False,
        summary="Prometheus exposition",
    )
    async def prometheus_metrics() -> Response:
        payload, content_type = metrics.render_latest()
        return Response(content=payload, media_type=content_type)

    return metrics_router
