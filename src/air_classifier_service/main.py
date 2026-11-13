"""The application factory — where every part of the service is joined up.

Two properties of this module matter more than the wiring itself.

**The factory takes its settings as an argument.** ``create_app(settings)``
never reaches for the environment on its own, so a test can assemble a complete
application — real tiers, real middleware, real error handlers — without
exporting a variable or clearing a cache. The process singleton is only the
default.

**A missing dependency is a deployment shape, not a failure.** The escalation
ladder exists so that a cheaper rung answers when an expensive one cannot, and
that promise has to hold at *startup* as well as per request: a deployment with
air-model-gateway entirely unreachable must boot, serve, and report itself honestly on
``/v1/ready`` and ``/v1/capabilities``. So every rung is built
inside :func:`_tier_guard`, which turns an unconfigured or broken dependency
into a warning and a shorter ladder. The only thing that can stop this service
from starting is something every rung needs — settings that will not validate,
or an API key file that will not parse.

Startup order is not arbitrary: logging comes up before tracing (which logs its
own status line), the cache before the rate limiter that stores buckets in it,
and the tiers before the orchestrator that runs them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

import orjson
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from air_classifier_service import __version__
from air_classifier_service.api.deps import AppState
from air_classifier_service.api.errors import install_error_handlers
from air_classifier_service.api.middleware import install_middleware
from air_classifier_service.api.v1.router import build_v1_router
from air_classifier_service.api.v1.system import (
    STATE_CACHE,
    STATE_SETTINGS,
    STATE_TIERS,
    build_metrics_router,
)
from air_classifier_service.audit.base import AuditSink
from air_classifier_service.audit.factory import build_audit_sink
from air_classifier_service.cache.base import CacheBackend
from air_classifier_service.cache.factory import build_cache
from air_classifier_service.config import Settings, get_settings
from air_classifier_service.constants import Provider
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.observability.logging import configure_logging, get_logger
from air_classifier_service.observability.tracing import configure_tracing, instrument_app
from air_classifier_service.pipeline.orchestrator import Orchestrator
from air_classifier_service.pipeline.policy import EscalationPolicy
from air_classifier_service.pipeline.service import AnalysisService
from air_classifier_service.pipeline.summary_service import SummaryService
from air_classifier_service.pipeline.tiers.base import Tier as TierProtocol
from air_classifier_service.pipeline.tiers.t0_rules import RulesTier
from air_classifier_service.pipeline.tiers.t1_classifier import ClassifierTier
from air_classifier_service.pipeline.tiers.t2_local_llm import LocalLLMTier
from air_classifier_service.pipeline.tiers.t3_cloud_llm import CloudLLMTier
from air_classifier_service.providers.base import LLMProvider
from air_classifier_service.providers.model_gateway_provider import ModelGatewayProvider
from air_classifier_service.security.api_keys import ApiKeyStore
from air_classifier_service.security.quotas import RateLimiter

__all__ = ["Ladder", "OrjsonResponse", "build_ladder", "create_app"]

_log = get_logger(__name__)

#: Attribute on ``app.state`` holding the process-wide :class:`AnalysisService`.
#: Routes build their own per request (see ``api/v1/router.get_analysis_service``);
#: this one is for anything that needs the ladder *outside* a request — a warmup
#: probe, a management command — without re-deriving the wiring.
STATE_ANALYSIS_SERVICE: Final[str] = "analysis_service"

_TITLE: Final[str] = "AIR Classifier Service"

_DESCRIPTION: Final[str] = """
Layered text classification with cost-aware escalation.

One entry point, `/v1/classify`, and one detailed response: sentiment, tone,
topics, intent, urgency, routing, a requires-human flag, aspect-level scoring
and rating consistency — whichever of those the text and the ladder's reach
actually support.

Every request enters at the cheapest rung of a four-tier ladder — deterministic
rules, a statistical classifier, a locally served LLM, a metered cloud LLM — and
climbs only as far as the confidence, margin and complexity of the input
actually require. The rung that decided is reported as `decided_by`, and the
reasoning is in `escalation_trace`.

A rung being unavailable is not an error. When a tier fails or a provider is
down, a lower rung answers and the response carries `degraded: true` with HTTP
200. The only failure the ladder can surface is `503 no-verdict`, which means
every enabled rung declined.

Errors are RFC 9457 `application/problem+json`.
""".strip()


# ── Serialisation ─────────────────────────────────────────────────────────────


class OrjsonResponse(JSONResponse):
    """``application/json`` rendered by orjson rather than the stdlib encoder.

    Written here rather than imported from ``fastapi.responses``: FastAPI ships
    the same class but deprecated it in 0.141, so using it would emit a warning
    on every single response.

    One trade-off is worth knowing before changing this. Setting *any* custom
    ``default_response_class`` opts the whole app out of FastAPI's newer fast
    path, which serialises a response model straight to bytes through pydantic's
    Rust core and skips the intermediate ``jsonable_encoder`` dict. Naming a
    class here therefore buys a uniform encoder — orjson also renders the routes
    that return a bare ``Response``, such as an idempotent replay — at the cost
    of that shortcut. If response latency ever shows up in the encoder, dropping
    this argument entirely is the fix.
    """

    def render(self, content: Any) -> bytes:
        # Non-string keys: `metadata` and `context` are caller-defined maps, and
        # a JSON object key that arrived as an int must not raise on the way out.
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)


# ── The ladder ────────────────────────────────────────────────────────────────


class _TierUnconfiguredError(Exception):
    """A rung's dependency is absent, so the rung is not built.

    Deliberately distinct from a build *failure*: "air-model-gateway is not configured"
    is a supported deployment, "the classifier raised while loading" is a
    defect, and an operator reading the boot log needs to tell those two
    apart at a glance.
    """


@dataclass(frozen=True, slots=True)
class Ladder:
    """Everything :func:`build_ladder` managed to construct.

    ``providers`` is kept separately from the tiers that use them because
    shutdown closes providers, and a provider that was built but whose tier was
    skipped still owns a connection pool. ``summary_provider`` is not a ladder
    rung at all — it backs the customer-summary narrative refresh
    (`pipeline/summary_service.py`) — but it is built the same non-fatal way
    and is already included in ``providers`` for shutdown.
    """

    tiers: tuple[TierProtocol, ...] = ()
    providers: tuple[LLMProvider, ...] = field(default_factory=tuple)
    summary_provider: LLMProvider | None = None


@contextmanager
def _tier_guard(component: str) -> Iterator[None]:
    """Let one optional component fail to build without taking the service down.

    This is the single most important behaviour in this module. Without it, a
    revoked air-model-gateway token or air-model-gateway itself being unreachable would turn a
    service designed to degrade into one that cannot start — and a crash loop
    at boot is strictly worse than a shorter ladder, because a shorter ladder
    still answers. Named generically rather than after :class:`TierId` because
    it also guards the customer-summary narrative provider, which is not a
    ladder rung — callers pass `.value` for an actual tier and a plain string
    for anything else.

    The exception's message is logged but never reaches a caller: it can carry a
    file path or a base URL, and ``/v1/capabilities`` already reports the
    outcome — which rungs exist — without reporting the cause.
    """
    try:
        yield
    except _TierUnconfiguredError as exc:
        _log.warning("startup.component_skipped", component=component, reason=str(exc))
    except Exception as exc:
        _log.warning(
            "startup.component_build_failed",
            component=component,
            error=type(exc).__name__,
            detail=str(exc),
        )


def build_ladder(settings: Settings) -> Ladder:
    """Construct every enabled rung whose dependencies are actually present.

    Rungs disabled in configuration are never constructed, so their absence from
    ``Ladder.tiers`` and ``tiers.<rung>.enabled = false`` are the same fact seen
    from two sides — which is exactly what ``/v1/capabilities`` reports.
    """
    enabled = set(settings.enabled_tiers())
    tiers: list[TierProtocol] = []
    providers: list[LLMProvider] = []
    summary_provider: LLMProvider | None = None

    if TierId.T0_RULES in enabled:
        with _tier_guard(TierId.T0_RULES.value):
            tiers.append(RulesTier(settings))

    if TierId.T1_CLASSIFIER in enabled:
        with _tier_guard(TierId.T1_CLASSIFIER.value):
            # `build_scorer` already falls back to the lexicon when the ONNX
            # artifact is missing, so this rung effectively cannot be skipped —
            # which is why a deployment with nothing else configured still has a
            # ladder two rungs deep.
            tiers.append(ClassifierTier(settings))

    if TierId.T2_LOCAL_LLM in enabled:
        with _tier_guard(TierId.T2_LOCAL_LLM.value):
            local = ModelGatewayProvider(settings, Provider.OLLAMA)
            # Constructing the provider does no I/O, so `configured` here means
            # "switched on", not "reachable". Reachability is a readiness
            # question, and the rung is built either way so that air-model-gateway
            # coming up after this service does not need a restart to be used.
            if not local.configured:
                raise _TierUnconfiguredError("model_gateway is disabled in configuration")
            providers.append(local)
            tiers.append(LocalLLMTier(settings, local))

    if TierId.T3_CLOUD_LLM in enabled:
        with _tier_guard(TierId.T3_CLOUD_LLM.value):
            cloud = ModelGatewayProvider(settings, Provider.ANTHROPIC)
            # Appended before the credential check so that shutdown still closes
            # a provider whose HTTP client was built but whose tier was skipped.
            providers.append(cloud)
            if not cloud.configured:
                raise _TierUnconfiguredError("model_gateway is disabled in configuration")
            tiers.append(CloudLLMTier(settings, cloud))

    # Not a ladder rung — the customer-summary narrative refresh
    # (pipeline/summary_service.py) — but built the same non-fatal way and its
    # own instance rather than reusing T3's, so its bulkhead/breaker are keyed
    # independently (see SummaryService) rather than sharing T3's fault state.
    with _tier_guard("summary_refresh"):
        summary = ModelGatewayProvider(settings, Provider.ANTHROPIC)
        providers.append(summary)
        if not summary.configured:
            raise _TierUnconfiguredError("model_gateway is disabled in configuration")
        summary_provider = summary

    _log.info(
        "startup.ladder_built",
        tiers=[tier.tier.value for tier in tiers],
        # Only the ones with credentials — a provider built without any is an
        # implementation detail of the check above. Named `configured`, not
        # `providers` or `available`: `configured` is a no-I/O settings check
        # (ModelGatewayProvider.configured's own docstring), not a reachability
        # probe, and this line runs once at boot before anything has actually
        # been dialled. Listing it here as if it were verified would read as
        # "air-model-gateway is available" at exactly the moment nobody has checked —
        # `/v1/ready` and `/v1/capabilities` are the honest, continuously
        # re-checked signal for that, not this line.
        configured=[p.name.value for p in providers if p.configured],
    )
    return Ladder(tiers=tuple(tiers), providers=tuple(providers), summary_provider=summary_provider)


# ── Lifespan ──────────────────────────────────────────────────────────────────


def _configure_observability(settings: Settings) -> None:
    """Bring up logging, then tracing.

    Order is load-bearing: ``configure_tracing`` emits its own status line, and
    a line emitted before the handler is installed is unstructured output in the
    middle of a JSON log stream. Both calls are idempotent, which is what lets
    the factory and the lifespan each make them.
    """
    configure_logging(settings)
    configure_tracing(settings)


def _lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the startup/shutdown context manager bound to ``settings``.

    A closure rather than a module-level function because the lifespan signature
    hands over the app but not the settings, and reading them from the process
    singleton here would quietly undo the whole point of an injectable factory.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_observability(settings)
        _log.info("startup.begin", env=settings.app.env, version=__version__)

        cache = await build_cache(settings)
        audit = build_audit_sink(settings)
        await audit.start()
        api_keys = ApiKeyStore.load(settings)
        # The limiter's buckets live in the cache, so it cannot precede it.
        rate_limiter = RateLimiter(cache, settings)

        ladder = build_ladder(settings)
        orchestrator = Orchestrator(ladder.tiers, EscalationPolicy(settings), settings)
        analysis_service = AnalysisService(orchestrator, settings, cache, audit)
        # Always constructed, even when the narrative provider is not — only
        # the narrative-refresh call inside it degrades. See
        # pipeline/summary_service.py.
        summary_service = SummaryService(analysis_service, ladder.summary_provider, settings)

        AppState(
            settings=settings,
            cache=cache,
            audit=audit,
            api_keys=api_keys,
            rate_limiter=rate_limiter,
            orchestrator=orchestrator,
            tiers=ladder.tiers,
            summary_service=summary_service,
        ).attach(app)

        # `api/v1/system.py` reads these three off `app.state` by name rather
        # than through `AppState`, so that its routes still answer in a test app
        # assembled without a lifespan. Published under the constants that
        # module exports, so a rename cannot leave the two sides disagreeing.
        setattr(app.state, STATE_SETTINGS, settings)
        setattr(app.state, STATE_TIERS, ladder.tiers)
        setattr(app.state, STATE_CACHE, cache)
        setattr(app.state, STATE_ANALYSIS_SERVICE, analysis_service)

        _log.info(
            "startup.complete",
            tiers=[tier.tier.value for tier in ladder.tiers],
            cache_backend=settings.cache.backend,
            audit_backend=settings.audit.backend,
            summary_refresh_configured=ladder.summary_provider is not None,
            api_keys=len(api_keys),
            allow_unauthenticated=settings.security.allow_unauthenticated,
            docs_enabled=settings.app.docs_enabled,
        )
        try:
            yield
        finally:
            await _shutdown(settings, cache=cache, audit=audit, providers=ladder.providers)

    return lifespan


async def _shutdown(
    settings: Settings,
    *,
    cache: CacheBackend,
    audit: AuditSink,
    providers: Sequence[LLMProvider],
) -> None:
    """Release every connection pool, bounded by the grace period.

    The bound matters: an unreachable Redis or a hung HTTP pool can make
    ``aclose`` wait, and a pod that will not exit is eventually killed with
    SIGKILL — which drops the in-flight requests that a graceful shutdown was
    supposed to protect. Timing out here is the lesser loss, so it is a warning
    and not an exception.
    """
    grace = settings.app.shutdown_grace_seconds
    _log.info("shutdown.begin", grace_seconds=grace)
    try:
        # A zero grace disables the bound rather than skipping the close:
        # closing pools is not the same as draining traffic, and leaking sockets
        # on the way out is never what an operator meant by "do not wait".
        async with asyncio.timeout(grace if grace > 0 else None):
            await _close_all(cache=cache, audit=audit, providers=providers)
    except TimeoutError:
        _log.warning("shutdown.grace_exceeded", grace_seconds=grace)
    _log.info("shutdown.complete")


async def _close_all(
    *, cache: CacheBackend, audit: AuditSink, providers: Sequence[LLMProvider]
) -> None:
    """Close the providers concurrently, then the cache and the audit sink.

    ``return_exceptions`` so one backend that fails to close does not leave the
    others open — at this point in the process's life there is nothing left to
    do about a failure except record it.
    """
    outcomes = await asyncio.gather(
        *(provider.aclose() for provider in providers), return_exceptions=True
    )
    for provider, outcome in zip(providers, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            _log.warning(
                "shutdown.provider_close_failed",
                provider=provider.name.value,
                error=type(outcome).__name__,
            )

    try:
        await cache.aclose()
    except Exception as exc:
        _log.warning("shutdown.cache_close_failed", error=type(exc).__name__)

    try:
        await audit.aclose()
    except Exception as exc:
        _log.warning("shutdown.audit_close_failed", error=type(exc).__name__)


# ── Factory ───────────────────────────────────────────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    """Assemble the application.

    Everything that has to happen before the first request happens here, because
    Starlette freezes its middleware stack the moment the app is first called —
    including ``instrument_app``, whose OpenTelemetry middleware would be
    rejected if it were added from the lifespan.
    """
    resolved = settings if settings is not None else get_settings()
    # Also called by the lifespan; both are idempotent. It has to happen here as
    # well because `instrument_app` below is a no-op unless a tracer already
    # exists, and by the time the lifespan runs it is too late to instrument.
    _configure_observability(resolved)

    docs = resolved.app.docs_enabled
    app = FastAPI(
        title=_TITLE,
        version=__version__,
        description=_DESCRIPTION,
        # All three are gated together: `/docs` without `/openapi.json` renders
        # an empty page, which looks like a broken deployment rather than a
        # deliberately closed one.
        openapi_url="/openapi.json" if docs else None,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        root_path=resolved.app.root_path,
        # Responses carry per-tier traces and float-heavy verdicts; orjson
        # serialises both several times faster than the stdlib encoder.
        default_response_class=OrjsonResponse,
        lifespan=_lifespan(resolved),
    )

    install_middleware(app, resolved)
    install_error_handlers(app)
    app.include_router(build_v1_router())
    # Outside `/v1` and outside auth, at a configured path — see
    # `build_metrics_router` for why that is safe only on an internal network.
    app.include_router(build_metrics_router(resolved))
    instrument_app(app)
    return app
