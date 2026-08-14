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
no Anthropic key and no Ollama daemon must boot, serve, and report itself
honestly on ``/v1/ready`` and ``/v1/capabilities``. So every rung is built
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

from air_sentiment import __version__
from air_sentiment.api.deps import AppState
from air_sentiment.api.errors import install_error_handlers
from air_sentiment.api.middleware import install_middleware
from air_sentiment.api.v1.router import build_v1_router
from air_sentiment.api.v1.system import (
    STATE_CACHE,
    STATE_SETTINGS,
    STATE_TIERS,
    build_metrics_router,
)
from air_sentiment.cache.base import CacheBackend
from air_sentiment.cache.factory import build_cache
from air_sentiment.config import Settings, get_settings
from air_sentiment.constants import Provider
from air_sentiment.constants import Tier as TierId
from air_sentiment.observability.logging import configure_logging, get_logger
from air_sentiment.observability.tracing import configure_tracing, instrument_app
from air_sentiment.pipeline.orchestrator import Orchestrator
from air_sentiment.pipeline.policy import EscalationPolicy
from air_sentiment.pipeline.service import AnalysisService
from air_sentiment.pipeline.tiers.base import Tier as TierProtocol
from air_sentiment.pipeline.tiers.t0_rules import RulesTier
from air_sentiment.pipeline.tiers.t1_classifier import ClassifierTier
from air_sentiment.pipeline.tiers.t2_local_llm import LocalLLMTier
from air_sentiment.pipeline.tiers.t3_cloud_llm import CloudLLMTier
from air_sentiment.providers.anthropic_provider import AnthropicProvider
from air_sentiment.providers.base import LLMProvider
from air_sentiment.providers.infra_provider import InfraGatewayProvider
from air_sentiment.providers.ollama_provider import OllamaProvider
from air_sentiment.providers.openai_provider import OpenAIProvider
from air_sentiment.providers.registry import ProviderRegistry
from air_sentiment.resilience.breaker import CircuitBreaker
from air_sentiment.security.api_keys import ApiKeyStore
from air_sentiment.security.quotas import RateLimiter

__all__ = ["Ladder", "OrjsonResponse", "build_ladder", "create_app"]

_log = get_logger(__name__)

#: Attribute on ``app.state`` holding the process-wide :class:`AnalysisService`.
#: Routes build their own per request (see ``api/v1/router.get_analysis_service``);
#: this one is for anything that needs the ladder *outside* a request — a warmup
#: probe, a management command — without re-deriving the wiring.
STATE_ANALYSIS_SERVICE: Final[str] = "analysis_service"

_TITLE: Final[str] = "AIR Sentiment"

_DESCRIPTION: Final[str] = """
Layered sentiment analysis with cost-aware escalation.

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

    Deliberately distinct from a build *failure*: "no Anthropic key" is a
    supported deployment, "the classifier raised while loading" is a defect, and
    an operator reading the boot log needs to tell those two apart at a glance.
    """


@dataclass(frozen=True, slots=True)
class Ladder:
    """Everything :func:`build_ladder` managed to construct.

    ``providers`` is kept separately from the tiers that use them because
    shutdown closes providers, and a provider that was built but whose tier was
    skipped still owns a connection pool.
    """

    tiers: tuple[TierProtocol, ...] = ()
    #: ``None`` when no cloud provider has credentials — a local-only
    #: deployment, not a misconfiguration.
    registry: ProviderRegistry | None = None
    providers: tuple[LLMProvider, ...] = field(default_factory=tuple)


@contextmanager
def _tier_guard(tier: TierId) -> Iterator[None]:
    """Let one rung fail to build without taking the service down.

    This is the single most important behaviour in this module. Without it, a
    revoked cloud key or a stopped Ollama daemon would turn a service designed
    to degrade into one that cannot start — and a crash loop at boot is strictly
    worse than a shorter ladder, because a shorter ladder still answers.

    The exception's message is logged but never reaches a caller: it can carry a
    file path or a base URL, and ``/v1/capabilities`` already reports the
    outcome — which rungs exist — without reporting the cause.
    """
    try:
        yield
    except _TierUnconfiguredError as exc:
        _log.warning("startup.tier_skipped", tier=tier.value, reason=str(exc))
    except Exception as exc:
        _log.warning(
            "startup.tier_build_failed",
            tier=tier.value,
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
    registry: ProviderRegistry | None = None

    if TierId.T0_RULES in enabled:
        with _tier_guard(TierId.T0_RULES):
            tiers.append(RulesTier(settings))

    if TierId.T1_CLASSIFIER in enabled:
        with _tier_guard(TierId.T1_CLASSIFIER):
            # `build_scorer` already falls back to the lexicon when the ONNX
            # artifact is missing, so this rung effectively cannot be skipped —
            # which is why a deployment with nothing else configured still has a
            # ladder two rungs deep.
            tiers.append(ClassifierTier(settings))

    if TierId.T2_LOCAL_LLM in enabled:
        with _tier_guard(TierId.T2_LOCAL_LLM):
            ollama = _local_provider(settings)
            # Constructing the provider does no I/O, so `configured` here means
            # "switched on", not "reachable". Reachability is a readiness
            # question, and the rung is built either way so that a daemon
            # starting up after the service does not need a restart to be used.
            if not ollama.configured:
                raise _TierUnconfiguredError("ollama is disabled in configuration")
            providers.append(ollama)
            tiers.append(LocalLLMTier(settings, ollama))

    if TierId.T3_CLOUD_LLM in enabled:
        with _tier_guard(TierId.T3_CLOUD_LLM):
            candidates = _cloud_providers(settings)
            # Appended before the credential check so that shutdown still closes
            # a provider whose SDK client was built but whose tier was skipped.
            providers.extend(candidates)
            usable = {p.name: p for p in candidates if p.configured}
            if not usable:
                raise _TierUnconfiguredError("no cloud provider has credentials")
            registry = ProviderRegistry(
                settings,
                usable,
                {
                    name: CircuitBreaker.from_settings(name.value, settings.resilience)
                    for name in usable
                },
            )
            tiers.append(CloudLLMTier(settings, registry))

    _log.info(
        "startup.ladder_built",
        tiers=[tier.tier.value for tier in tiers],
        # Only the usable ones: a provider that was constructed without
        # credentials is an implementation detail of the check above, and naming
        # it here would read as "anthropic is available" in the boot log.
        providers=[p.name.value for p in providers if p.configured],
    )
    return Ladder(tiers=tuple(tiers), registry=registry, providers=tuple(providers))


def _local_provider(settings: Settings) -> LLMProvider:
    """The Tier-2 backend: the air-infra gateway when enabled, else Ollama direct.

    In gateway mode the model asked for is still ``providers.ollama.model``; only
    the transport changes, so the rung's timeouts and prompt stay as they were.
    """
    if settings.providers.gateway.enabled:
        return InfraGatewayProvider(settings, Provider.OLLAMA)
    return OllamaProvider(settings)


def _cloud_providers(settings: Settings) -> list[LLMProvider]:
    """Both metered backends, configured or not.

    Built unconditionally because ``configured`` is the only honest way to ask
    whether an SDK is installed *and* a key is present, and because the
    registry's preference order is settings-driven — filtering here on
    ``providers.preference`` would let a reordering silently drop a backend.

    In gateway mode each backend is an :class:`InfraGatewayProvider` pinned to
    its provider identity, so the registry keys and fails over between them
    exactly as it does the direct SDK adapters.
    """
    if settings.providers.gateway.enabled:
        return [
            InfraGatewayProvider(settings, Provider.ANTHROPIC),
            InfraGatewayProvider(settings, Provider.OPENAI),
        ]
    return [AnthropicProvider(settings), OpenAIProvider(settings)]


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
        api_keys = ApiKeyStore.load(settings)
        # The limiter's buckets live in the cache, so it cannot precede it.
        rate_limiter = RateLimiter(cache, settings)

        ladder = build_ladder(settings)
        orchestrator = Orchestrator(ladder.tiers, EscalationPolicy(settings), settings)

        AppState(
            settings=settings,
            cache=cache,
            api_keys=api_keys,
            rate_limiter=rate_limiter,
            orchestrator=orchestrator,
            registry=ladder.registry,
            tiers=ladder.tiers,
        ).attach(app)

        # `api/v1/system.py` reads these three off `app.state` by name rather
        # than through `AppState`, so that its routes still answer in a test app
        # assembled without a lifespan. Published under the constants that
        # module exports, so a rename cannot leave the two sides disagreeing.
        setattr(app.state, STATE_SETTINGS, settings)
        setattr(app.state, STATE_TIERS, ladder.tiers)
        setattr(app.state, STATE_CACHE, cache)
        setattr(app.state, STATE_ANALYSIS_SERVICE, AnalysisService(orchestrator, settings, cache))

        _log.info(
            "startup.complete",
            tiers=[tier.tier.value for tier in ladder.tiers],
            cache_backend=settings.cache.backend,
            api_keys=len(api_keys),
            allow_unauthenticated=settings.security.allow_unauthenticated,
            docs_enabled=settings.app.docs_enabled,
        )
        try:
            yield
        finally:
            await _shutdown(settings, cache=cache, providers=ladder.providers)

    return lifespan


async def _shutdown(
    settings: Settings, *, cache: CacheBackend, providers: Sequence[LLMProvider]
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
            await _close_all(cache=cache, providers=providers)
    except TimeoutError:
        _log.warning("shutdown.grace_exceeded", grace_seconds=grace)
    _log.info("shutdown.complete")


async def _close_all(*, cache: CacheBackend, providers: Sequence[LLMProvider]) -> None:
    """Close the providers concurrently, then the cache.

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
