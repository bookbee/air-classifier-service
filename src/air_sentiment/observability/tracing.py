"""OpenTelemetry tracing, optional at runtime.

The otel packages live behind the ``otel`` extra, so this module is written to
be *absent-tolerant*: with the SDK missing, or with tracing switched off,
everything here degrades to a no-op that costs an attribute lookup. Call sites
instrument unconditionally and never branch on whether tracing is live —

    async with span("tier.t1_classifier", tier="t1_classifier"):
        ...

does the right thing in both worlds.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Final, Protocol

from air_sentiment.config import Settings
from air_sentiment.observability.logging import get_logger

try:  # The `otel` extra is optional; absence is a supported configuration.
    from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
        BatchSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import (  # type: ignore[import-not-found]
        ParentBased,
        TraceIdRatioBased,
    )

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by the dependency matrix, not by tests
    OTEL_AVAILABLE = False

__all__ = [
    "OTEL_AVAILABLE",
    "Span",
    "configure_tracing",
    "current_trace_id",
    "instrument_app",
    "span",
    "tracing_enabled",
]

_log = get_logger(__name__)

#: Set once :func:`configure_tracing` succeeds; ``None`` means "trace nothing".
_tracer: Any | None = None

#: Attribute types the OTLP wire format accepts as-is. Anything else is stringified.
_SCALARS: Final[tuple[type, ...]] = (str, bool, int, float)


class Span(Protocol):
    """The slice of the otel span API this service uses."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = ...) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

    def is_recording(self) -> bool: ...


class _NoopSpan:
    """Stand-in that swallows instrumentation when tracing is off."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def is_recording(self) -> bool:
        return False


#: Immutable and stateless, so one instance serves every untraced span.
_NOOP_SPAN: Final[Span] = _NoopSpan()


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider, or leave tracing off.

    Off is not a failure: a missing SDK or a missing endpoint downgrades to
    no-op spans and a log line, never to a startup error, because losing traces
    must not be able to take the service down.
    """
    global _tracer

    if not settings.obs.otel_enabled:
        return
    if not OTEL_AVAILABLE:
        _log.warning(
            "tracing.unavailable",
            reason="opentelemetry_not_installed",
            hint="install air-sentiment[otel]",
        )
        return
    if _tracer is not None:
        return

    resource = Resource.create(
        {
            "service.name": settings.obs.service_name,
            "deployment.environment": settings.app.env,
        }
    )
    # ParentBased keeps a sampled upstream trace intact end to end; sampling the
    # decision independently per service produces holes in the waterfall.
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.obs.otel_sample_ratio)),
    )

    if settings.obs.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
        except ImportError:
            _log.warning("tracing.exporter_unavailable", endpoint=settings.obs.otel_endpoint)
        else:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.obs.otel_endpoint))
            )

    otel_trace.set_tracer_provider(provider)
    _tracer = otel_trace.get_tracer(settings.obs.service_name)
    _log.info(
        "tracing.configured",
        endpoint=settings.obs.otel_endpoint,
        sample_ratio=settings.obs.otel_sample_ratio,
    )


def tracing_enabled() -> bool:
    """Whether spans are actually being recorded."""
    return _tracer is not None


@asynccontextmanager
async def span(name: str, **attributes: Any) -> AsyncIterator[Span]:
    """Open a span, or hand back a no-op that behaves like one.

    Exceptions are recorded and re-raised: the caller's error handling is the
    caller's business, tracing only observes it.
    """
    tracer = _tracer
    if tracer is None:
        yield _NOOP_SPAN
        return

    with tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            if value is not None:
                active.set_attribute(key, value if isinstance(value, _SCALARS) else str(value))
        yield active


def current_trace_id() -> str | None:
    """Hex trace id of the active span, for stamping onto log lines."""
    if _tracer is None:
        return None
    context = otel_trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def instrument_app(app: Any) -> None:
    """Add automatic FastAPI server spans when the instrumentation is installed."""
    if _tracer is None:
        return
    try:
        from opentelemetry.instrumentation.fastapi import (  # type: ignore[import-not-found]
            FastAPIInstrumentor,
        )
    except ImportError:
        _log.warning("tracing.fastapi_instrumentation_unavailable")
        return
    FastAPIInstrumentor.instrument_app(app)
