"""Structured logging.

Every log line in the service is emitted through structlog, including the ones
the service never wrote: uvicorn's access log and httpx's request log are routed
through the same processor chain so an operator greps one format, not three.

Two invariants matter more than the plumbing:

* ``request_id`` and ``trace_id`` ride on contextvars, so nothing has to thread
  them through call signatures to get correlated logs.
* Raw user text reaches a log line only through :func:`safe_text_fields`, and
  only when it has been explicitly enabled.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, Final, TextIO

import orjson
import structlog

from air_classifier_service.cache.base import text_fingerprint
from air_classifier_service.config import Settings

__all__ = [
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "current_api_key_id",
    "get_logger",
    "safe_text_fields",
]

#: Third-party loggers that ship their own handlers. Stripping those and letting
#: the records propagate to root is what makes their output match ours.
_ADOPTED_LOGGERS: Final[tuple[str, ...]] = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    "fastapi",
    "httpx",
    "httpcore",
)

#: The one handler this module owns. Kept so repeat configuration replaces it
#: rather than stacking a second copy of every log line on top of the first.
_handler: logging.Handler | None = None


class _StdoutHandler(logging.StreamHandler[TextIO]):
    """Stream handler that resolves ``sys.stdout`` at emit time.

    A plain StreamHandler captures the stream object when it is constructed,
    which goes wrong for anything that swaps ``sys.stdout`` afterwards — pytest's
    capture fixtures, uvicorn's reloader.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self) -> TextIO:
        return sys.stdout

    @stream.setter
    def stream(self, value: TextIO) -> None:
        """Absorb ``StreamHandler.__init__``'s assignment; the property decides."""


def _json_dumps(obj: Any, default: Callable[[Any], Any] | None = None, **_: Any) -> str:
    """orjson serializer adapted to structlog's ``json.dumps`` call signature.

    ``default`` is retained because event dicts routinely carry enums, ``Path``s
    and exceptions, and a log line must never be the thing that raises.
    """
    return orjson.dumps(obj, default=default or str).decode("utf-8")


def configure_logging(settings: Settings) -> None:
    """Install the structlog + stdlib logging pipeline. Safe to call repeatedly.

    Repeat calls replace the previous handler, so reconfiguring (tests, a
    settings reload) neither duplicates output nor strips handlers this module
    does not own.
    """
    global _handler

    as_json = settings.obs.log_format == "json"
    level = logging.getLevelNamesMapping()[settings.obs.log_level]

    # Applied to structlog and stdlib records alike, so a uvicorn line carries
    # the same request context as ours.
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Loggers are cached, but rendering happens in the stdlib formatter
        # below, so a later reconfiguration still changes the output format.
        cache_logger_on_first_use=True,
    )

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer(serializer=_json_dumps)
        if as_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    final: list[structlog.typing.Processor] = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta
    ]
    if as_json:
        # ConsoleRenderer formats tracebacks itself; JSONRenderer needs the
        # exception flattened into a string first.
        final.append(structlog.processors.format_exc_info)
    final.append(renderer)

    handler = _StdoutHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=final,
            foreign_pre_chain=[*shared, structlog.stdlib.ExtraAdder()],
        )
    )

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    root.addHandler(handler)
    root.setLevel(level)
    _handler = handler

    for name in _ADOPTED_LOGGERS:
        adopted = logging.getLogger(name)
        adopted.handlers.clear()
        adopted.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Bound logger for a module. Pass ``__name__`` at call sites."""
    if name is None:
        return structlog.stdlib.get_logger()
    return structlog.stdlib.get_logger(name)


def bind_request_context(**kw: Any) -> None:
    """Attach fields to every log line emitted in this async context.

    Middleware binds ``request_id``/``trace_id``/``api_key_id`` here once, and
    every downstream log line inherits them without any function having to
    accept or forward them.
    """
    structlog.contextvars.bind_contextvars(**kw)


def clear_request_context() -> None:
    """Drop everything :func:`bind_request_context` attached.

    Contextvars survive in reused worker tasks, so failing to clear leaks one
    request's identifiers into the next one's logs.
    """
    structlog.contextvars.clear_contextvars()


def current_api_key_id() -> str | None:
    """The caller's key id, if `api/deps.py::require_api_key` has bound one.

    Best-effort read of the same contextvar every log line already carries —
    for a consumer (the audit log) that wants the identifier without importing
    structlog directly or threading it through a call signature.
    """
    value = structlog.contextvars.get_contextvars().get("api_key_id")
    return value if isinstance(value, str) else None


def safe_text_fields(text: str, settings: Settings) -> dict[str, object]:
    """Loggable stand-in for user text.

    The single chokepoint between input text and the log stream: a fingerprint
    and a length are enough to correlate a report with a request, and the text
    itself appears only under an explicit, production-forbidden opt-in.
    """
    fields: dict[str, object] = {
        "text_sha256": text_fingerprint(text),
        "text_len": len(text),
    }
    if settings.security.log_raw_text:
        fields["text"] = text
    return fields
