"""Tests for structured logging and the optional tracing layer."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, Literal

import pytest
import structlog

from air_sentiment.cache.base import text_fingerprint
from air_sentiment.config import (
    AppSettings,
    ObservabilitySettings,
    SecuritySettings,
    Settings,
)
from air_sentiment.observability import logging as obs_logging
from air_sentiment.observability import tracing as obs_tracing
from air_sentiment.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    safe_text_fields,
)


def make_settings(
    *,
    log_format: Literal["json", "console"] = "json",
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_raw_text: bool = False,
    otel_enabled: bool = False,
    otel_endpoint: str | None = None,
) -> Settings:
    return Settings(
        obs=ObservabilitySettings(
            log_format=log_format,
            log_level=log_level,
            otel_enabled=otel_enabled,
            otel_endpoint=otel_endpoint,
        ),
        security=SecuritySettings(log_raw_text=log_raw_text),
    )


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Leave global logging state exactly as it was found.

    structlog and stdlib logging are process-wide singletons; without this every
    test here would leak its configuration into the rest of the suite.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_structlog = structlog.get_config()
    clear_request_context()

    yield

    clear_request_context()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    obs_logging._handler = None
    structlog.configure(**saved_structlog)


def emitted(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """Parse the JSON log lines written to stdout so far."""
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# ── Rendering ─────────────────────────────────────────────────────────────────


def test_json_output_shape(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    get_logger("air_sentiment.test").info("tier.decision", tier="t1_classifier", escalate=True)

    records = emitted(capsys)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "tier.decision"
    assert record["level"] == "info"
    assert record["logger"] == "air_sentiment.test"
    assert record["tier"] == "t1_classifier"
    assert record["escalate"] is True
    assert record["timestamp"].endswith("Z")


def test_console_format_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings(log_format="console"))
    get_logger("air_sentiment.test").info("tier.decision", tier="t0_rules")

    out = capsys.readouterr().out
    assert "tier.decision" in out
    assert "t0_rules" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.splitlines()[0])


def test_level_filters_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings(log_level="WARNING"))
    log = get_logger("air_sentiment.test")
    log.info("suppressed")
    log.warning("kept")

    assert [r["event"] for r in emitted(capsys)] == ["kept"]


def test_non_serialisable_values_do_not_raise(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    get_logger("air_sentiment.test").info("weird", value=object())

    assert emitted(capsys)[0]["value"].startswith("<object object")


def test_exception_is_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("air_sentiment.test").exception("tier.failed")

    record = emitted(capsys)[0]
    assert record["level"] == "error"
    assert "ValueError: boom" in record["exception"]


# ── Idempotency and stdlib adoption ───────────────────────────────────────────


def test_configure_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    settings = make_settings()
    before = len(logging.getLogger().handlers)
    configure_logging(settings)
    configure_logging(settings)
    configure_logging(settings)

    assert len(logging.getLogger().handlers) == before + 1
    get_logger("air_sentiment.test").info("once")
    assert len(emitted(capsys)) == 1


def test_stdlib_logs_are_routed_through_structlog(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    logging.getLogger("uvicorn.error").warning("Application startup complete")

    record = emitted(capsys)[0]
    assert record["event"] == "Application startup complete"
    assert record["level"] == "warning"
    assert record["logger"] == "uvicorn.error"


def test_adopted_logger_handlers_are_replaced() -> None:
    noisy = logging.getLogger("uvicorn.access")
    noisy.addHandler(logging.NullHandler())
    noisy.propagate = False

    configure_logging(make_settings())

    assert noisy.handlers == []
    assert noisy.propagate is True


# ── Request context ───────────────────────────────────────────────────────────


def test_bound_context_appears_on_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    bind_request_context(request_id="req_01J", trace_id="4bf9" * 8)
    get_logger("air_sentiment.test").info("http.request")

    record = emitted(capsys)[0]
    assert record["request_id"] == "req_01J"
    assert record["trace_id"] == "4bf9" * 8


def test_bound_context_reaches_stdlib_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    bind_request_context(request_id="req_02K")
    logging.getLogger("httpx").warning("HTTP Request: POST /v1/messages")

    assert emitted(capsys)[0]["request_id"] == "req_02K"


def test_clear_request_context_removes_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    log = get_logger("air_sentiment.test")
    bind_request_context(request_id="req_03L")
    log.info("during")
    clear_request_context()
    log.info("after")

    during, after = emitted(capsys)
    assert during["request_id"] == "req_03L"
    assert "request_id" not in after


# ── Raw-text chokepoint ───────────────────────────────────────────────────────


def test_safe_text_fields_omits_text_by_default() -> None:
    text = "the coffee machine is broken again"
    fields = safe_text_fields(text, make_settings(log_raw_text=False))

    assert fields == {"text_sha256": text_fingerprint(text), "text_len": len(text)}
    assert "text" not in fields


def test_safe_text_fields_includes_text_when_opted_in() -> None:
    text = "the coffee machine is broken again"
    fields = safe_text_fields(text, make_settings(log_raw_text=True))

    assert fields["text"] == text
    assert fields["text_sha256"] == text_fingerprint(text)
    assert fields["text_len"] == len(text)


def test_safe_text_fields_survives_a_log_round_trip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings()
    configure_logging(settings)
    text = "sécurité: 用户 feedback 🙂"
    get_logger("air_sentiment.test").info("tier.decision", **safe_text_fields(text, settings))

    record = emitted(capsys)[0]
    assert record["text_sha256"] == text_fingerprint(text)
    assert record["text_len"] == len(text)
    assert "text" not in record


def test_production_settings_reject_raw_text_logging() -> None:
    with pytest.raises(ValueError, match="log_raw_text"):
        Settings(
            app=AppSettings(env="production"),
            security=SecuritySettings(log_raw_text=True),
        )


# ── Tracing degradation ───────────────────────────────────────────────────────


def test_otel_is_absent_in_this_environment() -> None:
    """The rest of the tracing tests only mean something if otel really is missing."""
    assert obs_tracing.OTEL_AVAILABLE is False


def test_configure_tracing_is_a_noop_when_disabled() -> None:
    obs_tracing.configure_tracing(make_settings(otel_enabled=False))

    assert obs_tracing.tracing_enabled() is False
    assert obs_tracing.current_trace_id() is None


def test_configure_tracing_warns_but_survives_without_otel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(make_settings())
    obs_tracing.configure_tracing(
        make_settings(otel_enabled=True, otel_endpoint="http://collector:4318/v1/traces")
    )

    record = emitted(capsys)[0]
    assert record["event"] == "tracing.unavailable"
    assert record["level"] == "warning"
    assert obs_tracing.tracing_enabled() is False


async def test_span_yields_a_usable_noop() -> None:
    async with obs_tracing.span("tier.t1_classifier", tier="t1_classifier", cost=None) as active:
        active.set_attribute("label", "positive")
        active.add_event("escalated")
        active.record_exception(RuntimeError("ignored"))
        assert active.is_recording() is False


async def test_span_does_not_swallow_exceptions() -> None:
    with pytest.raises(RuntimeError, match="tier failed"):
        async with obs_tracing.span("tier.t3_cloud_llm"):
            raise RuntimeError("tier failed")


async def test_nested_spans_are_independent() -> None:
    async with obs_tracing.span("request") as outer, obs_tracing.span("preprocess") as inner:
        assert outer.is_recording() is False
        assert inner.is_recording() is False
        assert obs_tracing.current_trace_id() is None


def test_instrument_app_is_a_noop_without_otel() -> None:
    sentinel = object()
    obs_tracing.instrument_app(sentinel)  # must not raise
