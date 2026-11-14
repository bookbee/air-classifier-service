"""Backend selection and the null sink's never-retain contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from air_classifier_service.audit.base import AuditEntry, AuditSink
from air_classifier_service.audit.factory import NullAuditSink, build_audit_sink
from air_classifier_service.audit.mongo import MongoAuditSink
from air_classifier_service.config import AuditSettings, Settings
from air_classifier_service.constants import Tier


def settings_with(**audit_kwargs: Any) -> Settings:
    return Settings(audit=AuditSettings(**audit_kwargs))


def _entry() -> AuditEntry:
    return AuditEntry(
        request_id="req_1",
        text="great product",
        decided_by=Tier.T0_RULES,
        degraded=False,
        latency_ms=1.0,
        timestamp=datetime.now(UTC),
    )


# ── Dispatch ──────────────────────────────────────────────────────────────


def test_defaults_to_none() -> None:
    sink = build_audit_sink(Settings())
    assert isinstance(sink, NullAuditSink)


def test_none_backend_builds_null_sink() -> None:
    sink = build_audit_sink(settings_with(backend="none"))
    assert isinstance(sink, NullAuditSink)


def test_mongo_backend_builds_without_a_live_server() -> None:
    sink = build_audit_sink(settings_with(backend="mongo"))
    assert isinstance(sink, MongoAuditSink)


@pytest.mark.parametrize("backend", ["mongo", "none"])
def test_every_backend_satisfies_the_protocol(backend: str) -> None:
    sink = build_audit_sink(settings_with(backend=backend))
    assert isinstance(sink, AuditSink)


# ── NullAuditSink ─────────────────────────────────────────────────────────


async def test_null_sink_never_retains() -> None:
    sink = NullAuditSink()
    await sink.start()
    await sink.record(_entry())
    await sink.aclose()
