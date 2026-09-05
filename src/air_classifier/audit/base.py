"""Audit-log contract.

An operational record of classification jobs, for a support person to review —
not customer state, and not this service's only persistence (the cache already
holds that role). Every implementation must never raise, and must never let a
slow or down sink add latency to the request that triggered it: recording is
fire-and-forget from the caller's point of view.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from air_classifier.constants import Tier


class AuditEntry(BaseModel):
    """One classification job, as recorded for a support reviewer.

    ``text`` is already-redacted — the same text every tier saw
    (`pipeline/service.py::AnalysisService._prepare`), never raw pre-redaction
    text. ``timestamp`` is the field a TTL index expires entries against.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    text: str
    api_key_id: str | None = None
    decided_by: Tier
    degraded: bool
    latency_ms: float = Field(ge=0.0)
    timestamp: datetime


@runtime_checkable
class AuditSink(Protocol):
    """Fire-and-forget operational log.

    ``start`` exists here and not on the cache backend because a sink that
    persists somewhere real (Mongo) owns a background writer that must be
    explicitly launched and, on shutdown, explicitly drained — a lazily
    self-sufficient cache client has no equivalent lifecycle.
    """

    async def start(self) -> None:
        """Launch whatever background work this sink needs. Idempotent."""
        ...

    async def record(self, entry: AuditEntry) -> None:
        """Enqueue one entry. Must return immediately and never raise."""
        ...

    async def aclose(self) -> None:
        """Stop accepting writes and release resources. Must never raise."""
        ...
