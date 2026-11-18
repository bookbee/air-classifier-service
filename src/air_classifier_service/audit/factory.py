"""Audit-sink backend selection.

Which backend is running is a deployment decision, not a code path — everything
downstream sees only :class:`~air_classifier_service.audit.base.AuditSink`. Unlike the
cache, there is no in-process fallback: "none" (the default) is a true no-op
rather than a degraded-but-working local backend, because nobody should start
accumulating an audit trail nobody configured a home for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from air_classifier_service.audit.base import AuditEntry, AuditSink
from air_classifier_service.audit.mongo import MongoAuditSink
from air_classifier_service.observability.logging import get_logger

if TYPE_CHECKING:
    from air_classifier_service.config import Settings

logger = get_logger(__name__)

__all__ = ["NullAuditSink", "build_audit_sink"]


class NullAuditSink:
    """A sink that records nothing. The default — see module docstring."""

    async def start(self) -> None:
        return None

    async def record(self, entry: AuditEntry) -> None:
        return None

    async def aclose(self) -> None:
        return None


def build_audit_sink(settings: Settings) -> AuditSink:
    """Construct the configured sink. Never raises.

    A "mongo" backend that cannot actually reach Mongo (unreachable host,
    `motor` not installed) still returns a working `MongoAuditSink` — it
    degrades its own connect attempts internally, exactly as `RedisCache` does.
    """
    backend = settings.audit.backend
    if backend == "mongo":
        logger.info(
            "audit.configured",
            backend=backend,
            database=settings.audit.database,
            collection=settings.audit.collection,
        )
        return MongoAuditSink(settings.audit)

    logger.info("audit.configured", backend="none")
    return NullAuditSink()
