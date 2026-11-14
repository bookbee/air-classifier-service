"""MongoDB-backed audit sink.

Every write here is fire-and-forget from the caller's point of view: `record`
only ever enqueues, and a single background task owns the actual Mongo I/O. A
bounded queue in front of that task means a slow or fully-down Mongo can never
grow request-handling memory unbounded or add latency to a classify call — it
can only drop the newest entries once the queue is full, which is logged and
counted rather than silent.

TTL is Mongo's own `expireAfterSeconds` index, created once on first successful
connect. There is no cleanup job anywhere in this service; documents expire
themselves.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final

from air_classifier_service.audit.base import AuditEntry
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.observability.metrics import AuditEvent, record_audit_event

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

    from air_classifier_service.config import AuditSettings

logger = get_logger(__name__)

#: Deliberately short — a hung connect attempt must not stall `start()`.
_CONNECT_TIMEOUT_MS: Final[int] = 2000


class MongoAuditSink:
    """Audit sink over ``motor.motor_asyncio``.

    The client is built lazily, on `start()` or first `record()`, never in
    `__init__` — constructing the sink must never perform I/O or depend on
    Mongo being up at import or startup time, matching `RedisCache`'s own
    contract.
    """

    def __init__(
        self,
        settings: AuditSettings,
        *,
        client: AsyncIOMotorClient[dict[str, Any]] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._ttl_ensured = False
        self._queue: asyncio.Queue[AuditEntry] = asyncio.Queue(maxsize=settings.queue_max_size)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Launch the background consumer. Safe to call more than once."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume(), name="audit-mongo-consumer")

    async def record(self, entry: AuditEntry) -> None:
        """Enqueue one entry without blocking. Drops on a full queue."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning("audit.queue_full", request_id=entry.request_id)
            record_audit_event(AuditEvent.DROPPED_QUEUE_FULL)

    async def aclose(self) -> None:
        """Stop accepting writes, drain what is queued, then release the client."""
        self._closed = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=self._settings.drain_timeout_s)
            except TimeoutError:
                logger.warning("audit.drain_timed_out", pending=self._queue.qsize())
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._degraded("consumer_stop", exc)
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                self._degraded("aclose", exc)

    # ── Internals ─────────────────────────────────────────────────────────

    async def _consume(self) -> None:
        """Drain the queue, one document at a time, for the life of the sink."""
        while True:
            entry = await self._queue.get()
            try:
                await self._write(entry)
            finally:
                self._queue.task_done()

    async def _write(self, entry: AuditEntry) -> None:
        collection = await self._collection_or_none()
        if collection is None:
            record_audit_event(AuditEvent.FAILED)
            return
        try:
            await collection.insert_one(entry.model_dump(mode="json"))
            record_audit_event(AuditEvent.WRITTEN)
        except Exception as exc:
            self._degraded("insert_one", exc)
            record_audit_event(AuditEvent.FAILED)

    async def _collection_or_none(self) -> AsyncIOMotorCollection[dict[str, Any]] | None:
        """Return the target collection, connecting and indexing on first use."""
        client = await self._client_or_none()
        if client is None:
            return None
        collection = client[self._settings.database][self._settings.collection]
        if not self._ttl_ensured:
            await self._ensure_ttl_index(collection)
        return collection

    async def _client_or_none(self) -> AsyncIOMotorClient[dict[str, Any]] | None:
        client = self._client
        if client is not None:
            return client
        if self._closed:
            return None
        async with self._connect_lock:
            client = self._client
            if client is not None or self._closed:
                return client
            client = self._create_client()
            self._client = client
        return client

    def _create_client(self) -> AsyncIOMotorClient[dict[str, Any]] | None:
        """Construct the client, or None if it cannot be built.

        `motor` is an optional dependency, so a missing package is a
        configuration problem to log — not an ImportError raised from the
        middle of the background consumer.
        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            return AsyncIOMotorClient(
                self._settings.mongo_url,
                serverSelectionTimeoutMS=_CONNECT_TIMEOUT_MS,
            )
        except Exception as exc:
            self._degraded("connect", exc)
            return None

    async def _ensure_ttl_index(self, collection: AsyncIOMotorCollection[dict[str, Any]]) -> None:
        """Create the TTL index once. Never raises past this method.

        A conflicting index (the TTL value changed since it was first created)
        is logged as an operator-actionable warning rather than an attempt to
        drop and recreate it automatically — that is a deliberate operational
        decision, not something this service should do unattended.
        """
        try:
            await collection.create_index(
                "timestamp", expireAfterSeconds=self._settings.ttl_days * 86_400
            )
            self._ttl_ensured = True
        except Exception as exc:
            logger.warning(
                "audit.ttl_index_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                ttl_days=self._settings.ttl_days,
            )

    def _degraded(self, op: str, exc: BaseException) -> None:
        logger.warning(
            "audit.mongo_degraded",
            op=op,
            error=str(exc),
            error_type=type(exc).__name__,
        )
