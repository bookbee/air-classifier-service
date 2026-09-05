"""MongoAuditSink's fail-open contract, TTL indexing, and queue backpressure.

Mirrors ``test_cache_factory.py``'s conventions: a hand-rolled fake client
injected via the constructor's ``client=`` parameter, no live server and no
``unittest.mock``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from air_classifier.audit.base import AuditEntry
from air_classifier.audit.mongo import MongoAuditSink
from air_classifier.config import AuditSettings
from air_classifier.constants import Tier
from air_classifier.observability import metrics


def _entry(request_id: str = "req_1") -> AuditEntry:
    return AuditEntry(
        request_id=request_id,
        text="great product",
        decided_by=Tier.T0_RULES,
        degraded=False,
        latency_ms=1.0,
        timestamp=datetime.now(UTC),
    )


def _sample(name: str, **labels: str) -> float:
    value = metrics.get_registry().get_sample_value(name, labels)
    return 0.0 if value is None else value


# ── Fakes ─────────────────────────────────────────────────────────────────


class FakeCollection:
    def __init__(self, *, fail_insert: bool = False, fail_index: bool = False) -> None:
        self.documents: list[dict[str, Any]] = []
        self.index_calls: list[dict[str, Any]] = []
        self._fail_insert = fail_insert
        self._fail_index = fail_index

    async def insert_one(self, document: dict[str, Any]) -> None:
        if self._fail_insert:
            raise ConnectionError("mongo is down")
        self.documents.append(document)

    async def create_index(self, name: str, **kwargs: Any) -> str:
        if self._fail_index:
            raise ConnectionError("mongo is down")
        self.index_calls.append({"name": name, **kwargs})
        return name


class FakeDatabase:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collection


class FakeMongoClient:
    """Enough of ``AsyncIOMotorClient``'s surface for ``MongoAuditSink``."""

    def __init__(self, collection: FakeCollection | None = None) -> None:
        self.collection = collection or FakeCollection()
        self.closed = False

    def __getitem__(self, name: str) -> FakeDatabase:
        return FakeDatabase(self.collection)

    def close(self) -> None:
        self.closed = True


# ── Fail-open ─────────────────────────────────────────────────────────────


async def test_record_writes_through_on_drain() -> None:
    client = FakeMongoClient()
    sink = MongoAuditSink(AuditSettings(), client=client)
    await sink.start()

    await sink.record(_entry())
    await sink.aclose()

    assert client.collection.documents[0]["request_id"] == "req_1"


async def test_insert_failure_degrades_without_raising() -> None:
    client = FakeMongoClient(FakeCollection(fail_insert=True))
    sink = MongoAuditSink(AuditSettings(), client=client)
    await sink.start()

    await sink.record(_entry())
    await sink.aclose()  # must not raise


async def test_missing_motor_package_degrades_rather_than_raises() -> None:
    """No injected client and nothing listening — `_create_client` must not raise."""
    sink = MongoAuditSink(AuditSettings(mongo_url="mongodb://127.0.0.1:1/"))
    await sink.start()

    await sink.record(_entry())
    await sink.aclose()  # must not raise, and must not hang


async def test_aclose_is_safe_to_call_twice() -> None:
    sink = MongoAuditSink(AuditSettings(), client=FakeMongoClient())
    await sink.start()

    await sink.aclose()
    await sink.aclose()


# ── TTL index ─────────────────────────────────────────────────────────────


async def test_ttl_index_created_on_first_connect() -> None:
    client = FakeMongoClient()
    sink = MongoAuditSink(AuditSettings(ttl_days=30), client=client)
    await sink.start()

    await sink.record(_entry())
    await sink.aclose()

    assert client.collection.index_calls == [
        {"name": "timestamp", "expireAfterSeconds": 30 * 86_400}
    ]


async def test_ttl_index_failure_does_not_block_writes() -> None:
    collection = FakeCollection(fail_index=True)
    client = FakeMongoClient(collection)
    sink = MongoAuditSink(AuditSettings(), client=client)
    await sink.start()

    await sink.record(_entry())
    await sink.aclose()

    assert collection.documents[0]["request_id"] == "req_1"


# ── Queue backpressure ────────────────────────────────────────────────────


async def test_queue_full_drops_and_records_metric() -> None:
    # No `start()`: nothing ever drains the queue, so the second `record` finds
    # it full deterministically rather than racing a background consumer.
    sink = MongoAuditSink(AuditSettings(queue_max_size=1), client=FakeMongoClient())
    before = _sample("air_classifier_audit_events_total", result="dropped_queue_full")

    await sink.record(_entry("first"))
    await sink.record(_entry("second"))

    after = _sample("air_classifier_audit_events_total", result="dropped_queue_full")
    assert after == before + 1


async def test_record_after_close_is_a_no_op() -> None:
    client = FakeMongoClient()
    sink = MongoAuditSink(AuditSettings(), client=client)
    await sink.start()
    await sink.aclose()

    await sink.record(_entry())  # must not raise, must not reopen the client

    assert client.collection.documents == []
