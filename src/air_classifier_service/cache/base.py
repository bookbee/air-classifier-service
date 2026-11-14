"""Cache contract.

The service holds no database, so this is the only persistence it has. It serves
three jobs — verdict reuse, idempotency, and rate-limit counters — and the
backend behind it is a configuration choice, never a code change.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from air_classifier_service.schemas.common import Options


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal string key/value store with TTL.

    Implementations must never raise on a cache miss or a backend outage: a
    cache that fails closed would turn a Redis blip into a service outage. Log
    and return None instead.
    """

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr(self, key: str, ttl_seconds: int) -> int:
        """Atomically increment a counter, setting the TTL on first write."""
        ...

    async def ping(self) -> bool:
        """Reachability probe. Must not raise."""
        ...

    async def aclose(self) -> None: ...


#: Bumped whenever a change would make previously cached verdicts wrong —
#: a new tier, a changed prompt, a re-tuned threshold set. Cheaper and safer
#: than trying to selectively invalidate. v2: dropped the resource segment
#: when `/sentiment`, `/feedback` and `/reviews` were removed — one route
#: means one key shape, and the field never distinguished anything again.
#: v3: SIT fixes, 2026-11-14 — a lexicon-only Tier 1 verdict no longer clears
#: the floor, zero-confidence topics are dropped, `max_cost_usd` is enforced
#: before a paid rung runs and `degraded` covers skipped rungs; a v2 verdict
#: can be confidently wrong, or bought past a ceiling, in ways v3 never is.
CACHE_SCHEMA_VERSION = "v3"


def verdict_cache_key(
    *,
    prefix: str,
    text: str,
    options: Options,
    resource_fields: dict[str, object] | None = None,
) -> str:
    """Derive a stable cache key.

    Only the options that can change the *answer* participate. `include_trace`
    and `latency_budget_ms` deliberately do not: two requests differing only in
    those should share a cached verdict, and trace presence is shaped on the way
    out rather than recomputed.
    """
    material = "\x1f".join(
        [
            CACHE_SCHEMA_VERSION,
            text,
            str(options.min_tier.value if options.min_tier else "-"),
            options.max_tier.value,
            "1" if options.redact_pii else "0",
            ",".join(sorted(options.aspects or ())),
            _stable_fields(resource_fields or {}),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{prefix}verdict:{digest}"


def idempotency_cache_key(*, prefix: str, api_key_id: str, idempotency_key: str) -> str:
    """Idempotency is scoped per API key so two callers cannot collide."""
    scoped = f"{api_key_id}\x1f{idempotency_key}"
    digest = hashlib.sha256(scoped.encode("utf-8")).hexdigest()
    return f"{prefix}idem:{digest}"


def text_fingerprint(text: str) -> str:
    """Short, stable hash of input text, safe to log in place of the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_fields(fields: dict[str, object]) -> str:
    """Deterministic rendering of resource fields.

    Sorted because dict iteration order is insertion order, and two logically
    identical requests must not produce different keys.
    """
    return ";".join(f"{k}={fields[k]!r}" for k in sorted(fields) if fields[k] is not None)
