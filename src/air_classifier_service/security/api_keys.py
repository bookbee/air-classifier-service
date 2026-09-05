"""API key identity.

A raw key exists in exactly two places: the caller's `X-API-Key` header and
whatever handed the key to them. Configuration carries salted sha256 digests
only, and nothing in this module renders a raw key or a digest into a log line,
a repr, or an exception message.

The store is built once at startup and read-only afterwards — rotating a key is
a config change plus a restart, not a runtime mutation — so lookup needs no lock
and stays on the hot path without contention.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Iterable
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from air_classifier_service.config import Settings
from air_classifier_service.constants import Tier
from air_classifier_service.observability.logging import get_logger

logger = get_logger(__name__)

#: Raw keys carry this prefix so a leaked key is greppable in source dumps, logs
#: and secret scanners. It is not secret and adds no entropy.
KEY_PREFIX: Final[str] = "airc_"

SCOPE_CLASSIFY_READ: Final[str] = "classify:read"
SCOPE_ADMIN_READ: Final[str] = "admin:read"

#: What the development anonymous identity is granted.
ALL_SCOPES: Final[frozenset[str]] = frozenset({SCOPE_CLASSIFY_READ, SCOPE_ADMIN_READ})

#: Stable id for the unauthenticated identity. Appears in metric labels, so it
#: must stay a fixed string rather than anything per-request.
ANONYMOUS_KEY_ID: Final[str] = "anonymous"

_SHA256_HEX_LEN: Final[int] = 64
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_DEFAULT_KEY_BYTES: Final[int] = 32


# ── Hashing ───────────────────────────────────────────────────────────────────


def hash_key(raw: str, salt: str) -> str:
    """Salted sha256 of a raw key, hex encoded.

    A plain digest rather than a KDF on purpose: keys are 256 bits of CSPRNG
    output, not passwords, so there is no dictionary for a slow hash to defend
    against — bcrypt here would buy nothing and cost every request its latency.
    """
    return hashlib.sha256((salt + raw).encode("utf-8")).hexdigest()


def generate_key(salt: str, *, nbytes: int = _DEFAULT_KEY_BYTES) -> tuple[str, str]:
    """Mint a key.

    Returns ``(raw_key, key_hash)``. Only the hash is storable; the raw value
    must be handed to its owner at this moment or regenerated, never recovered.
    """
    raw = KEY_PREFIX + secrets.token_urlsafe(nbytes)
    return raw, hash_key(raw, salt)


def _is_sha256_hex(value: str) -> bool:
    return len(value) == _SHA256_HEX_LEN and all(char in _HEX_DIGITS for char in value)


# ── Records ───────────────────────────────────────────────────────────────────


class ApiKeyRecord(BaseModel):
    """One caller's identity, and the ceilings that travel with it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        min_length=1,
        max_length=64,
        description="Stable, non-secret; safe to log and to use as a metric label.",
    )
    name: str = Field(min_length=1, max_length=128)
    key_hash: str = Field(repr=False, description="sha256(salt + raw_key), hex.")
    scopes: set[str] = Field(default_factory=set)
    max_tier: Tier = Field(
        default=Tier.T3_CLOUD_LLM,
        description="Per-key ceiling on the ladder; a low-trust key never reaches the cloud.",
    )
    force_pii_redaction: bool = False
    rate_limit_rpm: int | None = Field(default=None, ge=1)
    burst: int | None = Field(default=None, ge=0)
    enabled: bool = True

    @field_validator("key_hash")
    @classmethod
    def _normalise_hash(cls, value: str) -> str:
        # Digests copied out of tooling arrive uppercase often enough that
        # rejecting them would just look like a wrong key.
        return value.strip().lower()

    def __repr__(self) -> str:
        # Pydantic's generated repr prints every field, digest included. A record
        # reaches logs, tracebacks and debuggers; none of them may see the hash.
        return f"ApiKeyRecord(id={self.id!r}, name={self.name!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def clamp_tier(self, requested: Tier) -> Tier:
        """The lower of what the caller asked for and what this key may reach."""
        return requested if requested.rank <= self.max_tier.rank else self.max_tier

    @classmethod
    def anonymous(cls) -> Self:
        """Synthetic identity for ``security.allow_unauthenticated`` deployments.

        Its empty digest can never collide with a real one — :func:`hash_key`
        always returns 64 hex characters — so this record is unreachable through
        lookup and can only be produced deliberately.
        """
        return cls(
            id=ANONYMOUS_KEY_ID,
            name="anonymous",
            key_hash="",
            scopes=set(ALL_SCOPES),
        )


# ── Store ─────────────────────────────────────────────────────────────────────


class ApiKeyStore:
    """Digest-indexed set of key records."""

    def __init__(
        self,
        records: Iterable[ApiKeyRecord],
        *,
        salt: str,
        allow_unauthenticated: bool = False,
    ) -> None:
        self._salt = salt
        self._allow_unauthenticated = allow_unauthenticated
        self._by_hash: dict[str, ApiKeyRecord] = {}
        for record in records:
            self._index(record)
        if not self._by_hash and not allow_unauthenticated:
            logger.warning("api_key_store_empty")

    def __len__(self) -> int:
        return len(self._by_hash)

    @classmethod
    def load(cls, settings: Settings) -> Self:
        """Build the store from ``security.api_keys_inline`` or ``..._file``.

        Inline wins when both are present: an image may bake in a key file, and
        an environment variable has to be able to override it without a rebuild.
        """
        security = settings.security
        payload: str | None = None
        source = "none"

        if security.api_keys_inline:
            payload, source = security.api_keys_inline, "inline"
        elif security.api_keys_file is not None:
            try:
                payload = security.api_keys_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"could not read security.api_keys_file: {exc}") from exc
            source = "file"

        records = _parse_records(payload) if payload else []
        logger.info(
            "api_key_store_loaded",
            source=source,
            count=len(records),
            allow_unauthenticated=security.allow_unauthenticated,
        )
        return cls(
            records,
            salt=security.hash_salt.get_secret_value(),
            allow_unauthenticated=security.allow_unauthenticated,
        )

    def lookup(self, raw_key: str | None) -> ApiKeyRecord | None:
        """Resolve a presented key. ``None`` means reject the request."""
        if raw_key:
            record = self._match(raw_key)
            if record is not None:
                if not record.enabled:
                    # Revocation is a config edit; a revoked key still hitting
                    # the service is worth an operator's attention.
                    logger.warning("api_key_disabled", key_id=record.id)
                    return None
                return record
        return ApiKeyRecord.anonymous() if self._allow_unauthenticated else None

    def _match(self, raw_key: str) -> ApiKeyRecord | None:
        candidate = hash_key(raw_key, self._salt)
        record = self._by_hash.get(candidate)
        if record is None:
            return None
        # The dict probe narrows to one candidate in O(1); this is the comparison
        # that authorises. compare_digest keeps it constant-time, so a digest
        # that matches on a prefix cannot be told from one that does not by
        # watching response latency.
        if not hmac.compare_digest(record.key_hash, candidate):
            return None
        return record

    def _index(self, record: ApiKeyRecord) -> None:
        digest = record.key_hash
        if not _is_sha256_hex(digest):
            # Nearly always a raw key pasted where a digest belongs. Such a
            # record can never match, so complain at startup rather than let it
            # be discovered as an unexplained 401 in production.
            logger.warning("api_key_hash_malformed", key_id=record.id)
            return
        existing = self._by_hash.get(digest)
        if existing is not None:
            logger.warning("api_key_hash_duplicate", key_id=record.id, shadowed=existing.id)
        self._by_hash[digest] = record


def _parse_records(payload: str) -> list[ApiKeyRecord]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"api key config is not valid JSON: {exc.msg} at position {exc.pos}"
        ) from exc
    if not isinstance(data, list):
        raise ValueError("api key config must be a JSON array of key records")

    records: list[ApiKeyRecord] = []
    for index, item in enumerate(data):
        try:
            records.append(ApiKeyRecord.model_validate(item))
        except ValidationError as exc:
            # `from None` rather than `from exc`: pydantic renders the offending
            # input value into its message, and that value is a key digest.
            fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors())
            raise ValueError(f"api key record {index} is invalid; check fields: {fields}") from None
    return records
