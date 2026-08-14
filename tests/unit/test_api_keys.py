"""Unit tests for the API key store."""

from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from air_sentiment.config import SecuritySettings, Settings
from air_sentiment.constants import Tier
from air_sentiment.security.api_keys import (
    ALL_SCOPES,
    ANONYMOUS_KEY_ID,
    KEY_PREFIX,
    ApiKeyRecord,
    ApiKeyStore,
    generate_key,
    hash_key,
)

SALT = "unit-test-salt"
RAW_KEY = "airs_known-raw-key"


def _record(**overrides: Any) -> ApiKeyRecord:
    fields: dict[str, Any] = {
        "id": "key-1",
        "name": "primary",
        "key_hash": hash_key(RAW_KEY, SALT),
        "scopes": {"sentiment:read"},
    }
    fields.update(overrides)
    return ApiKeyRecord(**fields)


def _store(*records: ApiKeyRecord, allow_unauthenticated: bool = False) -> ApiKeyStore:
    return ApiKeyStore(records, salt=SALT, allow_unauthenticated=allow_unauthenticated)


def _settings(**security: Any) -> Settings:
    return Settings(security=SecuritySettings(hash_salt=SecretStr(SALT), **security))


# ── Hashing ───────────────────────────────────────────────────────────────────


def test_hash_key_is_deterministic_lowercase_sha256_hex() -> None:
    digest = hash_key(RAW_KEY, SALT)
    assert digest == hash_key(RAW_KEY, SALT)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(char in "0123456789abcdef" for char in digest)


def test_hash_key_is_salted() -> None:
    assert hash_key(RAW_KEY, SALT) != hash_key(RAW_KEY, "other-salt")


def test_generate_key_is_prefixed_and_hashes_consistently() -> None:
    raw, digest = generate_key(SALT)
    assert raw.startswith(KEY_PREFIX)
    assert digest == hash_key(raw, SALT)
    assert len(digest) == 64


def test_generate_key_is_unique_per_call() -> None:
    first, _ = generate_key(SALT)
    second, _ = generate_key(SALT)
    assert first != second


# ── Lookup ────────────────────────────────────────────────────────────────────


def test_lookup_resolves_a_known_key() -> None:
    record = _record()
    assert _store(record).lookup(RAW_KEY) is record


def test_lookup_of_unknown_key_returns_none() -> None:
    assert _store(_record()).lookup("airs_not-a-real-key") is None


def test_lookup_of_empty_or_missing_key_returns_none() -> None:
    store = _store(_record())
    assert store.lookup(None) is None
    assert store.lookup("") is None


def test_lookup_of_disabled_key_returns_none() -> None:
    store = _store(_record(enabled=False))
    assert store.lookup(RAW_KEY) is None


def test_disabled_key_is_rejected_even_when_unauthenticated_is_allowed() -> None:
    store = _store(_record(enabled=False), allow_unauthenticated=True)
    assert store.lookup(RAW_KEY) is None


def test_lookup_uses_compare_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    real = hmac.compare_digest

    def spy(a: object, b: object) -> bool:
        assert isinstance(a, str) and isinstance(b, str)
        calls.append((a, b))
        return bool(real(a, b))

    monkeypatch.setattr(hmac, "compare_digest", spy)
    assert _store(_record()).lookup(RAW_KEY) is not None
    assert calls == [(hash_key(RAW_KEY, SALT), hash_key(RAW_KEY, SALT))]


def test_compare_digest_is_authoritative_not_decorative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: False)
    assert _store(_record()).lookup(RAW_KEY) is None


def test_records_with_a_malformed_hash_are_not_indexed() -> None:
    # A raw key pasted where a digest belongs must never become reachable.
    store = _store(_record(key_hash=RAW_KEY))
    assert len(store) == 0
    assert store.lookup(RAW_KEY) is None


def test_uppercase_hash_in_config_still_matches() -> None:
    store = _store(_record(key_hash=hash_key(RAW_KEY, SALT).upper()))
    assert store.lookup(RAW_KEY) is not None


# ── Record semantics ──────────────────────────────────────────────────────────


def test_repr_and_str_never_expose_the_hash() -> None:
    record = _record()
    for rendered in (repr(record), str(record), f"{record}"):
        assert record.key_hash not in rendered
        assert "key-1" in rendered
        assert "primary" in rendered


def test_record_defaults_match_the_contract() -> None:
    record = _record()
    assert record.max_tier is Tier.T3_CLOUD_LLM
    assert record.force_pii_redaction is False
    assert record.rate_limit_rpm is None
    assert record.burst is None
    assert record.enabled is True


def test_has_scope_and_clamp_tier() -> None:
    record = _record(scopes={"sentiment:read"}, max_tier=Tier.T1_CLASSIFIER)
    assert record.has_scope("sentiment:read")
    assert not record.has_scope("admin:read")
    assert record.clamp_tier(Tier.T3_CLOUD_LLM) is Tier.T1_CLASSIFIER
    assert record.clamp_tier(Tier.T0_RULES) is Tier.T0_RULES


# ── Loading ───────────────────────────────────────────────────────────────────


def test_load_from_inline_json_round_trips_every_field() -> None:
    payload = [
        {
            "id": "key-a",
            "name": "restricted",
            "key_hash": hash_key(RAW_KEY, SALT),
            "scopes": ["sentiment:read", "admin:read"],
            "max_tier": "t1_classifier",
            "force_pii_redaction": True,
            "rate_limit_rpm": 30,
            "burst": 5,
        }
    ]
    store = ApiKeyStore.load(_settings(api_keys_inline=json.dumps(payload)))

    record = store.lookup(RAW_KEY)
    assert record is not None
    assert record.id == "key-a"
    assert record.scopes == {"sentiment:read", "admin:read"}
    assert record.max_tier is Tier.T1_CLASSIFIER
    assert record.force_pii_redaction is True
    assert record.rate_limit_rpm == 30
    assert record.burst == 5


def test_load_from_file(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    payload = [{"id": "key-f", "name": "from-file", "key_hash": hash_key(RAW_KEY, SALT)}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    record = ApiKeyStore.load(_settings(api_keys_file=path)).lookup(RAW_KEY)
    assert record is not None
    assert record.id == "key-f"


def test_inline_takes_precedence_over_file(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps([{"id": "from-file", "name": "f", "key_hash": hash_key(RAW_KEY, SALT)}]),
        encoding="utf-8",
    )
    inline = json.dumps([{"id": "from-inline", "name": "i", "key_hash": hash_key(RAW_KEY, SALT)}])

    store = ApiKeyStore.load(_settings(api_keys_inline=inline, api_keys_file=path))
    record = store.lookup(RAW_KEY)
    assert record is not None
    assert record.id == "from-inline"


def test_load_with_no_configured_keys_yields_an_empty_store() -> None:
    store = ApiKeyStore.load(_settings())
    assert len(store) == 0
    assert store.lookup(RAW_KEY) is None


def test_load_rejects_non_array_json() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        ApiKeyStore.load(_settings(api_keys_inline='{"id": "key-a"}'))


def test_load_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        ApiKeyStore.load(_settings(api_keys_inline="[{"))


def test_invalid_record_error_names_fields_but_not_values() -> None:
    digest = hash_key(RAW_KEY, SALT)
    inline = json.dumps([{"name": "no-id", "key_hash": digest}])

    with pytest.raises(ValueError) as excinfo:
        ApiKeyStore.load(_settings(api_keys_inline=inline))

    message = str(excinfo.value)
    assert "id" in message
    assert digest not in message


def test_load_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="api_keys_file"):
        ApiKeyStore.load(_settings(api_keys_file=tmp_path / "absent.json"))


# ── Development fallback ──────────────────────────────────────────────────────


def test_anonymous_fallback_when_unauthenticated_is_allowed() -> None:
    store = ApiKeyStore.load(_settings(allow_unauthenticated=True))
    record = store.lookup(None)

    assert record is not None
    assert record.id == ANONYMOUS_KEY_ID
    assert record.scopes == set(ALL_SCOPES)
    assert record.enabled is True


def test_anonymous_record_is_unreachable_through_a_presented_key() -> None:
    # Its empty digest must never be matched by hashing anything a caller sends.
    store = _store(ApiKeyRecord.anonymous())
    assert len(store) == 0
    assert store.lookup("") is None
