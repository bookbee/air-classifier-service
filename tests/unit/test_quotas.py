"""Unit tests for the rate limiter."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from air_sentiment.config import CacheSettings, SecuritySettings, Settings
from air_sentiment.security.api_keys import ApiKeyRecord, hash_key
from air_sentiment.security.quotas import WINDOW_SECONDS, RateLimiter, TokenBucket


class FakeCache:
    """In-memory stand-in for a `CacheBackend`, with no TTL expiry."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        value = self.counters.get(key)
        return None if value is None else str(value)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.counters[key] = int(value)
        self.ttls[key] = ttl_seconds

    async def delete(self, key: str) -> None:
        self.counters.pop(key, None)

    async def incr(self, key: str, ttl_seconds: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        self.ttls.setdefault(key, ttl_seconds)
        return self.counters[key]

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class BrokenCache(FakeCache):
    async def incr(self, key: str, ttl_seconds: int) -> int:
        raise RuntimeError("redis is gone")


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _settings(**security: Any) -> Settings:
    return Settings(
        cache=CacheSettings(key_prefix="test:"),
        security=SecuritySettings(hash_salt=SecretStr("salt"), **security),
    )


# ── TokenBucket ───────────────────────────────────────────────────────────────


async def test_bucket_allows_up_to_the_limit_then_denies() -> None:
    bucket = TokenBucket(FakeCache(), clock=Clock())

    for expected_remaining in (2, 1, 0):
        result = await bucket.consume("key-1", limit=3)
        assert result.allowed
        assert result.remaining == expected_remaining

    denied = await bucket.consume("key-1", limit=3)
    assert not denied.allowed
    assert denied.remaining == 0


async def test_retry_after_is_positive_when_denied() -> None:
    clock = Clock()
    bucket = TokenBucket(FakeCache(), clock=clock)

    await bucket.consume("key-1", limit=1)
    denied = await bucket.consume("key-1", limit=1)

    assert not denied.allowed
    assert denied.retry_after_s > 0.0
    assert denied.retry_after_s <= WINDOW_SECONDS
    assert denied.reset_at_s > clock.now


async def test_retry_after_is_zero_while_allowed() -> None:
    result = await TokenBucket(FakeCache(), clock=Clock()).consume("key-1", limit=5)
    assert result.retry_after_s == 0.0


async def test_key_ids_do_not_share_a_bucket() -> None:
    bucket = TokenBucket(FakeCache(), clock=Clock())

    assert (await bucket.consume("key-1", limit=1)).allowed
    assert not (await bucket.consume("key-1", limit=1)).allowed
    # A second caller's budget is untouched by the first exhausting theirs.
    assert (await bucket.consume("key-2", limit=1)).allowed


async def test_window_rollover_resets_the_budget() -> None:
    clock = Clock()
    bucket = TokenBucket(FakeCache(), clock=clock)

    await bucket.consume("key-1", limit=1)
    assert not (await bucket.consume("key-1", limit=1)).allowed

    clock.advance(WINDOW_SECONDS)
    assert (await bucket.consume("key-1", limit=1)).allowed


async def test_counter_ttl_outlives_its_window() -> None:
    cache = FakeCache()
    await TokenBucket(cache, clock=Clock()).consume("key-1", limit=1)

    assert all(ttl > WINDOW_SECONDS for ttl in cache.ttls.values())


async def test_prefix_scopes_the_counter_key() -> None:
    cache = FakeCache()
    await TokenBucket(cache, prefix="test:", clock=Clock()).consume("key-1", limit=1)

    assert all(key.startswith("test:rl:") and key.endswith(":key-1") for key in cache.counters)


async def test_backend_failure_fails_open() -> None:
    result = await TokenBucket(BrokenCache(), clock=Clock()).consume("key-1", limit=1)

    assert result.allowed
    assert result.remaining == 1
    assert result.retry_after_s == 0.0


async def test_non_positive_limit_still_admits_one_request() -> None:
    bucket = TokenBucket(FakeCache(), clock=Clock())

    assert (await bucket.consume("key-1", limit=0)).allowed
    assert not (await bucket.consume("key-1", limit=0)).allowed


# ── Headers ───────────────────────────────────────────────────────────────────


async def test_headers_carry_retry_after_only_when_denied() -> None:
    bucket = TokenBucket(FakeCache(), clock=Clock())

    allowed = (await bucket.consume("key-1", limit=1)).headers()
    assert allowed["RateLimit-Limit"] == "1"
    assert allowed["RateLimit-Remaining"] == "0"
    assert "Retry-After" not in allowed

    denied = (await bucket.consume("key-1", limit=1)).headers()
    assert int(denied["Retry-After"]) >= 1


# ── RateLimiter ───────────────────────────────────────────────────────────────


async def test_limiter_window_admits_rpm_plus_burst() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=2, default_burst=1))

    for _ in range(3):
        assert (await limiter.check("key-1")).allowed
    denied = await limiter.check("key-1")

    assert not denied.allowed
    assert denied.limit == 3
    assert denied.retry_after_s > 0.0


async def test_limiter_explicit_arguments_override_defaults() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=600, default_burst=60))

    assert (await limiter.check("key-1", rpm=1, burst=0)).allowed
    assert not (await limiter.check("key-1", rpm=1, burst=0)).allowed


async def test_limiter_falls_back_to_defaults_per_argument() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=5, default_burst=1))

    result = await limiter.check("key-1", burst=0)
    assert result.limit == 5


async def test_check_key_uses_the_records_overrides() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=600, default_burst=60))
    record = ApiKeyRecord(
        id="key-slow",
        name="slow",
        key_hash=hash_key("airs_raw", "salt"),
        rate_limit_rpm=1,
        burst=0,
    )

    assert (await limiter.check_key(record)).allowed
    assert not (await limiter.check_key(record)).allowed


async def test_check_key_without_overrides_uses_service_defaults() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=2, default_burst=1))
    record = ApiKeyRecord(id="key-default", name="default", key_hash=hash_key("airs_raw", "salt"))

    result = await limiter.check_key(record)
    assert result.limit == 3


async def test_limiter_keeps_key_ids_isolated() -> None:
    limiter = RateLimiter(FakeCache(), _settings(default_rate_limit_rpm=1, default_burst=1))

    assert (await limiter.check("key-1")).allowed
    assert (await limiter.check("key-1")).allowed
    assert not (await limiter.check("key-1")).allowed
    assert (await limiter.check("key-2")).allowed


@pytest.mark.parametrize("replicas", [1, 3])
async def test_replicas_sharing_a_backend_share_one_budget(replicas: int) -> None:
    cache = FakeCache()
    settings = _settings(default_rate_limit_rpm=1, default_burst=1)
    limiters = [RateLimiter(cache, settings) for _ in range(replicas)]

    # The budget belongs to the key, not to whichever replica took the request.
    for index in range(2):
        assert (await limiters[index % replicas].check("key-1")).allowed
    assert not (await limiters[0].check("key-1")).allowed
