"""Backend selection, the null backend, and RedisCache's fail-open contract.

The Redis cases live here rather than in a module of their own because the
factory is what wires Redis up; none of them need a live server — the client is
either injected or pointed at a closed port.
"""

from __future__ import annotations

from typing import Any

import pytest

from air_sentiment.cache.base import CacheBackend
from air_sentiment.cache.factory import NullCache, build_cache
from air_sentiment.cache.memory import MemoryCache
from air_sentiment.cache.redis import RedisCache
from air_sentiment.config import CacheSettings, Settings

#: Nothing listens here, so RedisCache must degrade rather than hang or raise.
CLOSED_PORT_URL = "redis://127.0.0.1:1/0"


def settings_with(**cache_kwargs: Any) -> Settings:
    return Settings(cache=CacheSettings(**cache_kwargs))


# ── Dispatch ──────────────────────────────────────────────────────────────


async def test_defaults_to_memory() -> None:
    cache = await build_cache(Settings())
    assert isinstance(cache, MemoryCache)


async def test_memory_backend_honours_max_entries() -> None:
    cache = await build_cache(settings_with(backend="memory", max_entries=2))
    assert isinstance(cache, MemoryCache)

    for key in ("a", "b", "c"):
        await cache.set(key, key, 60)

    assert cache.size == 2


async def test_none_backend_builds_null_cache() -> None:
    cache = await build_cache(settings_with(backend="none"))
    assert isinstance(cache, NullCache)


async def test_redis_backend_builds_without_a_live_server() -> None:
    cache = await build_cache(settings_with(backend="redis", redis_url=CLOSED_PORT_URL))
    try:
        assert isinstance(cache, RedisCache)
    finally:
        await cache.aclose()


@pytest.mark.parametrize("backend", ["memory", "redis", "none"])
async def test_every_backend_satisfies_the_protocol(backend: str) -> None:
    cache = await build_cache(settings_with(backend=backend, redis_url=CLOSED_PORT_URL))
    try:
        assert isinstance(cache, CacheBackend)
    finally:
        await cache.aclose()


# ── NullCache ─────────────────────────────────────────────────────────────


async def test_null_cache_never_retains() -> None:
    cache = NullCache()
    await cache.set("k", "v", 60)

    assert await cache.get("k") is None
    assert await cache.incr("k", 60) == 0
    await cache.delete("k")
    assert await cache.ping() is True
    await cache.aclose()


# ── RedisCache: outages degrade to misses ─────────────────────────────────


class ExplodingRedis:
    """Every command fails, the way a Redis mid-failover does."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("redis is down")

    async def set(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("redis is down")

    async def delete(self, *keys: str) -> int:
        raise ConnectionError("redis is down")

    def pipeline(self, transaction: bool = True) -> Any:
        raise ConnectionError("redis is down")

    async def ping(self) -> bool:
        raise ConnectionError("redis is down")

    async def aclose(self) -> None:
        raise ConnectionError("redis is down")


async def test_get_degrades_to_a_miss() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    assert await cache.get("k") is None


async def test_set_swallows_backend_failure() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    await cache.set("k", "v", 60)


async def test_delete_swallows_backend_failure() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    await cache.delete("k")


async def test_incr_returns_zero_when_uncounted() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    assert await cache.incr("k", 60) == 0


async def test_ping_reports_false_rather_than_raising() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    assert await cache.ping() is False


async def test_aclose_swallows_backend_failure() -> None:
    cache = RedisCache("redis://unused", client=ExplodingRedis())
    await cache.aclose()


async def test_unreachable_server_yields_misses() -> None:
    cache = RedisCache(CLOSED_PORT_URL)
    try:
        assert await cache.ping() is False
        assert await cache.get("k") is None
        await cache.set("k", "v", 60)
        assert await cache.incr("k", 60) == 0
    finally:
        await cache.aclose()


async def test_operations_after_close_are_misses() -> None:
    cache = RedisCache(CLOSED_PORT_URL)
    await cache.aclose()

    assert await cache.get("k") is None
    assert await cache.ping() is False


# ── RedisCache: command shape against a fake server ───────────────────────


class FakePipeline:
    """Queues commands and applies them on execute, like MULTI/EXEC."""

    def __init__(self, server: FakeRedis) -> None:
        self._server = server
        self._queued: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def set(self, key: str, value: Any, **kwargs: Any) -> FakePipeline:
        self._queued.append(("set", (key, value), kwargs))
        return self

    def incr(self, key: str) -> FakePipeline:
        self._queued.append(("incr", (key,), {}))
        return self

    async def execute(self) -> list[Any]:
        return [self._server.apply(name, *args, **kwargs) for name, args, kwargs in self._queued]


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}
        #: One entry per TTL actually written, so tests can count them.
        self.expiry_writes: list[str] = []

    def apply(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "set":
            key, value = args
            if kwargs.get("nx") and key in self.values:
                return None
            self.values[key] = str(value)
            if kwargs.get("ex") is not None:
                self.expiries[key] = int(kwargs["ex"])
                self.expiry_writes.append(key)
            return True
        key = args[0]
        self.values[key] = str(int(self.values.get(key, "0")) + 1)
        return int(self.values[key])

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def get(self, key: str) -> bytes | None:
        raw = self.values.get(key)
        return raw.encode() if raw is not None else None

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        return bool(self.apply("set", key, value, ex=ex))

    async def delete(self, *keys: str) -> int:
        return sum(self.values.pop(key, None) is not None for key in keys)


async def test_roundtrip_against_fake_server() -> None:
    server = FakeRedis()
    cache = RedisCache("redis://unused", client=server)

    await cache.set("k", "v", 60)

    assert await cache.get("k") == "v"
    assert server.expiries["k"] == 60


async def test_set_without_ttl_leaves_key_unexpiring() -> None:
    server = FakeRedis()
    cache = RedisCache("redis://unused", client=server)

    await cache.set("k", "v", 0)

    assert "k" not in server.expiries


async def test_delete_removes_key_from_server() -> None:
    server = FakeRedis()
    cache = RedisCache("redis://unused", client=server)
    await cache.set("k", "v", 60)

    await cache.delete("k")

    assert await cache.get("k") is None


async def test_incr_sets_ttl_only_on_first_write() -> None:
    server = FakeRedis()
    cache = RedisCache("redis://unused", client=server)

    counts = [await cache.incr("c", 60) for _ in range(3)]

    assert counts == [1, 2, 3]
    assert server.values["c"] == "3"
    assert server.expiries["c"] == 60
    # The window is seeded once; later hits must not push its deadline out.
    assert server.expiry_writes == ["c"]
