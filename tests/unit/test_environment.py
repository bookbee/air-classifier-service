"""Pins the guarantees ``tests/conftest.py`` makes to every other test.

Worth its own file because these are the failures that produce no signal: if the
hermeticity fixture stops working, nothing goes red — the suite simply starts
reading the developer's environment, and a test that passes locally begins
failing in CI (or, worse, the other way round) for reasons no failure message
mentions.
"""

from __future__ import annotations

import os

import pytest

from air_sentiment.cache.memory import MemoryCache
from air_sentiment.config import Settings


def test_no_air_sentiment_environment_leaks_into_the_suite() -> None:
    assert [name for name in os.environ if name.upper().startswith("AIR_SENTIMENT__")] == []


def test_dotenv_is_not_read_during_tests() -> None:
    assert Settings.model_config["env_file"] is None


def test_settings_fixture_is_a_test_environment(settings: Settings) -> None:
    assert settings.app.env == "test"
    assert settings.app.is_production is False


@pytest.mark.parametrize("ttl", [60, 0])
async def test_memory_cache_fixture_is_usable(memory_cache: MemoryCache, ttl: int) -> None:
    await memory_cache.set("k", "v", ttl)
    assert await memory_cache.get("k") == "v"
    assert await memory_cache.incr("c", ttl) == 1


async def test_memory_cache_fixture_is_not_shared_between_tests(
    memory_cache: MemoryCache,
) -> None:
    assert memory_cache.size == 0
