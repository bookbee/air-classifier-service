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

from air_classifier.cache.memory import MemoryCache
from air_classifier.config import Settings


def test_no_air_classifier_environment_leaks_into_the_suite() -> None:
    assert [name for name in os.environ if name.upper().startswith("AIR_CLASSIFIER__")] == []


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


# ── Partial nested overrides ──────────────────────────────────────────────────
# Every per-rung threshold is documented in .env.example as an individually
# settable key, and the README promises the ladder can be retuned without a
# deploy. That only holds if setting one key leaves the rest of its rung alone —
# which is a property of how the defaults are declared, not something the type
# system enforces. These pin it.


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("AIR_CLASSIFIER__TIERS__T0_RULES__ENABLED", "true"),
        ("AIR_CLASSIFIER__TIERS__T0_RULES__TIMEOUT_MS", "7"),
    ],
)
def test_overriding_one_t0_key_preserves_the_rest_of_the_rung(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    """The 0.95 floor is the whole safety argument for the T0 shortcuts.

    Losing it to an unrelated override would let a weak lexicon match decide a
    request outright, and nothing would report that it had happened.
    """
    monkeypatch.setenv(key, value)
    assert Settings().tiers.t0_rules.confidence_floor == 0.95


def test_overriding_a_floor_preserves_that_rungs_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2's 6s ceiling was measured against its own p90; the base default is 1s."""
    monkeypatch.setenv("AIR_CLASSIFIER__TIERS__T2_LOCAL_LLM__CONFIDENCE_FLOOR", "0.99")
    tier = Settings().tiers.t2_local_llm
    assert (tier.confidence_floor, tier.margin_floor, tier.timeout_ms) == (0.99, 0.20, 6000)


def test_overriding_one_rung_leaves_the_others_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIR_CLASSIFIER__TIERS__T3_CLOUD_LLM__CONFIDENCE_FLOOR", "0.70")
    tiers = Settings().tiers
    assert (tiers.t3_cloud_llm.confidence_floor, tiers.t3_cloud_llm.timeout_ms) == (0.70, 8000)
    assert (tiers.t1_classifier.margin_floor, tiers.t1_classifier.timeout_ms) == (0.25, 400)
    assert tiers.t0_rules.timeout_ms == 5


def test_overriding_one_air_llm_key_preserves_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same hazard on the air_llm block, where a reset model id is silent."""
    monkeypatch.setenv("AIR_CLASSIFIER__AIR_LLM__TIMEOUT_S", "45")
    air_llm = Settings().air_llm
    assert air_llm.timeout_s == 45.0
    assert air_llm.cloud_model == "classify-default"
    assert air_llm.max_tokens == 2048
    assert air_llm.summary_model == "summarize-default"
    assert air_llm.summary_max_tokens == 512


def test_overriding_the_summary_model_preserves_its_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIR_CLASSIFIER__AIR_LLM__SUMMARY_MODEL", "summarize-custom")
    air_llm = Settings().air_llm
    assert air_llm.summary_model == "summarize-custom"
    assert air_llm.summary_max_tokens == 512
    assert air_llm.summary_timeout_s == 10.0
    assert air_llm.cloud_model == "classify-default"


def test_overriding_one_audit_key_preserves_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit has no zero-config fallback like cache does — "none" ships by default."""
    monkeypatch.setenv("AIR_CLASSIFIER__AUDIT__TTL_DAYS", "7")
    audit = Settings().audit
    assert audit.backend == "none"
    assert audit.ttl_days == 7
    assert audit.database == "air_classifier"
    assert audit.queue_max_size == 1000


def test_a_bare_tier_settings_merges_rather_than_resets() -> None:
    """`TiersSettings(t2=TierSettings(timeout_ms=...))` means change one knob."""
    from air_classifier.config import TierSettings, TiersSettings

    tier = TiersSettings(t2_local_llm=TierSettings(timeout_ms=4000)).t2_local_llm
    assert (tier.timeout_ms, tier.confidence_floor, tier.margin_floor) == (4000, 0.72, 0.20)
