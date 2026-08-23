"""The metric catalogue is a contract: dashboards and alerts bind to these exact
names and label sets, so a rename is a breaking change and gets to fail here."""

from __future__ import annotations

import prometheus_client
import pytest

from air_classifier.constants import EscalationReason, PIIKind, Provider, Tier
from air_classifier.observability import metrics
from air_classifier.observability.metrics import BreakerState, CacheEvent, TierOutcome

CATALOGUE = (
    "air_classifier_requests_total",
    "air_classifier_request_duration_seconds",
    "air_classifier_tier_decisions_total",
    "air_classifier_escalations_total",
    "air_classifier_tier_duration_seconds",
    "air_classifier_decided_by_total",
    "air_classifier_llm_tokens_total",
    "air_classifier_llm_cost_usd_total",
    "air_classifier_cache_events_total",
    "air_classifier_breaker_state",
    "air_classifier_degraded_responses_total",
    "air_classifier_pii_redactions_total",
)


def sample(name: str, **labels: str) -> float:
    """Current value of one series, treating 'never observed' as zero."""
    value = metrics.get_registry().get_sample_value(name, labels)
    return 0.0 if value is None else value


# ── Registry ──────────────────────────────────────────────────────────────────


def test_uses_a_private_registry() -> None:
    assert metrics.get_registry() is not prometheus_client.REGISTRY


def test_catalogue_is_exactly_the_documented_set() -> None:
    payload, _ = metrics.render_latest()
    rendered = payload.decode()
    for name in CATALOGUE:
        assert f"# HELP {name} " in rendered

    # `_created` series are prometheus_client bookkeeping, not part of the catalogue.
    declared = {
        line.split(" ")[2]
        for line in rendered.splitlines()
        if line.startswith("# HELP ") and not line.split(" ")[2].endswith("_created")
    }
    assert declared == set(CATALOGUE)


def test_render_latest_returns_payload_and_content_type() -> None:
    payload, content_type = metrics.render_latest()
    assert isinstance(payload, bytes)
    assert content_type.startswith("text/plain")


# ── Buckets ───────────────────────────────────────────────────────────────────


def test_tier_buckets_span_sub_millisecond_to_seconds() -> None:
    # T0 is measured in tens of microseconds and T3 in seconds; a bucket set that
    # cannot resolve both ends makes at least one tier's P95 meaningless.
    assert min(metrics.TIER_DURATION_BUCKETS) <= 0.001
    assert max(metrics.TIER_DURATION_BUCKETS) >= 10.0
    assert list(metrics.TIER_DURATION_BUCKETS) == sorted(metrics.TIER_DURATION_BUCKETS)


def test_request_buckets_reach_the_latency_budget_ceiling() -> None:
    # Options.latency_budget_ms tops out at 30_000.
    assert max(metrics.REQUEST_DURATION_BUCKETS) >= 30.0
    assert list(metrics.REQUEST_DURATION_BUCKETS) == sorted(metrics.REQUEST_DURATION_BUCKETS)


# ── Recording helpers ─────────────────────────────────────────────────────────


def test_record_request_labels_the_key_id_not_the_key() -> None:
    labels = {"endpoint": "/v1/classify", "status": "200", "api_key_id": "key_01"}
    before = sample("air_classifier_requests_total", **labels)

    metrics.record_request(endpoint="/v1/classify", status=200, api_key_id="key_01")

    assert sample("air_classifier_requests_total", **labels) == before + 1
    payload, _ = metrics.render_latest()
    assert b'api_key_id="key_01"' in payload


def test_record_request_times_when_given_a_duration() -> None:
    before = sample("air_classifier_request_duration_seconds_count", endpoint="/v1/classify")

    metrics.record_request(
        endpoint="/v1/classify", status=200, api_key_id="key_01", duration_s=0.25
    )

    assert sample("air_classifier_request_duration_seconds_count", endpoint="/v1/classify") == (
        before + 1
    )


def test_record_tier_decision() -> None:
    labels = {"tier": Tier.T1_CLASSIFIER.value, "outcome": TierOutcome.ESCALATE.value}
    before = sample("air_classifier_tier_decisions_total", **labels)

    metrics.record_tier_decision(Tier.T1_CLASSIFIER, TierOutcome.ESCALATE)

    assert sample("air_classifier_tier_decisions_total", **labels) == before + 1


def test_record_escalation() -> None:
    labels = {
        "from_tier": Tier.T1_CLASSIFIER.value,
        "to_tier": Tier.T2_LOCAL_LLM.value,
        "reason": EscalationReason.LOW_CONFIDENCE.value,
    }
    before = sample("air_classifier_escalations_total", **labels)

    metrics.record_escalation(
        Tier.T1_CLASSIFIER, Tier.T2_LOCAL_LLM, EscalationReason.LOW_CONFIDENCE
    )

    assert sample("air_classifier_escalations_total", **labels) == before + 1


def test_record_decided_by() -> None:
    before = sample("air_classifier_decided_by_total", tier=Tier.T0_RULES.value)

    metrics.record_decided_by(Tier.T0_RULES)

    assert sample("air_classifier_decided_by_total", tier=Tier.T0_RULES.value) == before + 1


def test_record_llm_usage_splits_tokens_by_billing_class() -> None:
    def tokens(kind: str) -> float:
        return sample(
            "air_classifier_llm_tokens_total",
            provider=Provider.ANTHROPIC.value,
            model="claude-haiku-4-5",
            kind=kind,
        )

    before = {kind: tokens(kind) for kind in ("in", "out", "cache_read", "cache_write")}
    before_cost = sample(
        "air_classifier_llm_cost_usd_total",
        provider=Provider.ANTHROPIC.value,
        model="claude-haiku-4-5",
    )

    metrics.record_llm_usage(
        Provider.ANTHROPIC,
        "claude-haiku-4-5",
        tokens_in=1200,
        tokens_out=300,
        cache_read=4096,
        cache_write=8192,
        cost_usd=0.0031,
    )

    assert tokens("in") == before["in"] + 1200
    assert tokens("out") == before["out"] + 300
    assert tokens("cache_read") == before["cache_read"] + 4096
    assert tokens("cache_write") == before["cache_write"] + 8192
    assert sample(
        "air_classifier_llm_cost_usd_total",
        provider=Provider.ANTHROPIC.value,
        model="claude-haiku-4-5",
    ) == pytest.approx(before_cost + 0.0031)


def test_record_llm_usage_skips_zero_valued_classes() -> None:
    metrics.record_llm_usage(
        Provider.OLLAMA, "llama3.2:3b", tokens_in=64, tokens_out=16, cost_usd=0.0
    )

    registry = metrics.get_registry()
    assert (
        registry.get_sample_value(
            "air_classifier_llm_tokens_total",
            {"provider": Provider.OLLAMA.value, "model": "llama3.2:3b", "kind": "cache_read"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "air_classifier_llm_cost_usd_total",
            {"provider": Provider.OLLAMA.value, "model": "llama3.2:3b"},
        )
        is None
    )


def test_record_cache_event() -> None:
    before = sample("air_classifier_cache_events_total", result=CacheEvent.HIT.value)

    metrics.record_cache_event(CacheEvent.HIT)

    assert sample("air_classifier_cache_events_total", result=CacheEvent.HIT.value) == before + 1


def test_breaker_state_is_a_gauge_not_a_counter() -> None:
    metrics.set_breaker_state(Provider.ANTHROPIC, BreakerState.OPEN)
    assert sample("air_classifier_breaker_state", provider=Provider.ANTHROPIC.value) == 2.0

    metrics.set_breaker_state(Provider.ANTHROPIC, BreakerState.HALF_OPEN)
    assert sample("air_classifier_breaker_state", provider=Provider.ANTHROPIC.value) == 1.0

    metrics.set_breaker_state(Provider.ANTHROPIC, BreakerState.CLOSED)
    assert sample("air_classifier_breaker_state", provider=Provider.ANTHROPIC.value) == 0.0


def test_breaker_state_from_name() -> None:
    assert BreakerState.from_name("half_open") is BreakerState.HALF_OPEN
    assert BreakerState.from_name("OPEN") is BreakerState.OPEN
    with pytest.raises(KeyError):
        BreakerState.from_name("melted")


def test_record_degraded() -> None:
    before = sample(
        "air_classifier_degraded_responses_total", reason=EscalationReason.TIER_FAILED.value
    )

    metrics.record_degraded(EscalationReason.TIER_FAILED)

    assert (
        sample("air_classifier_degraded_responses_total", reason=EscalationReason.TIER_FAILED.value)
        == before + 1
    )


def test_record_pii_redaction_counts_spans() -> None:
    before = sample("air_classifier_pii_redactions_total", kind=PIIKind.EMAIL.value)

    metrics.record_pii_redaction(PIIKind.EMAIL, count=3)
    metrics.record_pii_redaction(PIIKind.EMAIL)

    assert sample("air_classifier_pii_redactions_total", kind=PIIKind.EMAIL.value) == before + 4


# ── Timer ─────────────────────────────────────────────────────────────────────


def test_tier_timer_observes_a_duration() -> None:
    tier = Tier.T3_CLOUD_LLM.value
    before = sample("air_classifier_tier_duration_seconds_count", tier=tier)
    before_sum = sample("air_classifier_tier_duration_seconds_sum", tier=tier)

    with metrics.tier_timer(Tier.T3_CLOUD_LLM):
        pass

    assert sample("air_classifier_tier_duration_seconds_count", tier=tier) == before + 1
    assert sample("air_classifier_tier_duration_seconds_sum", tier=tier) >= before_sum


def test_tier_timer_records_a_failed_tier_and_reraises() -> None:
    tier = Tier.T2_LOCAL_LLM.value
    before = sample("air_classifier_tier_duration_seconds_count", tier=tier)

    with pytest.raises(RuntimeError, match="boom"), metrics.tier_timer(Tier.T2_LOCAL_LLM):
        raise RuntimeError("boom")

    assert sample("air_classifier_tier_duration_seconds_count", tier=tier) == before + 1
