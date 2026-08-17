"""Cost arithmetic against hand-computed figures.

The escalation policy stops the ladder on spend, so these numbers are control
flow, not reporting — a factor-of-1000 slip here would either strand every
request at T0 or ignore the ceiling entirely.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from air_classifier.observability import cost as cost_module
from air_classifier.observability.cost import (
    PRICE_TABLE,
    CostMeter,
    ModelPrice,
    estimate_cost,
    override_prices,
    price_for,
    reset_price_table,
)
from air_classifier.schemas.common import TierUsage

MTOK = 1_000_000


@pytest.fixture(autouse=True)
def _pristine_table() -> Iterator[None]:
    reset_price_table()
    yield
    reset_price_table()


@dataclass(frozen=True)
class _Recorder:
    """Stands in for the structlog logger."""

    calls: list[tuple[str, dict[str, Any]]]

    def warning(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


# ── The published table ───────────────────────────────────────────────────────


def test_haiku_list_prices() -> None:
    price = PRICE_TABLE["claude-haiku-4-5"]
    assert price.input_per_mtok == 1.00
    assert price.output_per_mtok == 5.00
    # Cache reads bill at ~0.1x input, 1h cache writes at ~2x input.
    assert price.cache_read_per_mtok == pytest.approx(0.10)
    assert price.cache_write_per_mtok == pytest.approx(2.00)


def test_sonnet_list_prices() -> None:
    price = PRICE_TABLE["claude-sonnet-5"]
    assert price.input_per_mtok == 3.00
    assert price.output_per_mtok == 15.00
    assert price.cache_read_per_mtok == pytest.approx(0.30)
    assert price.cache_write_per_mtok == pytest.approx(6.00)


def test_local_models_are_free() -> None:
    assert PRICE_TABLE["llama3.2:3b"].is_free
    # A retagged local model must not fall off the table into "unknown".
    for tag in ("llama3.1:8b", "qwen2.5:7b", "mistral:7b", "gemma2:9b"):
        priced = price_for(tag)
        assert priced is not None
        assert priced.is_free


# ── Arithmetic ────────────────────────────────────────────────────────────────


def test_haiku_one_million_each_way() -> None:
    # 1.00 in + 5.00 out
    assert estimate_cost("claude-haiku-4-5", tokens_in=MTOK, tokens_out=MTOK) == pytest.approx(6.00)


def test_haiku_realistic_call() -> None:
    # 4_000 in  → 4_000 / 1e6 * 1.00 = 0.004
    #   500 out →   500 / 1e6 * 5.00 = 0.0025
    assert estimate_cost("claude-haiku-4-5", tokens_in=4_000, tokens_out=500) == pytest.approx(
        0.0065
    )


def test_sonnet_arbitration_call() -> None:
    # 8_000 in  → 8_000 / 1e6 *  3.00 = 0.024
    # 1_000 out → 1_000 / 1e6 * 15.00 = 0.015
    assert estimate_cost("claude-sonnet-5", tokens_in=8_000, tokens_out=1_000) == pytest.approx(
        0.039
    )


def test_cache_tokens_bill_at_their_own_rates() -> None:
    # Reads at 0.1x input, writes at 2x input — the whole reason prompt caching
    # pays for itself only after the prefix is reused.
    assert estimate_cost(
        "claude-haiku-4-5", tokens_in=0, tokens_out=0, cache_read_tokens=MTOK
    ) == pytest.approx(0.10)
    assert estimate_cost(
        "claude-haiku-4-5", tokens_in=0, tokens_out=0, cache_write_tokens=MTOK
    ) == pytest.approx(2.00)


def test_cached_call_is_cheaper_than_the_uncached_one() -> None:
    uncached = estimate_cost("claude-haiku-4-5", tokens_in=8_192, tokens_out=200)
    cached = estimate_cost("claude-haiku-4-5", tokens_in=0, tokens_out=200, cache_read_tokens=8_192)
    assert cached < uncached


def test_full_call_sums_every_class() -> None:
    #  2_000 in         → 0.002
    #    400 out        → 0.002
    #  6_000 cache_read → 6_000 / 1e6 * 0.10 = 0.0006
    # 10_000 cache_write→ 10_000 / 1e6 * 2.00 = 0.02
    assert estimate_cost(
        "claude-haiku-4-5",
        tokens_in=2_000,
        tokens_out=400,
        cache_read_tokens=6_000,
        cache_write_tokens=10_000,
    ) == pytest.approx(0.0246)


def test_zero_tokens_costs_nothing() -> None:
    assert estimate_cost("claude-sonnet-5", tokens_in=0, tokens_out=0) == 0.0


def test_local_model_costs_nothing() -> None:
    assert estimate_cost("llama3.2:3b", tokens_in=50_000, tokens_out=2_000) == 0.0


def test_model_name_is_normalised() -> None:
    assert estimate_cost("  Claude-Haiku-4-5 ", tokens_in=MTOK, tokens_out=0) == pytest.approx(1.00)


# ── Unknown models ────────────────────────────────────────────────────────────


def test_unknown_model_is_free_not_fatal() -> None:
    assert estimate_cost("gpt-9-omega", tokens_in=MTOK, tokens_out=MTOK) == 0.0
    assert price_for("gpt-9-omega") is None


def test_unknown_model_warns_once_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(calls=[])
    monkeypatch.setattr(cost_module, "logger", recorder)

    for _ in range(100):
        estimate_cost("gpt-9-omega", tokens_in=10, tokens_out=10)
    estimate_cost("some-other-model", tokens_in=10, tokens_out=10)

    assert [call[1]["model"] for call in recorder.calls] == ["gpt-9-omega", "some-other-model"]


# ── Overrides ─────────────────────────────────────────────────────────────────


def test_override_prices_merges_over_the_table() -> None:
    override_prices(
        {"claude-haiku-4-5": ModelPrice.metered(input_per_mtok=2.00, output_per_mtok=10.00)}
    )

    assert estimate_cost("claude-haiku-4-5", tokens_in=MTOK, tokens_out=0) == pytest.approx(2.00)
    # Untouched entries survive the merge.
    assert estimate_cost("claude-sonnet-5", tokens_in=MTOK, tokens_out=0) == pytest.approx(3.00)

    reset_price_table()
    assert estimate_cost("claude-haiku-4-5", tokens_in=MTOK, tokens_out=0) == pytest.approx(1.00)


def test_override_can_price_a_previously_unknown_model() -> None:
    override_prices({"gpt-9-omega": ModelPrice.metered(input_per_mtok=4.00, output_per_mtok=8.00)})

    assert estimate_cost("gpt-9-omega", tokens_in=MTOK, tokens_out=0) == pytest.approx(4.00)


# ── CostMeter ─────────────────────────────────────────────────────────────────


def test_meter_starts_empty() -> None:
    assert CostMeter().total_usd == 0.0


def test_meter_prefers_the_provider_reported_cost() -> None:
    meter = CostMeter()
    usage = TierUsage(
        tokens_in=4_000,
        tokens_out=500,
        cost_usd=0.001,  # a discounted account: below our list-price estimate
        provider="anthropic",
        model="claude-haiku-4-5",
    )

    assert meter.add(usage) == pytest.approx(0.001)
    assert meter.total_usd == pytest.approx(0.001)


def test_meter_estimates_when_the_provider_reports_no_cost() -> None:
    meter = CostMeter()
    usage = TierUsage(tokens_in=4_000, tokens_out=500, model="claude-haiku-4-5")

    assert meter.add(usage) == pytest.approx(0.0065)


def test_meter_accumulates_across_the_ladder() -> None:
    meter = CostMeter()
    meter.add(TierUsage(tokens_in=4_000, tokens_out=500, model="claude-haiku-4-5"))
    meter.add(TierUsage(tokens_in=8_000, tokens_out=1_000, model="claude-sonnet-5"))

    assert meter.total_usd == pytest.approx(0.0455)
    assert dict(meter.by_model) == {
        "claude-haiku-4-5": pytest.approx(0.0065),
        "claude-sonnet-5": pytest.approx(0.039),
    }


def test_meter_ignores_free_and_zero_entries() -> None:
    meter = CostMeter()
    meter.add(TierUsage(tokens_in=900, tokens_out=90, model="llama3.2:3b"))
    meter.add(TierUsage())

    assert meter.total_usd == 0.0
    assert dict(meter.by_model) == {}


def test_meter_breakdown_is_read_only() -> None:
    meter = CostMeter()
    meter.add_usd("claude-haiku-4-5", 0.01)

    with pytest.raises(TypeError):
        meter.by_model["claude-haiku-4-5"] = 99.0  # type: ignore[index]


def test_meter_accepts_any_usage_shape() -> None:
    @dataclass(frozen=True)
    class _Result:
        tokens_in: int
        tokens_out: int
        cache_read_tokens: int
        cache_write_tokens: int
        cost_usd: float
        model: str

    meter = CostMeter()
    meter.add(
        _Result(
            tokens_in=1_000,
            tokens_out=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=0.0,
            model="claude-sonnet-5",
        )
    )

    assert meter.total_usd == pytest.approx(0.003)
