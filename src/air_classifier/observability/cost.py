"""Model prices and per-request cost accounting.

Every figure here is USD per **million** tokens, matching how providers publish
their rates, so the table can be eyeballed against a pricing page without
arithmetic.

Cost is a first-class control signal in this service, not a reporting
afterthought: the escalation policy stops the ladder when spend reaches
``Options.max_cost_usd``. That makes an *over*-estimate merely conservative and
an *under*-estimate a way to blow through a caller's budget, which is why the
table carries list prices rather than promotional ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Final, Protocol

from air_classifier.observability.logging import get_logger

logger = get_logger(__name__)

_TOKENS_PER_MTOK: Final[float] = 1_000_000.0

#: Anthropic bills a cache read at roughly 10% of the base input rate.
CACHE_READ_RATIO: Final[float] = 0.1

#: ...and a 1h cache write at roughly 2x input (a 5m write is ~1.25x). The
#: service pins the TTL to 1h (`providers.anthropic.prompt_cache_ttl`), so the
#: larger multiplier is the one that applies.
CACHE_WRITE_RATIO: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """List price for one model, in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float

    @classmethod
    def metered(cls, *, input_per_mtok: float, output_per_mtok: float) -> ModelPrice:
        """A cloud model, with cache rates derived from the published ratios."""
        return cls(
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            cache_read_per_mtok=input_per_mtok * CACHE_READ_RATIO,
            cache_write_per_mtok=input_per_mtok * CACHE_WRITE_RATIO,
        )

    @classmethod
    def free(cls) -> ModelPrice:
        """A model we host ourselves. Not free in reality, but free per token."""
        return cls(
            input_per_mtok=0.0,
            output_per_mtok=0.0,
            cache_read_per_mtok=0.0,
            cache_write_per_mtok=0.0,
        )

    @property
    def is_free(self) -> bool:
        return self.input_per_mtok == 0.0 and self.output_per_mtok == 0.0

    def cost_for(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """USD for one call at this price."""
        return (
            tokens_in * self.input_per_mtok
            + tokens_out * self.output_per_mtok
            + cache_read_tokens * self.cache_read_per_mtok
            + cache_write_tokens * self.cache_write_per_mtok
        ) / _TOKENS_PER_MTOK


#: Published list prices. Overridable at startup via :func:`override_prices`, so
#: a rate change is a deploy-time config edit rather than a code change.
#:
#: Sonnet 5 carries an introductory rate of 2.00/10.00 through 2026-08-31. The
#: table holds the standard 3.00/15.00 deliberately: when the promotion lapses,
#: budgets that were sized against the intro rate would silently start
#: overspending, whereas an over-estimate only ever stops the ladder early.
PRICE_TABLE: Final[Mapping[str, ModelPrice]] = MappingProxyType(
    {
        "claude-haiku-4-5": ModelPrice.metered(input_per_mtok=1.00, output_per_mtok=5.00),
        "claude-sonnet-5": ModelPrice.metered(input_per_mtok=3.00, output_per_mtok=15.00),
        # Ollama runs on hardware we already own; a token costs electricity, and
        # billing it against the caller's budget would suppress the cheap rung
        # the whole ladder is built around.
        "llama3.2:3b": ModelPrice.free(),
    }
)

#: Local-model name prefixes. Swapping the Ollama tag is a config change, so any
#: locally served model is free without needing a row of its own — otherwise a
#: retag would start logging unknown-model warnings for a model that costs
#: nothing.
_LOCAL_MODEL_PREFIXES: Final[tuple[str, ...]] = (
    "llama",
    "qwen",
    "mistral",
    "mixtral",
    "gemma",
    "phi",
    "deepseek",
)

_FREE: Final[ModelPrice] = ModelPrice.free()

_active_prices: dict[str, ModelPrice] = dict(PRICE_TABLE)


class UsageLike(Protocol):
    """Anything that reports what a call consumed.

    Structural rather than a concrete import so both ``TierUsage`` and
    ``ProviderResult`` satisfy it without this module depending on either.
    """

    @property
    def tokens_in(self) -> int: ...

    @property
    def tokens_out(self) -> int: ...

    @property
    def cache_read_tokens(self) -> int: ...

    @property
    def cache_write_tokens(self) -> int: ...

    @property
    def cost_usd(self) -> float: ...

    @property
    def model(self) -> str | None: ...


def override_prices(overrides: Mapping[str, ModelPrice]) -> None:
    """Merge operator-supplied prices over the built-in table.

    Called once at startup. Merging rather than replacing means an override for
    one model does not silently un-price every other one.
    """
    _active_prices.update({_normalise(name): price for name, price in overrides.items()})


def reset_price_table() -> None:
    """Restore the built-in table and re-arm the unknown-model warning."""
    _active_prices.clear()
    _active_prices.update(PRICE_TABLE)
    _warn_unknown_model.cache_clear()


def price_for(model: str) -> ModelPrice | None:
    """The price of a model, or None when we have never heard of it."""
    key = _normalise(model)
    price = _active_prices.get(key)
    if price is not None:
        return price
    if key.startswith(_LOCAL_MODEL_PREFIXES):
        return _FREE
    return None


def estimate_cost(
    model: str,
    *,
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD for one call. An unknown model costs 0.0 and warns once.

    Refusing to price an unknown model would fail a request over a bookkeeping
    gap; pricing it at zero degrades the cost ceiling instead, which is the same
    trade the rest of the service makes. The warning is what turns that into an
    operational signal rather than a silent hole in the budget.
    """
    price = price_for(model)
    if price is None:
        _warn_unknown_model(_normalise(model))
        return 0.0
    return price.cost_for(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


@dataclass(slots=True)
class CostMeter:
    """Running spend for a single request.

    One meter per request, threaded through the ladder alongside the deadline.
    It never raises on an over-budget add — deciding what a ceiling means is the
    escalation policy's job, and a meter that refused entries would lose the
    very number the policy needs to make that call.
    """

    _total_usd: float = 0.0
    _by_model: dict[str, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def by_model(self) -> Mapping[str, float]:
        """Per-model breakdown, for the trace and for cost attribution."""
        return MappingProxyType(self._by_model)

    def add(self, usage: UsageLike) -> float:
        """Fold one call's spend in, returning the new total.

        A provider-reported ``cost_usd`` wins over the local table: it reflects
        the account's actual rates, including discounts this table cannot know
        about. The table is the fallback for providers that report tokens only.
        """
        model = usage.model or "unknown"
        amount = usage.cost_usd
        if amount <= 0.0:
            amount = estimate_cost(
                model,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        return self.add_usd(model, amount)

    def add_usd(self, model: str | None, amount: float) -> float:
        """Fold a bare amount in, returning the new total."""
        if amount <= 0.0:
            return self._total_usd
        key = model or "unknown"
        self._total_usd += amount
        self._by_model[key] = self._by_model.get(key, 0.0) + amount
        return self._total_usd


def _normalise(model: str) -> str:
    return model.strip().lower()


@lru_cache(maxsize=256)
def _warn_unknown_model(model: str) -> None:
    """Warn once per model.

    Cached rather than guarded by a set so a hot loop over an unpriced model
    logs a single line, and so the memory this costs stays bounded no matter how
    many model names go past.
    """
    logger.warning(
        "cost.unknown_model",
        model=model,
        detail="no price entry; billed as zero and excluded from the cost ceiling",
    )
