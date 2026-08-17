"""Cloud provider selection and failover.

Failover here is **provider-level, not call-level**. The registry consults the
breaker *before* choosing, so once Anthropic's breaker opens the whole stream of
traffic moves to OpenAI at once. The alternative — try the preferred provider,
fail, then fall back — makes every single request pay one doomed round trip
against a dependency that has already been declared unhealthy, which is exactly
the latency the breaker exists to avoid.

Health is probed over the network, so it is cached for a few seconds. Without
that cache :meth:`ProviderRegistry.select` would add a round trip to every
request in order to save one, which is a bad trade in both directions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from air_classifier.config import AnthropicSettings, OpenAISettings, Settings
from air_classifier.constants import Provider
from air_classifier.observability.logging import get_logger
from air_classifier.providers.base import LLMProvider, ProviderError
from air_classifier.resilience.breaker import CircuitBreaker

if TYPE_CHECKING:
    from air_classifier.pipeline.tiers.base import TierContext

logger = get_logger(__name__)

#: Per-provider settings blocks that describe a metered cloud model. Both shapes
#: carry the same field names, which is what lets the tier read `model`,
#: `arbiter_model` and the arbitration knobs without caring who it is talking to.
CloudProviderSettings = AnthropicSettings | OpenAISettings

#: How long a health verdict stays good. Short enough that a recovered provider
#: is picked up within one breaker cooldown, long enough that a burst of requests
#: shares a single probe.
DEFAULT_HEALTH_TTL_S: Final[float] = 10.0


class NoProviderAvailableError(ProviderError):
    """Every configured cloud provider is unusable right now.

    Caught by Tier 3 and converted into a tier failure, never surfaced to a
    caller: a cloud outage degrades the ladder to a lower rung, it does not
    become a 5xx.
    """


@dataclass(frozen=True, slots=True)
class _HealthVerdict:
    healthy: bool
    checked_at: float


class ProviderRegistry:
    """Picks which cloud provider serves the next Tier-3 call."""

    def __init__(
        self,
        settings: Settings,
        providers: Mapping[Provider, LLMProvider],
        breakers: Mapping[Provider, CircuitBreaker],
        *,
        health_ttl_s: float = DEFAULT_HEALTH_TTL_S,
    ) -> None:
        self._settings = settings
        self._providers = dict(providers)
        self._breakers = dict(breakers)
        self._health_ttl_s = health_ttl_s
        self._health: dict[Provider, _HealthVerdict] = {}
        self._configs: dict[Provider, CloudProviderSettings] = {
            Provider.ANTHROPIC: settings.providers.anthropic,
            Provider.OPENAI: settings.providers.openai,
        }

    async def select(self, ctx: TierContext) -> tuple[LLMProvider, str]:
        """The first preferred provider that is configured, closed and healthy.

        Raises:
            NoProviderAvailableError: None of them qualified.
        """
        order = self._preference_order(ctx)
        chosen = await self._first_usable(order)
        if chosen is None:
            raise NoProviderAvailableError(
                "no cloud provider is configured, closed and healthy: "
                + ", ".join(p.value for p in order)
            )
        return chosen, self._configs[chosen.name].model

    async def available(self) -> bool:
        """Whether :meth:`select` would succeed. Never raises.

        Used by the tier's readiness probe and by the policy's lookahead, both of
        which need an answer rather than an exception.
        """
        return await self._first_usable(self._preference_order(None)) is not None

    def config_for(self, name: Provider) -> CloudProviderSettings:
        """The settings block for a cloud provider, including its arbiter knobs."""
        config = self._configs.get(name)
        if config is None:
            raise NoProviderAvailableError(f"{name.value} is not a metered cloud provider")
        return config

    def breaker_for(self, name: Provider) -> CircuitBreaker | None:
        """The breaker guarding one provider, if the caller supplied one."""
        return self._breakers.get(name)

    def invalidate_health(self, name: Provider | None = None) -> None:
        """Drop cached health, so the next selection re-probes.

        Called after a failure that the breaker has not yet acted on: continuing
        to trust a cached "healthy" for the rest of its TTL would keep sending
        traffic at a provider we have just watched fail.
        """
        if name is None:
            self._health.clear()
        else:
            self._health.pop(name, None)

    async def _first_usable(self, order: Sequence[Provider]) -> LLMProvider | None:
        """Walk the preference order; the breaker is consulted before the probe."""
        for name in order:
            provider = self._providers.get(name)
            if provider is None or not provider.configured:
                continue
            breaker = self._breakers.get(name)
            if breaker is not None and not breaker.is_closed:
                logger.info("registry.skip_open_breaker", provider=name.value)
                continue
            if await self._healthy(name, provider):
                return provider
        return None

    def _preference_order(self, ctx: TierContext | None) -> Sequence[Provider]:
        """Configured failover order, restricted to metered cloud providers.

        ``ctx`` is accepted because the order is meant to become per-key
        overridable; today it is purely config-driven. Ollama is filtered out
        rather than rejected — it is a legitimate provider, it simply belongs to
        Tier 2 and has no cloud model or arbiter to offer here.
        """
        return [p for p in self._settings.providers.preference if p in self._configs]

    async def _healthy(self, name: Provider, provider: LLMProvider) -> bool:
        """Cached reachability check.

        Deliberately unlocked: two concurrent requests racing to refresh a stale
        entry cost one extra probe, whereas serialising every selection behind a
        lock would cost every request the latency of the slowest one.
        """
        cached = self._health.get(name)
        now = time.monotonic()
        if cached is not None and now - cached.checked_at < self._health_ttl_s:
            return cached.healthy

        healthy = await provider.health()
        self._health[name] = _HealthVerdict(healthy=healthy, checked_at=now)
        if not healthy:
            logger.warning("registry.provider_unhealthy", provider=name.value)
        return healthy
