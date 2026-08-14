"""Provider registry tests.

The property under test throughout is that selection is decided *before* the
call: an open breaker moves the whole stream of traffic to the next provider
rather than letting each request discover the outage for itself. The health-cache
tests guard the other half of that bargain — a selector that probed on every
request would spend the round trip it exists to save.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from air_sentiment.config import (
    AnthropicSettings,
    AppSettings,
    OpenAISettings,
    ProvidersSettings,
    Settings,
)
from air_sentiment.constants import Provider, ResourceKind
from air_sentiment.pipeline.features import extract_features
from air_sentiment.pipeline.tiers.base import TierContext
from air_sentiment.providers.base import ProviderResult
from air_sentiment.providers.registry import NoProviderAvailableError, ProviderRegistry
from air_sentiment.resilience.breaker import CircuitBreaker
from air_sentiment.schemas.common import LanguageInfo, Options, SafetyReport

TEXT = "Great battery but the camera is awful."


class _FakeProvider:
    """A provider that reports whatever health the test asked for, and counts probes."""

    def __init__(self, name: Provider, *, configured: bool = True, healthy: bool = True) -> None:
        self.name = name
        self._configured = configured
        self.healthy = healthy
        self.probes = 0

    @property
    def configured(self) -> bool:
        return self._configured

    async def health(self) -> bool:
        self.probes += 1
        return self.healthy

    async def classify(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[BaseModel],
        model: str,
        max_tokens: int,
        cache_prefix: bool = True,
    ) -> ProviderResult:
        raise AssertionError("the registry must never call a provider")

    async def aclose(self) -> None:
        return None


def _ctx() -> TierContext:
    return TierContext(
        request_id="req_registry",
        resource=ResourceKind.SENTIMENT,
        text=TEXT,
        original_length=len(TEXT),
        language=LanguageInfo(code="en", confidence=0.9),
        features=extract_features(TEXT),
        safety=SafetyReport(),
        options=Options(),
        deadline_at=time.monotonic() + 5.0,
    )


def _settings(preference: list[Provider] | None = None, **kwargs: Any) -> Settings:
    return Settings(
        app=AppSettings(env="test"),
        providers=ProvidersSettings(
            preference=preference
            if preference is not None
            else [Provider.ANTHROPIC, Provider.OPENAI],
            anthropic=AnthropicSettings(api_key=SecretStr("sk-ant"), **kwargs),
            openai=OpenAISettings(api_key=SecretStr("sk-oai")),
        ),
    )


def _registry(
    *,
    anthropic: _FakeProvider | None = None,
    openai: _FakeProvider | None = None,
    breakers: dict[Provider, CircuitBreaker] | None = None,
    preference: list[Provider] | None = None,
    health_ttl_s: float = 10.0,
) -> tuple[ProviderRegistry, dict[Provider, _FakeProvider]]:
    providers: dict[Provider, _FakeProvider] = {
        Provider.ANTHROPIC: anthropic or _FakeProvider(Provider.ANTHROPIC),
        Provider.OPENAI: openai or _FakeProvider(Provider.OPENAI),
    }
    registry = ProviderRegistry(
        _settings(preference),
        dict(providers),
        breakers or {},
        health_ttl_s=health_ttl_s,
    )
    return registry, providers


async def _open(breaker: CircuitBreaker) -> None:
    """Drive a breaker to OPEN through its own state machine, not by poking it."""
    for _ in range(32):
        if not breaker.is_closed:
            return
        await breaker.record_failure()
    raise AssertionError("breaker never opened")


# ── Selection ─────────────────────────────────────────────────────────────────


async def test_first_preference_wins_when_it_is_healthy() -> None:
    registry, _ = _registry()

    provider, model = await registry.select(_ctx())

    assert provider.name is Provider.ANTHROPIC
    assert model == "claude-haiku-4-5"


async def test_preference_order_is_honoured() -> None:
    registry, _ = _registry(preference=[Provider.OPENAI, Provider.ANTHROPIC])

    provider, model = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI
    assert model == "gpt-4.1-mini"


async def test_open_breaker_moves_traffic_wholesale_to_the_next_provider() -> None:
    breaker = CircuitBreaker("anthropic", threshold=2, cooldown_s=60.0)
    await _open(breaker)
    registry, providers = _registry(breakers={Provider.ANTHROPIC: breaker})

    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI
    # The point of provider-level failover: the cut-off provider is not probed,
    # let alone called, so no request pays for the outage twice.
    assert providers[Provider.ANTHROPIC].probes == 0


async def test_a_recovered_breaker_takes_traffic_back() -> None:
    breaker = CircuitBreaker("anthropic", threshold=2, cooldown_s=60.0)
    await _open(breaker)
    registry, _ = _registry(breakers={Provider.ANTHROPIC: breaker})
    assert (await registry.select(_ctx()))[0].name is Provider.OPENAI

    await breaker.reset()

    assert (await registry.select(_ctx()))[0].name is Provider.ANTHROPIC


async def test_unconfigured_provider_is_skipped() -> None:
    registry, _ = _registry(anthropic=_FakeProvider(Provider.ANTHROPIC, configured=False))

    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI


async def test_unhealthy_provider_is_skipped() -> None:
    registry, _ = _registry(anthropic=_FakeProvider(Provider.ANTHROPIC, healthy=False))

    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI


async def test_all_providers_down_raises() -> None:
    registry, _ = _registry(
        anthropic=_FakeProvider(Provider.ANTHROPIC, healthy=False),
        openai=_FakeProvider(Provider.OPENAI, healthy=False),
    )

    with pytest.raises(NoProviderAvailableError):
        await registry.select(_ctx())


async def test_every_breaker_open_raises() -> None:
    breakers = {
        Provider.ANTHROPIC: CircuitBreaker("anthropic", threshold=1, cooldown_s=60.0),
        Provider.OPENAI: CircuitBreaker("openai", threshold=1, cooldown_s=60.0),
    }
    for breaker in breakers.values():
        await _open(breaker)
    registry, _ = _registry(breakers=breakers)

    with pytest.raises(NoProviderAvailableError):
        await registry.select(_ctx())


async def test_missing_provider_entry_is_skipped_rather_than_crashing() -> None:
    registry = ProviderRegistry(_settings(), {Provider.OPENAI: _FakeProvider(Provider.OPENAI)}, {})

    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI


async def test_ollama_in_the_preference_list_is_ignored() -> None:
    # Ollama is a real provider, but it belongs to Tier 2 and has no cloud model
    # or arbiter to offer here, so it must be filtered rather than selected.
    registry, _ = _registry(preference=[Provider.OLLAMA, Provider.OPENAI])

    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI


# ── Health caching ────────────────────────────────────────────────────────────


async def test_health_is_probed_once_across_many_selections() -> None:
    registry, providers = _registry()

    for _ in range(5):
        await registry.select(_ctx())

    assert providers[Provider.ANTHROPIC].probes == 1


async def test_health_is_reprobed_once_the_cache_expires() -> None:
    registry, providers = _registry(health_ttl_s=0.0)

    await registry.select(_ctx())
    await registry.select(_ctx())

    assert providers[Provider.ANTHROPIC].probes == 2


async def test_invalidating_health_forces_a_reprobe() -> None:
    registry, providers = _registry()
    await registry.select(_ctx())

    registry.invalidate_health(Provider.ANTHROPIC)
    await registry.select(_ctx())

    assert providers[Provider.ANTHROPIC].probes == 2


async def test_invalidating_everything_forces_a_reprobe() -> None:
    registry, providers = _registry(preference=[Provider.OPENAI])
    await registry.select(_ctx())

    registry.invalidate_health()
    await registry.select(_ctx())

    assert providers[Provider.OPENAI].probes == 2


async def test_a_cached_unhealthy_verdict_is_honoured() -> None:
    unhealthy = _FakeProvider(Provider.ANTHROPIC, healthy=False)
    registry, _ = _registry(anthropic=unhealthy)
    await registry.select(_ctx())

    unhealthy.healthy = True
    provider, _ = await registry.select(_ctx())

    assert provider.name is Provider.OPENAI
    assert unhealthy.probes == 1


# ── Accessors ─────────────────────────────────────────────────────────────────


async def test_available_reports_true_when_anything_can_serve() -> None:
    registry, _ = _registry(anthropic=_FakeProvider(Provider.ANTHROPIC, healthy=False))

    assert await registry.available() is True


async def test_available_reports_false_when_nothing_can() -> None:
    registry, _ = _registry(
        anthropic=_FakeProvider(Provider.ANTHROPIC, healthy=False),
        openai=_FakeProvider(Provider.OPENAI, healthy=False),
    )

    assert await registry.available() is False


async def test_config_for_returns_the_provider_block() -> None:
    registry, _ = _registry()

    assert registry.config_for(Provider.ANTHROPIC).arbiter_model == "claude-sonnet-5"
    assert registry.config_for(Provider.OPENAI).arbiter_model == "gpt-4.1"


async def test_config_for_rejects_a_non_cloud_provider() -> None:
    registry, _ = _registry()

    with pytest.raises(NoProviderAvailableError):
        registry.config_for(Provider.OLLAMA)


async def test_breaker_for_returns_what_was_supplied() -> None:
    breaker = CircuitBreaker("anthropic")
    registry, _ = _registry(breakers={Provider.ANTHROPIC: breaker})

    assert registry.breaker_for(Provider.ANTHROPIC) is breaker
    assert registry.breaker_for(Provider.OPENAI) is None
