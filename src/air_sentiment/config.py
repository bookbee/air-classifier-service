"""Application settings.

Configuration is 12-factor: every value below is settable from the environment
using the ``AIR_SENTIMENT__`` prefix and ``__`` as the nesting delimiter, e.g.::

    AIR_SENTIMENT__TIERS__T1__CONFIDENCE_FLOOR=0.85
    AIR_SENTIMENT__PROVIDERS__ANTHROPIC__MODEL=claude-haiku-4-5

Escalation thresholds live here rather than in code on purpose: tuning the
ladder's cost/accuracy balance must never require deploying new logic, only new
values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from air_sentiment.constants import Provider, Tier

# ── Application ───────────────────────────────────────────────────────────────


class AppSettings(BaseModel):
    """Process-level knobs."""

    env: Literal["development", "staging", "production", "test"] = "development"
    host: str = "0.0.0.0"  # noqa: S104 — containers bind all interfaces by design
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=2, ge=1, le=64)
    root_path: str = ""
    docs_enabled: bool = True
    max_body_bytes: int = Field(default=1_048_576, ge=1024)
    max_batch_items: int = Field(default=100, ge=1, le=1000)
    batch_concurrency: int = Field(default=8, ge=1, le=128)
    shutdown_grace_seconds: float = Field(default=15.0, ge=0.0)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# ── Tiers ─────────────────────────────────────────────────────────────────────


class TierSettings(BaseModel):
    """Per-rung gating. `margin_floor` of 0 disables the margin check."""

    enabled: bool = True
    confidence_floor: float = Field(default=0.80, ge=0.0, le=1.0)
    margin_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=1000, ge=1, le=120_000)


class TiersSettings(BaseModel):
    """Defaults mirror the thresholds table in docs/02-lld.md §5."""

    t0_rules: TierSettings = Field(
        default_factory=lambda: TierSettings(confidence_floor=0.95, timeout_ms=5)
    )
    t1_classifier: TierSettings = Field(
        default_factory=lambda: TierSettings(
            confidence_floor=0.80, margin_floor=0.25, timeout_ms=200
        )
    )
    #: 6s, not the 4s the LLD first guessed. Measured on llama3.2:3b/CPU with the
    #: compact prompt: p50 1.5s, p90 3.3s. A 4s ceiling left no headroom over p90
    #: and lost every cold model load outright. Note the request-level
    #: `latency_budget_ms` is usually the binding constraint anyway — the
    #: orchestrator takes min(tier timeout, remaining budget).
    t2_local_llm: TierSettings = Field(
        default_factory=lambda: TierSettings(
            confidence_floor=0.72, margin_floor=0.20, timeout_ms=6000
        )
    )
    t3_cloud_llm: TierSettings = Field(
        default_factory=lambda: TierSettings(
            confidence_floor=0.65, margin_floor=0.15, timeout_ms=8000
        )
    )

    #: Escalate when input is longer than this many tokens and other signals are weak.
    long_input_tokens: int = Field(default=120, ge=1)
    #: Negations at or above this count trigger escalation.
    negation_threshold: int = Field(default=2, ge=1)
    #: Contrastive conjunctions only trigger above this token count (avoids "but ok").
    contrastive_min_tokens: int = Field(default=8, ge=1)

    def for_tier(self, tier: Tier) -> TierSettings:
        return getattr(self, tier.value)  # type: ignore[no-any-return]


# ── Providers ─────────────────────────────────────────────────────────────────


class AnthropicSettings(BaseModel):
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str = "claude-haiku-4-5"
    arbiter_model: str = "claude-sonnet-5"
    arbitration_enabled: bool = True
    arbitration_floor: float = Field(default=0.65, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    arbiter_max_tokens: int = Field(default=2048, ge=64, le=16384)
    timeout_s: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)
    prompt_cache_enabled: bool = True
    prompt_cache_ttl: Literal["5m", "1h"] = "1h"

    @property
    def configured(self) -> bool:
        return self.api_key is not None


class OpenAISettings(BaseModel):
    api_key: SecretStr | None = None
    base_url: str | None = None
    model: str = "gpt-4.1-mini"
    arbiter_model: str = "gpt-4.1"
    arbitration_enabled: bool = True
    arbitration_floor: float = Field(default=0.65, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    arbiter_max_tokens: int = Field(default=2048, ge=64, le=16384)
    timeout_s: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)

    @property
    def configured(self) -> bool:
        return self.api_key is not None


class OllamaSettings(BaseModel):
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout_s: float = Field(default=15.0, gt=0)
    #: Must exceed the compact system prompt plus the longest input we accept.
    #: Undersizing it does not error — Ollama silently truncates the prompt, so
    #: the model loses its rubric and we never find out. `test_prompts.py` guards
    #: the relationship between this value and the compact prompt's length.
    num_ctx: int = Field(default=8192, ge=512)
    keep_alive: str = "10m"


class GatewaySettings(BaseModel):
    """Route every model call through the air-infra gateway instead of the SDKs.

    When ``enabled``, the tiers' providers become thin clients that POST to
    air-infra, which owns the provider keys, routing, cost and cache. The models
    named in ``anthropic``/``openai``/``ollama`` above still choose *which* model
    each rung asks for; only the transport changes. Off by default so a
    deployment without air-infra keeps talking to the providers directly.
    """

    enabled: bool = False
    base_url: str = "http://localhost:8080"
    #: air-infra service token, sent as ``X-API-Key``. Replaces the raw provider
    #: keys, which in gateway mode live only in air-infra.
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)

    @property
    def configured(self) -> bool:
        return self.enabled and self.api_key is not None


class ProvidersSettings(BaseModel):
    #: Cloud failover order. The first healthy, configured, closed-breaker entry wins.
    preference: list[Provider] = Field(
        default_factory=lambda: [Provider.ANTHROPIC, Provider.OPENAI]
    )
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)


# ── Classifier (Tier 1) ───────────────────────────────────────────────────────


class ClassifierSettings(BaseModel):
    """Tier-1 backend.

    ``model_path`` points at the ONNX artifact baked into the image. When it is
    absent the tier falls back to the pure-lexicon scorer so the service still
    runs — degraded, and reported as such by ``/v1/capabilities``.
    """

    model_path: Path | None = None
    tokenizer_path: Path | None = None
    #: Temperature-scaling parameter fitted on the calibration set. 1.0 = uncalibrated.
    calibration_temperature: float = Field(default=1.0, gt=0.0, le=10.0)
    #: Weight of the lexicon score when blending with the neural score.
    lexicon_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    max_input_chars: int = Field(default=4000, ge=64)
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])


# ── Cache ─────────────────────────────────────────────────────────────────────


class CacheSettings(BaseModel):
    backend: Literal["memory", "redis", "none"] = "memory"
    ttl_seconds: int = Field(default=3600, ge=0)
    idempotency_ttl_seconds: int = Field(default=86_400, ge=0)
    max_entries: int = Field(default=10_000, ge=1)
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "air:sent:"


# ── Resilience ────────────────────────────────────────────────────────────────


class ResilienceSettings(BaseModel):
    breaker_threshold: int = Field(default=5, ge=1)
    breaker_cooldown_s: float = Field(default=30.0, gt=0)
    breaker_half_open_max: int = Field(default=1, ge=1)
    llm_concurrency: int = Field(default=8, ge=1, le=256)
    bulkhead_acquire_timeout_s: float = Field(default=2.0, gt=0)
    retry_attempts: int = Field(default=2, ge=0, le=8)
    retry_base_delay_ms: int = Field(default=120, ge=1)
    retry_max_delay_ms: int = Field(default=4000, ge=1)


# ── Security ──────────────────────────────────────────────────────────────────


class SecuritySettings(BaseModel):
    #: JSON array of key records; see security/api_keys.py for the shape.
    api_keys_inline: str | None = None
    api_keys_file: Path | None = None
    #: Development escape hatch. Refused at startup when env == production.
    allow_unauthenticated: bool = False
    hash_salt: SecretStr = SecretStr("air-sentiment-dev-salt")
    default_rate_limit_rpm: int = Field(default=600, ge=1)
    default_burst: int = Field(default=60, ge=1)
    #: Deliberate opt-in. When false, only a hash and length of input text is logged.
    log_raw_text: bool = False


# ── Observability ─────────────────────────────────────────────────────────────


class ObservabilitySettings(BaseModel):
    service_name: str = "air-sentiment"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    otel_enabled: bool = False
    otel_endpoint: str | None = None
    #: Sampling ratio for traces, 0.0 to 1.0.
    otel_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


# ── Root ──────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Root settings object. Obtain via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="AIR_SENTIMENT__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    tiers: TiersSettings = Field(default_factory=TiersSettings)
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    obs: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode="after")
    def _guard_production(self) -> Self:
        """Fail fast rather than run production with a development escape hatch open."""
        if self.app.is_production:
            if self.security.allow_unauthenticated:
                raise ValueError(
                    "security.allow_unauthenticated must be false when app.env is production"
                )
            if self.security.log_raw_text:
                raise ValueError("security.log_raw_text must be false when app.env is production")
        return self

    def enabled_tiers(self) -> tuple[Tier, ...]:
        """Ladder rungs that are switched on, cheapest first."""
        return tuple(t for t in Tier.ordered() if self.tiers.for_tier(t).enabled)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that import-time consumers and request-time consumers agree. Tests
    that mutate the environment should call ``get_settings.cache_clear()``.
    """
    return Settings()
