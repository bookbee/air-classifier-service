"""Application settings.

Configuration is 12-factor: every value below is settable from the environment
using the ``AIR_CLASSIFIER__`` prefix and ``__`` as the nesting delimiter, e.g.::

    AIR_CLASSIFIER__TIERS__T1__CONFIDENCE_FLOOR=0.85
    AIR_CLASSIFIER__AIR_LLM__CLOUD_MODEL=auto

Escalation thresholds live here rather than in code on purpose: tuning the
ladder's cost/accuracy balance must never require deploying new logic, only new
values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from air_classifier_service.constants import ModelClass, Tier

# ── Application ───────────────────────────────────────────────────────────────


class AppSettings(BaseModel):
    """Process-level knobs."""

    env: Literal["development", "staging", "production", "test"] = "development"
    host: str = "0.0.0.0"  # noqa: S104 — containers bind all interfaces by design
    # 8082 in the AIR port map (air-infra/README.md). Model calls go to
    # air-llm (its own port, `air_llm.base_url`), not to anything on this map.
    port: int = Field(default=8082, ge=1, le=65535)
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


# Each rung's defaults live on its own subclass rather than in a
# `default_factory` on TiersSettings, and that is load-bearing rather than
# stylistic. pydantic-settings builds a nested model directly from whichever
# leaves the environment supplies — the factory is never called — so with the
# factory form, setting a single documented key like
# `AIR_CLASSIFIER__TIERS__T0_RULES__ENABLED=true` silently reset that rung's
# siblings to the *base class* defaults: T0's confidence floor fell 0.95 -> 0.80
# (the floor is the whole safety argument for the shortcut rules) and T2's
# timeout fell 6000ms -> 1000ms, below its own p50. As field defaults they are
# part of the model, so a partial override changes only what it names.


class T0RulesSettings(TierSettings):
    """Rules. The floor is high because a shortcut either matched or it did not."""

    confidence_floor: float = Field(default=0.95, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=5, ge=1, le=120_000)


class T1ClassifierSettings(TierSettings):
    """400ms, not the 200ms this service first shipped with.

    That number was sized for a two- or three-class sequence-classification
    head (~10ms warm). The default backend since docs/02-lld.md §12.6 is a
    zero-shot NLI scorer running four forward passes per call, and it is
    genuinely slower — measured warm, *inside the built container* rather
    than on the host running it (Docker Desktop's virtualisation overhead
    turned out to matter here): p50 173ms, p90 193ms, max 199ms against the
    old 200ms ceiling. That is not "close to the edge", it already lost
    samples to it — a fraction of production traffic would have silently paid
    T2's latency for a rung that was in fact available and would have
    answered a moment later. 400ms leaves roughly 2x headroom over the
    observed max, the same margin T2's own timeout was tuned to when this
    codebase last measured a tier's real behaviour instead of guessing.
    """

    confidence_floor: float = Field(default=0.80, ge=0.0, le=1.0)
    margin_floor: float = Field(default=0.25, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=400, ge=1, le=120_000)


class T2LocalLlmSettings(TierSettings):
    """6s, not the 4s the LLD first guessed.

    Measured on the model `air_llm.local_model` currently resolves to, on CPU,
    with the compact prompt: p50 1.5s, p90 3.3s. A 4s ceiling left no headroom
    over p90 and lost every cold model load outright. That measurement is
    specific to today's local model — a different one behind the same alias
    could need this revisited. Note the request-level `latency_budget_ms` is
    usually the binding constraint anyway — the orchestrator takes min(tier
    timeout, remaining budget).
    """

    confidence_floor: float = Field(default=0.72, ge=0.0, le=1.0)
    margin_floor: float = Field(default=0.20, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=6000, ge=1, le=120_000)


class T3CloudLlmSettings(TierSettings):
    confidence_floor: float = Field(default=0.65, ge=0.0, le=1.0)
    margin_floor: float = Field(default=0.15, ge=0.0, le=1.0)
    timeout_ms: int = Field(default=8000, ge=1, le=120_000)


class TiersSettings(BaseModel):
    """Defaults mirror the thresholds table in docs/02-lld.md §5."""

    t0_rules: T0RulesSettings = Field(default_factory=T0RulesSettings)
    t1_classifier: T1ClassifierSettings = Field(default_factory=T1ClassifierSettings)
    t2_local_llm: T2LocalLlmSettings = Field(default_factory=T2LocalLlmSettings)
    t3_cloud_llm: T3CloudLlmSettings = Field(default_factory=T3CloudLlmSettings)

    @field_validator("t0_rules", "t1_classifier", "t2_local_llm", "t3_cloud_llm", mode="before")
    @classmethod
    def _merge_onto_rung_defaults(cls, value: object) -> object:
        """Let a caller hand in a bare ``TierSettings`` and keep the rung's defaults.

        ``TiersSettings(t2_local_llm=TierSettings(timeout_ms=4000))`` reads as
        "change the timeout", not "reset this rung and change the timeout", and
        the base class carries generic defaults that would quietly do the latter.
        Forwarding only the fields the caller actually set gives programmatic
        construction the same merge semantics the environment path has.
        """
        if type(value) is TierSettings:
            return value.model_dump(exclude_unset=True)
        return value

    #: Escalate when input is longer than this many tokens and other signals are weak.
    long_input_tokens: int = Field(default=120, ge=1)
    #: Negations at or above this count trigger escalation.
    negation_threshold: int = Field(default=2, ge=1)
    #: Contrastive conjunctions only trigger above this token count (avoids "but ok").
    contrastive_min_tokens: int = Field(default=8, ge=1)

    def for_tier(self, tier: Tier) -> TierSettings:
        return getattr(self, tier.value)  # type: ignore[no-any-return]


# ── Model access ──────────────────────────────────────────────────────────────


class AirLlmSettings(BaseModel):
    """Every model call — Tier 2, Tier 3 and the summary narrative — goes through
    air-model-gateway's ``POST /v1/generate``.

    The gateway is the platform's single model gateway: it owns provider keys,
    cross-vendor routing and failover, cost accounting, budgets and caching.
    This service never names a vendor or a model. Each call names a
    :class:`ModelClass` (``local``, ``auto`` or ``cloud``), and the gateway
    resolves that class to an ordered chain from its ``deploy/routing.yaml``,
    where ``overrides.air-classifier-service`` tunes the chains for this
    service alone. What stays this service's job is the escalation decision:
    is the answer that came back good enough, or does the ladder climb
    (§6, docs/02-lld.md)?

    The three classes are not interchangeable, and which rung asks for which
    is deliberate:

    - ``local_model`` (Tier 2) is ``local``: in this service's override a
      single local rung with no fallback. A local runtime being down must
      fail this call cleanly so *this* service's ladder escalates to Tier 3,
      rather than the gateway quietly substituting a cloud model under the
      Tier-2 label, which would blur tier attribution and cost accounting.
      (The old route did this with a bare ``provider:model`` bypass; on
      ``/v1/generate`` that would be a ``pin``, which needs a grant this
      service does not have and does not need.)
    - ``cloud_model`` (Tier 3's default call) is ``auto``: in this service's
      override a cloud-only, cheap-end, cross-vendor chain. An outage at one
      vendor must not remove the whole tier, and that failover is the
      gateway's job.
    - ``arbiter_model`` (Tier 3's second opinion) is ``cloud``: the
      strongest models, also cross-vendor.

    The gateway does not accept the alias names the retired
    ``/v1/inference`` route did (``classify-local``, ``classify-default``,
    ...): those are now a startup validation error here rather than a 422 on
    every call.
    """

    enabled: bool = True
    base_url: str = "http://localhost:8083"
    #: Service token, sent as ``X-API-Key``. The gateway maps it to the
    #: ``air-classifier-service`` identity, whose policy entry grants
    #: ``generate`` and sets this service's rate limit and daily budget.
    token: SecretStr | None = None
    timeout_s: float = Field(default=20.0, gt=0)

    # Tier 2: local only, one rung, no fallback (see class docstring).
    local_model: ModelClass = ModelClass.LOCAL

    # Tier 3: cross-vendor cloud chains, cheap first, strongest for arbitration.
    cloud_model: ModelClass = ModelClass.AUTO
    arbiter_model: ModelClass = ModelClass.CLOUD
    arbitration_enabled: bool = True
    arbitration_floor: float = Field(default=0.65, ge=0.0, le=1.0)
    #: Sized for whichever provider a class's chain might actually reach, not
    #: just the primary one — some providers in the configured chains bill
    #: reasoning ("thinking") tokens as output, which can starve a ceiling
    #: sized only for the cheapest member before the model ever reaches the
    #: closing brace of the structured response. Verified live against this
    #: service's own ~8.8k-token system prompt (§6, docs/02-lld.md's
    #: changelog): 1024 was not enough headroom for that failure mode and
    #: read as a schema-parsing bug rather than what it was.
    max_tokens: int = Field(default=2048, ge=64, le=8192)
    arbiter_max_tokens: int = Field(default=4096, ge=64, le=16384)

    # Customer-summary narrative refresh (pipeline/summary_service.py): local,
    # like Tier 2, with no fallback on purpose. A failed narrative refresh
    # degrades to the caller's prior narrative unchanged, so there is no
    # ladder above it to protect and no reason to pay for a cloud model.
    summary_model: ModelClass = ModelClass.LOCAL
    summary_max_tokens: int = Field(default=512, ge=64, le=4096)
    summary_timeout_s: float = Field(default=10.0, gt=0)

    @property
    def configured(self) -> bool:
        return self.enabled and self.token is not None


# ── Classifier (Tier 1) ───────────────────────────────────────────────────────


class ClassifierSettings(BaseModel):
    """Tier-1 backend.

    ``model_path`` points at the ONNX artifact baked into the image. When it is
    absent the tier falls back to the pure-lexicon scorer so the service still
    runs — degraded, and reported as such by ``/v1/capabilities``.
    """

    model_path: Path | None = None
    tokenizer_path: Path | None = None
    #: Which ONNX backend `model_path` is expected to be — the two are not
    #: interchangeable at the same artifact (`pipeline/tiers/scorers.py`'s
    #: module docstring). `zero_shot_nli` is the default: a quantised
    #: DeBERTa-v3 entailment head, general-purpose and not tied to one fixed
    #: label set. `sequence_classification` is the SST-2-family shape this
    #: service shipped with first, kept working for a deployment that already
    #: baked that artifact in.
    scorer_kind: Literal["zero_shot_nli", "sequence_classification"] = "zero_shot_nli"
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
    key_prefix: str = "air:clf:"


# ── Audit ─────────────────────────────────────────────────────────────────────


class AuditSettings(BaseModel):
    """An operational log of classification jobs, for support review — not
    customer state. Unlike `cache`, there is no zero-config in-process
    fallback: `backend` defaults to `"none"` so nobody starts writing to a
    database nobody configured.
    """

    backend: Literal["mongo", "none"] = "none"
    mongo_url: str = "mongodb://localhost:27017"
    database: str = "air_classifier_service"
    collection: str = "audit_log"
    #: Native Mongo TTL index; entries self-expire, no cleanup job needed.
    ttl_days: int = Field(default=30, ge=1)
    #: Bounded queue in front of the background writer. A slow or down Mongo
    #: drops the newest entries past this size rather than blocking requests.
    queue_max_size: int = Field(default=1000, ge=1)
    drain_timeout_s: float = Field(default=2.0, gt=0)


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
    hash_salt: SecretStr = SecretStr("air-classifier-service-dev-salt")
    default_rate_limit_rpm: int = Field(default=600, ge=1)
    default_burst: int = Field(default=60, ge=1)
    #: Deliberate opt-in. When false, only a hash and length of input text is logged.
    log_raw_text: bool = False


# ── Observability ─────────────────────────────────────────────────────────────


class ObservabilitySettings(BaseModel):
    service_name: str = "air-classifier-service"
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
        env_prefix="AIR_CLASSIFIER__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    tiers: TiersSettings = Field(default_factory=TiersSettings)
    air_llm: AirLlmSettings = Field(default_factory=AirLlmSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
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
