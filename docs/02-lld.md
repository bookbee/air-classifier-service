# AIR Sentiment — Low-Level Design

**Status:** Draft for review · **Companion docs:** [Plan](00-plan.md) · [HLD](01-hld.md)

This document is the implementation contract. Code should follow it; where it turns out to
be wrong, the document gets amended rather than silently diverged from.

---

## 1. Repository layout

```
air-sentiment/
├── pyproject.toml                 # deps, ruff, mypy, pytest config
├── Dockerfile                     # multi-stage, non-root runtime
├── docker-compose.yml             # app + ollama + redis + prometheus + grafana
├── Makefile                       # dev / test / lint / run / image targets
├── .env.example                   # every settings key, documented
├── docs/                          # 00-plan · 01-hld · 02-lld · api · operations
└── src/air_sentiment/
    ├── __init__.py                # __version__
    ├── main.py                    # create_app(), lifespan, middleware wiring
    ├── config.py                  # Settings (pydantic-settings), get_settings()
    ├── constants.py               # Tier / Label / Provider enums
    │
    ├── api/
    │   ├── deps.py                # api_key_auth, rate_limit, request_context
    │   ├── errors.py              # AppError hierarchy + RFC 9457 handlers
    │   ├── middleware.py          # request-id, access log, timing, body-size cap
    │   └── v1/
    │       ├── router.py          # aggregates the v1 routers
    │       ├── sentiment.py       # POST /v1/sentiment(/batch)
    │       ├── feedback.py        # POST /v1/feedback(/batch)
    │       ├── reviews.py         # POST /v1/reviews(/batch)
    │       └── system.py          # /v1/health, /v1/ready, /v1/capabilities
    │
    ├── schemas/
    │   ├── common.py              # Label, Tier, RequestContext, Options, Emotion, Aspect
    │   ├── verdict.py             # TierVerdict, EscalationStep, LLMSentimentOutput
    │   ├── requests.py            # SentimentRequest, FeedbackRequest, ReviewRequest (+batch)
    │   └── responses.py           # SentimentResponse, FeedbackResponse, ReviewResponse
    │
    ├── pipeline/
    │   ├── orchestrator.py        # Orchestrator.run(): the ladder
    │   ├── policy.py              # EscalationPolicy.decide()
    │   ├── features.py            # ComplexityFeatures extraction
    │   └── tiers/
    │       ├── base.py            # Tier protocol + TierContext
    │       ├── t0_rules.py
    │       ├── t1_classifier.py
    │       ├── t2_local_llm.py
    │       └── t3_cloud_llm.py
    │
    ├── providers/
    │   ├── base.py                # LLMProvider protocol, ProviderResult
    │   ├── anthropic_provider.py
    │   ├── openai_provider.py
    │   ├── ollama_provider.py
    │   ├── registry.py            # routing + failover + health
    │   └── prompts/               # system.md, rubric.md, fewshot.md, per-resource.md
    │
    ├── preprocess/
    │   ├── normalize.py           # unicode NFKC, whitespace, control chars, length cap
    │   ├── language.py            # fast language identification
    │   ├── pii.py                 # detect + redact, returns PIIReport
    │   └── lexicon.py             # polarity/negation/intensifier/emoji tables
    │
    ├── cache/
    │   ├── base.py                # CacheBackend protocol
    │   ├── memory.py              # TTLCache, default
    │   └── redis.py               # optional, config-selected
    │
    ├── resilience/
    │   ├── breaker.py             # CircuitBreaker
    │   ├── bulkhead.py            # bounded-concurrency semaphore
    │   └── guard.py               # @guarded: timeout + retry + breaker + bulkhead
    │
    ├── observability/
    │   ├── logging.py             # structlog config, context binding
    │   ├── metrics.py             # Prometheus registry + metric objects
    │   ├── tracing.py             # OpenTelemetry setup
    │   └── cost.py                # CostMeter, price table
    │
    └── security/
        ├── api_keys.py            # ApiKeyStore, hashing, scopes, per-key tier ceiling
        └── quotas.py              # token bucket rate limiter
```

**Tests** mirror the source tree under `tests/`, split into `unit/`, `integration/`
(testcontainers for Redis/Ollama), and `contract/` (schema snapshots + golden fixtures).

## 2. Core enums and protocols

```python
# constants.py

class Tier(StrEnum):
    T0_RULES      = "t0_rules"
    T1_CLASSIFIER = "t1_classifier"
    T2_LOCAL_LLM  = "t2_local_llm"
    T3_CLOUD_LLM  = "t3_cloud_llm"

    @property
    def rank(self) -> int: ...        # ordering for min_tier/max_tier comparisons

class Label(StrEnum):
    POSITIVE = "positive"; NEGATIVE = "negative"
    NEUTRAL  = "neutral";  MIXED    = "mixed"; UNKNOWN = "unknown"

class Provider(StrEnum):
    ANTHROPIC = "anthropic"; OPENAI = "openai"; OLLAMA = "ollama"
```

```python
# pipeline/tiers/base.py

class TierContext(BaseModel):
    """Everything a tier needs, assembled once by the orchestrator."""
    request_id: str
    resource: Literal["sentiment", "feedback", "reviews"]
    text: str                       # normalised, possibly redacted
    original_length: int
    language: LanguageInfo
    features: ComplexityFeatures
    context: dict[str, Any]
    metadata: dict[str, Any]
    options: Options
    prior_verdicts: list[TierVerdict]
    deadline_at: float              # monotonic clock deadline for the whole request
    cost_spent_usd: float

class Tier(Protocol):
    tier: Tier
    model_version: str
    async def is_available(self) -> bool: ...
    async def analyse(self, ctx: TierContext) -> TierVerdict: ...
```

```python
# providers/base.py

class LLMProvider(Protocol):
    name: Provider
    async def health(self) -> bool: ...
    async def classify(
        self,
        *,
        system_prompt: str,
        user_content: str,
        output_model: type[BaseModel],
        model: str,
        max_tokens: int,
        cache_prefix: bool = True,
    ) -> ProviderResult: ...

class ProviderResult(BaseModel):
    parsed: BaseModel               # validated instance of output_model
    provider: Provider
    model: str
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float
    cost_usd: float
```

A single `LLMProvider` protocol covers both the local and the cloud tiers. Ollama is not a
special case — it is one more provider that happens to be free and nearby, which means
Tier 2 and Tier 3 share their prompt-building and validation code entirely.

## 3. Pydantic contracts

### Shared

```python
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_tier: Tier | None = None
    max_tier: Tier = Tier.T3_CLOUD_LLM
    latency_budget_ms: int = Field(default=5000, ge=50, le=30_000)
    max_cost_usd: float = Field(default=0.05, ge=0.0)
    include_trace: bool = True
    redact_pii: bool = True
    aspects: list[str] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _tier_bounds(self) -> Self:
        if self.min_tier and self.min_tier.rank > self.max_tier.rank:
            raise ValueError("min_tier cannot exceed max_tier")
        return self

class SentimentResult(BaseModel):
    label: Label
    polarity: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=500)

class Aspect(BaseModel):
    name: str; label: Label; confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = None

class Emotion(BaseModel):
    name: str; score: float = Field(ge=0.0, le=1.0)
```

### Verdict and trace

```python
class TierVerdict(BaseModel):
    """The single shape every tier produces. The heart of the design."""
    tier: Tier
    model_version: str
    result: SentimentResult
    aspects: list[Aspect] = []
    emotions: list[Emotion] = []
    label_margin: float = 0.0          # top-1 minus top-2 probability
    latency_ms: float
    succeeded: bool = True
    error: str | None = None
    usage: TierUsage = TierUsage()

class EscalationStep(BaseModel):
    tier: Tier
    label: Label
    confidence: float
    latency_ms: float
    escalated_because: str | None      # comma-joined reason codes; None = terminal
```

### Requests

```python
class BaseAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20_000)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    options: Options = Field(default_factory=Options)

class SentimentRequest(BaseAnalysisRequest): ...

class FeedbackRequest(BaseAnalysisRequest):
    channel: str | None = None            # app | email | support | survey | ...
    user_segment: str | None = None
    subject: str | None = Field(default=None, max_length=500)

class ReviewRequest(BaseAnalysisRequest):
    rating: float | None = Field(default=None, ge=0, le=10)
    rating_scale_max: float = 5.0
    product_id: str | None = None
    verified_purchase: bool | None = None
    title: str | None = Field(default=None, max_length=500)

class BatchRequest[T](BaseModel):
    items: list[T] = Field(min_length=1, max_length=100)
    options: Options | None = None        # per-batch default, item options win
```

### Responses

```python
class BaseAnalysisResponse(BaseModel):
    request_id: str
    sentiment: SentimentResult
    emotions: list[Emotion] = []
    aspects: list[Aspect] = []
    language: LanguageInfo
    decided_by: Tier
    degraded: bool = False
    escalation_trace: list[EscalationStep] = []
    safety: SafetyReport
    usage: RequestUsage
    cached: bool = False
    latency_ms: float
    model_versions: dict[str, str] = {}

class FeedbackResponse(BaseAnalysisResponse):
    urgency: Literal["low", "medium", "high", "critical"]
    actionability: Literal["none", "informational", "actionable", "blocking"]
    suggested_route: str | None = None

class ReviewResponse(BaseAnalysisResponse):
    rating_consistency: RatingConsistency | None = None   # agrees | prose_more_positive |
                                                          # prose_more_negative + delta
```

### LLM output model (Tiers 2 and 3)

This model is what gets converted to a JSON Schema and handed to the provider as a
structured-output constraint. It is deliberately **flatter and stricter** than the API
response — the model's job is to fill fields, not to design an envelope.

```python
class LLMSentimentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Label
    polarity: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=300)
    emotions: list[Emotion] = Field(default_factory=list, max_length=5)
    aspects: list[Aspect] = Field(default_factory=list, max_length=10)
    sarcasm_detected: bool = False
    language_code: str
```

Structured-output schema support excludes recursion and numeric range constraints, so
`ge`/`le` bounds are enforced by Pydantic on our side after the provider returns. A value
out of range is a validation failure, which counts as a tier failure and triggers one
bounded retry before escalation.

## 4. Orchestrator

```python
async def run(self, ctx: TierContext) -> OrchestrationResult:
    verdicts: list[TierVerdict] = []
    steps:    list[EscalationStep] = []
    degraded = False

    for tier in self._enabled_tiers_within(ctx.options):
        if tier.rank < (ctx.options.min_tier.rank if ctx.options.min_tier else 0):
            continue                                   # floor: skip cheaper tiers

        with tracer.span(f"tier.{tier.tier}"), metrics.tier_timer(tier.tier):
            try:
                verdict = await asyncio.wait_for(
                    tier.analyse(ctx), timeout=self._budget_for(tier, ctx))
            except (TimeoutError, TierError) as exc:
                verdict = TierVerdict.failed(tier, exc)
                degraded = True

        verdicts.append(verdict)
        ctx = ctx.model_copy(update={"prior_verdicts": verdicts,
                                     "cost_spent_usd": ctx.cost_spent_usd + verdict.usage.cost_usd})

        decision = self.policy.decide(verdict, ctx)
        steps.append(EscalationStep.from_(verdict, decision))
        metrics.record_decision(tier.tier, decision)

        if not decision.escalate:
            break

    best = self.policy.select_best(verdicts)
    if best is None:
        raise NoVerdictError()                          # the only path to a 5xx
    return OrchestrationResult(best=best, verdicts=verdicts,
                               steps=steps, degraded=degraded)
```

Three properties worth noting. The loop is **tier-agnostic** — it never names a concrete
tier. `ctx` is **immutably threaded**, so a tier cannot corrupt state for its successors.
And a tier failure is *not* an exception escaping the loop: it becomes a failed verdict, the
ladder continues, and `degraded` is set — which is exactly the "degrade, don't fail"
principle made mechanical.

## 5. Escalation policy

```python
class EscalationDecision(BaseModel):
    escalate: bool
    reasons: list[str] = []      # stable reason codes, exported as metric labels
```

Hard limits are evaluated first and short-circuit. The quality rules are then all
evaluated together, accumulating reasons. Availability is checked **last**.

| # | Condition | Outcome |
| --- | --- | --- |
| 1 | `verdict.tier == max_tier` | **stop** — ceiling reached |
| 2 | Latency budget exhausted (`now > deadline`) | **stop** (`budget_exhausted`) |
| 3 | `cost_spent >= max_cost_usd` | **stop** (`cost_ceiling`) |
| 5 | `min_tier` not yet reached | escalate (`min_tier_floor`) |
| 6 | Verdict failed, or abstained | escalate (`tier_failed` / `no_shortcut_matched`) |
| 7 | `confidence < floor[tier]` | escalate (`low_confidence`) |
| 8 | `label_margin < margin_floor[tier]` | escalate (`narrow_margin`) |
| 9 | Complexity trigger fires (below) | escalate (`<feature name>`) |
| 10 | Disagreement with a prior tier's label | escalate (`tier_disagreement`) |
| 11 | Language unsupported by the deciding tier | escalate (`unsupported_language`) |
| — | no reason to climb | **stop** (`confident`) |
| 4 | wanted to climb, but next tier is down / breaker open | **stop** (`next_tier_unavailable`) |

> **Why rule 4 moved to the end.** It was originally specified fourth, which made it
> short-circuit before the quality rules ran. The outcome was correct — the ladder stops
> either way — but the *reported reason* was not: a rung that answered confidently at 0.92
> with no cloud provider configured reported `next_tier_unavailable`, telling an operator
> they were blocked by an outage when the ladder had simply finished. Since rules 5–11 are
> pure computation, evaluating them first costs nothing and makes `next_tier_unavailable`
> mean what it says: *we wanted to climb and could not*. Explainability is a stated
> first-class output of this service, so a trace reason that misleads is a real defect, not
> a cosmetic one. Rule numbers are kept as originally assigned so the reason codes and their
> tests stay traceable to this table.

### Default thresholds (all configurable, no code change to tune)

| Tier | Confidence floor | Margin floor |
| --- | --- | --- |
| T0 rules | 0.95 | — |
| T1 classifier | 0.80 | 0.25 |
| T2 local LLM | 0.72 | 0.20 |
| T3 cloud (Haiku) | 0.65 → arbitrate to Sonnet 5 | 0.15 |

### Complexity triggers (`pipeline/features.py`)

```python
class ComplexityFeatures(BaseModel):
    token_count: int
    negation_count: int              # not, never, no, n't, hardly, barely
    contrastive_count: int           # but, however, although, though, yet
    polarity_flip: bool              # positive and negative lexicon hits both present
    emoji_text_conflict: bool        # emoji polarity opposes lexical polarity
    intensifier_count: int           # very, extremely, absolutely
    uppercase_ratio: float
    exclamation_count: int
    question_only: bool
    has_url_or_code: bool
```

Escalation triggers when `contrastive_count > 0 and token_count > 8`, or `polarity_flip`,
or `emoji_text_conflict`, or `negation_count >= 2`. These are heuristics that route *whom to
ask*, not classifiers — being liberal costs a cheap tier hop, and being conservative costs
accuracy, so they are tuned toward escalation.

## 6. Tier implementations

### T0 — Rules

Ordered short-circuits, each returning immediately on match:
cache hit → empty/whitespace/punctuation-only → explicit `rating` in a review request that
is unambiguous (≤20% or ≥80% of scale) *and* prose is short → emoji-only text → pure
lexicon match with no negation and no contrastive marker and `token_count <= 5` → blocklist.
No match yields `Label.UNKNOWN` with confidence `0.0`, which always escalates.

### T1 — Classifier

Ensemble of a quantised ONNX sentiment model (INT8, CPU, batched) and a lexicon score.
Softmax probabilities give both `confidence` (top-1) and `label_margin` (top-1 − top-2).
Runs in a thread-pool executor so the event loop is never blocked. Calibrated with
temperature scaling against a held-out set so that a reported 0.8 means roughly 80% accuracy
— without this, the whole policy is built on sand.

### T2 — Local LLM (Ollama)

Prompt = system rubric + resource-specific instructions + few-shot + the input. Requests the
JSON schema derived from `LLMSentimentOutput`, with `format` constrained so the model cannot
emit prose. `temperature: 0`, deterministic seed. One retry on schema-validation failure with
a repair instruction appended; a second failure marks the tier failed and escalates.

### T3 — Cloud LLM

```python
async def analyse(self, ctx: TierContext) -> TierVerdict:
    provider, model = self.registry.select(ctx)          # preference order + breaker state
    result = await provider.classify(
        system_prompt=self.prompts.system_for(ctx.resource),   # cache-anchored prefix
        user_content=self.prompts.user_for(ctx),
        output_model=LLMSentimentOutput,
        model=model,                                     # claude-haiku-4-5 by default
        max_tokens=1024,
        cache_prefix=True,
    )
    if result.parsed.confidence < self.arbitration_floor and self.arbitration_enabled:
        result = await provider.classify(..., model="claude-sonnet-5", max_tokens=2048)
    return TierVerdict.from_llm(self.tier, result)
```

#### Anthropic adapter notes

- Uses the SDK's structured-output parse path with `LLMSentimentOutput` as the output model,
  so a validated instance comes back rather than a string to parse.
- **Prompt caching** marks the end of the system prompt + rubric + few-shot block as the
  cacheable prefix. This only takes effect above the model's minimum cacheable prefix —
  **4096 tokens for Haiku 4.5, 1024 for Sonnet 5** — so the prompt bundle is sized past the
  Haiku threshold deliberately, and `cache_read_tokens` is asserted non-zero in an
  integration test. Below the threshold nothing errors; the cache just silently never fills.
- The prefix must be **byte-stable**: no timestamps, no request IDs, no per-caller
  interpolation above the cache breakpoint. Volatile content goes in the user turn only.
- Classification wants no extended thinking. Haiku 4.5 has no `effort` parameter; Sonnet 5
  supports `effort` and defaults to adaptive thinking, so the arbitration call sets
  thinking off (or `effort: low`) explicitly rather than paying for reasoning tokens on a
  labelling task.
- Sonnet 5 rejects non-default sampling parameters, so determinism is steered by prompt and
  schema, not by `temperature`.
- Non-2xx handling uses the SDK's typed exception classes in a most-specific-first chain
  (not-found → rate-limit → status → connection), so retryable and non-retryable failures
  are distinguished rather than collapsed into one `except`.

#### Provider registry

```python
class ProviderRegistry:
    async def select(self, ctx) -> tuple[LLMProvider, str]:
        for candidate in self._preference_order(ctx):   # config-driven, per-key overridable
            if self._breaker[candidate.name].is_closed and await candidate.health():
                return candidate, self._model_for(candidate, ctx)
        raise NoProviderAvailableError()
```

Failover is *provider-level*, not call-level: once a breaker opens, traffic moves wholesale
until the half-open probe succeeds, rather than every request paying a failed attempt first.

## 7. Configuration

`pydantic-settings`, prefix `AIR_SENTIMENT__`, nested via `__`. Every key in `.env.example`.

| Key | Default | Purpose |
| --- | --- | --- |
| `APP__ENV` | `development` | Environment name |
| `APP__WORKERS` | `2` | Uvicorn workers |
| `APP__MAX_BODY_BYTES` | `1048576` | Ingress body cap |
| `TIERS__ENABLED` | `t0,t1,t2,t3` | Which rungs exist at all |
| `TIERS__T1__CONFIDENCE_FLOOR` | `0.80` | Escalation threshold |
| `TIERS__T1__MARGIN_FLOOR` | `0.25` | Top-2 margin threshold |
| `TIERS__T2__CONFIDENCE_FLOOR` | `0.72` | — |
| `TIERS__T3__ARBITRATION_FLOOR` | `0.65` | Haiku→Sonnet arbitration trigger |
| `TIERS__*__TIMEOUT_MS` | `5 / 200 / 6000 / 8000` | Per-tier timeout (T2 measured, see below) |
| `PROVIDERS__PREFERENCE` | `anthropic,openai` | Cloud failover order |
| `PROVIDERS__ANTHROPIC__MODEL` | `claude-haiku-4-5` | Default cloud model |
| `PROVIDERS__ANTHROPIC__ARBITER_MODEL` | `claude-sonnet-5` | Arbitration model |
| `PROVIDERS__ANTHROPIC__API_KEY` | — | Secret, injected |
| `PROVIDERS__OLLAMA__BASE_URL` | `http://localhost:11434` | Local LLM |
| `PROVIDERS__OLLAMA__MODEL` | `llama3.2:3b` | Local model tag |
| `PROVIDERS__OLLAMA__NUM_CTX` | `8192` | Local context window; must hold compact prompt + longest input |
| `CACHE__BACKEND` | `memory` | `memory` \| `redis` |
| `CACHE__TTL_SECONDS` | `3600` | Verdict cache TTL |
| `RESILIENCE__BREAKER_THRESHOLD` | `5` | Consecutive failures to open |
| `RESILIENCE__BREAKER_COOLDOWN_S` | `30` | Half-open delay |
| `RESILIENCE__LLM_CONCURRENCY` | `8` | Bulkhead width |
| `SECURITY__LOG_RAW_TEXT` | `false` | Deliberate opt-in only |
| `OBS__OTEL_ENDPOINT` | — | Trace exporter |

## 8. Observability catalogue

### Metrics

| Name | Type | Labels |
| --- | --- | --- |
| `air_sentiment_requests_total` | counter | `endpoint`, `status`, `api_key_id` |
| `air_sentiment_request_duration_seconds` | histogram | `endpoint` |
| `air_sentiment_tier_decisions_total` | counter | `tier`, `outcome` (`stop`/`escalate`/`failed`) |
| `air_sentiment_escalations_total` | counter | `from_tier`, `to_tier`, `reason` |
| `air_sentiment_tier_duration_seconds` | histogram | `tier` |
| `air_sentiment_decided_by_total` | counter | `tier` |
| `air_sentiment_llm_tokens_total` | counter | `provider`, `model`, `kind` (`in`/`out`/`cache_read`/`cache_write`) |
| `air_sentiment_llm_cost_usd_total` | counter | `provider`, `model` |
| `air_sentiment_cache_events_total` | counter | `result` (`hit`/`miss`/`store`) |
| `air_sentiment_breaker_state` | gauge | `provider` (0 closed / 1 half-open / 2 open) |
| `air_sentiment_degraded_responses_total` | counter | `reason` |
| `air_sentiment_pii_redactions_total` | counter | `kind` |

`decided_by_total` divided by `requests_total` **is** the tier-distribution KPI from the
plan — the primary dashboard panel and the number that says whether the ladder is earning
its keep.

### Logs

One access log per request, plus one structured event per tier decision:

```json
{"event":"tier.decision","request_id":"req_01J...","trace_id":"4bf9...",
 "tier":"t1_classifier","label":"positive","confidence":0.52,"margin":0.11,
 "latency_ms":11.2,"escalate":true,"reasons":["low_confidence","contrastive_conjunction"],
 "text_sha256":"9f2b…","text_len":58}
```

Raw text never appears unless `SECURITY__LOG_RAW_TEXT` is explicitly enabled.

### Traces

Root span per request; child spans `preprocess`, `tier.{name}`, `provider.{name}`,
`cache.{op}`. Provider spans carry model, tokens, cache-read tokens, and cost.

## 9. Error model (RFC 9457)

```json
{ "type": "https://air.dev/errors/validation-error",
  "title": "Request validation failed",
  "status": 422,
  "detail": "text: String should have at least 1 character",
  "instance": "/v1/sentiment",
  "request_id": "req_01J...",
  "errors": [ { "field": "text", "code": "too_short" } ] }
```

| Status | Type slug | Raised when |
| --- | --- | --- |
| 400 | `malformed-request` | Unparseable body, oversized payload |
| 401 | `missing-api-key` | No `X-API-Key` header |
| 403 | `insufficient-scope` | Key lacks the scope, or requests a tier above its ceiling |
| 404 | `not-found` | Unknown route |
| 409 | `idempotency-conflict` | Same key, different body |
| 422 | `validation-error` | Pydantic validation failure |
| 429 | `rate-limited` | Quota exhausted; includes `Retry-After` |
| 500 | `internal-error` | Unhandled — should be alert-worthy and rare |
| 503 | `no-verdict` | Every enabled tier failed |

Note what is *absent*: there is no "provider unavailable" error surfaced to callers. A cloud
outage produces a `200` with `degraded: true` and a lower-tier verdict.

## 10. Testing strategy

| Layer | Coverage |
| --- | --- |
| **Unit** | Policy truth table (each rule in §5 fires exactly when it should); feature extraction; PII redaction; lexicon scoring; cost arithmetic; breaker state machine |
| **Tier fakes** | Orchestrator driven by four scripted fake tiers — verifies ordering, `min_tier`/`max_tier`, budget and cost stops, failure→degrade, and best-verdict selection with **no network at all** |
| **Contract** | Golden request/response fixtures per endpoint; OpenAPI snapshot diffed in CI so a schema change cannot land silently |
| **Provider** | Recorded-cassette tests for Anthropic/OpenAI/Ollama adapters; a live smoke test gated behind a marker |
| **Integration** | Full stack via testcontainers (Redis + Ollama); asserts trace completeness, metric emission, and non-zero `cache_read_tokens` on the second identical cloud call |
| **Load** | Locust profile at target RPS; asserts P95 and tier distribution stay inside the §7 plan targets |
| **Accuracy** | Labelled evaluation set; per-tier confusion matrices; ladder-vs-cloud-only agreement; regression gate on merge |

## 11. Implementation order

Matching the plan's phases, the concrete build order is:

1. `config.py` → `main.py` → `api/errors.py` → `observability/logging.py` → health endpoints
2. `schemas/*` complete → routers with stub verdicts → OpenAPI review → auth + idempotency
3. `preprocess/*` → `pipeline/features.py` → `tiers/t0` → `tiers/t1` → `policy` →
   `orchestrator` (+ the fake-tier suite, which should exist *before* real tiers do)
4. `providers/base` + `ollama_provider` → `tiers/t2` → `resilience/*`
5. `providers/anthropic_provider` → `openai_provider` → `registry` → `tiers/t3` →
   `observability/cost`
6. `security/quotas` → metrics + tracing → batch concurrency → load test → Docker/compose
   finalisation → `docs/api.md` + runbook

Writing the fake-tier orchestrator suite in step 3, before any real tier exists, is
deliberate: it pins the ladder's semantics while they are still cheap to change.

---

## 12. Amendments from implementation

The LLD is the implementation contract, so where building the service proved it wrong, it is
amended here rather than silently diverged from.

### 12.1 The prompt library is split, not shared

§6 said Tiers 2 and 3 share their prompt entirely. They share the *builder* and the *output
schema*, but not the system prefix.

The remote prefix is deliberately ~8,800 tokens to clear Anthropic's per-model prompt-cache
floor (4096 on Haiku 4.5), because from the second request onward a cached prefix bills at
roughly a tenth of the input rate. **That argument is worth nothing on a locally served
model:** Ollama has no cross-request prefix cache, so every one of those tokens is
re-processed on every call.

Measured on `llama3.2:3b`, the full prefix took **~9.9 s** against **~1.5 s** for a compact
one — and because it was also 2.2× the configured `num_ctx`, Ollama was *silently truncating
it*. The tier paid the latency, lost the rubric, and blew its timeout on every request, with
nothing logged and nothing raised. The ladder degraded to Tier 1 exactly as designed, which
is precisely why this could have shipped unnoticed.

So `prompts/system_compact.md` (~1,100 tokens) serves the local rung. It keeps the label
taxonomy, the polarity anchors, and — critically — the confidence-calibration rubric, because
the escalation policy gates on the confidence Tier 2 reports. It drops the long-form
discussion and most worked examples. `tests/unit/test_prompts.py` guards both the remote
prompt's cache floor and the compact prompt's fit inside `num_ctx`; both failures are
invisible at runtime, so a test is the only thing that catches them.

### 12.2 Tier-2 timeout and the request latency budget

§5 guessed 4000 ms for Tier 2. Measured warm on `llama3.2:3b`/CPU with the compact prompt:
**p50 1.5 s, p90 3.3 s**. A 4 s ceiling left no headroom over p90 and lost every cold model
load outright, so the default is now **6000 ms**.

The more important correction is conceptual: the per-tier timeout is rarely the binding
constraint. The orchestrator gives each rung `min(tier timeout, remaining request budget)`,
so `options.latency_budget_ms` is the real governor. Its default rose from 5000 ms to
**8000 ms** so that a request can actually reach the cloud tier on the default path.

### 12.3 Rule 4 evaluates last

See the note under §5. `next_tier_unavailable` now reports only when the policy genuinely
wanted to climb.
