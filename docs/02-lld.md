# AIR Classifier — Low-Level Design

**Status:** Draft for review · **Companion docs:** [Plan](00-plan.md) · [HLD](01-hld.md)

This document is the implementation contract. Code should follow it; where it turns out to
be wrong, the document gets corrected rather than silently diverged from. §12 keeps a short
log of the corrections that mattered enough to leave a trace of — this section itself
describes the system as it stands today, not as it was first drafted.

---

## 1. Repository layout

```
air-classifier/
├── pyproject.toml                 # deps, ruff, mypy, pytest config
├── Dockerfile                     # multi-stage, non-root runtime
├── docker-compose.yml             # this service only; deps come from air-infra
├── Makefile                       # dev / test / lint / run / image targets
├── .env.example                   # every settings key, documented
├── docs/                          # 00-plan · 01-hld · 02-lld
├── scripts/
│   ├── fetch_model.py             # downloads a pinned, pre-built ONNX artifact
│   └── export_zeroshot_model.py   # produces one from scratch (torch + optimum)
└── src/air_classifier/
    ├── __init__.py                # __version__
    ├── main.py                    # create_app(), lifespan, middleware wiring
    ├── config.py                  # Settings (pydantic-settings), get_settings()
    ├── constants.py                # Tier / Label / Tone / Topic / Intent / ... enums
    │
    ├── api/
    │   ├── deps.py                # api_key_auth, rate_limit, idempotency
    │   ├── errors.py              # AppError hierarchy + RFC 9457 handlers
    │   ├── middleware.py          # request-id, access log, timing, body-size cap
    │   └── v1/
    │       ├── router.py          # request-scoped AnalysisService, error catalogue, aggregation
    │       ├── classify.py        # POST /v1/classify(/batch) — the whole surface
    │       └── system.py          # /v1/health, /v1/ready, /v1/capabilities, /metrics
    │
    ├── schemas/
    │   ├── common.py              # Label, Tier, Options, LabeledScore[T], Emotion, Aspect, ...
    │   ├── verdict.py             # TierVerdict, EscalationStep, LLMSentimentOutput
    │   ├── classify.py            # ClassifyRequest, ClassificationResponse (+ batch envelopes)
    │   └── responses.py           # HealthResponse, ReadyResponse — system endpoints only
    │
    ├── pipeline/
    │   ├── orchestrator.py        # Orchestrator.run(): the ladder
    │   ├── policy.py              # EscalationPolicy.decide()
    │   ├── service.py             # HTTP↔ladder seam: preprocess, cache, shape, batch isolation
    │   ├── features.py            # ComplexityFeatures extraction
    │   └── tiers/
    │       ├── base.py            # Tier protocol + TierContext
    │       ├── scorers.py         # Tier-1 backends: ZeroShotOnnxScorer, OnnxScorer, LexiconScorer
    │       ├── t0_rules.py
    │       ├── t1_classifier.py
    │       ├── t2_local_llm.py
    │       └── t3_cloud_llm.py
    │
    ├── providers/
    │   ├── base.py                # LLMProvider protocol, ProviderResult
    │   ├── air_llm_provider.py    # the one adapter: POST /v1/inference on air-llm
    │   └── prompts/                # system.md, system_compact.md, loader.py
    │
    ├── preprocess/
    │   ├── normalize.py           # unicode NFKC, whitespace, control chars, length cap
    │   ├── language.py            # fast language identification
    │   ├── pii.py                 # detect + redact, returns PIIReport
    │   └── lexicon.py             # polarity/negation/intensifier/emoji tables
    │
    ├── cache/
    │   ├── base.py                # CacheBackend protocol, verdict_cache_key
    │   ├── memory.py               # TTLCache, default
    │   ├── redis.py                # optional, config-selected
    │   └── factory.py
    │
    ├── resilience/
    │   ├── breaker.py             # CircuitBreaker
    │   ├── bulkhead.py            # bounded-concurrency semaphore
    │   └── guard.py               # guarded(): timeout + retry + breaker + bulkhead composed
    │
    ├── observability/
    │   ├── logging.py             # structlog config, context binding
    │   ├── metrics.py             # Prometheus registry + metric objects
    │   └── tracing.py             # OpenTelemetry setup
    │
    └── security/
        ├── api_keys.py            # ApiKeyStore, hashing, scopes, per-key tier ceiling
        └── quotas.py               # token bucket rate limiter
```

**Tests** mirror the source tree under `tests/`, split into `unit/`, `integration/` (needs
Redis and a reachable air-llm), and `contract/` (OpenAPI snapshot + golden response fixtures).

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
    ANTHROPIC = "anthropic"; OPENAI = "openai"; GEMINI = "gemini"; OLLAMA = "ollama"

# Also here: Tone, Topic, Intent, RoutingPriority, Urgency, Actionability,
# RatingAgreement — the closed vocabularies §3's response models use.
```

```python
# pipeline/tiers/base.py

class TierContext(BaseModel):
    """Everything a tier needs, assembled once by the orchestrator."""
    request_id: str
    text: str                       # normalised, possibly redacted
    original_length: int
    language: LanguageInfo
    features: ComplexityFeatures
    context: dict[str, Any]
    metadata: dict[str, Any]
    resource_fields: dict[str, Any] # the caller's optional fields — rating, channel, ...
    options: Options
    prior_verdicts: list[TierVerdict]
    deadline_at: float              # monotonic clock deadline for the whole request
    cost_spent_usd: float

class Tier(Protocol):
    tier: Tier
    model_version: str
    async def is_available(self) -> bool: ...
    async def unavailable_reason(self) -> str | None: ...  # why, when unavailable — feeds /v1/ready's detail
    async def analyse(self, ctx: TierContext) -> TierVerdict: ...
```

```python
# providers/base.py

class LLMProvider(Protocol):
    name: Provider
    async def health(self) -> bool: ...
    @property
    def configured(self) -> bool: ...       # credentials/endpoint present; no I/O
    @property
    def health_detail(self) -> str | None: ...  # why the last health() failed, or None
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
    async def aclose(self) -> None: ...

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

A single `LLMProvider` protocol covers both the local and the cloud tiers. The local tier is
not a special case — it is a model that happens to be free and nearby, with which concrete
model that even is left entirely to air-llm's own routing — which means Tier 2 and Tier 3
share their prompt-building and validation code entirely (though not their system prompt —
§6).

## 3. Pydantic contracts

### Shared

```python
class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_tier: Tier | None = None
    max_tier: Tier = Tier.T3_CLOUD_LLM
    latency_budget_ms: int = Field(default=8000, ge=50, le=30_000)
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

class LabeledScore[T](BaseModel):
    """A closed-vocabulary label with a confidence — tone, urgency, intent, each topic."""
    label: T
    confidence: float = Field(ge=0.0, le=1.0)

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
    tone: LabeledScore[Tone] | None = None       # LLM tiers only, when asked
    topics: list[LabeledScore[Topic]] = []       # LLM tiers only, when asked
    intent: LabeledScore[Intent] | None = None   # LLM tiers only, when asked
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
    succeeded: bool
    escalated_because: str | None      # comma-joined reason codes; None = terminal,
                                        # except `next_tier_unavailable` — a capacity
                                        # gap is terminal too but stays visible here
                                        # rather than collapsing to null like other
                                        # terminal reasons, so a caller can tell "the
                                        # ladder was satisfied" from "it wanted more
                                        # and couldn't get it" (§5 rule 4, §4)
```

### The request

One shape for the whole surface. Every field beyond `text` is optional and read only where
it is relevant — a caller with a star rating fills `rating`; a caller with neither fills
neither.

```python
class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20_000)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    options: Options = Field(default_factory=Options)

    channel: str | None = None            # app | email | support | survey | ...
    user_segment: str | None = None
    subject: str | None = Field(default=None, max_length=500)

    rating: float | None = Field(default=None, ge=0)
    rating_scale_max: float = Field(default=5.0, gt=0, le=100)
    product_id: str | None = None
    verified_purchase: bool | None = None
    title: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _rating_within_scale(self) -> Self:
        if self.rating is not None and self.rating > self.rating_scale_max:
            raise ValueError("rating cannot exceed rating_scale_max")
        return self

class ClassifyBatchRequest(BaseModel):
    items: list[ClassifyRequest] = Field(min_length=1, max_length=1000)
    options: Options | None = None        # per-batch default, item options win
```

### The response

```python
class RoutingInfo(BaseModel):
    queue: str | None = None
    priority: RoutingPriority          # P1 (critical) .. P4 (low) — a rank mapping of urgency

class ModelInfo(BaseModel):
    name: str                          # the deciding tier's own model_version
    version: str                       # this service's own release

class ClassificationResponse(BaseModel):
    request_id: str

    sentiment: SentimentResult                    # label, polarity, confidence, rationale
    tone: LabeledScore[Tone] | None = None         # LLM-only — see §6
    urgency: LabeledScore[Urgency]                 # heuristic, always present
    topics: list[LabeledScore[Topic]] = []         # LLM-only
    intent: LabeledScore[Intent] | None = None     # LLM-only
    routing: RoutingInfo                           # heuristic, always present
    requires_human: bool                           # heuristic, always present
    actionability: Actionability                   # heuristic, always present

    aspects: list[Aspect] = []                     # LLM-only, per-facet sentiment
    rating_consistency: RatingConsistency | None = None   # only when `rating` was given
    emotions: list[Emotion] = []

    language: LanguageInfo
    model: ModelInfo

    decided_by: Tier
    degraded: bool = False
    escalation_trace: list[EscalationStep] = []
    safety: SafetyReport
    usage: RequestUsage
    cached: bool = False
    latency_ms: float
    model_versions: dict[str, str] = {}
    metadata: dict[str, Any] = {}
```

`tone`, `topics`, `intent` and `aspects` need real language understanding no lexicon or
classifier head can supply — they are populated only when the ladder reached an LLM tier that
was asked to fill them (§6's prompt framing always asks). `urgency`, `routing`,
`requires_human` and `actionability` are the opposite: deterministic heuristics computed in
`pipeline/service.py` from whichever verdict decided, present on every response regardless of
which rung answered. `rating_consistency` is `None` unless the request carried a `rating`.

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
    # Optional; filled only when the prompt's task framing asks for them.
    tone: LabeledScore[Tone] | None = None
    topics: list[LabeledScore[Topic]] = Field(default_factory=list, max_length=6)
    intent: LabeledScore[Intent] | None = None
```

Structured-output schema support excludes recursion and numeric range constraints, so
`ge`/`le` bounds are enforced by Pydantic on our side after the provider returns. A value
out of range is a validation failure, which counts as a tier failure and triggers one
bounded retry before escalation.

## 4. Orchestrator

```python
async def run(self, ctx: TierContext) -> OrchestrationResult:
    ladder = self._ladder_for(ctx)      # enabled tiers within [min_tier, max_tier]
    verdicts: list[TierVerdict] = []
    steps:    list[EscalationStep] = []
    degraded = False

    position = 0
    while position < len(ladder):
        tier = ladder[position]
        timeout_s = min(self._settings.tiers.for_tier(tier.tier).timeout_ms, ctx.remaining_ms)
        if timeout_s <= 0:
            break                                       # out of budget; stop, do not fail

        verdict = await self._analyse(tier, ctx, timeout_s)  # timeout/error → failed verdict
        degraded = degraded or not verdict.succeeded
        verdicts.append(verdict)
        ctx = ctx.with_verdict(verdict)

        # Scans forward past any dead rung to the nearest one that can serve —
        # not just the adjacent one. See §12's note on why that distinction matters.
        next_position = await self._next_available_position(ladder, position + 1)
        decision = self.policy.decide(verdict, ctx, next_available=next_position is not None)
        steps.append(EscalationStep.build(verdict, decision))

        # A verdict that succeeded but was not good enough, with nothing further
        # up the ladder to hand it to, is a capacity gap rather than an outright
        # failure — but the caller is still getting a worse answer than the
        # ladder's own thresholds call acceptable, so it counts as degraded too.
        if not decision.escalate and decision.primary_reason is NEXT_TIER_UNAVAILABLE:
            degraded = True
            _log.warning("ladder.capacity_gap", stopped_at=tier.tier.value, ...)

        if not decision.escalate:
            break
        position = next_position

    best = self.policy.select_best(verdicts)
    if best is None:
        raise NoVerdictError()                          # the only path to a 5xx
    return OrchestrationResult(best=best, verdicts=verdicts, steps=steps, degraded=degraded)
```

Four properties worth noting. The loop is **tier-agnostic** — it never names a concrete
tier. `ctx` is **immutably threaded**, so a tier cannot corrupt state for its successors. A
tier failure is *not* an exception escaping the loop: it becomes a failed verdict, the
ladder continues, and `degraded` is set — which is exactly the "degrade, don't fail"
principle made mechanical. And `degraded` has a second trigger beyond outright failure: a
capacity gap, where the ladder wanted to climb but had nowhere left to go — see rule 4 in §5.

## 5. Escalation policy

```python
class EscalationDecision(BaseModel):
    escalate: bool
    reasons: list[str] = []      # stable reason codes, exported as metric labels
```

Hard limits are evaluated first and short-circuit. The quality rules are then all
evaluated together, accumulating reasons. Availability is checked **last**, deliberately —
see the note below the table.

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
| 4 | wanted to climb, but nothing further up the ladder can serve | **stop** (`next_tier_unavailable`) |

Rule 4 runs last on purpose: the quality rules (5–11) are pure computation and cost nothing
to evaluate, so checking them first is what lets `next_tier_unavailable` mean what it says —
*we wanted to climb and could not* — rather than firing ahead of a confident verdict and
misreporting a finished ladder as a blocked one. Rule numbers are kept as originally assigned
so the reason codes and their tests stay traceable to this table.

Stopping on rule 4 also marks the response `degraded: true` and logs a distinct
`ladder.capacity_gap` warning (§4) — the caller got the best available verdict, but a
dependency being down, not the verdict's own quality, is why the ladder stopped there.

The **lookahead behind rule 4 spans the whole remaining ladder**, not just the adjacent rung:
a rung that is down does not mask a healthy one further up (§4's orchestrator sketch,
`_next_available_position`). One dead rung is a hop now, not a ceiling.

### Default thresholds (all configurable, no code change to tune)

| Tier | Confidence floor | Margin floor | Timeout |
| --- | --- | --- | --- |
| T0 rules | 0.95 | — | 5 ms |
| T1 classifier | 0.80 | 0.25 | 400 ms |
| T2 local LLM | 0.72 | 0.20 | 6000 ms |
| T3 cloud (default model) | 0.65 → arbitrate to the stronger model | 0.15 | 8000 ms |

T1's timeout reflects the zero-shot backend's real cost (§6), not a distilled head's — see
§12 changelog for the measurement behind the 400 ms figure specifically.

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

Ordered short-circuits, each returning immediately on match: cache hit → empty/whitespace/
punctuation-only → an explicit `rating` on the request that is unambiguous (≤20% or ≥80% of
scale) *and* whose prose is short → emoji-only text → pure lexicon match with no negation and
no contrastive marker and `token_count <= 5` → blocklist. No match yields `Label.UNKNOWN` with
confidence `0.0`, which always escalates.

### T1 — Classifier

The default backend is `ZeroShotOnnxScorer`: a quantised DeBERTa-v3 **entailment** head
(`MoritzLaurer/deberta-v3-base-zeroshot-v1`), not a head trained against one fixed label set.
It scores each of the four labels by templating a hypothesis — *"This text expresses a
positive sentiment."* — and checking whether the input entails it, batched as one inference
call, then softmaxes the entailment logit across the four labels. This is the same procedure
behind HF's `zero-shot-classification` pipeline in single-label mode, and it earns a concrete
win over a fixed-label alternative: it can score `MIXED` directly, where a two- or three-class
head never could and needed the lexicon blend to manufacture that mass instead.

`OnnxScorer` — a plain sequence-classification head, the SST-2 family — is still supported
behind `classifier.scorer_kind: sequence_classification` for a deployment that already baked
that artifact in. The two are not interchangeable at the same artifact file, so the setting is
explicit rather than autodetected from the model's output shape.

Both blend with a lexicon score for an ensemble read. Softmax probabilities give both
`confidence` (top-1) and `label_margin` (top-1 − top-2). Scoring runs in a thread-pool
executor so the event loop is never blocked. **Calibration matters regardless of backend**:
`classifier.calibration_temperature` defaults to `1.0` (uncalibrated), and every threshold in
the table above is a claim about probability that an uncalibrated model does not actually
make true.

INT8 quantization of the zero-shot checkpoint is not a drop-in step — naive per-tensor
dynamic quantization measurably destroys its ability to discriminate between hypotheses.
`scripts/export_zeroshot_model.py` is pinned to a corrected configuration
(`per_channel=True`); see the §12 changelog for the measurement.

### T2 — Local LLM (via air-llm)

Prompt = a compact system prefix (not the cloud tier's — see below) + one task framing +
the caller's optional fields + the input. Requests the JSON schema derived from
`LLMSentimentOutput`, with `format` constrained so the model cannot emit prose.
`temperature: 0`, deterministic seed. One retry on schema-validation failure with a repair
instruction appended; a second failure marks the tier failed and escalates.

The call itself is one `AirLlmProvider.classify()`, targeting `air_llm.local_model` — an
air-llm routing alias owned by this service (`classify-local`, by default), but deliberately
a single-entry one with no failover of its own, so a down local daemon fails this one call
cleanly and the ladder escalates to Tier 3, rather than air-llm quietly substituting a
cloud model under the Tier-2 label. The verdict reports `f"{result.provider.value}:
{result.model}"` — the resolved backend, not the alias name — the same shape Tier 3 uses,
so a caller can always see who actually answered regardless of which rung decided.

### T3 — Cloud LLM (via air-llm)

```python
async def analyse(self, ctx: TierContext) -> TierVerdict:
    primary = await self._call(
        ctx, model=self._config.cloud_model,           # "classify-default" alias, by default
        max_tokens=self._config.max_tokens,
        user_text=user_content(ctx),
    )
    output = _output_of(primary)
    final, usage = await self._arbitrate(ctx, primary=primary, output=output)
    return output.to_verdict(
        self.tier, model_version=f"{final.provider.value}:{final.model}",
        latency_ms=..., usage=usage,
    )

async def _arbitrate(self, ctx, *, primary, output):
    if output.confidence >= self._config.arbitration_floor or not self._config.arbitration_enabled:
        return primary, primary.to_usage()
    arbiter = await self._call(ctx, model=self._config.arbiter_model,     # "classify-arbiter" alias
                                max_tokens=self._config.arbiter_max_tokens,
                                user_text=user_content(ctx))               # hinted with primary's verdict
    return arbiter, primary.to_usage() + arbiter.to_usage()
```

`self._call` wraps one `AirLlmProvider.classify()` behind this rung's own timeout, circuit
breaker and bulkhead — the guards that decide *whether to attempt* the call at all. Which
provider actually answers is air-llm's decision, made per request by walking the named
alias's failover chain; this tier never sees or chooses between the cloud vendors directly.
`final.provider`/`final.model` in the verdict above are read from air-llm's own response, so
the trace reports who actually answered even when a chain failed over mid-request.

#### air-llm adapter notes

- `AirLlmProvider.classify()` POSTs one `InferenceRequest` (`task: "chat"`) to air-llm's
  `/v1/inference`, with the caller's JSON Schema riding along as `json_schema` — air-llm binds
  it to whichever structured-output mechanism the answering provider uses internally, so this
  service never parses prose or retries a schema failure at the HTTP layer. Which mechanism
  that is for a given vendor is air-llm's own concern, not this service's.
- **Prompt caching** marks the end of the system prompt as the cacheable prefix. This only
  takes effect above the model's minimum cacheable prefix, which is provider- and
  model-specific and re-measured in `providers/prompts/loader.py`'s own docstring rather than
  duplicated here (`tests/unit/test_prompts.py` asserts it). Below the threshold nothing
  errors; the cache just silently never fills. This is still this service's own prompt to size
  correctly — air-llm forwards `cache_prefix` but has no opinion on how big the prefix should be.
- The prefix must be **byte-stable**: no timestamps, no request IDs, no per-caller
  interpolation above the cache breakpoint. Volatile content — the text, the caller's
  optional fields, what cheaper tiers already concluded — goes in the user turn only.
- `AirLlmProvider.classify()` reads the response's own `provider` field for
  `ProviderResult.provider`, not the adapter's own constructed-with label — cost and usage
  attribution has to follow who actually answered, which can differ from the label whenever
  an alias's failover chain moved to a different vendor mid-request.
- Non-2xx handling maps air-llm's own status codes (429 → retryable rate limit, 5xx →
  unavailable) rather than any one vendor's SDK exceptions — air-llm has already absorbed
  that per-vendor distinction.

**The remote and local system prompts are different files, not one shared string.** The
remote one is deliberately long to clear the cache floor above — a cached prefix bills at
roughly a tenth of the input rate from the second call onward. That argument is worth nothing
locally: whichever runtime serves Tier 2 today has no cross-request cache, so every token of a
long prefix is re-processed on every call. The local prompt
(`providers/prompts/system_compact.md`) keeps the label taxonomy and the
confidence-calibration rubric — the policy gates on the confidence T2 reports — and drops the
long-form discussion. Both share the same task framing and output schema in the *user* turn
(`providers/prompts/loader.py`).

## 7. Configuration

`pydantic-settings`, prefix `AIR_CLASSIFIER__`, nested via `__`. Every key in `.env.example`.

| Key | Default | Purpose |
| --- | --- | --- |
| `APP__ENV` | `development` | Environment name |
| `APP__WORKERS` | `2` | Uvicorn workers |
| `APP__MAX_BODY_BYTES` | `1048576` | Ingress body cap |
| `APP__MAX_BATCH_ITEMS` | `100` | Effective batch ceiling (schema allows up to 1000) |
| `CLASSIFIER__SCORER_KIND` | `zero_shot_nli` | `zero_shot_nli` \| `sequence_classification` — which Tier-1 backend `MODEL_PATH` is |
| `CLASSIFIER__CALIBRATION_TEMPERATURE` | `1.0` | Temperature scaling; `1.0` = uncalibrated |
| `TIERS__T0_RULES__TIMEOUT_MS` | `5` | — |
| `TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR` | `0.80` | Escalation threshold |
| `TIERS__T1_CLASSIFIER__MARGIN_FLOOR` | `0.25` | Top-2 margin threshold |
| `TIERS__T1_CLASSIFIER__TIMEOUT_MS` | `400` | Sized for the zero-shot backend — §6 and the §12 changelog |
| `TIERS__T2_LOCAL_LLM__CONFIDENCE_FLOOR` | `0.72` | — |
| `AIR_LLM__ENABLED` | `true` | Switches Tiers 2 and 3 off together when false — a local-only deployment is a supported shape |
| `AIR_LLM__BASE_URL` | `http://localhost:8083` | air-llm's address |
| `AIR_LLM__TOKEN` | — | Secret, injected; air-llm's service token for this caller |
| `AIR_LLM__LOCAL_MODEL` | `classify-local` | Tier 2 — an air-llm alias, deliberately single-entry with no failover (§6's T2 note) |
| `AIR_LLM__CLOUD_MODEL` | `classify-default` | Tier 3 — an air-llm routing alias with its own cross-vendor failover |
| `AIR_LLM__ARBITER_MODEL` | `classify-arbiter` | Tier 3 arbitration trigger's target alias |
| `AIR_LLM__ARBITRATION_FLOOR` | `0.65` | Confidence below which the arbiter alias is called |
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
| `air_classifier_requests_total` | counter | `endpoint`, `status`, `api_key_id` |
| `air_classifier_request_duration_seconds` | histogram | `endpoint` |
| `air_classifier_tier_decisions_total` | counter | `tier`, `outcome` (`stop`/`escalate`/`failed`) |
| `air_classifier_escalations_total` | counter | `from_tier`, `to_tier`, `reason` |
| `air_classifier_tier_duration_seconds` | histogram | `tier` |
| `air_classifier_decided_by_total` | counter | `tier` |
| `air_classifier_llm_tokens_total` | counter | `provider`, `model`, `kind` (`in`/`out`/`cache_read`/`cache_write`) |
| `air_classifier_llm_cost_usd_total` | counter | `provider`, `model` |
| `air_classifier_cache_events_total` | counter | `result` (`hit`/`miss`/`store`) |
| `air_classifier_breaker_state` | gauge | `provider` (0 closed / 1 half-open / 2 open) |
| `air_classifier_degraded_responses_total` | counter | `reason` |
| `air_classifier_pii_redactions_total` | counter | `kind` |

`decided_by_total` divided by `requests_total` **is** the tier-distribution KPI from the
plan — the primary dashboard panel and the number that says whether the ladder is earning
its keep.

### Logs

One access log per request, plus one structured event per tier decision:

```json
{"event":"tier.decision","request_id":"req_01J...","trace_id":"4bf9...",
 "tier":"t1_classifier","label":"positive","confidence":0.52,"margin":0.11,
 "latency_ms":95.2,"escalate":true,"reasons":["low_confidence","contrastive_conjunction"],
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
  "instance": "/v1/classify",
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
| **Unit** | Policy truth table (each rule in §5 fires exactly when it should); feature extraction; PII redaction; lexicon scoring; breaker state machine; the derivation heuristics (`_derive_urgency`, `_derive_routing`, ...) |
| **Tier fakes** | Orchestrator driven by scripted fake tiers — verifies ordering, `min_tier`/`max_tier`, budget and cost stops, failure→degrade, best-verdict selection, and the multi-hop lookahead, with **no network at all** |
| **Contract** | Golden request/response fixtures; OpenAPI snapshot diffed in CI so a schema change cannot land silently |
| **Provider** | Scripted-stub tests for `AirLlmProvider` against air-llm's `/v1/inference` contract (health, classify, error mapping) — no network, no real air-llm |
| **Integration** | Real Tier 0/Tier 1 and the real `Orchestrator`/`EscalationPolicy` wired together (`tests/integration/`); Tiers 2/3 are scripted stubs — no external services, nothing opens a socket |
| **Accuracy** | `evals/golden_set.jsonl` — labelled examples run through the real app (`tests/eval/`), split CI-safe (T0/T1, `make test-eval`) vs. needs-air-llm (T2/T3, `pytest.mark.live`, `make test-eval-live`); a directional calibration check asserts confident cases score meaningfully higher than deliberately-hard ones, not full confusion matrices or ladder-vs-cloud-only agreement measurement — see the file's own docstring for what it does and does not claim |

## 11. Implementation order

Matching the plan's phases, the concrete build order was:

1. `config.py` → `main.py` → `api/errors.py` → `observability/logging.py` → health endpoints
2. `schemas/*` complete → the route with a stub verdict → OpenAPI review → auth + idempotency
3. `preprocess/*` → `pipeline/features.py` → `tiers/t0` → `tiers/t1` → `policy` →
   `orchestrator` (+ the fake-tier suite, which should exist *before* real tiers do)
4. `providers/base` + `ollama_provider` → `tiers/t2` → `resilience/*`
5. `providers/anthropic_provider` → `openai_provider` → `registry` → `tiers/t3` →
   `observability/cost`
6. `security/quotas` → metrics + tracing → batch concurrency → load test → Docker/compose
   finalisation

Writing the fake-tier orchestrator suite in step 3, before any real tier exists, was
deliberate: it pins the ladder's semantics while they are still cheap to change.

---

## 12. Changelog

Corrections to this document worth leaving a trace of, newest first. Each entry says what
changed and why; the *current* behaviour is described in the sections above, not here.

- **`/v1/capabilities` requires `X-API-Key` and is rate-limited, same as `/v1/classify`.**
  Flagged in peer review: it was the only `/v1` route besides `/v1/health`/`/v1/ready` with
  neither, despite disclosing configured model/alias names and which Tier-1 backend actually
  loaded — real deployment detail, not the liveness boolean a kubelet or load balancer needs
  to probe without credentials (§4, §7's Security note). `/v1/health` and `/v1/ready` stay
  exempt on purpose; `/v1/capabilities` no longer does.

- **`CloudLLMTier` self-builds a bulkhead when none is passed, exactly as `LocalLLMTier`
  already did.** Also flagged in peer review: the air-llm cutover's constructor rewrite (§6)
  gave both tiers the same `bulkhead: Bulkhead | None = None` shape, but nothing had
  actually verified `main.py`'s production call — `CloudLLMTier(settings, cloud)`, no
  bulkhead argument — exercised the self-built default rather than silently running
  unprotected (`guard.py`'s `guarded()` skips bulkhead protection outright when handed
  `None`, rather than raising). It does; a regression test now pins it at both the
  constructor level and through `build_ladder()` itself, since the bug this guards against
  lives in the wiring, not only in the tier class.

- **`air_llm.max_tokens`/`arbiter_max_tokens` default to 2048/4096, not 1024/2048.**
  Collapsing three provider-specific token ceilings into one shared setting (the air-llm
  cutover below) initially kept Anthropic's old default, 1024 — generous for Anthropic, but
  both `classify-default` and `smart` carry a Gemini fallback rung, and Gemini's 2.5 family
  bills thinking tokens as output. Verified live: this service's ~8.8k-token system prompt
  left Gemini so little of a 1024 budget after reasoning that it never reached the closing
  brace, which surfaced as a schema-validation failure and looked like a parsing bug rather
  than what it was — the same trap §6's Anthropic-adapter-notes-turned-air-llm-adapter-notes
  already named for Gemini specifically, rediscovered the moment a shared ceiling made it
  live traffic's problem instead of one vendor's configuration detail.

- **Every model call — Tier 2 and Tier 3 both — routes through air-llm, not a provider SDK
  or air-infra's gateway.** `providers/anthropic_provider.py`, `openai_provider.py`,
  `gemini_provider.py`, `ollama_provider.py`, `infra_provider.py`, and `providers/registry.py`
  (`ProviderRegistry`, cross-vendor failover, per-provider health caching) are gone,
  replaced by one `AirLlmProvider` (§6) that speaks air-llm's `POST /v1/inference`. air-llm
  is a separate, independently-deployed platform service — the model-hosting half of what
  used to be air-infra's `gateway/` module, extracted into its own repo because air-infra
  itself is being scoped down to local-development convenience only and will never run in
  production. What this service keeps is exactly what it should: the escalation decision
  (confidence floors, arbitration) and the prompt/schema construction — "is this answer good
  enough" stays here; "where and how to execute the call" is entirely air-llm's now. One
  side effect worth naming: Tier 3's `model_version` no longer advertises "the first
  configured provider in preference order" (that concept is gone, along with the
  multi-provider `ProvidersSettings` block) — it is simply the configured `cloud_model`,
  which may itself be an air-llm routing alias rather than a bare model id.

- **One route, not three.** `/v1/sentiment`, `/v1/feedback` and `/v1/reviews` are removed.
  `/v1/classify` (§3, §6's tier notes) absorbed every field any of the three produced —
  `aspects`/`rating_consistency` from reviews, `actionability` from feedback — behind one
  request and one response shape. `ResourceKind` and the per-resource dispatch it drove
  (prompt framing, response shaping, a `REVIEWS`-only gate on the T0 rating shortcut) are
  gone with it: there is one path now, not a choice between four.

- **Tier 1's default backend is a zero-shot NLI head, not a fixed classifier.** No
  general-purpose, non-domain-locked sequence-classification checkpoint exists for the
  architecture this deployment wants (DeBERTa-v3-base), so the default became an entailment
  head scored per-label against a templated hypothesis (§6) — which, as a side effect, can
  express `MIXED` directly instead of relying on the lexicon blend for it. Two corrections
  came with it: **naive INT8 quantization silently destroys this checkpoint's ability to
  discriminate** (`per_channel=True` fixes it — `scripts/export_zeroshot_model.py` is pinned
  to that configuration), and **Tier 1's timeout rose from 200ms to 400ms** — four batched
  forward passes on a 184M-parameter encoder measured p90 193ms / max 199ms inside the built
  container, already losing samples to the old ceiling.

- **The orchestrator's lookahead spans the whole remaining ladder**, not just the adjacent
  rung (§4, §5). A dead rung used to cap every request below it even when a healthy rung sat
  further up, because the policy's "can I climb" check only ever looked one hop ahead.

- **Rule 4 (§5) evaluates last.** Originally fourth, which made `next_tier_unavailable` fire
  ahead of the quality rules and misreport a confidently-finished ladder as a blocked one.

- **Tier-2's timeout is 6000ms, not the original 4000ms guess** — measured warm on
  `llama3.2:3b`/CPU: p50 1.5s, p90 3.3s. `options.latency_budget_ms` rose from 5000ms to
  8000ms alongside it, since it is the binding constraint far more often than any single
  tier's own timeout.

- **Tier 2 and Tier 3 do not share one system prompt.** The remote prefix is sized to clear
  Anthropic's prompt-cache floor; that argument does not transfer to a model with no
  cross-request cache, and feeding Ollama the long prefix cost ~9.9s (against ~1.5s for a
  compact one written for it) while silently exceeding `num_ctx` and truncating itself.
