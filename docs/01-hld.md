# AIR Classifier — High-Level Design

**Status:** Draft for review · **Companion docs:** [Plan](00-plan.md) · [LLD](02-lld.md)

---

## 1. System context

```
   air-client / air-platform / external callers
                     │  HTTPS + X-API-Key
                     ▼
        ┌────────────────────────────────┐
        │      air-classifier (this)      │
        │  FastAPI · stateless · Docker  │
        └───┬────────┬─────────┬─────────┘
            │        │         │
     in-proc│  HTTP  │    HTTPS│
      cache │        │         │
            ▼        ▼         ▼
        Redis*    Ollama   Anthropic / OpenAI
       (optional) (local)      (cloud)

   * optional; in-process cache is the default
```

The service owns no database. Everything it needs to answer a request arrives in the
request, is loaded at boot (model artifacts, lexicons, config), or is fetched from a tier.

## 2. Architectural principles

1. **Cheapest sufficient answer.** Every request enters at the cheapest tier and climbs
   only when the policy says the current answer is not good enough.
2. **One verdict shape, four producers.** Every tier emits the same `TierVerdict`. The
   orchestrator and the policy engine are therefore tier-agnostic, and adding a fifth tier
   is a registration, not a refactor.
3. **Contracts are Pydantic.** Request, response, verdict, and provider payload models are
   all Pydantic v2. OpenAPI, validation, and provider structured-output schemas are all
   *derived* from them — never hand-maintained in parallel.
4. **Degrade, don't fail.** A failure at tier N returns the best verdict from tiers < N with
   `degraded: true`. A 5xx is reserved for "no tier produced anything".
5. **Every remote call is guarded.** Timeout + retry + circuit breaker + bulkhead, applied
   uniformly by decorator, not ad hoc per call site.
6. **Explainability is a first-class output.** The escalation trace ships in the response by
   default. Operators and callers can both see exactly why a verdict cost what it cost.

## 3. The escalation ladder

| Tier | Name | Technology | Typical latency | Marginal cost | Resolves |
| --- | --- | --- | --- | --- | --- |
| **T0** | Rules & shortcuts | Pure Python, lexicons, regex | < 1 ms | ~0 | Cache hits, empty/near-empty text, explicit rating metadata, emoji-only, blocklist |
| **T1** | Statistical classifier | ONNX quantised transformer + VADER-style lexicon ensemble | 5–25 ms | ~0 | Clear polarity, conventional phrasing, single-aspect |
| **T2** | Local LLM | Ollama · `llama3.2:3b` · JSON-schema constrained | 200 ms – 2 s | ~0 (own hardware) | Mixed sentiment, mild sarcasm, aspect extraction, domain phrasing |
| **T3** | Cloud LLM | Claude Haiku 4.5 → Sonnet 5 arbitration; OpenAI adapter as alternate | 0.8 – 3 s | $ per call | Genuine ambiguity, subtle sarcasm/irony, multilingual edge cases, high-stakes routing |

### Why this ordering

T0 is not a classifier — it is an *escape hatch*. A cached identical input, a review that
already carries `star_rating: 5`, or a two-emoji comment does not need a model at all.

T1 is a real classifier but a cheap one; it is confident on the large, easy majority and
usefully *unconfident* on everything else — which is precisely the signal the policy needs.

T2 buys genuine language understanding for the price of local compute. It is the workhorse
for anything T1 flagged as ambiguous, and it keeps data on our own hardware.

T3 is the arbiter of last resort, and internally has two rungs of its own: Haiku 4.5 handles
the bulk, and Sonnet 5 is invoked only when Haiku itself reports low confidence or when the
caller demands maximum accuracy.

### Escalation signals

The policy escalates on any of:

- **Low confidence** — tier confidence below the tier's configured floor.
- **Narrow margin** — gap between the top two candidate labels below a threshold.
- **Input complexity** — length, negation density, contrastive conjunctions ("but",
  "however"), mixed polarity between clauses, emoji/text disagreement.
- **Unsupported language** — detected language outside the Tier-1 model's training set.
- **Caller demand** — `options.min_tier`, or a `sensitivity: high` hint.
- **Disagreement** — T0 and T1 produce conflicting labels.

The policy *stops* escalating on any of: `options.max_tier` reached, latency budget spent,
cost ceiling reached, circuit breaker open with no failover, or a confident verdict.

## 4. API surface

All endpoints are versioned under `/v1`, authenticated with an `X-API-Key` header, and
respond with `application/json` (errors as `application/problem+json`, RFC 9457).

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sentiment` | Generic text sentiment |
| `POST` | `/v1/sentiment/batch` | Up to N texts in one call |
| `POST` | `/v1/feedback` | Feedback-shaped input; adds urgency, actionability, routing category |
| `POST` | `/v1/feedback/batch` | Batch variant |
| `POST` | `/v1/reviews` | Review-shaped input; adds aspect-based sentiment and rating consistency |
| `POST` | `/v1/reviews/batch` | Batch variant |
| `GET`  | `/v1/health` | Liveness — process is up |
| `GET`  | `/v1/ready` | Readiness — every enabled tier probed |
| `GET`  | `/v1/capabilities` | Tier inventory, model versions, supported languages, limits |
| `GET`  | `/metrics` | Prometheus exposition (separate port in production) |

### Why three resources rather than one endpoint with a `type` field

The three share a common core (`text`, `context`, `metadata`, `options`) but differ in
**required inputs and returned fields**:

- `/feedback` may carry `channel`, `user_segment`, and returns `urgency`,
  `actionability`, `suggested_route`.
- `/reviews` may carry `rating`, `product_id`, `verified_purchase`, and returns
  `aspects[]` plus `rating_consistency` (does the prose match the stars?).
- `/sentiment` is the unopinionated base.

Distinct Pydantic models per resource give each endpoint an honest OpenAPI schema and let
each carry its own prompt template and validation, without a discriminated union that would
make every field conditionally optional.

### Request envelope (shared core)

```jsonc
{
  "text": "The battery life is superb but the camera is a letdown.",
  "context":  { "locale": "en-GB", "channel": "app_store", "product_id": "sku-42" },
  "metadata": { "any": "caller-defined key/values, echoed back, usable by rules" },
  "options": {
    "min_tier": null,             // force a floor, e.g. "t2_local_llm"
    "max_tier": "t3_cloud_llm",   // cap the ladder (cost/privacy control)
    "latency_budget_ms": 3000,
    "include_trace": true,
    "redact_pii": true,
    "aspects": ["battery", "camera"]   // reviews: hint aspects to score
  }
}
```

### Response envelope

```jsonc
{
  "request_id": "req_01J...",
  "sentiment": {
    "label": "mixed",              // positive | negative | neutral | mixed
    "polarity": 0.12,              // -1.0 .. +1.0
    "confidence": 0.88,
    "rationale": "Praises battery, criticises camera."
  },
  "emotions": [ { "name": "satisfaction", "score": 0.4 },
                { "name": "disappointment", "score": 0.55 } ],
  "aspects":  [ { "name": "battery", "label": "positive", "confidence": 0.94 },
                { "name": "camera",  "label": "negative", "confidence": 0.91 } ],
  "language": { "code": "en", "confidence": 0.99 },
  "decided_by": "t2_local_llm",
  "degraded": false,
  "escalation_trace": [
    { "tier": "t0_rules",      "label": "unknown",  "confidence": 0.00,
      "latency_ms": 0.4,  "escalated_because": "no_shortcut_matched" },
    { "tier": "t1_classifier", "label": "positive", "confidence": 0.52,
      "latency_ms": 11.2, "escalated_because": "confidence_below_floor,contrastive_conjunction" },
    { "tier": "t2_local_llm",  "label": "mixed",    "confidence": 0.88,
      "latency_ms": 640.0, "escalated_because": null }
  ],
  "safety": { "pii_redacted": false, "flagged": false },
  "usage": { "tiers_run": 3, "tokens_in": 210, "tokens_out": 96, "est_cost_usd": 0.0 },
  "cached": false,
  "latency_ms": 652.9,
  "model_versions": { "t1_classifier": "sst2-onnx-1.3.0", "t2_local_llm": "llama3.2:3b" }
}
```

The envelope is **identical across all three resources**; `/feedback` and `/reviews` add
their extra blocks (`urgency`, `rating_consistency`, …) alongside it. A caller that only
understands the core can ignore the extras safely.

## 5. Component architecture

```
┌──────────────────────── API layer ─────────────────────────┐
│ routers  · auth dep · rate-limit dep · idempotency         │
│ request-context middleware · RFC 9457 error handlers       │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌──────────────────── Orchestration layer ───────────────────┐
│  Preprocessor → Orchestrator ⇄ EscalationPolicy            │
│  (normalise, detect lang, redact PII, extract features)    │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌────────────────────── Tier layer ──────────────────────────┐
│  T0 Rules  │  T1 Classifier  │  T2 LocalLLM │ T3 CloudLLM  │
│            all implement the Tier protocol                 │
└───────────────────────────┬────────────────────────────────┘
                            ▼
┌──────────────── Provider & infrastructure ─────────────────┐
│ ProviderRegistry → Anthropic │ OpenAI │ Ollama adapters    │
│ Cache · Resilience (breaker/timeout/bulkhead) · Cost meter │
│ Observability (logs · metrics · traces)                    │
└────────────────────────────────────────────────────────────┘
```

**Dependency direction is strictly downward.** Tiers know nothing about HTTP; providers know
nothing about tiers; the orchestrator knows the `Tier` protocol and nothing about any
concrete tier's internals. This is what makes each layer independently testable — the whole
ladder can be exercised with four fake tiers and no network.

## 6. Request lifecycle

1. **Ingress** — middleware assigns `request_id`, starts an OTel span, binds log context.
2. **Auth** — `X-API-Key` hashed and looked up; scopes, quota, and `max_tier` ceiling for
   the key are attached to the request context.
3. **Rate limit** — token bucket per key; `429` with `Retry-After` on exhaustion.
4. **Validation** — Pydantic parses the body; failures become RFC 9457 `422`.
5. **Idempotency** — if `Idempotency-Key` present and cached, return the stored response.
6. **Preprocess** — normalise whitespace/unicode, detect language, redact PII (if enabled),
   compute complexity features, derive a cache key from `(normalised_text, resource,
   relevant_options)`.
7. **Cache** — on hit, return with `cached: true` (T0 shortcut).
8. **Ladder** — orchestrator runs T0; asks the policy after each tier whether to continue;
   accumulates a `TierVerdict` per rung and merges into the final response.
9. **Egress** — build the resource-specific response, record metrics and cost, store cache
   and idempotency entries, close the span, emit an access log.

## 7. Cross-cutting concerns

### Security

- API keys presented in `X-API-Key`, stored as salted hashes, never logged.
- Per-key scopes (`sentiment:read`, `admin:read`) and a per-key `max_tier` ceiling, so a
  low-trust key can be prevented from ever reaching a third-party provider.
- PII redaction (email, phone, card-like, national-ID patterns) before any text leaves the
  process; redaction is recorded in `safety.pii_redacted`.
- Request-body size caps and batch-size caps to bound work per call.
- No raw input text at `INFO`; a stable hash plus length is logged instead. Raw-text logging
  is a deliberate, per-environment opt-in.

### Resilience

| Mechanism | Applied to | Behaviour |
| --- | --- | --- |
| Timeout | Every tier | Per-tier budget; expiry counts as tier failure |
| Retry | Remote providers only | Jittered exponential backoff on 429/5xx/connection errors |
| Circuit breaker | Each provider | Opens after N consecutive failures, half-open probe after cooldown |
| Bulkhead | Local + cloud LLM calls | Bounded-concurrency semaphore with acquisition timeout |
| Failover | Cloud tier | Registry falls to the next provider in preference order |
| Degradation | Whole ladder | Best lower-tier verdict returned with `degraded: true` |

### Observability

- **Logs** — `structlog` JSON, one access log per request plus one event per tier decision,
  all carrying `request_id` and `trace_id`.
- **Traces** — OpenTelemetry; a span per tier and per provider call, attributed with tier,
  model, tokens, and escalation reason.
- **Metrics** — Prometheus (full catalogue in the LLD): request counters, tier-decision and
  escalation counters, per-tier latency histograms, token and cost counters, cache hit
  ratio, breaker state gauges.
- **Cost** — a per-request cost meter converts provider token usage into USD using a
  configured price table, exported as both a counter and a response field.

## 8. Deployment

- **Image** — multi-stage Dockerfile: builder installs dependencies into a virtualenv;
  runtime is a slim Python 3.12 base running as a non-root user, with the Tier-1 model
  artifact baked in. Healthcheck hits `/v1/health`.
- **Process** — `uvicorn` with `uvloop`; worker count from config; graceful shutdown drains
  in-flight requests within a configurable window.
- **Compose** — this service alone, joined to air-infra's `air-net` network. Redis, Ollama,
  Prometheus and Grafana are shared platform components owned by air-infra and started once
  for the whole platform; each is a URL in configuration, so production points at managed
  instances without a code change and this file is never deployed.
- **Config** — 12-factor via environment variables mapped through `pydantic-settings`;
  secrets injected, never baked into the image; `.env.example` documents every key.
- **Probes** — `/v1/health` for liveness (process only), `/v1/ready` for readiness (probes
  each enabled tier; a cloud-tier outage marks the service *degraded*, not unready, because
  it can still serve from lower tiers).

## 9. Cloud-tier model choices

| Model | ID | Context | Price (in/out per MTok) | Role |
| --- | --- | --- | --- | --- |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1 / $5 | Default cloud classifier |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3 / $15 (intro $2 / $10 through 2026‑08‑31) | Arbitration for low-confidence or high-sensitivity requests |

Two provider-level constraints shape the design and are easy to get wrong:

1. **Structured output is produced by the provider, not by parsing.** Anthropic's structured
   outputs bind the response to a JSON Schema derived from our Pydantic model, so there is
   no regex-scraping of prose and no retry-on-parse-failure loop.
2. **Prompt caching has a minimum cacheable prefix, and it differs per model** — 4096 tokens
   on Haiku 4.5, 1024 on Sonnet 5. A terse classification prompt will silently fail to cache
   on Haiku. The system prompt, rubric, and few-shot block must therefore be sized
   deliberately, and cache-hit metrics must be asserted rather than assumed.

The OpenAI adapter implements the same `LLMProvider` protocol using that provider's own
JSON-schema response format, so the registry can route or fail over between them without the
tier knowing which one answered.
