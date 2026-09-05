# AIR Classifier Service — High-Level Design

**Status:** Draft for review · **Companion docs:** [Plan](00-plan.md) · [LLD](02-lld.md)

---

## 1. System context

```
   air-client / air-platform / external callers
                     │  HTTPS + X-API-Key
                     ▼
        ┌─────────────────────────────────┐
        │  air-classifier-service (this)  │
        │  FastAPI · stateless · Docker   │
        └───┬─────────────────────┬────────┘
            │ in-proc            │ HTTPS + X-API-Key
            │ cache (optional)   ▼
            ▼             ┌──────────────────┐
         Redis*           │      air-llm      │
                          │ (its own service) │
                          └──────────────────┘

   * optional; in-process cache is the default
```

The service owns no database, and no model is reached directly — every model call, Tier 2
and Tier 3 both, goes through air-llm, the platform's own model gateway (a separate,
independently-deployed service; see its own README for what it owns and how). air-llm is
the enabler that makes a more accurate, more expensive answer reachable at all; *which*
model or vendor actually executes a call, and at what price, is entirely its decision —
opaque from here, and deliberately so (§4's API surface, §9). Everything else this service
needs to answer a request arrives in the request, is loaded at boot (the Tier-1 model
artifact, lexicons, config), or is fetched from air-llm.

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
| **T2** | Local LLM | air-llm → a local model, JSON-schema constrained (which one is air-llm's config) | 200 ms – 2 s | ~0 (own hardware) | Mixed sentiment, mild sarcasm, aspect extraction, domain phrasing |
| **T3** | Cloud LLM | air-llm → a cheap model, then arbitration to a stronger one (vendor is air-llm's routing choice) | 0.8 – 3 s | $ per call | Genuine ambiguity, subtle sarcasm/irony, multilingual edge cases, high-stakes routing |

### Why this ordering

T0 is not a classifier — it is an *escape hatch*. A cached identical input, a review that
already carries `star_rating: 5`, or a two-emoji comment does not need a model at all.

T1 is a real classifier but a cheap one; it is confident on the large, easy majority and
usefully *unconfident* on everything else — which is precisely the signal the policy needs.

T2 buys genuine language understanding for the price of local compute. It is the workhorse
for anything T1 flagged as ambiguous, and it keeps data on our own hardware.

T3 is the arbiter of last resort, and internally has two rungs of its own: a cheap default
model handles the bulk, and a stronger one is invoked only when the default itself reports
low confidence or when the caller demands maximum accuracy. Both are air-llm routing
aliases (§9) — which vendor answers either one is not a decision this service makes.

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

All endpoints are versioned under `/v1` and respond with `application/json` (errors as
`application/problem+json`, RFC 9457). Most require an `X-API-Key` header; `/v1/health` and
`/v1/ready` deliberately do not (§7's Security note explains why).

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v1/classify` | Required | The single entry point: sentiment, tone, topics, intent, urgency, routing, requires-human, actionability, and (when the caller supplied a rating or wants aspects) rating consistency and per-aspect sentiment |
| `POST` | `/v1/classify/batch` | Required | Batch variant, up to `app.max_batch_items` |
| `POST` | `/v1/summary/refresh` | Required | Stateless customer-summary refresh: classifies new feedback/review items, merges them onto a caller-supplied `existing_summary`, and returns the updated rollup and narrative. This service stores nothing — the caller persists the result |
| `GET`  | `/v1/health` | None | Liveness — process is up |
| `GET`  | `/v1/ready` | None | Readiness — every enabled tier probed |
| `GET`  | `/v1/capabilities` | Required | Tier inventory, model versions, supported languages, limits — deployment detail for an operator, not a liveness boolean |
| `GET`  | `/metrics` | None | Prometheus exposition (network-restrict in production) |

One route rather than a family of resource-shaped ones: every caller's optional fields
(`channel`, `rating`, `product_id`, …) live on the same flat request, read only where
relevant, and every derived field (`urgency`, `aspects`, `rating_consistency`, …) is present
on the same response, populated only when the input or the deciding tier supports it. A
caller sending only `text` gets a smaller-looking response, not a different one.

### Request envelope

```jsonc
{
  "text": "The battery life is superb but the camera is a letdown.",
  "context":  { "locale": "en-GB" },
  "metadata": { "any": "caller-defined key/values, echoed back, usable by rules" },
  "options": {
    "min_tier": null,             // force a floor, e.g. "t2_local_llm"
    "max_tier": "t3_cloud_llm",   // cap the ladder (cost/privacy control)
    "latency_budget_ms": 8000,
    "include_trace": true,
    "redact_pii": true,
    "aspects": ["battery", "camera"]   // hint aspects to score
  },
  "channel": "app_store",
  "product_id": "sku-42",
  "rating": 4, "rating_scale_max": 5
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
  "tone": { "label": "frustrated", "confidence": 0.61 },
  "urgency": { "label": "low", "confidence": 0.7 },
  "topics": [ { "label": "battery_life", "confidence": 0.8 },
              { "label": "camera_quality", "confidence": 0.76 } ],
  "intent": { "label": "feedback", "confidence": 0.66 },
  "routing": { "queue": null, "priority": "p3" },
  "requires_human": false,
  "actionability": "informational",
  "emotions": [ { "name": "satisfaction", "score": 0.4 },
                { "name": "disappointment", "score": 0.55 } ],
  "aspects":  [ { "name": "battery", "label": "positive", "confidence": 0.94 },
                { "name": "camera",  "label": "negative", "confidence": 0.91 } ],
  "rating_consistency": { "agreement": "consistent", "expected_label": "positive" },
  "language": { "code": "en", "confidence": 0.99 },
  "model": { "name": "ollama:llama3.2:3b", "version": "0.4.0" },
  "decided_by": "t2_local_llm",
  "degraded": false,
  "escalation_trace": [
    { "tier": "t0_rules",      "label": "unknown",  "confidence": 0.00,
      "latency_ms": 0.4,  "escalated_because": "no_shortcut_matched" },
    { "tier": "t1_classifier", "label": "positive", "confidence": 0.52,
      "latency_ms": 95.2, "escalated_because": "low_confidence,contrastive_conjunction" },
    { "tier": "t2_local_llm",  "label": "mixed",    "confidence": 0.88,
      "latency_ms": 640.0, "escalated_because": null }
  ],
  "safety": { "pii_redacted": false, "flagged": false },
  "usage": { "tiers_run": 3, "tokens_in": 210, "tokens_out": 96, "est_cost_usd": 0.0 },
  "cached": false,
  "latency_ms": 652.9,
  "model_versions": { "t1_classifier": "zeroshot-deberta-v3-onnx-int8", "t2_local_llm": "ollama:llama3.2:3b" }
}
```

`tone`, `topics`, `intent`, and `aspects` come only from an LLM tier that was asked to fill
them; `rating_consistency` is present only when the request carried a `rating`. Every other
field in the envelope above is always present, regardless of which tier decided.

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
│ AirLlmProvider — one adapter, one call: POST /v1/inference │
│ on air-llm. Vendor routing/failover/cost/cache are its job │
│ Cache · Resilience (breaker/timeout/bulkhead)               │
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
- `/v1/health` and `/v1/ready` are the two deliberate exceptions to "everything needs a
  key": a kubelet or load balancer probing them carries no credential, and a liveness
  check that 401s looks identical to a dead process from outside. `/v1/capabilities`
  is not exempted the same way — it discloses configured model/alias names and which
  Tier-1 backend actually loaded, which is operator detail rather than a boolean a
  probe needs, so it carries the same `X-API-Key` requirement `/v1/classify` does.
- Per-key scopes (`classify:read`, `admin:read`) and a per-key `max_tier` ceiling, so a
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
| Retry | air-llm calls only | Jittered exponential backoff on 429/5xx/connection errors |
| Circuit breaker | Each rung's air-llm call (T2, T3) | Opens after N consecutive failures, half-open probe after cooldown |
| Bulkhead | Local + cloud LLM calls | Bounded-concurrency semaphore with acquisition timeout |
| Failover | Cloud tier | air-llm's own routing alias walks its cross-vendor chain first; if air-llm itself cannot serve, this service's breaker opens and the ladder degrades |
| Degradation | Whole ladder | Best lower-tier verdict returned with `degraded: true` |

### Observability

- **Logs** — `structlog` JSON, one access log per request plus one event per tier decision,
  all carrying `request_id` and `trace_id`.
- **Traces** — OpenTelemetry; a span per tier and per provider call, attributed with tier,
  model, tokens, and escalation reason.
- **Metrics** — Prometheus (full catalogue in the LLD): request counters, tier-decision and
  escalation counters, per-tier latency histograms, token and cost counters, cache hit
  ratio, breaker state gauges.
- **Cost** — air-llm reports the authoritative `cost_usd` for each call it serves; this
  service holds no price table of its own, it folds that figure into a running per-request
  spend, exported as both a counter and a response field.
- **Audit log (optional)** — a MongoDB-backed, TTL-expiring record of classification jobs
  (redacted text plus verdict) for support review, off by default. Written fire-and-forget
  behind a bounded queue, so a slow or unreachable Mongo degrades to dropped entries rather
  than adding request latency — the same optional-dependency contract Redis and air-llm
  already have with this service.

## 8. Deployment

- **Image** — multi-stage Dockerfile: builder installs dependencies into a virtualenv;
  runtime is a slim Python 3.12 base running as a non-root user. The Tier-1 model artifact
  ships empty by default — produce one with `scripts/export_zeroshot_model.py` and mount it
  (compose already does) rather than rebuild the image; absent, the tier degrades to its
  lexicon fallback rather than failing. Healthcheck hits `/v1/health`.
- **Process** — `uvicorn` with `uvloop`; worker count from config; graceful shutdown drains
  in-flight requests within a configurable window.
- **Compose** — this service alone, joined to air-infra's `air-net` network for Redis, which
  is the only shared platform component it still reaches directly. Model access (Tier 2 and
  Tier 3 both) is air-llm's — a separate, independently-deployed service on its own network,
  reached over the host (`host.docker.internal`) rather than a shared compose network,
  because air-llm is meant to be reachable the same way in every environment, local included.
- **Config** — 12-factor via environment variables mapped through `pydantic-settings`;
  secrets injected, never baked into the image; `.env.example` documents every key.
- **Probes** — `/v1/health` for liveness (process only), `/v1/ready` for readiness (probes
  each enabled tier; a cloud-tier outage marks the service *degraded*, not unready, because
  it can still serve from lower tiers).

## 9. Working with air-llm

This service does not choose which vendor answers a Tier-3 call — air-llm does, per its own
`deploy/routing.yaml`. What stays this service's own decision is *which named model or
routing alias* to ask for, and whether the answer that comes back is good enough:

- `air_llm.local_model` — an air-llm routing alias owned by this service (default
  `classify-local`), but deliberately a single-entry one with no failover of its own. A down
  local daemon must fail this call cleanly so *this* service's own ladder escalates to Tier 3,
  rather than air-llm silently substituting a cloud model under the Tier-2 label — cross-vendor
  failover inside the alias would hide exactly the signal Tier 2's escalation depends on.
- `air_llm.cloud_model` / `arbiter_model` — air-llm routing aliases, also owned by this
  service (default `classify-default` and `classify-arbiter`), each already carrying its own
  cross-vendor failover chain. Cross-vendor failover *is* wanted here — an outage on one cloud
  vendor should not remove Tier 3 outright — and that is exactly what an alias is for.

All three aliases are named and scoped to this service specifically, rather than pointed at
air-llm's shared ones — a change another consumer makes to a shared alias must not silently
change this service's escalation behaviour.

Two constraints shape this service's own prompt design and are easy to get wrong regardless
of who calls the provider API:

1. **Structured output is produced by the provider, not by parsing.** Whichever provider an
   alias currently resolves to binds the response to a JSON Schema derived from our Pydantic
   model, so there is no regex-scraping of prose and no retry-on-parse-failure loop.
2. **Prompt caching has a minimum cacheable prefix, and it differs by provider and model.** A
   terse classification prompt can silently fail to cache on whichever model an alias resolves
   to today. The system prompt, rubric, and few-shot block must therefore be sized
   deliberately, and cache-hit metrics must be asserted rather than assumed.

Model IDs and prices are DevOps-owned config in air-llm's own repo
(`deploy/routing.yaml`, `deploy/pricing.yaml`), not duplicated here — this service reports
whatever `model.name` air-llm's response says actually answered, rather than assuming.
