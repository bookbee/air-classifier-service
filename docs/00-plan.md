# AIR Classifier — Plan

**Status:** Draft for review · **Owner:** Vikas Roy · **Date:** 2026-08-08

---

## 1. Problem statement

`air-classifier` is the AIR platform's sentiment assessment service. It accepts arbitrary
user-authored text — plus surrounding context and metadata — and returns a structured,
explainable sentiment verdict over a REST API.

The defining constraint is **cost/latency vs. accuracy**. Most real-world text ("Great
product, thanks!", "this is broken") is trivially classifiable. A minority is genuinely
hard (sarcasm, mixed sentiment, domain jargon, negation chains, code-switching). Sending
100% of traffic to a cloud LLM is the naive answer: accurate, slow, and expensive by two
to three orders of magnitude versus what the easy cases need.

So the service is built as a **layered escalation ladder**: cheap deterministic logic
first, then a local classifier, then a local LLM, and finally a cloud LLM — with each rung
climbed only when the rung below is not confident enough.

## 2. Goals

| # | Goal | Acceptance signal |
| --- | --- | --- |
| G1 | Layered escalation with per-tier confidence gating | ≥85% of production traffic resolved below the cloud tier |
| G2 | Typed, versioned REST API for sentiment / feedback / reviews | OpenAPI 3.1 published; Pydantic models are the single source of truth |
| G3 | Robust integration surface | API-key auth, idempotency, RFC 9457 errors, deterministic response envelope |
| G4 | Production-grade resilience | No 5xx when *any* tier produced a verdict; circuit breakers on every remote call |
| G5 | Full observability | Structured logs, OTel traces, Prometheus metrics, per-request cost attribution |
| G6 | Reproducible deployment | Multi-stage Docker image, compose stack, health/readiness probes |
| G7 | Precise documentation | Endpoint reference, escalation-policy reference, runbook |

## 3. Non-goals (v1)

- **No datastore.** The service is stateless; a cache is the only persistence (see §4).
  Audit trails live in logs, and shipping them to a warehouse is the platform's job.
- **No async job API.** v1 is synchronous request/response plus synchronous batch.
- **No training pipeline.** The Tier-1 classifier is a versioned artifact loaded at boot;
  training lives elsewhere.
- **No streaming (SSE).** Deferred until a consumer needs tier-by-tier progressive results.
- **No UI.** `air-client` covers exercising the API.

## 4. Decisions taken

| Decision | Choice | Rationale |
| --- | --- | --- |
| Cloud tier | **Multi-provider from day one** — Anthropic + OpenAI behind one `LLMProvider` protocol | Avoids a rewrite when routing/failover is needed; the protocol is the expensive part, and writing it once is cheaper than retrofitting |
| Persistence | **Stateless + cache only** | No DB to operate, back up, or migrate. Cache serves dedup, idempotency, and cost control |
| API shape | **Sync single + sync batch** | Bounded latency budget with hard per-tier timeouts makes sync viable; batch amortises overhead |
| Language / runtime | Python 3.12, FastAPI, Pydantic v2 | Matches the Pydantic-first requirement; async-native; best-in-class LLM SDK support |
| Local LLM | Ollama (`llama3.2:3b` present on the dev box) | Already installed; HTTP interface keeps it swappable for vLLM/llama.cpp |
| Cloud default | Claude Haiku 4.5, arbitrating to Claude Sonnet 5 | Haiku is the cost/latency sweet spot for classification; Sonnet is reserved for genuine ambiguity |

### Resolved — approved 2026-08-08

1. **Tier-1 model artifact — bake it in.** A pinned ONNX sentiment model is downloaded during
   the Docker build and baked into the image. Reproducibility and zero cold-start network
   beat image size. *Consequence:* the classifier tier is written against a pluggable
   `ScorerBackend`, with an ONNX backend and a pure-lexicon fallback, so the service (and the
   test suite) still runs when the artifact is absent. `/v1/capabilities` reports which
   backend is live, and readiness degrades — not fails — when the fallback is in use.
2. **Cache backend — ship both, default in-process.** `CACHE__BACKEND=memory` is the default;
   `redis` is selected by config alone. Scaling past one replica is a config change, never a
   code change.
3. **PII redaction — on by default, caller-overridable, key-enforceable.** `options.redact_pii`
   defaults to `true`. A caller may set it to `false`, *unless* their API key carries
   `force_pii_redaction`, in which case the override is rejected with `403
   insufficient-scope`. This keeps the safe default for everyone, allows opt-out for callers
   who have their own upstream redaction, and gives the operator a hard lever per key.
   Redaction always runs before any text leaves the process, and the outcome is reported in
   `safety.pii_redacted`.

## 5. Phased delivery

Each phase ends in a runnable, testable state.

### Phase 0 — Skeleton (foundation)

Project scaffolding, `pyproject.toml`, settings via `pydantic-settings`, FastAPI app factory
with lifespan, structured logging, `/v1/health` + `/v1/ready`, error handlers, Dockerfile,
compose, Makefile, CI-ready test harness.
*Exit:* `docker compose up` serves a healthy app.

### Phase 1 — Contracts

All Pydantic request/response/verdict models. Endpoint stubs for sentiment, feedback,
reviews (single + batch) returning a fixed stub verdict. OpenAPI generated and reviewed.
API-key auth dependency, idempotency handling, request-context middleware.
*Exit:* `air-client` can call every endpoint and get a schema-valid response.

### Phase 2 — Tiers 0 and 1

Normalisation, language detection, PII redaction, rules/lexicon tier, classical classifier
tier, calibrated confidence, and the escalation policy engine wiring T0→T1.
*Exit:* Service returns real verdicts with zero external dependencies.

### Phase 3 — Tier 2 (local LLM)

Ollama provider, JSON-schema-constrained prompting, timeout + circuit breaker, escalation
T1→T2, trace enrichment.
*Exit:* Nuanced inputs resolve locally without touching the cloud.

### Phase 4 — Tier 3 (cloud LLM)

`LLMProvider` protocol, Anthropic and OpenAI adapters, provider registry with routing and
failover, structured outputs, prompt caching, cost accounting, Haiku→Sonnet arbitration.
*Exit:* Full ladder operational; cost per request is measured and exported.

### Phase 5 — Hardening & ops

Rate limiting and quotas per key, bulkheads, batch concurrency control, Prometheus metrics,
OpenTelemetry traces, load test, runbook, `docs/api.md` finalised, image published.
*Exit:* Ready for a production pilot.

## 6. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Confidence scores are not comparable across tiers | Escalation policy misfires — over-escalating (cost) or under-escalating (accuracy) | Calibrate every tier onto a common 0–1 scale against a shared labelled set; treat thresholds as config, not code |
| Cloud provider outage or rate limit | Requests fail at the top rung | Circuit breaker → failover provider → graceful degradation to the best lower-tier verdict with `degraded: true`. Never 5xx when a lower tier answered |
| Prompt-cache minimum not met | Cloud cost stays high despite caching | Haiku 4.5 requires a ≥4096-token cacheable prefix; Sonnet 5 requires ≥1024. Size the system prompt + rubric deliberately, and assert cache hits in metrics |
| Local LLM saturates CPU/GPU under load | Tier-2 latency spikes drag P99 | Bulkhead semaphore + queue timeout; shed to Tier 3 (or return Tier-1 verdict) rather than queue indefinitely |
| Sensitive text sent to a third party | Compliance exposure | PII redaction before the cloud tier; `max_tier` cap enforceable per API key; opt-out per request |
| Escalation runs away on adversarial input | Cost blowout | Hard per-request cost ceiling and latency budget; ladder stops at budget exhaustion and returns best-so-far |

## 7. Success metrics

- **Tier distribution:** ≥60% T0/T1, ≤15% T3 on representative traffic.
- **Latency:** P50 < 25 ms, P95 < 400 ms, P99 < 2.5 s end-to-end.
- **Cost:** < $0.30 per 1,000 requests blended.
- **Availability:** 99.9% non-5xx; degraded-but-answered counts as success.
- **Agreement:** ≥92% label agreement with the cloud tier on a held-out set, measured by
  shadow-running T3 on a sample of traffic that resolved lower.

## 8. Review checklist

Please confirm or push back on each before coding starts:

- [ ] The four-tier ladder and its ordering
- [ ] The three resource endpoints and their differentiating fields (§HLD 4)
- [ ] The response envelope, especially `escalation_trace` being on by default
- [ ] Default thresholds in the escalation policy (§LLD 5)
- [ ] Haiku 4.5 as the cloud default with Sonnet 5 arbitration
- [ ] The three open questions in §4
