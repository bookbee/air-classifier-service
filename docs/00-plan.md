# AIR Classifier Service — Plan

**Status:** Draft for review · **Owner:** Vikas Roy · **Date:** 2026-08-08

---

## 1. Problem statement

`air-classifier-service` is the AIR platform's sentiment assessment service. It accepts arbitrary
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
| G2 | Typed, versioned REST API for text classification | OpenAPI 3.1 published; Pydantic models are the single source of truth |
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
| Model access | One `LLMProvider` protocol, one adapter — every call (local and cloud) routed through [air-llm](../../air-llm/README.md), the platform's model gateway | Multi-vendor routing/failover, cost accounting and caching are air-llm's job now, not this repo's; this service keeps only the escalation decision (LLD §6, and its changelog) |
| Persistence | **Stateless + cache only** | No DB to operate, back up, or migrate. Cache serves dedup, idempotency, and cost control |
| API shape | **Sync single + sync batch** | Bounded latency budget with hard per-tier timeouts makes sync viable; batch amortises overhead |
| Language / runtime | Python 3.12, FastAPI, Pydantic v2 | Matches the Pydantic-first requirement; async-native |
| Tier 2 (local) | Whichever model `air_llm.local_model`'s alias resolves to — this service's own decision is only the alias name, not the model | A local rung at near-zero marginal cost is the whole point of climbing past Tier 1 before paying for Tier 3; which runtime serves it is air-llm's config, swappable without a code change here |
| Tier 3 (cloud) | A cheap model by default, arbitrating to a stronger one below a confidence floor — both air-llm's routing choice | The cost/latency shape (cheap-first, escalate-on-doubt) is this service's decision; which vendor embodies "cheap" or "strong" at any given time is not |

### Resolved — approved 2026-08-08

1. **Tier-1 model artifact — mounted, not baked in.** The classifier tier is written against
   a pluggable `ScorerBackend` (a zero-shot NLI ONNX backend by default, plus a
   pure-lexicon fallback), and the image ships without the artifact baked in: it is produced
   locally with `scripts/export_zeroshot_model.py` and bind-mounted (`docker-compose.yml`
   already wires `./models:/opt/models:ro`), so rebuilding the image is never required to
   pick up a new export. Absent the mount, the service (and the test suite) still runs on the
   lexicon fallback. `/v1/capabilities` reports which backend is live, and readiness degrades
   — not fails — when the fallback is in use.
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

### Addendum — decided after this plan

4. **Customer-summary refresh — stateless, added 2026-09-05.** `POST /v1/summary/refresh` lets
   a caller fold new feedback/review items into a caller-held rolling summary (a narrative plus
   a structured rollup) without this service persisting anything — §3's "no datastore" stance
   holds unchanged; the caller owns storage and resends the summary each call. See
   `docs/01-hld.md` §4 and `docs/02-lld.md` for the full contract.
5. **Optional audit log — added 2026-09-05.** A MongoDB-backed, TTL-expiring log of
   classification jobs (redacted text + verdict) for support review, off by default
   (`AuditSettings.backend = "none"`). This is operational tooling, not the datastore §3 rules
   out — no verdict, cache, or customer data lives there, only a support-facing activity trail
   that self-expires. See `docs/02-lld.md`.

## 5. Phased delivery

Each phase ends in a runnable, testable state.

### Phase 0 — Skeleton (foundation)

Project scaffolding, `pyproject.toml`, settings via `pydantic-settings`, FastAPI app factory
with lifespan, structured logging, `/v1/health` + `/v1/ready`, error handlers, Dockerfile,
compose, Makefile, CI-ready test harness.
*Exit:* `docker compose up` serves a healthy app.

### Phase 1 — Contracts

All Pydantic request/response/verdict models. An endpoint stub for classify (single +
batch) returning a fixed stub verdict. OpenAPI generated and reviewed. API-key auth
dependency, idempotency handling, request-context middleware.
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
| Cloud provider outage or rate limit | Requests fail at the top rung | air-llm's own routing alias fails over across vendors first; if air-llm itself cannot serve, this service's circuit breaker trips and degrades to the best lower-tier verdict with `degraded: true`. Never 5xx when a lower tier answered |
| Prompt-cache minimum not met | Cloud cost stays high despite caching | Each provider behind a cloud alias has its own minimum cacheable prefix (currently 4096/1024 tokens for the two in play). Size the system prompt + rubric past the larger one deliberately, and assert cache hits in metrics |
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
- [ ] The single `/v1/classify` endpoint and its full field set (§HLD 4)
- [ ] The response envelope, especially `escalation_trace` being on by default
- [ ] Default thresholds in the escalation policy (§LLD 5)
- [ ] A cheap-first, arbitrate-on-doubt cloud tier, with air-llm choosing the vendor
- [ ] The three open questions in §4

## 9. Future extensions (not v1)

`docs/AIR-Classifier_Architecture_Design.pdf` (not authored as part of this plan; kept in
this repo as reference) sketches a considerably larger deployment — an ecommerce-domain
classifier serving both an interactive caller and high-volume peer services behind one API.
None of it is adopted here: it describes a different product shape than the general-purpose
service this plan actually built, and several pieces directly reverse decisions §3/§4 already
made deliberately (stateless-only, one generic taxonomy, sync-only). It is summarised below
so the thinking isn't lost, not as a queued backlog.

**Domain-specific taxonomy + a published contract.** A versioned ecommerce label set
(aspect sentiment over shipping/pricing/sizing/etc., a retail-shaped intent list, abuse and
spam/fake-review integrity scores) exposed at `GET /v1/taxonomy` so consumers validate
against it rather than hardcoding a copy, with `taxonomy_version` stamped on every verdict.
This service's taxonomy today is generic on purpose (docs/02-lld.md's changelog: the whole
point of the `/v1/classify` cutover was *not* re-narrowing to one domain) — adopting an
ecommerce-specific set would be a scope decision, not a technical one. The underlying
principle — additive-only label changes, a retraining that shifts distributions counts as
breaking — is adopted now regardless (`constants.py`).

**Spec-driven development — considered and declined, 2026-08-23.** `spec/openapi.yaml` +
`spec/taxonomy/v1.yaml` + `spec/slo.yaml` as the generation source for models, routes and
typed client packages, regenerated in CI with a build failure on drift. SDD was the stated
intent at the start of this engagement; what shipped is Pydantic-first instead — models are
still the single source of truth for OpenAPI (HLD principle 3), just hand-authored rather
than generated from a separate spec file, with `tests/contract/test_openapi.py`'s round-trip
test playing the drift-on-CI role a codegen step would otherwise play.

Deliberately not revisited before deployment: the core guarantee SDD exists to provide —
contract and implementation cannot silently diverge — already holds here, from the other
direction. Retrofitting spec-first now would mean re-expressing every existing model as YAML
(several carry hand-tuned cross-field validators that do not survive typical codegen
cleanly) for a service whose only consumers today are Python, in-family, with no observed
drift incident driving it. Revisit only if a concrete forcing function shows up — a
non-Python consumer needing a generated client, a non-engineering team needing to version the
taxonomy independently of code, or a governance requirement mandating contract-first review.

**Two traffic lanes.** A sync/interactive path (air-platform, tight SLO) and an async
job-based batch path (`POST /v1/classify/batch` returning a `job_id`, polled via
`GET /v1/jobs/{id}`) on separate replica pools with admission control — API key → tenant,
quota, priority; interactive preempts batch; batch sheds before interactive ever queues.
Today's `/v1/classify/batch` is fully synchronous, bounded by `app.max_batch_items` — a
genuinely different scaling model, not an incremental change. The document's own interactive
SLO (p95 < 60ms) is worth naming as a real tension if this is ever pursued: it is 1–2 orders
of magnitude tighter than this ladder's own Tier-2/Tier-3 timeouts, so "interactive" in that
model could only ever mean Tiers 0–1 — Tier 2 and 3 would have to move to the batch/async
lane entirely, not just get a tighter budget.

**New persistence.** Postgres for batch job state, an API-key/quota registry, a usage ledger,
and a label-feedback table (corrections feeding the next training round); object storage for
large batch result payloads. Directly reverses §3's "no datastore" — the current cache-only
model has no place to put any of this without it.

**Kubernetes deployment.** Two HPA-scaled pools (interactive on inflight-request count, batch
on queue depth), NetworkPolicy restricting callers by namespace, no direct internet exposure.
Today's deployment is Docker Compose only (§HLD 8) — a real gap if a GKE rollout is actually
next, not something this plan attempts to anticipate.

**Multi-head model ensemble.** One model head per taxonomy attribute (sentiment, tone,
urgency, abuse, integrity, …) rather than one escalating verdict. This is a different serving
architecture from the T0–T3 ladder this service is built around — worth a dedicated design
pass of its own rather than folding in as an amendment, since the two models (one verdict
that escalates vs. several independent per-attribute heads) don't obviously compose.

**air-llm's role narrows in that vision** to batch-pool zero-shot labelling and new-taxonomy
bootstrapping only, never the interactive path — a different relationship than today's, where
air-llm serves Tiers 2 and 3 on the one synchronous path that exists.

**Operational principles worth carrying forward regardless of which (if any) of the above
gets built** — several are already true of this service today, noted where so: per-language
*and* per-tenant metrics reporting, never one blended number (n/a — English-only, no tenant
concept yet); calibrated, per-class thresholds rather than a single accuracy figure (already
true — `TierSettings.confidence_floor`/`margin_floor` per tier, `classifier.
calibration_temperature`); PII scrubbed before it is cached, stored or logged, never after
(already true — `preprocess/pii.py`, `options.redact_pii`); a scheduled drift check against a
rolling labelled sample, not only at release (not built — `evals/` and a labelled set are
future work, per §3's "no training pipeline"); adversarial label classes (spam, fake-review)
retrained on a faster cadence than sentiment and their scores kept internal (n/a — no
integrity/abuse heads exist); cost tracked per thousand classifications, cache hit rate per
attribute (the raw metrics already exist —
`air_classifier_llm_cost_usd_total`/`air_classifier_cache_events_total` — this is a dashboard
query away, not new instrumentation).
