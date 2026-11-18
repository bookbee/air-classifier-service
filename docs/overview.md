# AIR Classifier Service — Overview

**For:** a team presentation, and whoever picks this project up next.
**Detail lives elsewhere:** this is the map, not the territory — see [00-plan.md](00-plan.md)
(why/decisions), [01-hld.md](01-hld.md) (architecture), [02-lld.md](02-lld.md)
(implementation + changelog) for the full reasoning behind everything summarized here.

---

## 1. The need

Most text sent for sentiment/classification is trivial — "great product," "this is
broken." A minority is genuinely hard: sarcasm, mixed sentiment, negation chains, damning
praise. Routing 100% of traffic to a cloud LLM is the naive answer: accurate, and two to
three orders of magnitude more expensive and slower than the easy cases need.

air-classifier-service exists to make **cost and confidence a dial, not a coin flip**: a request
enters at the cheapest tier and climbs only as far as it has to. Every rung costs more and
answers better than the one below it; the service's whole job is deciding, per request, when
that trade is worth making.

It is one service in the AIR platform, consumed by other AIR services and by `air-client`'s
demo UI. It holds no model weights or provider credentials of its own beyond Tier 1's local
classifier artifact — every LLM call (Tiers 2 and 3) is delegated to **air-model-gateway**,
the platform's model gateway.

## 2. Architecture

```
Caller → air-classifier-service
           │
           ├─ Tier 0  rules/lexicon shortcuts        (in-process, ~0ms, free)
           ├─ Tier 1  local ONNX classifier           (in-process, ~200ms, free)
           ├─ Tier 2  local LLM   ──┐
           └─ Tier 3  cloud LLM   ──┴──→ air-model-gateway ──→ whichever provider
                                     POST /v1/generate      the class's chain reaches
```

Each tier implements one `Tier` protocol (`is_available`, `unavailable_reason`, `analyse`,
`supports_language`) and emits one `TierVerdict` shape — the orchestrator and escalation
policy are tier-agnostic; adding a fifth tier is a registration, not a refactor.

**Dependency direction is strictly downward** — tiers don't know about HTTP, the
orchestrator doesn't know any tier's internals — which is what lets the whole ladder be
exercised in tests with four fakes and no network.

**Where it sits in the platform:**
- **air-infra** — shared Redis (verdict/idempotency cache), reached over `air-net`.
- **air-model-gateway** — the only path to any model, local or cloud. Each rung asks for a
  model class (`local` for Tier 2 and the summary, `auto` for Tier 3, `cloud` for the
  arbiter) rather than naming a vendor or model anywhere in this service's code. The
  gateway's `deploy/routing.yaml` (`overrides.air-classifier-service`) decides which models
  serve each class and in what failover order.
- **air-client** — the Streamlit console used to exercise and demo this API (and every
  other AIR service) by hand.

## 3. Design principles

1. **Cheapest sufficient answer.** Escalate only when the policy — not a person, not a
   hardcoded rule — decides the current answer isn't good enough.
2. **Contracts are Pydantic, all the way down.** Request, response, verdict, and even the
   provider wire payload to the gateway are Pydantic v2 models. OpenAPI and validation are
   *derived* from them, never hand-maintained in parallel.
3. **Degrade, don't fail.** A tier failing, or the ladder wanting to climb and finding
   nothing available, returns the best verdict any tier produced, with `degraded: true` and
   a specific reason — never a silent wrong answer, and a 5xx only when *no* tier produced
   anything at all.
4. **This service decides *whether* to escalate and *how much* to spend; the gateway
   decides *who answers*.** No vendor name, model family, or provider SDK appears anywhere in
   this codebase's logic — only the model classes it asks for, and scripted test fixtures.
5. **Resilience is per-dependency, not global.** A circuit breaker and bulkhead guard each
   remote call independently, so one dependency being down degrades exactly the tiers that
   need it and nothing else.

## 4. Implementation summary

| Layer | What it is |
| --- | --- |
| API | FastAPI; `/v1/classify`, `/v1/classify/batch`, `/v1/summary/refresh`, `/v1/health`, `/v1/ready`, `/v1/capabilities`, `/metrics`; API-key auth, idempotency, RFC 9457 errors |
| Orchestration | `Orchestrator` runs the ladder; `EscalationPolicy` is a pure, fully-tested rule table (11 rules) deciding stop-vs-climb after every tier |
| Tier 0 | Deterministic rules + lexicon shortcuts |
| Tier 1 | Zero-shot NLI ONNX classifier (INT8, per-channel quantized) blended with the lexicon; falls back to lexicon-only if the artifact isn't mounted |
| Tier 2 / 3 | One `ModelGatewayProvider` adapter, one call — `POST /v1/generate` on air-model-gateway, asking for a model class; Tier 3 additionally arbitrates to a stronger model when confidence is still low |
| Summary refresh | `/v1/summary/refresh` — stateless: classifies new items via the same ladder, merges a rollup, refreshes a narrative through a separately-guarded gateway call that degrades (never fails) on its own |
| Audit log | Optional, off by default: MongoDB-backed, TTL-expiring record of classification jobs for support review — fire-and-forget, never affects a response (`audit/`) |
| Resilience | Timeout + circuit breaker + bulkhead per remote dependency (`resilience/`) |
| Observability | `structlog` JSON logs, OpenTelemetry traces, Prometheus metrics, cost attribution straight from the gateway's own reported `cost_usd` (no local price table) |
| Config | `pydantic-settings`, 12-factor via env vars, every key documented in `.env.example` |
| Tests | Unit (policy truth table, each module behind fakes) → tier fakes (orchestrator wiring, no network) → contract (OpenAPI + golden fixtures) → provider (scripted gateway contract) → integration (`tests/integration/`, real internal wiring, no external services required) |

Full endpoint/config/testing reference: [02-lld.md](02-lld.md).

## 5. Running a demo

This story is strongest performed live, in this order — it was run end-to-end this way
during final review and every transition below was verified with **zero restarts** of
air-classifier-service itself:

1. **Start clean.** `make up` here, plus air-infra and air-model-gateway up. `bash scripts/doctor.sh`
   confirms everything is reachable before you start talking.
2. **The ladder, cheap to expensive** — either via `bash scripts/demo-tiers.sh` (scripted,
   narrated output) or by hand through air-client's classifier tab (its tier-probe buttons
   pin a request to one rung so the audience sees exactly which tier answered):
   - `"absolutely terrible"` → stops at **T0**, sub-millisecond, free.
   - A plain, unambiguous sentence → stops at **T1**, no LLM touched.
   - `"Battery lasts forever, if by forever you mean until lunchtime."` → T1 confidently
     gets this *wrong* (`neutral`, ~0.36); **T2** catches the sarcasm (`negative`, ~0.92).
     This is the single best line for showing *why* the ladder exists.
   - `"Support replied within minutes, every single time, to tell me they could not help."`
     → escalates to **T3** for arbitration — damning praise a cheaper model reads as
     positive.
3. **The resilience story — the part worth building the demo around.** Stop air-infra and
   air-model-gateway entirely (air-classifier-service keeps running — it has no `depends_on` on either).
   Re-run the sarcasm case: T0/T1 still answer instantly; the natural (unpinned) request
   climbs, finds nothing above T1, and returns `200` with `degraded: true` and
   `escalated_because: "next_tier_unavailable"` — not a 503, not a hang. Show
   `/v1/ready`: `degraded`, each down component naming *why* (`"air-model-gateway unreachable at
   ...: ConnectError"`), not a bare "unavailable." Then bring air-infra up (cache recovers
   on the next request, no restart), then air-model-gateway (T2/T3 clear on the next request, no
   restart). This is the actual engineering the rest of the ladder demo doesn't show.
4. **Introspection.** `/v1/capabilities` for what's configured and live per tier;
   `/metrics` for the Prometheus catalogue; `docker logs -f` filtered to `tier.decision` /
   `ladder.capacity_gap` for a narrated trace of exactly what the policy just did and why.

## 6. Current gaps

Specific, not hand-wavy — each of these is either directly observed this review cycle or
explicitly called out as unbuilt in `00-plan.md`:

| Gap | Why it matters | Next step |
| --- | --- | --- |
| Accuracy-evaluation harness is a starting set, not a complete one | `evals/golden_set.jsonl` (48 entries) plus `tests/eval/` now runs in the build and directionally checks calibration (`make test-eval`) — this was fully unbuilt before; what is still missing is scale (the "100-200" a real programme wants), true per-class confusion matrices, and drift tracking across model/prompt updates over time, not just a pass/fail at one point in time | Grow the set as real production misclassifications are found; add a confusion-matrix report, not just pass/fail |
| T1/T3 per-tier timeouts are local-dev stopgaps, not calibrated defaults | `.env` currently overrides both above their `config.py` defaults, with comments explaining *why* (CPU contention from a concurrently-running local model; a schema-repair round-trip against whichever cloud provider is live occasionally exceeding 8s) — real, reproduced live, but tuned for one developer's machine under one specific load pattern | Re-measure on production-shaped hardware; treat the current `.env` values as a starting point, not a verified default |
| The gateway's per-service routing override can silently drift | The class names themselves are validated at startup, but what each class *means* for this service lives in **air-model-gateway's** `deploy/routing.yaml` (`overrides.air-classifier-service`), a separate repo. Without that override, `local` falls back to the global 3-rung local chain (Tier 2 no longer fails cleanly when its model is down) and `cloud` to the global cheap chain (arbitration would ask the same tier of model again). Nothing here detects that | A boot-time or smoke check against the gateway's `GET /v1/capabilities`, asserting `local` is one rung and `cloud` leads with a stronger model than `auto` |
| Injection defence is tested at the prompt-construction level only | `tests/unit/test_prompts.py` proves the caller's text always lands inside the fences and that a fence-spoofing attempt is logged (`prompt.fence_marker_in_input`) — it cannot prove a real model won't be fooled by adversarial phrasing that *doesn't* try to spoof the fence itself | A live/recorded-cassette red-team pass against the real provider, not just the prompt-construction code |
| The remaining-budget header to the gateway is one-sided | `ModelGatewayProvider` sends `X-Remaining-Budget-Ms` on every call, but air-model-gateway does not read it today; it only has value once the gateway uses it as the call's deadline | Raise with the air-model-gateway owners (their `docs/00-plan.md` §6 is where future work is tracked) |
| No multi-tenant concept | Metrics, cost, and quotas are per-API-key, not per-tenant | Add a tenant dimension if/when a consumer needs isolated cost or quota reporting |
| No integrity/abuse/spam-style heads | Only sentiment/tone/urgency/intent/aspects today | Would be a genuinely new taxonomy dimension, not an incremental addition — see roadmap |
| Docker Compose only | No Kubernetes manifests, no HPA, no NetworkPolicy | Real gap only if/when a cluster rollout is actually next |
| Batch is synchronous only | `/v1/classify/batch` is bounded by `app.max_batch_items`, not a job queue | Fine at current scale; becomes a real limit if batch volume or per-item latency grows |
| No persistence beyond cache and the optional audit log | No usage ledger, no label-feedback table for retraining. The audit log (`audit/`, off by default) is a support-facing operational record, not queryable customer history or a datastore in the §3 sense | Deliberate (`00-plan.md` §3); revisit only if a consumer needs queryable history beyond what the audit log already gives support |

## 7. Future roadmap

`00-plan.md` §9 has the full reasoning behind each of these — summarized here as a
pointer, not duplicated:

- **Domain-specific taxonomy** (ecommerce aspect sentiment, a published `/v1/taxonomy`
  contract) — deliberately deferred; today's taxonomy is generic on purpose.
- **Two traffic lanes** (sync interactive + async batch with a job queue) — a genuinely
  different scaling model, not a tweak; would move Tiers 2/3 out of the interactive path
  entirely if a sub-100ms interactive SLO is ever required.
- **New persistence** (Postgres for batch state, quotas, a usage ledger, label feedback) —
  directly reverses the current stateless-plus-cache decision; only worth it once one of
  those consumers is real.
- **Kubernetes deployment** — two HPA-scaled pools, NetworkPolicy, no direct internet
  exposure.
- **Multi-head model ensemble** — one head per taxonomy attribute instead of one escalating
  verdict. A different serving architecture, worth its own design pass rather than an
  amendment to the ladder.
- **Spec-driven development** — considered and explicitly declined (2026-08-23): Pydantic
  models already are the single source of truth, with `tests/contract/test_openapi.py`
  playing the drift-detection role a codegen step would otherwise play. Revisit only if a
  non-Python consumer needs a generated client, or a non-engineering team needs to version
  the taxonomy independently of code.

## 8. Handoff notes for whoever picks this up

- Read `02-lld.md` §12 (Changelog) before changing anything in the escalation policy or T1's
  quantization pipeline — several non-obvious, previously-measured failure modes are
  recorded there so they aren't rediscovered the hard way.
- The `unavailable_reason()` / `health_detail` diagnostic path (feeding `/v1/ready` and
  `/v1/capabilities`) exists specifically so an operator never has to read this service's
  own logs to find out *why* a tier is down — extend it rather than routing around it if a
  new failure mode needs surfacing.
- Every provider-facing contract (the gateway's request/response, this service's own request/
  response/verdict) is a Pydantic model on purpose — if you find yourself reaching for a
  raw `dict[str, Any]` at a service boundary, that's very likely a gap to close, not a
  shortcut to take.
