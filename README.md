# air-classifier-service

Text classification for the AIR platform. One versioned REST API takes text plus optional
context (rating, channel, subject, …) and returns a structured, explainable verdict:
sentiment, tone, topics, intent, urgency, routing, a requires-human flag, aspect-level scoring
and rating consistency — whichever of those the text and the ladder's reach actually support.

| If you are… | Start with |
| --- | --- |
| New to the service | [How it works](#how-it-works), then [Quickstart](#quickstart) |
| Calling the API | [API](#api) and [Customer summary refresh](#customer-summary-refresh) |
| Deploying or operating it | [Configuration](#configuration), [Deployment and operations](#deployment-and-operations) |
| Changing the code | [Development](#development), then [`CLAUDE.md`](CLAUDE.md) and [`docs/`](#further-reading) |
| Picking up open work | [Known issues and open items](#known-issues-and-open-items) |

## How it works

The service is a **four-tier escalation ladder**. Most real text ("Great product, thanks!") is
trivially classifiable, and a cloud LLM is two to three orders of magnitude more expensive than
it needs to be. So every request enters at the cheapest rung and climbs only when the rung
below is not confident enough — and the response says exactly how far it climbed and why
(`escalation_trace`).

| Tier | Technology | Typical latency | Marginal cost | Resolves |
| --- | --- | --- | --- | --- |
| `t0_rules` | Lexicons, regex, cache | < 1 ms | ~0 | Cache hits, empty text, explicit ratings, emoji-only |
| `t1_classifier` | Quantised zero-shot NLI encoder (ONNX) + lexicon ensemble | 80–200 ms | ~0 | Clear polarity, conventional phrasing |
| `t2_local_llm` | air-model-gateway, `local` class — a local model, schema-constrained | 200 ms – 2 s | ~0 (own hardware) | Mixed sentiment, mild sarcasm, tone/topics/intent |
| `t3_cloud_llm` | air-model-gateway, `auto` class — a cheap cloud model; below a confidence floor, a second opinion from the `cloud` class | 0.8 – 3 s | $ per call | Genuine ambiguity, subtle irony, multilingual edges |

Rules every engineer should know:

- **Deterministic fields are not model outputs.** `sentiment` comes from the deciding tier;
  `urgency`, `routing`, `requires_human`, `actionability` and `rating_consistency` are always
  present and derived deterministically from that verdict (`pipeline/service.py`,
  `# Derivations`). `tone`, `topics` and `intent` need real language understanding, so they
  are populated only when the ladder reaches T2 or T3 (`options.min_tier` guarantees that).
- **Failure degrades, it never errors.** A tier failure returns the best lower-tier verdict
  with `degraded: true`. The only 503 is "no tier produced anything" (`no-verdict`).
- **Every escalation decision lives in one place**, `pipeline/policy.py::EscalationPolicy`,
  checked in a fixed order: hard ceilings (tier ceiling, latency budget, cost budget), then
  quality checks (confidence floor, margin floor, complexity triggers, tier disagreement,
  unsupported language), then "is there anywhere to climb to". The thresholds are
  configuration, not code, so retuning cost against accuracy never needs a logic deploy.
- **All model access goes through [air-model-gateway](../air-model-gateway/README.md).** This
  repo holds no provider SDKs and no provider keys. The gateway owns credentials, vendor
  routing and failover, cost accounting, per-service budgets and caching. This service owns
  the decision of *whether* to escalate and *how much* to spend.

### Request flow

```
POST /v1/classify
  → api/v1/classify.py              HTTP, auth, rate limit, idempotency
  → pipeline/service.py             AnalysisService: normalise, redact PII, detect language,
                                    cache key; after the ladder, the deterministic derivations
  → pipeline/orchestrator.py        climbs the ladder; turns every tier failure into a failed
                                    verdict, never an exception
  → pipeline/policy.py              decides climb / stop at each rung
  → pipeline/tiers/t0..t3           the rungs (one Tier protocol; adding a fifth is a registration)
  → providers/model_gateway_provider.py   T2/T3's only HTTP client: POST /v1/generate
```

Tiers only ever see already-redacted text. Each LLM rung has its own timeout, circuit breaker
and bulkhead (`resilience/`).

### How the service uses air-model-gateway

Each rung asks the gateway for a **model class**, never a model or a vendor. The gateway
resolves the class to an ordered chain in its `deploy/routing.yaml`, where
`overrides.air-classifier-service` tunes the chains for this service alone:

| Caller | Setting (`AIR_CLASSIFIER__MODEL_GATEWAY__…`) | Class | What this service's override provides |
| --- | --- | --- | --- |
| Tier 2 | `LOCAL_MODEL` | `local` | One local rung, no fallback: a local outage fails the call cleanly and *this* ladder escalates |
| Tier 3 | `CLOUD_MODEL` | `auto` | Cloud-only, cheap end, cross-vendor failover |
| Tier 3 arbiter | `ARBITER_MODEL` | `cloud` | The strongest models, asked only below `ARBITRATION_FLOOR` (0.65) |
| Summary narrative | `SUMMARY_MODEL` | `local` | Local; a failed refresh degrades, it never escalates |

Any value other than `local`, `auto` or `cloud` fails validation at startup. The gateway's
`GET /v1/capabilities` shows which concrete models currently serve each class.

How gateway responses map to ladder behaviour (`providers/model_gateway_provider.py`):

| Gateway response | Raised as | Effect on the ladder |
| --- | --- | --- |
| 5xx, timeout, unreachable | `ProviderUnavailableError` | Capacity gap: climb on or degrade (`degraded: true`) |
| 429 `rate_limited` | `ProviderRateLimitError` | Rung fails; ladder continues |
| 429 `budget_exceeded` | `ProviderBudgetExceededError` | Rung fails until the next UTC day (this service's daily budget is spent) |
| Any other 4xx | `ProviderRequestError`, logged at **error** level | Rung recorded as *failed*: almost always a config or contract bug on this side |
| 200 with `refusal` or `finish_reason: length` | `ProviderRefusalError` / `ProviderSchemaError` | One repair retry for schema errors, then the rung fails |

The gateway's request body is closed (`extra="forbid"`): sending any field it doesn't declare
is a 422 on every call. Every call forwards `X-Request-ID`, so one id follows a request through
both services' logs.

## Quickstart

Requires Python 3.12 and `make`. Docker is only needed for container runs.

```bash
make dev        # .venv with the service, runtime extras, pytest, ruff, mypy
make env        # .env from .env.example (if absent), with a local dev API key
make run        # http://127.0.0.1:8082, reloads on save; Ctrl+C stops it
```

```bash
curl -s http://127.0.0.1:8082/v1/classify \
  -H 'Content-Type: application/json' -H 'X-API-Key: airc_local_dev_key' \
  -d '{"text": "The battery life is superb but the camera is a letdown."}'
```

With nothing else running, the service answers on Tiers 0 and 1 only (Tier 1 on its lexicon
fallback). To enable Tiers 2 and 3, start the gateway and give this service its token:

```bash
cd ../air-model-gateway && make run      # or `make up`; serves on :8083
```

```bash
# in .env, then `make restart` (.env is read once at start-up)
AIR_CLASSIFIER__MODEL_GATEWAY__BASE_URL=http://127.0.0.1:8083
AIR_CLASSIFIER__MODEL_GATEWAY__TOKEN=air-classifier-dev     # provisioned by the gateway's own .env.example
```

`make doctor` checks every dependency and says how to fix what's missing. `/v1/ready` names
*why* a rung is down.

### Ways to run it locally

| Command | What runs | Needs |
| --- | --- | --- |
| `make run` | Native, foreground, autoreload | `.venv` |
| `make start` / `stop` / `restart` / `status` | Native, background, logs to `.run/service.log` | `.venv` |
| `make up` | Container on air-infra's `air-net` network, Redis cache | air-infra up (creates `air-net` and Redis); fails loudly otherwise, on purpose |
| `make up INFRA=none` | Container on its own network, in-process cache (`docker-compose.standalone.yml`) | Docker only |
| `make down` / `make logs` | Stop / tail the container (either mode) | — |

`make stop` kills whatever is *listening* on the port, never a client merely connected to it.
In both container modes the gateway is reached over the host (`host.docker.internal:8083`),
and the container reads the gateway token from `.env`.

## API

All routes are under `/v1`. Errors are RFC 9457 `application/problem+json`. The full contract
is [`docs/openapi.json`](docs/openapi.json), which `air-client` generates from; it's also
served at `/docs` when `APP__DOCS_ENABLED=true`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v1/classify` | `X-API-Key` | Analyse one text |
| `POST` | `/v1/classify/batch` | `X-API-Key` | Analyse up to `APP__MAX_BATCH_ITEMS` (100) texts; per-item failures are isolated |
| `POST` | `/v1/summary/refresh` | `X-API-Key` | Fold new feedback into a caller-held customer summary; this service stores nothing |
| `GET` | `/v1/health` | none | Liveness: the process only |
| `GET` | `/v1/ready` | none | Readiness: `ok`, `degraded` (still 200), or `not_ready` (503, only when no tier can answer) |
| `GET` | `/v1/capabilities` | `X-API-Key` | Tier inventory, configured model classes, live Tier-1 backend |
| `GET` | `/metrics` | none — restrict at the network | Prometheus exposition |

Request headers: `X-API-Key` (required), `X-Request-ID` (optional; echoed back and forwarded to
the gateway), `Idempotency-Key` (optional; a repeat with a different body is a 409).

Useful request options (`"options": {…}`): `min_tier` / `max_tier` pin or cap the ladder
(`"max_tier": "t2_local_llm"` keeps every byte on your own hardware), `max_cost_usd` caps spend
per request (default $0.05), `include_trace` (default `true`). Pass `rating` /
`rating_scale_max` for `rating_consistency`, or `channel` to sharpen `urgency`.

```jsonc
{
  "request_id": "req_01JQ…",
  "sentiment": { "label": "mixed", "polarity": 0.12, "confidence": 0.88, "rationale": null },
  "tone": { "label": "disappointed", "confidence": 0.74 },
  "urgency": { "label": "low", "confidence": 1.0 },
  "topics": [{ "label": "product_quality", "confidence": 0.81 }],
  "intent": { "label": "information", "confidence": 0.66 },
  "routing": { "queue": "product_quality", "priority": "P4" },
  "requires_human": false,
  "actionability": "informational",
  "decided_by": "t2_local_llm",
  "degraded": false,
  "escalation_trace": [
    { "tier": "t0_rules",      "label": "unknown",  "confidence": 0.00, "latency_ms": 0.4,
      "escalated_because": "no_shortcut_matched" },
    { "tier": "t1_classifier", "label": "positive", "confidence": 0.52, "latency_ms": 95.2,
      "escalated_because": "low_confidence,contrastive_conjunction" },
    { "tier": "t2_local_llm",  "label": "mixed",    "confidence": 0.88, "latency_ms": 640.0,
      "escalated_because": null }
  ],
  "usage": { "tiers_run": 3, "tokens_in": 210, "tokens_out": 96, "est_cost_usd": 0.0 },
  "latency_ms": 735.6
}
```

### Error catalogue

| Status | Type | When |
| --- | --- | --- |
| 400 | `malformed-request` | Unparseable or oversized body (`APP__MAX_BODY_BYTES`, 1 MiB) |
| 401 | `missing-api-key` | No `X-API-Key` |
| 403 | `insufficient-scope` | Key lacks the scope, or asks for a tier above its `max_tier` |
| 404 | `not-found` | Unknown route |
| 409 | `idempotency-conflict` | Same `Idempotency-Key`, different body |
| 422 | `validation-error` | Request fails validation |
| 429 | `rate-limited` | Per-key quota exhausted; carries `Retry-After` |
| 500 | `internal-error` | Unhandled; should be rare and alert-worthy |
| 503 | `no-verdict` | Every enabled tier failed |

## Customer summary refresh

`POST /v1/summary/refresh` keeps a running, human-readable summary of one customer's feedback
and reviews **without this service storing anything**. Send back the `summary` you received
last time as `existing_summary` (omit it the first time) together with only the new items, and
store what comes back in its place.

```bash
curl -s http://127.0.0.1:8082/v1/summary/refresh \
  -H 'Content-Type: application/json' -H 'X-API-Key: airc_local_dev_key' \
  -d '{
    "customer_id": "cust_482",
    "items": [
      { "text": "The battery life is superb but the camera is a letdown.",
        "source_type": "review", "rating": 4 },
      { "text": "Support replied within minutes, very helpful team.",
        "source_type": "feedback" }
    ]
  }'
```

The response carries `summary` (`narrative`, a cumulative `rollup` of counts and ratings, and
`version`), `items` (one `ClassifyBatchItem` per input, the same shape as
`/v1/classify/batch`), `degraded` and `usage`.

- Every item goes through the same ladder as `/v1/classify`; one failed item doesn't affect
  the others or the rollup.
- The narrative is rewritten by one guarded gateway call (`SUMMARY_MODEL`, `local`). If that
  call fails, the response is still `200` with a fresh rollup, `degraded: true`, and the
  previous narrative unchanged.

## Configuration

12-factor: every setting is an environment variable prefixed `AIR_CLASSIFIER__` and nested with
`__` (`AIR_CLASSIFIER__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.85`). Types and defaults live in
[`src/air_classifier_service/config.py`](src/air_classifier_service/config.py);
[`.env.example`](.env.example) documents every key and marks the secrets. Precedence:
environment, then `.env`, then the default.

**Unknown keys are ignored, not rejected.** A misspelt or renamed key is silently dropped and
its default applies — check `/v1/capabilities` and `/v1/ready` after any configuration change.

The settings an operator usually sets (all prefixed `AIR_CLASSIFIER__`):

| Setting | Default | Notes |
| --- | --- | --- |
| `APP__ENV` | `development` | `production` refuses `ALLOW_UNAUTHENTICATED=true` and `LOG_RAW_TEXT=true` at startup |
| `APP__DOCS_ENABLED` | `true` | Serves `/docs`; consider `false` in production |
| `SECURITY__API_KEYS_INLINE` / `SECURITY__API_KEYS_FILE` | — | **Secret.** API key records (JSON); see [API keys](#api-keys) |
| `SECURITY__HASH_SALT` | dev salt | **Secret.** Must be overridden outside development |
| `SECURITY__DEFAULT_RATE_LIMIT_RPM` / `DEFAULT_BURST` | `600` / `60` | Per key, unless the key record overrides it |
| `MODEL_GATEWAY__ENABLED` | `true` | `false` runs on Tiers 0–1 only |
| `MODEL_GATEWAY__BASE_URL` | `http://localhost:8083` | The gateway's address |
| `MODEL_GATEWAY__TOKEN` | — | **Secret.** The gateway maps it to the `air-classifier-service` identity |
| `MODEL_GATEWAY__TIMEOUT_S` | `20.0` | HTTP timeout; each tier's own budget is usually tighter |
| `MODEL_GATEWAY__LOCAL_MODEL` / `CLOUD_MODEL` / `ARBITER_MODEL` / `SUMMARY_MODEL` | `local` / `auto` / `cloud` / `local` | Model classes; see the table above |
| `MODEL_GATEWAY__ARBITRATION_ENABLED` / `ARBITRATION_FLOOR` | `true` / `0.65` | The arbiter is a second billed call |
| `TIERS__T2_LOCAL_LLM__TIMEOUT_MS` / `TIERS__T3_CLOUD_LLM__TIMEOUT_MS` | `6000` / `8000` | Per-rung budgets, capped by the request's remaining budget |
| `CACHE__BACKEND` | `memory` | `memory`, `redis` or `none` |
| `CACHE__REDIS_URL` | — | **Secret** if it carries a password. Needed for a cache shared across replicas |
| `AUDIT__BACKEND` / `AUDIT__MONGO_URL` | `none` / — | Optional MongoDB audit log; treat the URL as a secret if it has credentials |
| `OBS__LOG_FORMAT` / `OBS__LOG_LEVEL` | `json` / `INFO` | `console` for local reading |
| `OBS__OTEL_ENABLED` / `OBS__OTEL_ENDPOINT` | `false` / — | OpenTelemetry traces |
| `RESILIENCE__BREAKER_THRESHOLD` / `BREAKER_COOLDOWN_S` / `LLM_CONCURRENCY` | `5` / `30` / `8` | Per-dependency circuit breaker and bulkhead |

### API keys

Keys are never stored raw. Each record is
`{"id", "name", "key_hash", "scopes", "max_tier"?, "rate_limit_rpm"?, "burst"?, "enabled"?}`,
where `key_hash = sha256(HASH_SALT + raw_key)` and scopes are `classify:read` (analysis routes)
and `admin:read` (`/v1/capabilities`). `max_tier` caps how far that key's requests may climb
(default `t3_cloud_llm`). To produce a hash:

```bash
.venv/bin/python -c 'from air_classifier_service.security.api_keys import hash_key; print(hash_key("<raw key>", "<salt>"))'
```

`make env` seeds one development key (`airc_local_dev_key`, both scopes), valid only against
the development salt.

## Deployment and operations

Production does not use the compose files. Every dependency is reached through an environment
variable pointing at a managed instance. This repo contains **no Kubernetes manifests and no CI
workflow** (see [Known issues](#known-issues-and-open-items)).

### Image

| | |
| --- | --- |
| Build | `make image` (`docker build -t air-classifier-service:dev .`); multi-stage, `python:3.12-slim` |
| Runs as | Non-root user `air` (uid 1001) |
| Listens on | Container port **8080** (`uvicorn … --proxy-headers`). The host port 8082 is only this repo's local convention |
| Workers | One uvicorn process per container. Scale with replicas; `APP__WORKERS` only applies to `python -m air_classifier_service` |
| Healthcheck | Built in: `GET /v1/health` every 30 s |
| Tier-1 model | Not baked in by default. Mount `model.onnx` and `tokenizer.json` at `/opt/models`, or bake them in with the `MODEL_URL` / `MODEL_SHA256` / `TOKENIZER_URL` / `TOKENIZER_SHA256` build args. Without them Tier 1 runs on its lexicon fallback, which is less accurate |

### Probes

- **Liveness → `/v1/health`.** The process only; never restart a pod for a dependency outage.
- **Readiness → `/v1/ready`.** Returns 200 with `ok` or `degraded` (for example, the gateway is
  down but Tiers 0–1 still answer), and 503 `not_ready` only when no tier can produce a verdict.
  Don't treat `degraded` as unready: that would take the whole fleet out of rotation for an
  outage the ladder is built to absorb.

### Production checklist

- [ ] `APP__ENV=production`.
- [ ] Real `SECURITY__HASH_SALT`, and API key records hashed with it (`API_KEYS_INLINE` or
      `API_KEYS_FILE`). The dev salt is **not** rejected automatically in production.
- [ ] `MODEL_GATEWAY__BASE_URL` and `MODEL_GATEWAY__TOKEN` set. In the gateway's repo, this
      service has an entry in its `deploy/policy/policy.*.yaml` (grants, rate limit,
      `daily_budget_usd`) and its routing override in `deploy/routing.yaml`. Those files are
      the gateway team's to change.
- [ ] `CACHE__BACKEND=redis` with a `CACHE__REDIS_URL` when running more than one replica.
- [ ] `/metrics` reachable by Prometheus and nothing else.
- [ ] The Tier-1 artifact mounted or baked in, then confirm `/v1/capabilities` does not report
      the lexicon scorer.
- [ ] `APP__DOCS_ENABLED` decided; `OBS__LOG_FORMAT=json`.

### Observability

- **Logs:** structured JSON (`structlog`), one access line per request and one `tier.decision`
  event per rung, all carrying `request_id`. Raw text is never logged unless
  `SECURITY__LOG_RAW_TEXT=true`, which is refused in production.
- **Metrics:** all prefixed `air_classifier_`: `requests_total`, `request_duration_seconds`,
  `tier_decisions_total`, `escalations_total{reason}`, `tier_duration_seconds`,
  `decided_by_total`, `llm_tokens_total`, `llm_cost_usd_total`, `cache_events_total`,
  `breaker_state`, `degraded_responses_total`, `pii_redactions_total`, `audit_events_total`.
  The full catalogue is in [`docs/02-lld.md`](docs/02-lld.md) §8. Metric names are an external
  contract; don't rename them.
- **Traces:** OpenTelemetry, off by default (`OBS__OTEL_*`), with a span per tier and per
  gateway call.

### Troubleshooting

| Symptom | Likely cause | Check / fix |
| --- | --- | --- |
| Every answer has `decided_by: t1_classifier` and no tone/topics/intent | Gateway not configured or unreachable | `/v1/ready` names the reason; `make doctor`; check `MODEL_GATEWAY__BASE_URL` and `MODEL_GATEWAY__TOKEN` |
| `/v1/ready` says "air-model-gateway not configured" | No token, or `MODEL_GATEWAY__ENABLED=false` | Set the token. A misspelt key is ignored silently |
| T2/T3 marked failed with `ProviderRequestError` in the trace; error log `provider.model_gateway.request_rejected` | The gateway rejected the request: bad token (401), missing grant (403), invalid body or model class (422) | The log line has the status and the gateway's problem `code`; fix the config or the request body |
| T3 failing with `ProviderBudgetExceededError` | This service's daily budget at the gateway is spent | Clears at 00:00 UTC; raise `daily_budget_usd` in the gateway's policy, or reduce T3 traffic |
| Tier 2 times out intermittently on the first call after idle | Local model cold-loading (Ollama unloads idle models after 5 min) | Expected; `scripts/demo-tiers.sh` warms it first. Raise `TIERS__T2_LOCAL_LLM__TIMEOUT_MS` if it happens in production |
| `/v1/capabilities` shows `scorer_kind: lexicon` | No Tier-1 ONNX artifact | Mount or bake it in ([Image](#image)) |
| `make up`: "network air-net … not found" | air-infra isn't running | Start air-infra, or `make up INFRA=none` |
| Service won't start: validation error on a `*_MODEL` setting | A value other than `local` / `auto` / `cloud` | Use a class name; vendor and model names belong in the gateway's routing config |

## Development

```bash
make dev                # .venv + dev toolchain (once)
make check              # lint + mypy --strict + full test suite (the CI gate)
make fmt                # ruff format + autofix
make help               # every target
.venv/bin/pytest tests/unit/test_policy.py::test_name -q     # one test
```

### Repository map

| Path | What's there |
| --- | --- |
| `src/air_classifier_service/api/` | Routes, auth/rate-limit/idempotency dependencies, error rendering |
| `src/air_classifier_service/pipeline/` | `service.py` (AnalysisService + derivations), `orchestrator.py`, `policy.py`, `tiers/`, `summary_service.py` |
| `src/air_classifier_service/providers/` | `model_gateway_provider.py` (the only gateway client), `prompts/` (system prompts T2/T3 send) |
| `src/air_classifier_service/preprocess/` | Normalisation, language detection, lexicon |
| `src/air_classifier_service/cache/`, `resilience/`, `security/`, `observability/`, `audit/` | Cross-cutting infrastructure |
| `src/air_classifier_service/constants.py` | Every public value: response vocabulary, escalation reasons, model classes |
| `src/air_classifier_service/config.py` | All settings |
| `tests/` | `unit/` (mirrors `src/`), `integration/`, `contract/`, `eval/` |
| `evals/golden_set.jsonl` | Labelled accuracy and calibration cases |
| `scripts/` | `doctor.sh`, `demo-tiers.sh`, Tier-1 model export and fetch |
| `docs/` | Plan, HLD, LLD, overview, `openapi.json` |

### Tests

| Command | Runs | External services |
| --- | --- | --- |
| `make test` | Everything | None |
| `make test-unit` | `tests/unit/` | None |
| `make test-integration` | `tests/integration/`: real components wired together, the gateway mocked at the HTTP layer | None |
| `make test-eval` | Golden set, Tier 0/1 cases | None |
| `make test-eval-live` | Golden set, Tier 2/3 cases (`live` marker) | A running gateway; set `MODEL_GATEWAY_TOKEN` (and `MODEL_GATEWAY_BASE_URL` if not `http://127.0.0.1:8083`). Costs a few cents |

Pytest runs with `--strict-markers`; the only markers are `integration` and `live`.

### Conventions that matter here

- **Ruff and `mypy --strict`** are enforced by `make check`; no `print` (use `structlog`).
- **Changing the API** (a route, a schema, an error): run `make openapi` and commit
  `docs/openapi.json`. `tests/contract/test_openapi.py` fails if the committed spec drifts or
  the `/v1` surface changes from the HLD's.
- **Response vocabulary** (`Label`, `Tone`, `Topic`, `Intent`, `Urgency`, … in
  `constants.py`): adding a member is safe. Removing one or changing its meaning is not, and
  needs `cache.base.CACHE_SCHEMA_VERSION` bumped.
- **Escalation logic** changes go in `pipeline/policy.py` only; `EscalationReason` is a closed
  set because it's also a metric label.
- **No vendor SDKs.** New model calls go through `providers/model_gateway_provider.py`, never a
  vendor client, and never the gateway's reference client (it ships with the gateway server's
  dependencies).
- **Docstrings explain *why*.** Module docstrings carry design rationale; read them before
  changing a module's behaviour.

### Tier-1 model

No artifact ships in the image (no public ONNX export of the default zero-shot checkpoint
exists). Produce one:

```bash
pip install "torch>=2.2,<3" "optimum[exporters]>=1.20" onnx
python scripts/export_zeroshot_model.py --output ./models
```

Both `make run` and `make up` pick up `./models`.

### Demo: one request per tier

`bash scripts/demo-tiers.sh` runs a scripted set of requests, each pinned to one tier
(`min_tier` = `max_tier`) so the verdict cannot have come from anywhere else. Sections: `t0`,
`t1`, `t2`, `t3`, `nat` (unpinned) and `climb` (forces all four rungs). Before Tier 2 it warms
the local model through the gateway with the gateway's response cache bypassed. The script
also contains two negative controls that expect a `503`.

| Tier | Sample text |
| --- | --- |
| `t0_rules` | `"absolutely terrible"` |
| `t1_classifier` | `"Genuinely delighted with this purchase, it works beautifully."` |
| `t2_local_llm` | `"Battery lasts forever, if by forever you mean until lunchtime."` (sarcasm) |
| `t3_cloud_llm` | `"Support replied within minutes, every single time, to tell me they could not help."` |

## Dependencies

None is a hard dependency: a bare checkout with nothing configured boots and serves on Tiers
0–1.

| Dependency | Owner | Used for | When absent |
| --- | --- | --- | --- |
| [air-model-gateway](../air-model-gateway/README.md) | Gateway team | Tiers 2 and 3, summary narrative | Ladder stops at `t1_classifier`; `degraded: true` where it wanted to climb |
| Tier-1 ONNX artifact | This repo | Tier 1's real model | Lexicon fallback, less accurate |
| Redis | air-infra (locally) or managed (production) | Shared verdict cache, idempotency, quotas | In-process cache; correct, but per-replica |
| MongoDB | Optional | Audit log (`AUDIT__BACKEND=mongo`) | No audit log (the default) |
| Prometheus / Grafana | air-infra (locally) | Scraping `/metrics` | No dashboards |

## Known issues and open items

| Item | Detail |
| --- | --- |
| 5 golden-set Tier 1 cases fail | `t1-pos-2`, `t1-neg-3`, `t1-multilingual`, `t1-short-neg`, `t1-pos-4` fail with the lexicon fallback (no ONNX artifact). They fail on `develop` too; re-check with the real artifact before treating them as bugs |
| `make check` fails its format step | 5 files predate the current ruff formatter (`orchestrator.py`, `pipeline/service.py`, `tiers/scorers.py`, `tests/eval/test_golden_set.py`, `tests/unit/test_prompts.py`); `make fmt` fixes them as one commit |
| Gateway routing override can drift unnoticed | If the gateway's `overrides.air-classifier-service` were removed, `local` would gain fallback rungs and `cloud` would become the cheap chain. Nothing here detects that yet; a check against the gateway's `GET /v1/capabilities` would |
| Dev salt not rejected in production | The production guard covers `ALLOW_UNAUTHENTICATED` and `LOG_RAW_TEXT` only; see the [checklist](#production-checklist) |
| No CI workflow or Kubernetes manifests in this repo | `make check` is the intended CI gate; deployment manifests live wherever DevOps keeps them |
| air-infra repository name | Local docs and the Makefile refer to `../air-infra`; air-model-gateway calls it `air-infrastructure-service`. Confirm the name and update `INFRA_DIR` and the docs if it changed |
| `X-Remaining-Budget-Ms` is ignored | Sent on every gateway call, but the gateway doesn't read it yet |
| Gateway migration note | air-model-gateway's `docs/06-consumer-migration.md` still lists this service as not migrated; update it from that repo |

## Further reading

- [`docs/overview.md`](docs/overview.md): team guide, demo script, current gaps
- [`docs/00-plan.md`](docs/00-plan.md): problem, goals, decisions, phased delivery
- [`docs/01-hld.md`](docs/01-hld.md): architecture, the ladder, API surface, deployment
- [`docs/02-lld.md`](docs/02-lld.md): implementation contract — layout, schemas, policy table,
  metric catalogue, error model, and a dated changelog (§12) of every change since the design
- [`CLAUDE.md`](CLAUDE.md): condensed engineering guide, also used by coding agents
- [air-model-gateway](../air-model-gateway/README.md): its own `docs/04-integration.md` covers
  consumer access, errors and retries
