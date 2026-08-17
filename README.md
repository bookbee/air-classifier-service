# air-classifier

Text classification for the AIR platform. Takes text plus its surrounding context and returns
a structured, explainable verdict over a versioned REST API.

Three routes share one ladder, one request envelope and one response shape, each specialising
the same verdict for a different job:

| Route | For | Adds to the verdict |
| --- | --- | --- |
| `/v1/sentiment` | Any text, no assumptions about where it came from | — |
| `/v1/feedback` | Triaging inbound product and support feedback | `urgency`, `actionability`, `suggested_route` |
| `/v1/reviews` | Marketplace reviews, prose read against the stars | `aspects`, `rating_consistency` |

The service is built as a **four-tier escalation ladder**. Most real text ("Great product,
thanks!") is trivially classifiable and a cloud LLM is two to three orders of magnitude more
expensive than it needs to be. So every request enters at the cheapest rung and climbs only
when the rung below is not confident enough — and the response says exactly how far it
climbed and why.

| Tier | Technology | Typical latency | Marginal cost | Resolves |
| --- | --- | --- | --- | --- |
| `t0_rules` | Lexicons, regex, cache | < 1 ms | ~0 | Cache hits, empty text, explicit ratings, emoji-only |
| `t1_classifier` | Quantised ONNX transformer + lexicon ensemble | 5–25 ms | ~0 | Clear polarity, conventional phrasing |
| `t2_local_llm` | Ollama · `llama3.2:3b`, schema-constrained | 200 ms – 2 s | ~0 (own hardware) | Mixed sentiment, mild sarcasm, aspects |
| `t3_cloud_llm` | Claude Haiku 4.5 → Sonnet 5 arbitration | 0.8 – 3 s | $ per call | Genuine ambiguity, subtle irony, multilingual edges |

A tier failure never becomes a 5xx: the best lower-tier verdict is returned with
`degraded: true`. The only 503 is "no tier produced anything".

## Quickstart

```bash
make install                                   # .venv + the service and its runtime extras
make env                                       # .env from .env.example, if absent
make run                                       # http://127.0.0.1:8082 · Ctrl+C stops it
```

`make run` holds the terminal and reloads on save. To run it in the background
and drive it from anywhere: `make start`, `make status`, `make restart`,
`make stop` — logging to `.run/service.log`. `stop` kills whatever is
*listening* on the port, whoever started it, and never a process merely
connected to it. A restart is needed after editing `.env`, which is read once at
start-up.

`make env` seeds a local key store holding one known development key, so a fresh
checkout authenticates exactly the way staging and production do rather than
running with auth switched off:

| | |
| --- | --- |
| Raw key | `airc_local_dev_key` — send it as `X-API-Key` |
| Stored as | `sha256(AIR_CLASSIFIER__SECURITY__HASH_SALT + raw key)` |
| Scopes | `sentiment:read`, `admin:read` |

It is valid only against the development salt, and any real deployment overrides
both halves. `SECURITY__ALLOW_UNAUTHENTICATED` stays `false`.

```bash
curl -s http://127.0.0.1:8082/v1/sentiment \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: airc_local_dev_key' \
  -d '{"text": "The battery life is superb but the camera is a letdown."}'
```

```jsonc
{
  "request_id": "req_01JQ…",
  "sentiment": { "label": "mixed", "polarity": 0.12, "confidence": 0.88,
                 "rationale": "Praises battery, criticises camera." },
  "decided_by": "t2_local_llm",
  "degraded": false,
  "escalation_trace": [
    { "tier": "t0_rules",      "label": "unknown",  "confidence": 0.00, "latency_ms": 0.4,
      "escalated_because": "no_shortcut_matched" },
    { "tier": "t1_classifier", "label": "positive", "confidence": 0.52, "latency_ms": 11.2,
      "escalated_because": "low_confidence,contrastive_conjunction" },
    { "tier": "t2_local_llm",  "label": "mixed",    "confidence": 0.88, "latency_ms": 640.0,
      "escalated_because": null }
  ],
  "usage": { "tiers_run": 3, "tokens_in": 210, "tokens_out": 96, "est_cost_usd": 0.0 },
  "latency_ms": 652.9
}
```

`escalation_trace` is on by default; set `options.include_trace` to `false` to drop it.
Cap the ladder per request with `options.max_tier` — `"t2_local_llm"` keeps every byte on
your own hardware.

Without a running Ollama the ladder stops at `t1_classifier` and the trace says
`next_tier_unavailable`; without an ONNX artifact on disk Tier 1 runs its lexicon fallback.
Both are degradations, not failures.

## Dependencies

This repo runs **one process and nothing else**. Redis, Ollama, Prometheus and
Grafana are shared platform components owned by
[air-infra](../air-infra/README.md), which starts them once for every AIR service
rather than each repo standing up a private copy on ports the platform has
already allocated.

```bash
cd ../air-infra && make up      # redis, postgres, mongo, ollama, gateway
make ollama-pull                # the Tier-2 model, once
```

In production the question does not arise: every endpoint below is an
environment variable, DevOps points them at managed instances, and this compose
file is not deployed at all.

| Dependency | Used for | Absent ⇒ |
| --- | --- | --- |
| Ollama | Tier 2 | Ladder stops at `t1_classifier`, trace says `next_tier_unavailable` |
| Redis | Verdict cache, idempotency, quotas | Every lookup is a miss; the service is correct but colder |
| Prometheus / Grafana | Scrapes this service's `/metrics` | No dashboards; the endpoint still serves |
| air-infra gateway | Optional — routes T2+T3 through air-infra instead of the SDKs | Providers are called directly (the default) |

Nothing here is a hard dependency: each one degrades rather than fails.

```bash
make doctor    # what is reachable, in whichever mode, and what to do about it
```

### Running it as a container

```bash
make up        # this service on :8082, joined to air-infra's air-net network
make logs
make down      # stops this service; air-infra's stack keeps running
```

`make up` refuses to start if air-infra is not up, naming the fix — a missing
network is a better failure than a service that boots healthy and then times out
on every call.

### Ollama: container or native

Docker Desktop on macOS has no Metal passthrough, so a containerised Ollama runs
**CPU-only**. Running it natively is several times faster:

```bash
cd ../air-infra && OLLAMA=native make up    # frees :11434 for the host's own
OLLAMA=native make up                       # here, so the container dials the host
```

The native inner loop (`make run` / `make start`) needs no switch — it reads
`PROVIDERS__OLLAMA__BASE_URL` from `.env`, and `localhost:11434` addresses
whichever instance is serving. Do not run both copies at once: they can bind
11434 simultaneously on different stacks, and which one you reach then depends on
resolution order. `make doctor` flags it.

#### Warm the model, or the first request degrades

Ollama evicts and **reloads** a model when a request asks for a different
`num_ctx` than the resident copy. This service asks for 8192, and that reload
routinely outlasts Tier 2's 6s timeout — so the first request after Ollama
starts (or after any other client loaded the model at a different context size)
comes back `decided_by: t1_classifier`, `degraded: true`, with the T2 trace entry
at ~6000ms and `succeeded: false`.

Nothing is broken when that happens, which is exactly why it misleads. Warm it
once and the same request resolves at T2 in ~1.3s:

```bash
cd ../air-infra && make ollama-warm      # loads at num_ctx=8192, keep_alive 30m
```

`make doctor` compares the resident context size against the one this service
requests and warns before you hit it.

## Configuration

12-factor throughout: every value is an environment variable prefixed `AIR_CLASSIFIER__`,
nested with `__` — `AIR_CLASSIFIER__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.85`. Escalation
thresholds live in configuration on purpose: retuning the ladder's cost/accuracy balance
must never require a deploy of new logic.

[`.env.example`](.env.example) documents every key with its default and marks the secrets.
The types live in [`src/air_classifier/config.py`](src/air_classifier/config.py).

## API

All endpoints are under `/v1`, authenticated with `X-API-Key`, and return RFC 9457
`application/problem+json` on error.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sentiment`, `/v1/sentiment/batch` | Generic text |
| `POST` | `/v1/feedback`, `/v1/feedback/batch` | Adds urgency, actionability, routing |
| `POST` | `/v1/reviews`, `/v1/reviews/batch` | Adds aspect sentiment, rating consistency |
| `GET` | `/v1/health`, `/v1/ready`, `/v1/capabilities` | Liveness, tier readiness, tier inventory |
| `GET` | `/metrics` | Prometheus exposition |

`make openapi` writes the generated spec to `docs/openapi.json`.

## Design

- [`docs/00-plan.md`](docs/00-plan.md) — problem, goals, decisions, phased delivery
- [`docs/01-hld.md`](docs/01-hld.md) — architecture, the ladder, API surface, deployment
- [`docs/02-lld.md`](docs/02-lld.md) — implementation contract: layout, schemas, policy table,
  metric catalogue, error model

## Development

```bash
make dev          # runtime extras plus pytest, ruff and mypy
make check        # lint + mypy --strict + the full suite
make test-unit    # no external service required
make image        # build the runtime image
make help         # every target
```

The Tier-1 ONNX model is baked into the image at build time from a pinned revision
(`scripts/fetch_model.py`), so the runtime has no cold-start network dependency. If the
download fails the build still succeeds and the service starts on its lexicon fallback.

Tests mirror the source tree under `tests/`, split into `unit/`, `integration/` (marked
`integration`, needs Redis and Ollama) and `contract/`.
