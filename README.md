# air-sentiment

Sentiment analysis for the AIR platform. Takes text plus its surrounding context and returns
a structured, explainable verdict over a versioned REST API.

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
export AIR_SENTIMENT__SECURITY__ALLOW_UNAUTHENTICATED=true   # local dev has no key store
make run                                       # http://127.0.0.1:8080
```

```bash
curl -s http://127.0.0.1:8080/v1/sentiment \
  -H 'Content-Type: application/json' \
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

## The full stack

```bash
make up                                        # app + ollama + redis + prometheus + grafana
docker compose exec ollama ollama pull llama3.2:3b
make logs
make down
```

App on `:8080`, Prometheus on `:9090` scraping `/metrics`, Grafana on `:3000`.

## Configuration

12-factor throughout: every value is an environment variable prefixed `AIR_SENTIMENT__`,
nested with `__` — `AIR_SENTIMENT__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.85`. Escalation
thresholds live in configuration on purpose: retuning the ladder's cost/accuracy balance
must never require a deploy of new logic.

[`.env.example`](.env.example) documents every key with its default and marks the secrets.
The types live in [`src/air_sentiment/config.py`](src/air_sentiment/config.py).

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
