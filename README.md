# air-classifier

Text classification for the AIR platform. Takes text plus its surrounding context and returns
a structured, explainable verdict over a versioned REST API — one route, one response shape:
sentiment, tone, topics, intent, urgency, routing, a requires-human flag, aspect-level scoring
and rating consistency, whichever of those the text and the ladder's reach actually support.

The service is built as a **four-tier escalation ladder**. Most real text ("Great product,
thanks!") is trivially classifiable and a cloud LLM is two to three orders of magnitude more
expensive than it needs to be. So every request enters at the cheapest rung and climbs only
when the rung below is not confident enough — and the response says exactly how far it
climbed and why.

| Tier | Technology | Typical latency | Marginal cost | Resolves |
| --- | --- | --- | --- | --- |
| `t0_rules` | Lexicons, regex, cache | < 1 ms | ~0 | Cache hits, empty text, explicit ratings, emoji-only |
| `t1_classifier` | Quantised zero-shot NLI encoder + lexicon ensemble | 80–200 ms | ~0 | Clear polarity, conventional phrasing |
| `t2_local_llm` | air-llm → a local model, schema-constrained (which one is air-llm's own config) | 200 ms – 2 s | ~0 (own hardware) | Mixed sentiment, mild sarcasm, tone/topics/intent |
| `t3_cloud_llm` | air-llm → a metered cloud model, then arbitration to a stronger one (vendor is air-llm's choice) | 0.8 – 3 s | $ per call | Genuine ambiguity, subtle irony, multilingual edges |

`tone`, `topics` and `intent` need real language understanding no lexicon or classifier head
can supply, so they are populated only when the ladder reaches an LLM tier (T2 or T3); set
`options.min_tier` to guarantee that. Everything else — `sentiment`, `urgency`, `routing`,
`requires_human`, `actionability` — is always present, deterministically derived from whichever
rung answered.

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
| Scopes | `classify:read`, `admin:read` |

It is valid only against the development salt, and any real deployment overrides
both halves. `SECURITY__ALLOW_UNAUTHENTICATED` stays `false`.

```bash
curl -s http://127.0.0.1:8082/v1/classify \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: airc_local_dev_key' \
  -d '{"text": "The battery life is superb but the camera is a letdown."}'
```

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

`escalation_trace` is on by default; set `options.include_trace` to `false` to drop it.
Cap the ladder per request with `options.max_tier` — `"t2_local_llm"` keeps every byte on
your own hardware. A rated or channel-tagged text gets more out of the same call — pass
`rating`/`rating_scale_max` for `rating_consistency`, or `channel` to sharpen `urgency`.

Without air-llm reachable, both Tiers 2 and 3 stand down and the ladder stops at
`t1_classifier`, trace says `next_tier_unavailable`. Without an ONNX artifact on disk, Tier 1
runs its lexicon fallback. Both are degradations, not failures.

## Enabling Tiers 2 and 3

Every model call — local and cloud both — goes through
[air-llm](../air-llm/README.md), the platform's single model gateway. This service holds
no provider SDKs and no provider API keys of its own; air-llm owns the credentials, the
cross-vendor routing and failover, cost accounting and caching. Bring it up once
(`cd ../air-llm && make up`; see its own README for provider keys and local model runtime setup),
then point this service at it:

```bash
# in .env, then `make restart`
AIR_CLASSIFIER__AIR_LLM__BASE_URL=http://127.0.0.1:8083   # host.docker.internal:8083 under compose
AIR_CLASSIFIER__AIR_LLM__TOKEN=air-classifier-dev          # air-llm's own .env.example provisions this
```

`/v1/capabilities` reports each rung's live `model_version`, so it is the quickest way to
confirm the token landed and air-llm is answering.

Which model each rung asks air-llm for is still this service's own decision, but *only* as
an alias name it owns — never a vendor or model directly. `AIR_LLM__LOCAL_MODEL` (default
`classify-local`), `AIR_LLM__CLOUD_MODEL` (default `classify-default`) and
`AIR_LLM__ARBITER_MODEL` (default `classify-arbiter`) are all air-llm routing aliases
(air-llm's `deploy/routing.yaml`) defined for this service specifically — which concrete
model or vendor answers behind each one is entirely air-llm's config to change, and this
service does not need to know or care which. Arbitration means one request can be **two**
billed calls: the cheap model answers, and below `AIR_LLM__ARBITRATION_FLOOR` (0.65) the
arbiter model is asked as well. `options.max_cost_usd` (default $0.05) caps a single
request, and `AIR_LLM__ARBITRATION_ENABLED=false` turns the second call off entirely.

`AIR_LLM__SUMMARY_MODEL` (default `summarize-default`) is a fourth alias in the same
family, asked only by `/v1/summary/refresh`'s narrative step (below) — its own alias
because a narrative-refresh failure degrades gracefully rather than escalating, so it
needs no cross-vendor failover chain of its own.

## Customer summary refresh

`POST /v1/summary/refresh` gives a caller a running, human-readable summary of one
customer's feedback and reviews over time — without this service storing anything about
that customer. Every call is self-contained: send the summary you were handed last time
back as `existing_summary` (or omit it, for a customer's first-ever summary) alongside
the new text to fold in, and store whatever comes back in its place.

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

```jsonc
{
  "request_id": "req_01JQ…",
  "customer_id": "cust_482",
  "summary": {
    "narrative": "A generally positive customer who values responsive support; product quality is a mild, recurring concern.",
    "rollup": {
      "total_items": 2,
      "counts_by_source_type": { "review": 1, "feedback": 1 },
      "sentiment_counts": { "mixed": 1, "positive": 1 },
      "urgency_counts": { "medium": 1, "low": 1 },
      "topic_counts": { "product_quality": 1 },
      "rating_count": 1, "rating_sum": 4.0, "average_rating": 4.0,
      "recent_ratings": [4.0], "recent_average_rating": 4.0,
      "first_seen_at": "2026-09-05T15:17:09.535Z",
      "last_seen_at": "2026-09-05T15:17:09.535Z"
    },
    "version": 1
  },
  "items": [ /* one ClassifyBatchItem per request item, same shape as /v1/classify/batch's */ ],
  "degraded": false,
  "usage": { "tiers_run": 4, "tokens_in": 0, "tokens_out": 0, "est_cost_usd": 0.0 },
  "latency_ms": 192.7
}
```

On the *next* call for the same customer, send `summary` from this response back as
`existing_summary` alongside only the newest items — `rollup` accumulates onto it
(histograms grow, `recent_ratings` slides its window) and `narrative` is rewritten with
both old and new items in view, not just the latest batch.

Two properties worth relying on:

- **Every item is classified through the same ladder `/v1/classify` uses** — `items` in
  the response is a list of `ClassifyBatchItem`, byte-for-byte the same shape
  `/v1/classify/batch` returns, so existing tooling that already parses one parses the
  other. A per-item failure isolates the same way: that item's slot carries a
  `ProblemDetail`, its siblings and the rollup are unaffected.
- **The narrative step degrades, it never fails the call.** If air-llm (or specifically
  `AIR_LLM__SUMMARY_MODEL`) is unreachable or unconfigured, the response still comes back
  `200` with a freshly computed `rollup`, `degraded: true`, and `narrative` carried over
  unchanged from `existing_summary`. The only way this route ever fails outright is the
  same one `/v1/classify` shares — a batch-size or auth/validation problem, never "the
  summarizer was down."

## Demo — exercising every tier on purpose

```bash
bash scripts/demo-tiers.sh          # every tier, in order
bash scripts/demo-tiers.sh t2       # one section: t0 | t1 | t2 | t3 | nat | climb
```

Each case pins `options.min_tier`/`max_tier` to the *same* value, so the verdict cannot have
come from anywhere else — a pinned tier that cannot answer returns `503` rather than quietly
falling back, which is what makes this a test rather than a demo. `nat` runs the same texts
unpinned, to watch the ladder actually decide for itself where to stop; `climb` forces every
rung to answer and then escalate anyway, so you see all four fire in one request.

Before any section that touches Tier 2, the script fires one throwaway completion straight at
air-llm for the exact alias Tier 2 will ask for (`classify-local` by default) and waits for
it to come back. The local runtime that answers today evicts an idle model after 5 minutes
by default, and air-llm's adapter does not override that, so a gap longer than that between
requests means the *next* real call pays a full cold load on top of generation time — which
routinely exceeds Tier 2's 6s timeout and reads as a broken tier when it is really just a
cold cache. The warmup step exists specifically so a demo run doesn't intermittently fail on
exactly that.

### Sample texts, one confident case per tier

Each of these is written to land on exactly the tier named, pinned the same way the script
pins them (`"options": {"min_tier": "…", "max_tier": "…"}`) — useful for a quick `curl` without
running the whole script:

| Tier | Sample text | Why it lands there |
| --- | --- | --- |
| `t0_rules` | `"absolutely terrible"` | Short, unambiguous lexicon term — clears the 0.95 shortcut floor with no model involved |
| `t0_rules` | `"Arrived on time."` + `"rating": 5` | An extreme, unambiguous star rating on short prose is its own shortcut |
| `t1_classifier` | `"Genuinely delighted with this purchase, it works beautifully."` | Clear single polarity, conventional phrasing — no T0 shortcut fits, no ambiguity needs an LLM |
| `t1_classifier` | `"Der Akku ist gut, aber die Kamera enttäuscht."` | Same shape, non-English — the classifier is multilingual, T0's lexicon is not |
| `t2_local_llm` | `"Battery lasts forever, if by forever you mean until lunchtime."` | Sarcasm — a bag-of-terms classifier cannot see the reversal; needs real language understanding |
| `t2_local_llm` | `"The screen is gorgeous but the battery dies by lunchtime."` | Praise and complaint in one sentence, which must come back `mixed`, not averaged to neutral |
| `t3_cloud_llm` | `"Well, at least the box it arrived in was sturdy."` | Irony subtle enough that a 3B local model is itself unconfident and escalates |
| `t3_cloud_llm` | `"Support replied within minutes, every single time, to tell me they could not help."` | Structurally polite, substantively damning — genuine ambiguity between literal and intended sentiment |

```bash
curl -s http://127.0.0.1:8082/v1/classify \
  -H 'Content-Type: application/json' -H 'X-API-Key: airc_local_dev_key' \
  -d '{"text": "Battery lasts forever, if by forever you mean until lunchtime.",
       "options": {"min_tier": "t2_local_llm", "max_tier": "t2_local_llm"}}'
```

The full set — including two deliberately-unresolvable negative controls that expect a `503`,
not a `200` — lives in [`scripts/demo-tiers.sh`](scripts/demo-tiers.sh) itself; treat this
table as the confident subset worth memorising, not the complete case list.

## Dependencies

None of these are hard dependencies — each one degrades the service rather than breaking it,
and a bare checkout with nothing configured still boots and serves on a two-rung ladder
(T0 + T1's lexicon fallback).

| Dependency | Kind | Used for | Absent ⇒ |
| --- | --- | --- | --- |
| An ONNX artifact on disk | Optional | Tier 1's real model, instead of the lexicon fallback | `/v1/capabilities` reports `scorer_kind: lexicon`; answers still work, less accurately |
| [air-llm](../air-llm/README.md) | Optional | Tiers 2 and 3 both — local and cloud models, routing, cost, cache | Ladder stops at `t1_classifier` |
| Redis | Optional | Verdict cache, idempotency, quotas | Every lookup is a miss; correct, just colder |
| Prometheus / Grafana | Optional | Scrapes this service's `/metrics` | No dashboards; the endpoint still serves |

Redis and Prometheus are shared platform components owned by
[air-infra](../air-infra/README.md); air-llm is its own standalone service, started and
configured independently (its own README covers provider keys and its own local model runtime setup —
container vs. native, warming the model, all of it is air-llm's concern now, not this
repo's):

```bash
cd ../air-infra && make up      # redis, postgres, mongo
cd ../air-llm && make up        # every model this service will ever call
```

In production the question does not arise: every endpoint above is an environment variable,
DevOps points them at managed instances, and neither compose file is deployed at all.

```bash
make doctor    # what is reachable, and what to do about it
```

### Running it as a container

```bash
make up        # this service on :8082, joined to air-infra's air-net network for redis
make logs
make down      # stops this service; air-infra's and air-llm's stacks keep running
```

`make up` refuses to start if air-infra is not up, naming the fix — a missing
network is a better failure than a service that boots healthy and then times out
on every call. It does not check air-llm the same way: a missing air-llm is a
supported, degraded shape (`make doctor` reports it), not a reason to refuse to boot.

## Configuration

12-factor throughout: every value is an environment variable prefixed `AIR_CLASSIFIER__`,
nested with `__` — `AIR_CLASSIFIER__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.85`. Escalation
thresholds live in configuration on purpose: retuning the ladder's cost/accuracy balance
must never require a deploy of new logic.

[`.env.example`](.env.example) documents every key with its default and marks the secrets.
The types live in [`src/air_classifier/config.py`](src/air_classifier/config.py).

## API

Two analysis routes plus a stateless summary-refresh route and three operational
ones, returning RFC 9457 `application/problem+json` on error.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v1/classify`, `/v1/classify/batch` | `X-API-Key` | Analyse one text, or many in one call |
| `POST` | `/v1/summary/refresh` | `X-API-Key` | Fold new feedback/reviews into a caller-supplied customer summary and return the updated one; this service stores nothing |
| `GET` | `/v1/health`, `/v1/ready` | None | Liveness, tier readiness — open for a kubelet or load balancer |
| `GET` | `/v1/capabilities` | `X-API-Key` | Tier inventory, model versions, live scorer backend — deployment detail, not a probe |
| `GET` | `/metrics` | None (network-restrict in production) | Prometheus exposition |

`make openapi` writes the generated spec to `docs/openapi.json`.

## Design

- [`docs/00-plan.md`](docs/00-plan.md) — problem, goals, decisions, phased delivery
- [`docs/01-hld.md`](docs/01-hld.md) — architecture, the ladder, API surface, deployment
- [`docs/02-lld.md`](docs/02-lld.md) — implementation contract: layout, schemas, policy table,
  metric catalogue, error model, and the amendments log of what changed after each was written

## Development

```bash
make dev          # runtime extras plus pytest, ruff and mypy
make check        # lint + mypy --strict + the full suite
make test-unit    # no external service required
make image        # build the runtime image
make help         # every target
```

### Getting a Tier-1 model

No artifact is baked into the image by default — see the Dockerfile's `model` stage for why
(no public ONNX export of the default zero-shot checkpoint exists to fetch). Produce one
yourself:

```bash
pip install "torch>=2.2,<3" "optimum[exporters]>=1.20" onnx
python scripts/export_zeroshot_model.py --output ./models
```

That is enough for both a host-native run and `make up` (the compose file bind-mounts
`./models` into the container). For a self-contained Docker build instead, host the two output
files somewhere fetchable and set the Dockerfile's `MODEL_URL`/`MODEL_SHA256`/`TOKENIZER_URL`/
`TOKENIZER_SHA256` build args to match — the same mechanism `scripts/fetch_model.py` already
uses, just pointed at a different artifact. Without either, the service still boots and serves;
Tier 1 runs its lexicon fallback and `/v1/capabilities` says so.

Tests mirror the source tree under `tests/`, split into `unit/`, `integration/` (marked
`integration`, needs Redis and air-llm) and `contract/`.
