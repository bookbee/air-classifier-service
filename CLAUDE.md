# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service is

Text classification for the AIR platform. One versioned REST API takes text plus optional
context (rating, channel, subject, etc.) and returns a structured, explainable verdict:
sentiment, tone, topics, intent, urgency, routing, a requires-human flag, aspect-level scoring
and rating consistency — whichever of those the text and the ladder's reach actually support.

The core design is a **four-tier escalation ladder** (cheapest first): every request enters at
the cheapest rung and climbs only when the rung below isn't confident enough, and the response
records exactly how far it climbed and why (`escalation_trace`).

| Tier | Technology | Resolves |
| --- | --- | --- |
| `t0_rules` | Lexicons, regex, cache | Cache hits, empty text, explicit ratings, emoji-only |
| `t1_classifier` | Quantised zero-shot NLI encoder + lexicon ensemble | Clear polarity, conventional phrasing |
| `t2_local_llm` | air-llm → a local model, schema-constrained | Mixed sentiment, mild sarcasm, tone/topics/intent |
| `t3_cloud_llm` | air-llm → cloud model + arbitration | Genuine ambiguity, subtle irony, multilingual edges |

`tone`, `topics`, `intent` are only populated when the ladder actually reaches an LLM tier (T2/T3).
Everything else (`sentiment`, `urgency`, `routing`, `requires_human`, `actionability`) is always
present and is a deterministic derivation, not a model output.

A tier failure never becomes a 5xx: the best lower-tier verdict is returned with `degraded: true`.
The only 503 is "no tier produced anything at all" (`NoVerdictError`).

Every model call (local and cloud) goes through **air-llm** (`../air-llm`), the platform's single
model gateway — this service holds no provider SDKs or API keys of its own. air-llm owns
credentials, cross-vendor routing/failover, cost accounting and caching. Redis and Prometheus are
shared platform components owned by **air-infra** (`../air-infra`). All three of these dependencies
are optional — a bare checkout with nothing configured still boots and serves on a two-rung ladder
(T0 + T1's lexicon fallback); `make doctor` reports what's reachable.

## Commands

`make help` lists every target with a one-line description.

Single test: `.venv/bin/pytest tests/unit/test_policy.py::test_name -q`.
Tests mirror `src/` under `tests/unit/`; `tests/integration/` and `tests/contract/` cover
cross-module wiring and the OpenAPI/response-shape contract respectively.

Local dev auth: `make env` seeds a dev API key (`airc_local_dev_key`, header `X-API-Key`) so a
fresh checkout authenticates the same way staging/prod do. `SECURITY__ALLOW_UNAUTHENTICATED`
always stays `false`.

## Architecture

### Request flow

`api/v1/classify.py` (HTTP) → `pipeline/service.py::AnalysisService` (the HTTP/ladder seam) →
`pipeline/orchestrator.py::Orchestrator` (climbs the ladder) → `pipeline/tiers/*` (the rungs) →
`providers/air_llm_provider.py` (T2/T3's HTTP client to air-llm).

- **`AnalysisService`** (`pipeline/service.py`) is the *only* place that normalises, redacts PII,
  detects language, and computes the cache key — tiers only ever see already-redacted text via
  `TierContext.text`, by construction. It also derives everything that isn't a model
  prediction: `urgency`, `actionability`, `requires_human`, `routing`, `rating_consistency` are
  all deterministic heuristics computed here from the winning verdict, in the `# Derivations`
  section at the bottom of the file — never a tier output.
- **`Orchestrator`** (`pipeline/orchestrator.py`) is tier-agnostic: it holds no reference to any
  concrete `Tier` member, just a `dict[Tier, Tier-instance]` and an `EscalationPolicy`. Context is
  threaded immutably (`TierContext.with_verdict`), and every tier failure (timeout, `TierError`,
  or an outright defect) is converted into a failed `TierVerdict` rather than an exception — the
  ladder always keeps going. Its lookahead (`_next_available_position`) scans the *rest* of the
  ladder, not just the next rung, so one dead rung doesn't mask a healthy one two hops up.
- **`EscalationPolicy`** (`pipeline/policy.py`) is where every "should we climb?" decision lives,
  and nowhere else — the orchestrator only asks it and obeys. Checks run in a fixed, documented
  order: hard ceilings first (tier ceiling, latency budget, cost budget — these override
  everything), then quality checks (confidence floor, margin floor, complexity triggers, tier
  disagreement, unsupported language), then "is there anywhere to climb to" last. That ordering is
  itself part of the contract: it's what makes the trace distinguish "confident, stopped on
  purpose" from "wanted to climb, nothing was there" (`NEXT_TIER_UNAVAILABLE` → `degraded: true`,
  a capacity gap rather than a real failure). `EscalationReason` (`constants.py`) is a closed set
  on purpose — it's also a Prometheus label.
- **Tiers** (`pipeline/tiers/`) each implement the `Tier` protocol (`tiers/base.py`): `analyse`,
  `is_available`, `model_version`. Adding a fifth tier is a registration, not a refactor.
- **`providers/air_llm_provider.py`** is the only code that talks to air-llm; `providers/prompts/`
  holds the system prompts T2/T3 send, loaded via `prompts/loader.py`, which decides — based on
  the request — which optional fields (tone/topics/intent) to even ask the model for.
- **`POST /v1/summary/refresh`** (`api/v1/summary.py`, `pipeline/summary_service.py`) is a second,
  stateless entry point: given a caller-supplied `existing_summary` (or none) plus new
  feedback/review items, it classifies each item through `AnalysisService.analyse_batch`,
  deterministically merges the rollup, and refreshes a narrative through its own guarded air-llm
  call (`AIR_LLM__SUMMARY_MODEL`, a dedicated bulkhead/breaker). This service persists nothing —
  the caller stores and resends `summary`. A narrative-refresh failure degrades
  (`degraded: true`, prior narrative unchanged); it never fails the request.
- **`audit/`** is an optional MongoDB-backed operational log of classification jobs (redacted
  text + verdict, TTL-expiring), unrelated to the summary feature above — wired into
  `AnalysisService.analyse()` as a fire-and-forget write behind a bounded queue. Controlled by
  `AuditSettings` (`config.py`); defaults to off (`backend: "none"`).

### Config

12-factor: every setting is an env var prefixed `AIR_CLASSIFIER__`, nested with `__`
(`AIR_CLASSIFIER__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.85`). Types live in `config.py`
(`Settings`, pydantic-settings). Escalation thresholds live in config *on purpose* — retuning the
ladder's cost/accuracy trade must never require a logic deploy. `.env.example` documents every key
with its default and marks secrets. Which model an alias name resolves to (`AIR_LLM__LOCAL_MODEL`,
`AIR_LLM__CLOUD_MODEL`, `AIR_LLM__ARBITER_MODEL`) is this service's decision; which vendor/model
answers behind that alias is air-llm's routing config (`../air-llm/deploy/routing.yaml`), not
something this service knows or should hardcode around.

### Constants and compatibility

`constants.py` holds every value that's part of the public surface: appears in API responses,
Prometheus labels, or config keys. `Label`, `Tone`, `Topic`, `Intent`, `Urgency`,
`RoutingPriority`, `Actionability`, `RatingAgreement` are the response vocabulary — adding a
member is safe, removing one or changing its meaning is not (bump `cache.base.CACHE_SCHEMA_VERSION`
if you do, since a cached verdict from before the change would misrepresent one from after it).

### Errors

Errors are RFC 9457 `application/problem+json` (`schemas/errors.py::ProblemDetail`), built from a
fixed catalogue (`docs/02-lld.md` §9). An exception can render its own problem document via a
duck-typed `to_problem(instance, request_id)` method (see `_self_rendered` in `pipeline/service.py`)
— this is how a caller-side policy refusal from the HTTP layer surfaces its own status code without
`pipeline/service.py` importing anything from `api/`.

### Design docs

- `docs/00-plan.md` — problem, goals, decisions, phased delivery
- `docs/01-hld.md` — architecture, the ladder, API surface, deployment
- `docs/02-lld.md` — implementation contract: layout, schemas, the policy table, metric catalogue,
  error model, and an amendments log of what changed after each was written — check this before
  assuming the original design doc is still accurate.

## Conventions specific to this repo

- Ruff + mypy `--strict` are both enforced in `make check`; `S` (bandit) and `T20` (no `print`) are
  enabled — use `structlog` (`observability/logging.py`) instead of `print`.
- Docstrings in this codebase explain *why*, not what — module docstrings in particular carry real
  design rationale (see `orchestrator.py`, `service.py`, `policy.py`, `constants.py`). Read them
  before changing behavior in that file; match that density when adding non-obvious logic, but
  don't restate what the code already makes obvious.
- No per-vendor LLM SDKs belong in this repo (see `pyproject.toml`'s comment on this) — any new
  model-calling code goes through `providers/air_llm_provider.py`, never a direct vendor client.
