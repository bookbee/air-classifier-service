#!/usr/bin/env bash
#
# Exercise each rung of the ladder in isolation.
#
# `min_tier` removes cheaper rungs entirely and `max_tier` removes dearer ones
# (pipeline/policy.py::enabled_tiers_for), so setting them to the same value
# runs exactly one tier — the verdict cannot have come from anywhere else.
# A pinned tier that cannot answer returns 503 rather than falling back, which
# is what makes this a test rather than a demo.
#
#   bash scripts/demo-tiers.sh            # every tier
#   bash scripts/demo-tiers.sh t2         # one section
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8082}"
KEY="${KEY:-airc_local_dev_key}"
only="${1:-all}"

# air-llm's own address, not this service's — T2 goes through it now, and
# whichever local runtime currently answers the `local` class has its own
# eviction window between calls (minutes, not hours) after which the *next*
# T2 call pays a full cold model load on top of real generation time, which
# routinely exceeds TIERS__T2_LOCAL_LLM__TIMEOUT_MS (6s default) — that
# failure is a stale model cache, not a broken tier. Warmed once here, every
# T2 case below stays warm for the rest of this run.
AIR_LLM_BASE="${AIR_LLM_BASE:-http://127.0.0.1:8083}"
AIR_LLM_TOKEN="${AIR_LLM_TOKEN:-air-classifier-dev}"
AIR_LLM_LOCAL_MODEL="${AIR_LLM_LOCAL_MODEL:-local}"

warm_local_llm() {
  # cache:false: the gateway caches identical requests per service, and a
  # cached "hi" would return instantly without ever loading the model.
  printf '\n\033[2mwarming %s via air-llm…\033[0m' "$AIR_LLM_LOCAL_MODEL"
  started=$(date +%s%N)
  curl -s -m 60 -X POST "$AIR_LLM_BASE/v1/generate" \
    -H "X-API-Key: $AIR_LLM_TOKEN" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$AIR_LLM_LOCAL_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"cache\":false}" \
    -o /dev/null -w '%{http_code}' | { read -r code
      elapsed_ms=$(( ($(date +%s%N) - started) / 1000000 ))
      if [ "$code" = "200" ]; then
        printf ' done (%sms)\n' "$elapsed_ms"
      else
        printf ' HTTP %s — air-llm unreachable or the model is not pulled; T2 cases below will 503\n' "$code"
      fi
    }
}

run() {  # run <label> <route> <json>
  printf '\n\033[1m%s\033[0m\n  %s\n' "$1" "$3"
  resp=$(curl -s -m 90 -w '\n%{http_code}' "$BASE/v1/$2" \
           -H "X-API-Key: $KEY" -H 'Content-Type: application/json' -d "$3")
  printf '%s' "$(printf '%s' "$resp" | sed '$d')" | python3 -c "
import sys, json
code = '$(printf '%s' "$resp" | tail -1)'
try: d = json.load(sys.stdin)
except Exception: print(f'  -> HTTP {code}: no JSON body'); raise SystemExit
if code != '200':
    print(f\"  -> HTTP {code}  {d.get('title','')} :: {str(d.get('detail',''))[:100]}\"); raise SystemExit
s = d['sentiment']
print(f\"  -> {d['decided_by']:<15} label={s['label']:<9} conf={s['confidence']:.2f} degraded={d['degraded']}\")
for t in d.get('escalation_trace') or []:
    print(f\"       {t['tier']:<15} ok={str(t['succeeded']):<5} {t['latency_ms']:>8.1f}ms  {t['escalated_because'] or '-'}\")
"
}
want() { [ "$only" = all ] || [ "$only" = "$1" ]; }

# Once, regardless of which sections below actually touch T2 — nat and climb
# both do, same as t2 itself.
if want t2 || want nat || want climb; then
  warm_local_llm
fi

# ── T0 · rules ───────────────────────────────────────────────────────────────
# One case per shortcut rule. Each must clear tiers.t0_rules.confidence_floor
# (0.95) or the policy discards it, so these confidences are flat by design.
if want t0; then
  echo "════ T0 · rules only ════"
  run "T0-a contentless (0.99)"   classify '{"text":"   ","options":{"min_tier":"t0_rules","max_tier":"t0_rules"}}'
  run "T0-b star rating (0.97)"   classify '{"text":"Arrived on time.","rating":5,"options":{"min_tier":"t0_rules","max_tier":"t0_rules"}}'
  run "T0-c emoji only (0.96)"    classify '{"text":"😍😍","options":{"min_tier":"t0_rules","max_tier":"t0_rules"}}'
  run "T0-d lexicon term (0.96)"  classify '{"text":"absolutely terrible","options":{"min_tier":"t0_rules","max_tier":"t0_rules"}}'
  # Negative control: no rule matches, and a pinned T0 has nowhere to go. The
  # 503 is the correct answer, not a failure.
  run "T0-e no rule matches"      classify '{"text":"The delivery was scheduled for Tuesday according to the tracking page.","options":{"min_tier":"t0_rules","max_tier":"t0_rules"}}'
fi

# ── T1 · classifier ──────────────────────────────────────────────────────────
# Conventional phrasing with one clear polarity: no T0 shortcut fits (too long
# for the lexicon rule), and no ambiguity that would need an LLM.
if want t1; then
  echo; echo "════ T1 · classifier only ════"
  run "T1-a clear negative"  classify '{"text":"The build quality feels cheap and the app crashes constantly.","options":{"min_tier":"t1_classifier","max_tier":"t1_classifier"}}'
  run "T1-b clear positive"  classify '{"text":"Genuinely delighted with this purchase, it works beautifully.","options":{"min_tier":"t1_classifier","max_tier":"t1_classifier"}}'
  run "T1-c multilingual"    classify '{"text":"Der Akku ist gut, aber die Kamera enttäuscht.","options":{"min_tier":"t1_classifier","max_tier":"t1_classifier"}}'
fi

# ── T2 · local LLM ───────────────────────────────────────────────────────────
# What a bag of terms cannot see: sarcasm, and praise and complaint in one
# sentence that must come back as `mixed` rather than averaged to neutral.
if want t2; then
  echo; echo "════ T2 · local LLM only ════"
  run "T2-a sarcasm"      classify '{"text":"Battery lasts forever, if by forever you mean until lunchtime.","options":{"min_tier":"t2_local_llm","max_tier":"t2_local_llm"}}'
  run "T2-b mixed"        classify '{"text":"The screen is gorgeous but the battery dies by lunchtime.","options":{"min_tier":"t2_local_llm","max_tier":"t2_local_llm"}}'
  run "T2-c aspects"      classify '{"text":"Fast shipping, sturdy build, but the instructions were useless.","rating":3,"options":{"min_tier":"t2_local_llm","max_tier":"t2_local_llm"}}'
fi

# ── T3 · cloud LLM ───────────────────────────────────────────────────────────
# Needs air-llm reachable with a cloud key configured on its side. Without one
# the tier is skipped at startup and a pinned request 503s — see the README on
# configuring AIR_LLM__BASE_URL / AIR_LLM__TOKEN.
if want t3; then
  echo; echo "════ T3 · cloud LLM only ════"
  run "T3-a subtle irony"   classify '{"text":"Well, at least the box it arrived in was sturdy.","options":{"min_tier":"t3_cloud_llm","max_tier":"t3_cloud_llm"}}'
  run "T3-b damning praise" classify '{"text":"Support replied within minutes, every single time, to tell me they could not help.","options":{"min_tier":"t3_cloud_llm","max_tier":"t3_cloud_llm"}}'
fi

# ── The ladder itself ────────────────────────────────────────────────────────
if want nat; then
  echo; echo "════ natural escalation (no pinning) ════"
  run "NAT trivial  → stops at T0" classify '{"text":"absolutely terrible"}'
  run "NAT clear    → stops at T1" classify '{"text":"Genuinely delighted with this purchase, it works beautifully."}'
  run "NAT sarcasm  → climbs to T2" classify '{"text":"Battery lasts forever, if by forever you mean until lunchtime."}'
  # Same text twice. The second is a T0 cache hit, but `decided_by` and the
  # trace are REPLAYED from the stored verdict — provenance is preserved, so the
  # hit is invisible in the body and shows only in wall-clock time (~1.6s -> ~2ms).
  # Time it: `curl -o /dev/null -w '%{time_total}\n' ...` twice with the same text.
  run "NAT cache hit (repeat)"      classify '{"text":"Battery lasts forever, if by forever you mean until lunchtime."}'
fi

# ── The whole ladder in one request ──────────────────────────────────────────
# No text reliably lands below every rung's confidence floor, so a "watch it
# climb all four" demo cannot be built from wording alone — it needs the floors
# raised. This runs a throwaway instance with T1 and T2 set to 0.99 so each rung
# answers and then escalates anyway, ending at T3.
#
# The raised latency budget matters too: the orchestrator gives each rung
# min(its timeout, what is left of the request budget), and the 8s default is
# spent before T3 on a cold local model.
if want climb; then
  echo; echo "════ full climb · T0 -> T1 -> T2 -> T3 ════"
  port=8098
  # Cache off: a repeat run would otherwise replay the stored verdict with the
  # original trace, showing four rungs that did not actually execute.
  AIR_CLASSIFIER__TIERS__T1_CLASSIFIER__CONFIDENCE_FLOOR=0.99 \
  AIR_CLASSIFIER__TIERS__T2_LOCAL_LLM__CONFIDENCE_FLOOR=0.99 \
  AIR_CLASSIFIER__CACHE__BACKEND=none \
    .venv/bin/uvicorn air_classifier_service.main:create_app --factory \
      --host 127.0.0.1 --port "$port" >/tmp/air-climb.log 2>&1 &
  climb_pid=$!
  trap 'kill $climb_pid 2>/dev/null' EXIT
  for _ in $(seq 20); do
    curl -sf -m 1 "http://127.0.0.1:$port/v1/health" >/dev/null 2>&1 && break
    sleep 1
  done
  BASE="http://127.0.0.1:$port" run "every rung answers, then escalates" classify \
    '{"text":"The packaging was immaculate, which made the dead pixels easier to photograph.","options":{"latency_budget_ms":30000}}'
  kill $climb_pid 2>/dev/null
  trap - EXIT
fi
