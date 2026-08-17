#!/usr/bin/env bash
#
# Check that everything this service depends on is reachable, and that the
# service itself is serving. Works whether air-classifier is running natively
# (`make run` / `make start`) or as a container (`make up`).
#
# Shared dependencies belong to air-infra; this script only verifies they are
# there and answering, and says where to go when they are not.
#
# Run via `make doctor`.
set -uo pipefail

OLLAMA="${OLLAMA:-container}"
OLLAMA_URL="${OLLAMA_URL:-http://ollama:11434}"
NETWORK="${NETWORK:-air-net}"
INFRA_DIR="${INFRA_DIR:-../air-infra}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"
MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

fail=0
warn=0

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  \033[33m!\033[0m %s\n' "$1"; warn=$((warn + 1)); }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

hdr "air-infra (shared dependencies)"
if ! docker info >/dev/null 2>&1; then
  bad "docker daemon unreachable — start Docker Desktop"
elif docker network inspect "$NETWORK" >/dev/null 2>&1; then
  ok "network '$NETWORK' is up"
else
  bad "network '$NETWORK' is missing — run 'cd $INFRA_DIR && make up'"
fi

# Addressed on the host, which is how the native inner loop reaches them and how
# a developer verifies them regardless of where this service is running.
if docker exec air-infra-redis-1 redis-cli -a localdevpass ping 2>/dev/null | grep -q PONG; then
  ok "redis answers an authenticated PING"
else
  bad "air-infra's redis is not answering — 'cd $INFRA_DIR && make ps'"
fi

hdr "Tier 2 — ollama"
# Report what the service is ACTUALLY configured with, not what this make
# invocation happens to default to — a container started with `OLLAMA=native
# make up` keeps its URL regardless of how doctor is later called.
container=$(docker ps --filter 'name=air-classifier-air-classifier' --format '{{.Names}}' 2>/dev/null | head -1)
if [ -n "$container" ]; then
  effective=$(docker exec "$container" printenv AIR_CLASSIFIER__PROVIDERS__OLLAMA__BASE_URL 2>/dev/null)
  num_ctx=$(docker exec "$container" python -c \
    'from air_classifier.config import Settings; print(Settings().providers.ollama.num_ctx)' 2>/dev/null)
  ok "container $container dials ${effective:-<unset>}"
else
  effective="(native run — from .env)"
  num_ctx=$(grep -E '^AIR_CLASSIFIER__PROVIDERS__OLLAMA__NUM_CTX=' .env 2>/dev/null | cut -d= -f2)
  ok "no container running; the native loop reads .env (mode flag: $OLLAMA → $OLLAMA_URL)"
fi
num_ctx="${num_ctx:-8192}"

if curl -sf -m 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ok "reachable on 127.0.0.1:11434"
  if curl -s -m 3 http://127.0.0.1:11434/api/tags | grep -q "\"${MODEL}\""; then
    ok "model '$MODEL' is present"
  else
    bad "model '$MODEL' is NOT pulled — Tier 2 reports unavailable and the ladder stops at t1"
    note "  fix: cd $INFRA_DIR && OLLAMA=$OLLAMA make ollama-pull"
  fi

  # Ollama evicts and reloads a model when a request asks for a different
  # num_ctx than the resident copy. That reload routinely outlasts Tier 2's
  # timeout, so the first real request degrades even though nothing is broken —
  # which reads as a bug and is not one.
  resident=$(curl -s -m 3 http://127.0.0.1:11434/api/ps 2>/dev/null)
  if printf '%s' "$resident" | grep -q "\"${MODEL}\""; then
    loaded_ctx=$(printf '%s' "$resident" | tr ',' '\n' | grep -m1 context_length | grep -oE '[0-9]+')
    if [ "${loaded_ctx:-0}" = "$num_ctx" ]; then
      ok "resident at num_ctx=$num_ctx — first request will not trigger a reload"
    else
      note "resident at num_ctx=${loaded_ctx:-?}, this service asks for $num_ctx"
      note "  the next request forces a reload and will likely blow Tier 2's timeout once"
      note "  fix: cd $INFRA_DIR && make ollama-warm OLLAMA_NUM_CTX=$num_ctx"
    fi
  else
    note "model not resident — the first request pays the load and may exceed Tier 2's timeout"
    note "  fix: cd $INFRA_DIR && make ollama-warm OLLAMA_NUM_CTX=$num_ctx"
  fi
else
  note "nothing on 127.0.0.1:11434 — the ladder degrades to t1_classifier, which is not an error"
fi

hdr "This service"
base="http://${HOST}:${PORT}"
if curl -sf -m 3 "${base}/v1/health" >/dev/null 2>&1; then
  ok "healthy on ${base}"

  ready=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${base}/v1/ready" 2>/dev/null)
  if [ "$ready" = "200" ]; then
    ok "ready (tiers report usable)"
  else
    note "/v1/ready returned $ready — a tier is degraded; /v1/capabilities says which"
  fi

  caps=$(curl -s -m 5 -H 'X-API-Key: airc_local_dev_key' "${base}/v1/capabilities" 2>/dev/null)
  if [ -n "$caps" ]; then
    printf '      capabilities: %s\n' "$(printf '%s' "$caps" | head -c 300)"
  else
    note "/v1/capabilities returned nothing — check the API key in .env"
  fi
else
  note "not serving on ${base} — 'make run', 'make start' or 'make up'"
fi

hdr "Port hygiene"
# This repo must publish exactly one host port. Anything else here means a
# shared component has crept back in and is re-binding a port air-infra owns.
mine=$(docker ps --filter 'label=com.docker.compose.project=air-classifier' \
  --format '{{.Names}} {{.Ports}}' 2>/dev/null)
if [ -z "$mine" ]; then
  ok "no air-classifier containers running (native mode)"
else
  shared=$(printf '%s\n' "$mine" | grep -Eo '0\.0\.0\.0:(6379|11434|9090|3000)' | sort -u)
  if [ -z "$shared" ]; then
    ok "publishes only :$PORT — no shared ports re-bound"
  else
    bad "re-binding port(s) air-infra owns: $(printf '%s ' $shared)"
  fi
fi

# A gateway URL is only meaningful when gateway mode is on.
if grep -qE '^AIR_CLASSIFIER__PROVIDERS__GATEWAY__ENABLED=true' .env 2>/dev/null; then
  hdr "air-infra gateway (mode: on)"
  if curl -sf -m 3 http://127.0.0.1:8080/v1/health >/dev/null 2>&1; then
    ok "gateway healthy on :8080 — T2 and T3 route through it"
  else
    bad "gateway mode is ON but :8080 is not healthy — model calls will fail over to nothing"
  fi
fi

hdr "Summary"
if [ "$fail" -eq 0 ] && [ "$warn" -eq 0 ]; then
  printf '  \033[32mall checks passed\033[0m\n\n'
elif [ "$fail" -eq 0 ]; then
  printf '  \033[33m%d warning(s), no failures\033[0m\n\n' "$warn"
else
  printf '  \033[31m%d failure(s)\033[0m, %d warning(s)\n\n' "$fail" "$warn"
fi
exit $((fail > 0 ? 1 : 0))
