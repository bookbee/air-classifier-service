#!/usr/bin/env bash
#
# Check that everything this service depends on is reachable, and that the
# service itself is serving. Works whether air-classifier-service is running natively
# (`make run` / `make start`) or as a container (`make up`).
#
# Redis is air-infra's; every model call (Tier 2 and Tier 3 both) is air-llm's.
# This script only verifies they are there and answering, and says where to go
# when they are not.
#
# Run via `make doctor`.
set -uo pipefail

NETWORK="${NETWORK:-air-net}"
INFRA_DIR="${INFRA_DIR:-../air-infra}"
AIR_LLM_DIR="${AIR_LLM_DIR:-../air-llm}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"

fail=0
warn=0

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  \033[33m!\033[0m %s\n' "$1"; warn=$((warn + 1)); }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

hdr "air-infra (redis)"
if ! docker info >/dev/null 2>&1; then
  bad "docker daemon unreachable — start Docker Desktop"
elif docker network inspect "$NETWORK" >/dev/null 2>&1; then
  ok "network '$NETWORK' is up"
else
  bad "network '$NETWORK' is missing — run 'cd $INFRA_DIR && make up'"
fi

# Addressed on the host, which is how the native inner loop reaches it and how
# a developer verifies it regardless of where this service is running.
if docker exec air-infra-redis-1 redis-cli -a localdevpass ping 2>/dev/null | grep -q PONG; then
  ok "redis answers an authenticated PING"
else
  bad "air-infra's redis is not answering — 'cd $INFRA_DIR && make ps'"
fi

hdr "air-llm (tiers 2 and 3)"
# Report what the service is ACTUALLY configured with, not a guess — a
# container started against one air-llm keeps that URL regardless of what is
# reachable when doctor is later called.
container=$(docker ps --filter 'name=air-classifier-service-air-classifier-service' --format '{{.Names}}' 2>/dev/null | head -1)
if [ -n "$container" ]; then
  effective=$(docker exec "$container" printenv AIR_CLASSIFIER__AIR_LLM__BASE_URL 2>/dev/null)
  ok "container $container dials ${effective:-<unset, defaults to http://localhost:8083>}"
  probe_url="${effective:-http://127.0.0.1:8083}"
else
  effective="(native run — from .env)"
  ok "no container running; the native loop reads .env"
  probe_url="http://127.0.0.1:8083"
fi
# host.docker.internal only resolves from inside a container; probe the host
# loopback address here regardless of what the container itself is configured with.
probe_url=$(printf '%s' "$probe_url" | sed 's/host\.docker\.internal/127.0.0.1/')

if curl -sf -m 3 "${probe_url}/v1/ready" >/dev/null 2>&1; then
  ok "air-llm answers on ${probe_url}"
else
  note "air-llm is not answering on ${probe_url} — ladder degrades to t1_classifier, which is not an error"
  note "  fix: cd $AIR_LLM_DIR && make up"
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
# shared component has crept back in and is re-binding a port air-infra or
# air-llm owns.
mine=$(docker ps --filter 'label=com.docker.compose.project=air-classifier-service' \
  --format '{{.Names}} {{.Ports}}' 2>/dev/null)
if [ -z "$mine" ]; then
  ok "no air-classifier-service containers running (native mode)"
else
  shared=$(printf '%s\n' "$mine" | grep -Eo '0\.0\.0\.0:(6379|11434|8083|9090|3000)' | sort -u)
  if [ -z "$shared" ]; then
    ok "publishes only :$PORT — no shared ports re-bound"
  else
    bad "re-binding port(s) air-infra/air-llm owns: $(printf '%s ' $shared)"
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
