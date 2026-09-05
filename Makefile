VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn
COMPOSE ?= docker compose

IMAGE   ?= air-classifier-service
TAG     ?= dev
HOST    ?= 127.0.0.1
# 8082 in the AIR port map: 8080 is air-infra, 8081 is air-platform, 8083 is
# air-llm — every model call this service makes goes there.
PORT    ?= 8082

SERVE_LOG ?= .run/service.log

# The network air-infra publishes. Redis and Prometheus live there, not here.
# Model access (Tier 2 and Tier 3 both) is air-llm's, reached over the host —
# see docker-compose.yml — never through this network.
NETWORK ?= air-net
INFRA_DIR ?= ../air-infra
AIR_LLM_DIR ?= ../air-llm

# The listening server, and only it. `lsof -ti tcp:$(PORT)` on its own also
# matches every client *connected* to that port — a browser tab, air-client,
# a curl in flight — which `make stop` would then kill.
LISTENER = lsof -ti tcp:$(PORT) -sTCP:LISTEN 2>/dev/null

SOURCES := src tests scripts

.DEFAULT_GOAL := help
.PHONY: help install dev env run start stop restart status test test-unit test-integration \
        lint fmt typecheck check image up down logs clean openapi doctor infra-check

help: ## Show this help
	@echo "air-classifier-service — make <target>"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Overridable: IMAGE=$(IMAGE) TAG=$(TAG) HOST=$(HOST) PORT=$(PORT)"

$(PY):
	python3.12 -m venv $(VENV)

install: $(PY) ## Create .venv and install the service with every runtime extra
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

dev: install ## Add the development toolchain (pytest, ruff, mypy)
	$(PIP) install -e ".[all,dev]"

env: ## Create .env from .env.example if it does not exist yet
	@test -f .env && echo ".env exists — leaving it alone." \
		|| { cp .env.example .env; echo ".env created. Local API key: airc_local_dev_key"; }

run: env ## Serve in this terminal with autoreload; Ctrl+C stops it
	$(UVICORN) air_classifier_service.main:create_app --factory --reload --host $(HOST) --port $(PORT)

start: env ## Serve in the background, logging to .run/service.log
	@if [ -n "$$($(LISTENER))" ]; then \
		echo "already serving on :$(PORT) — 'make restart' to replace it"; exit 1; fi
	@mkdir -p $(dir $(SERVE_LOG))
	@nohup $(UVICORN) air_classifier_service.main:create_app --factory --reload \
		--host $(HOST) --port $(PORT) > $(SERVE_LOG) 2>&1 & \
		sleep 3; echo "serving http://$(HOST):$(PORT) · logs: $(SERVE_LOG)"

stop: ## Stop the service (whatever is serving on PORT)
	@pid=$$($(LISTENER) | xargs); \
	if [ -z "$$pid" ]; then echo "nothing serving on :$(PORT)"; else \
		kill $$pid 2>/dev/null; sleep 1; \
		if kill -0 $$pid 2>/dev/null; then kill -9 $$pid 2>/dev/null; fi; \
		echo "stopped :$(PORT) (pid $$pid)"; fi

restart: stop start ## Stop it and start again in the background

status: ## Say whether the service is serving, and where
	@pid=$$($(LISTENER) | xargs); \
	if [ -z "$$pid" ]; then echo "not serving on :$(PORT)"; else \
		echo "serving http://$(HOST):$(PORT) (pid $$pid)"; fi

test: ## Run the whole suite
	$(PYTEST)

test-unit: ## Run only tests that need no external service
	$(PYTEST) tests/unit -m "not integration and not live"

test-integration: ## Run the multi-component wiring suite (tests/integration/, no external services)
	$(PYTEST) tests/integration -m integration

test-eval: ## Run the golden accuracy/calibration set (tests/eval/, no external services) — put this in CI
	$(PYTEST) tests/eval -m "not live"

test-eval-live: ## Run the golden set's T2/T3 cases against a reachable air-llm (AIR_LLM_TOKEN, AIR_LLM_BASE_URL)
	$(PYTEST) tests/eval -m live

lint: ## Ruff lint + format check
	$(PY) -m ruff check $(SOURCES)
	$(PY) -m ruff format --check $(SOURCES)

fmt: ## Ruff format and autofix
	$(PY) -m ruff format $(SOURCES)
	$(PY) -m ruff check --fix $(SOURCES)

typecheck: ## mypy --strict
	$(PY) -m mypy

check: lint typecheck test ## Everything CI runs

openapi: ## Write the OpenAPI 3.1 spec to docs/openapi.json
	$(PY) -c "import json, pathlib; from air_classifier_service.main import create_app; \
		pathlib.Path('docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n')"
	@echo "wrote docs/openapi.json"

image: ## Build the runtime image
	docker build -t $(IMAGE):$(TAG) .

infra-check: ## Fail early if air-infra's shared stack is not up
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || { \
		echo "air-infra is not up: network '$(NETWORK)' does not exist."; \
		echo "Redis and Prometheus live there."; \
		echo; \
		echo "    cd $(INFRA_DIR) && make up"; \
		echo; \
		exit 1; }

up: infra-check ## Start this service as a container, against air-infra + air-llm
	$(COMPOSE) up -d --build
	@echo
	@echo "api      http://$(HOST):$(PORT)/v1/health"
	@echo "air-llm  http://host.docker.internal:8083 (cd $(AIR_LLM_DIR) && make up if it is not)"
	@echo "metrics  scraped by air-infra's prometheus (make up-observability there)"

down: ## Stop this service. Shared dependencies keep running in air-infra/air-llm.
	$(COMPOSE) down --remove-orphans

logs: ## Tail the app's logs
	$(COMPOSE) logs -f --tail=100 air-classifier-service

doctor: ## Check this service's dependencies are reachable
	@NETWORK=$(NETWORK) INFRA_DIR=$(INFRA_DIR) AIR_LLM_DIR=$(AIR_LLM_DIR) \
		HOST=$(HOST) PORT=$(PORT) bash scripts/doctor.sh

clean: ## Remove build and tool caches (leaves .venv alone)
	rm -rf build dist .coverage coverage.xml htmlcov
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
