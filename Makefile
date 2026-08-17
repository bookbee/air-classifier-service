VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn
COMPOSE ?= docker compose

IMAGE   ?= air-classifier
TAG     ?= dev
HOST    ?= 127.0.0.1
# 8082 in the AIR port map: 8080 is air-infra's gateway, 8081 is air-platform.
PORT    ?= 8082

SERVE_LOG ?= .run/service.log

# The network air-infra publishes. Shared dependencies (redis, ollama,
# prometheus, grafana) live there, not here.
NETWORK ?= air-net
INFRA_DIR ?= ../air-infra

# ---- ollama backend (compose only) -----------------------------------------
# container | native. Mirrors air-infra's switch of the same name and must agree
# with it: `native` means the laptop's own `ollama serve`, which on macOS is the
# only one with GPU access — Docker Desktop has no Metal passthrough.
#
# Only affects `make up`. The native inner loop (`make run` / `make start`)
# reads PROVIDERS__OLLAMA__BASE_URL from .env, where localhost:11434 addresses
# whichever instance is serving.
OLLAMA ?= container

ifeq ($(OLLAMA),container)
OLLAMA_URL := http://ollama:11434
else ifeq ($(OLLAMA),native)
OLLAMA_URL := http://host.docker.internal:11434
else
$(error OLLAMA must be `container` or `native`, got `$(OLLAMA)`)
endif

export AIR_CLASSIFIER_OLLAMA_URL := $(OLLAMA_URL)

# The listening server, and only it. `lsof -ti tcp:$(PORT)` on its own also
# matches every client *connected* to that port — a browser tab, air-client,
# a curl in flight — which `make stop` would then kill.
LISTENER = lsof -ti tcp:$(PORT) -sTCP:LISTEN 2>/dev/null

SOURCES := src tests scripts

.DEFAULT_GOAL := help
.PHONY: help install dev env run start stop restart status test test-unit test-integration \
        lint fmt typecheck check image up down logs clean openapi doctor infra-check

help: ## Show this help
	@echo "air-classifier — make <target>"
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
	$(UVICORN) air_classifier.main:create_app --factory --reload --host $(HOST) --port $(PORT)

start: env ## Serve in the background, logging to .run/service.log
	@if [ -n "$$($(LISTENER))" ]; then \
		echo "already serving on :$(PORT) — 'make restart' to replace it"; exit 1; fi
	@mkdir -p $(dir $(SERVE_LOG))
	@nohup $(UVICORN) air_classifier.main:create_app --factory --reload \
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

test-integration: ## Run tests that need redis/ollama (see `make up`)
	$(PYTEST) tests/integration -m integration

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
	$(PY) -c "import json, pathlib; from air_classifier.main import create_app; \
		pathlib.Path('docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n')"
	@echo "wrote docs/openapi.json"

image: ## Build the runtime image
	docker build -t $(IMAGE):$(TAG) .

infra-check: ## Fail early if air-infra's shared stack is not up
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || { \
		echo "air-infra is not up: network '$(NETWORK)' does not exist."; \
		echo "Shared dependencies (redis, ollama, prometheus, grafana) live there."; \
		echo; \
		echo "    cd $(INFRA_DIR) && make up"; \
		echo; \
		exit 1; }

up: infra-check ## Start this service as a container, against air-infra's stack
	$(COMPOSE) up -d --build
	@echo
	@echo "api     http://$(HOST):$(PORT)/v1/health"
	@echo "ollama  $(OLLAMA) at $(OLLAMA_URL)"
	@echo "metrics scraped by air-infra's prometheus (make up-observability there)"

down: ## Stop this service. Shared dependencies keep running in air-infra.
	$(COMPOSE) down --remove-orphans

logs: ## Tail the app's logs
	$(COMPOSE) logs -f --tail=100 air-classifier

doctor: ## Check this service's dependencies are reachable, in whichever mode
	@OLLAMA=$(OLLAMA) OLLAMA_URL=$(OLLAMA_URL) NETWORK=$(NETWORK) \
		INFRA_DIR=$(INFRA_DIR) HOST=$(HOST) PORT=$(PORT) bash scripts/doctor.sh

clean: ## Remove build and tool caches (leaves .venv alone)
	rm -rf build dist .coverage coverage.xml htmlcov
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
