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
# air-model-gateway — every model call this service makes goes there.
PORT    ?= 8082

SERVE_LOG ?= .run/service.log

# The network air-infra publishes. Redis and Prometheus live there, not here.
# Model access (Tier 2 and Tier 3 both) is air-model-gateway's, reached over the
# host — see docker-compose.yml — never through this network.
NETWORK ?= air-net
INFRA_DIR ?= ../air-infrastructure-service

# Container runs only: `shared` (default) joins air-infra's air-net and its
# Redis, and refuses to start without them; `none` runs this service on its own
# network with an in-process cache (docker-compose.standalone.yml). The native
# loop (`make run`) never touches either network.
INFRA ?= shared
STANDALONE_NETWORK := air-classifier-service-net
ifeq ($(INFRA),shared)
COMPOSE_FILES := -f docker-compose.yml
UP_DEPS := infra-check
else ifeq ($(INFRA),none)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.standalone.yml
UP_DEPS :=
else
$(error INFRA must be `shared` or `none`, got `$(INFRA)`)
endif
MODEL_GATEWAY_DIR ?= ../air-model-gateway

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
	@echo "Overridable: IMAGE=$(IMAGE) TAG=$(TAG) HOST=$(HOST) PORT=$(PORT) INFRA=$(INFRA) (shared|none)"

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

test-eval-live: ## Run the golden set's T2/T3 cases against a reachable air-model-gateway (MODEL_GATEWAY_TOKEN, MODEL_GATEWAY_BASE_URL)
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

image: ## Build the runtime image locally as $(IMAGE):$(TAG) (registry: make image-push)
	docker build -t $(IMAGE):$(TAG) .

infra-check: ## Fail early if air-infra's shared stack is not up
	@docker network inspect $(NETWORK) >/dev/null 2>&1 || { \
		echo "air-infra is not up: network '$(NETWORK)' does not exist."; \
		echo "Redis and Prometheus live there."; \
		echo; \
		echo "    cd $(INFRA_DIR) && make up"; \
		echo; \
		echo "Or run this service without them (own network, in-memory cache):"; \
		echo; \
		echo "    make up INFRA=none"; \
		echo; \
		exit 1; }

up: $(UP_DEPS) ## Start this service as a container (INFRA=none: without air-infra's network and Redis)
	$(COMPOSE) $(COMPOSE_FILES) up -d --build
	@echo
	@echo "api      http://$(HOST):$(PORT)/v1/health"
	@echo "models   http://host.docker.internal:8083, air-model-gateway (cd $(MODEL_GATEWAY_DIR) && make up if it is not)"
ifeq ($(INFRA),shared)
	@echo "metrics  scraped by air-infra's prometheus (make up-observability there)"
else
	@echo "cache    in-process (INFRA=none: no air-net, no shared Redis)"
endif

down: ## Stop this service. Shared dependencies keep running in air-infra/air-model-gateway.
	$(COMPOSE) $(COMPOSE_FILES) down --remove-orphans
	@# A standalone run stopped without INFRA=none leaves its own network behind,
	@# because the base file only knows the external air-net. Remove it if unused.
	@docker network rm $(STANDALONE_NETWORK) >/dev/null 2>&1 || true

logs: ## Tail the app's logs
	$(COMPOSE) $(COMPOSE_FILES) logs -f --tail=100 air-classifier-service

doctor: ## Check this service's dependencies are reachable
	@NETWORK=$(NETWORK) INFRA_DIR=$(INFRA_DIR) MODEL_GATEWAY_DIR=$(MODEL_GATEWAY_DIR) \
		INFRA=$(INFRA) STANDALONE_NETWORK=$(STANDALONE_NETWORK) \
		HOST=$(HOST) PORT=$(PORT) bash scripts/doctor.sh

# ---- image and deployment (UAT and production) ------------------------------
# This repository owns its Dockerfile and image. The Helm chart, the Terraform
# and the deployment scripts for every core service live in
# air-infrastructure-service (gke/), where DevOps keeps them. The targets below
# build and push the image here, then hand over to that repository, checked out
# next to this one (INFRA_DIR). Its docs/03-gke-runbook.md is the runbook.
#
#   DEPLOY_ENV        uat | prod
#   IMAGE_REPOSITORY  <region>-docker.pkg.dev/<project>/air/air-classifier-service
#   IMAGE_TAG         defaults to this checkout's short commit
#   IMAGE_DIGEST      what CI should deploy: a digest can't be re-pointed
#   KUBE_CONTEXT      required by deploy / deploy-smoke; never defaulted, so a
#                     deploy can't land on whatever cluster happens to be current
#   GCP_PROJECT       the environment's project (Secret Manager)
#   MODEL_URL, MODEL_SHA256, TOKENIZER_URL, TOKENIZER_SHA256
#                     image-push only, optional: bake the Tier 1 model into the
#                     image (see the Dockerfile). Without them Tier 1 runs on
#                     its weaker fallback unless the files are mounted
#   KEYS_FILE         deploy-smoke only, optional: an env file of raw API keys,
#                     to add one authenticated call to the readiness checks
DEPLOY_ENV ?= uat
SERVICE    := air-classifier-service
MODEL_BUILD_ARGS = $(foreach v,MODEL_URL MODEL_SHA256 TOKENIZER_URL TOKENIZER_SHA256,$(if $($(v)),--build-arg $(v)=$($(v))))
IMAGE_TAG  ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo dev)
INFRA_MAKE  = $(MAKE) --no-print-directory -C $(INFRA_DIR) SERVICE=$(SERVICE) DEPLOY_ENV=$(DEPLOY_ENV) \
	KUBE_CONTEXT="$(KUBE_CONTEXT)" GCP_PROJECT="$(GCP_PROJECT)" KEYS_FILE="$(KEYS_FILE)" \
	IMAGE_REPOSITORY="$(IMAGE_REPOSITORY)" IMAGE_DIGEST="$(IMAGE_DIGEST)" \
	IMAGE_TAG="$(if $(IMAGE_DIGEST),,$(IMAGE_TAG))"

.PHONY: require-infra-repo
require-infra-repo:
	@test -f $(INFRA_DIR)/gke/scripts/release.sh || { \
		echo "air-infrastructure-service is not at $(INFRA_DIR): the charts and deploy scripts live there."; \
		echo "Check it out next to this repository, or pass INFRA_DIR=<path>."; exit 1; }

.PHONY: require-image-repository
require-image-repository:
	@test -n "$(IMAGE_REPOSITORY)" || { echo "set IMAGE_REPOSITORY, e.g. <region>-docker.pkg.dev/<project>/air/$(SERVICE)"; exit 1; }

.PHONY: image-push
image-push: require-image-repository ## Build for linux/amd64 and push IMAGE_REPOSITORY:IMAGE_TAG (gcloud auth configure-docker first)
	docker build --platform linux/amd64 $(MODEL_BUILD_ARGS) -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
	docker push $(IMAGE_REPOSITORY):$(IMAGE_TAG)

.PHONY: deploy-render
deploy-render: require-infra-repo ## Print the manifests DEPLOY_ENV would get (SAMPLE=1 fills the project placeholders)
	@$(INFRA_MAKE) SAMPLE="$(SAMPLE)" gke-render

.PHONY: deploy
deploy: require-infra-repo ## Deploy this service to DEPLOY_ENV (uat|prod) in KUBE_CONTEXT: helm upgrade --install
	@$(INFRA_MAKE) gke-deploy

.PHONY: deploy-smoke
deploy-smoke: require-infra-repo ## Smoke-test this service in KUBE_CONTEXT: rollout, health, readiness (KEYS_FILE adds a real call)
	@$(INFRA_MAKE) gke-smoke

clean: ## Remove build and tool caches (leaves .venv alone)
	rm -rf build dist .coverage coverage.xml htmlcov
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
