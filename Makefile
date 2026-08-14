VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn
COMPOSE ?= docker compose

IMAGE   ?= air-sentiment
TAG     ?= dev
HOST    ?= 127.0.0.1
PORT    ?= 8080

SOURCES := src tests scripts

.DEFAULT_GOAL := help
.PHONY: help install dev run test test-unit test-integration lint fmt typecheck check \
        image up down logs clean openapi

help: ## Show this help
	@echo "air-sentiment — make <target>"
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

run: ## Serve the API locally with autoreload
	$(UVICORN) air_sentiment.main:create_app --factory --reload --host $(HOST) --port $(PORT)

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
	$(PY) -c "import json, pathlib; from air_sentiment.main import create_app; \
		pathlib.Path('docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n')"
	@echo "wrote docs/openapi.json"

image: ## Build the runtime image
	docker build -t $(IMAGE):$(TAG) .

up: ## Start the local stack (app + ollama + redis + prometheus + grafana)
	$(COMPOSE) up -d --build
	@echo "api http://$(HOST):$(PORT)/v1/health · prometheus :9090 · grafana :3000"

down: ## Stop the local stack, keeping the ollama model volume
	$(COMPOSE) down --remove-orphans

logs: ## Tail the app's logs
	$(COMPOSE) logs -f --tail=100 air-sentiment

clean: ## Remove build and tool caches (leaves .venv alone)
	rm -rf build dist .coverage coverage.xml htmlcov
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
