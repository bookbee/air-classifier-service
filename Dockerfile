# syntax=docker/dockerfile:1.7
#
# Multi-stage build. Three stages so the two slow, independent jobs — resolving
# Python wheels and pulling the model artifact — neither block nor invalidate
# each other.

ARG PYTHON_IMAGE=python:3.12-slim
ARG VENV=/opt/venv
ARG MODEL_DIR=/opt/models


# ── Builder ───────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

ARG VENV
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv "${VENV}"
ENV PATH="${VENV}/bin:${PATH}"

WORKDIR /build

# Dependencies resolve from pyproject alone. A stub package satisfies hatchling
# so this layer — the expensive one — survives every source edit.
COPY pyproject.toml README.md ./
RUN mkdir -p src/air_classifier_service && touch src/air_classifier_service/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install ".[all]"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --force-reinstall .


# ── Model bake-in (currently inert; see below) ─────────────────────────────────
# This stage exists to bake the Tier-1 artifact into the image — reproducible
# builds and zero cold-start network beat image size — but ships empty by
# default: MODEL_URL/TOKENIZER_URL below have no value to default to yet.
#
# Tier 1's default backend (docs/02-lld.md §6) is a quantised DeBERTa-v3
# zero-shot NLI head, not the SST-2 sequence-classification head this service
# shipped with first — `pipeline/tiers/scorers.py`'s module docstring explains
# why. Unlike that first artifact, nobody has published a ready ONNX export of
# `MoritzLaurer/deberta-v3-base-zeroshot-v1`, so `fetch_model.py`'s plain
# download cannot get one from a URL that does not exist yet. `scripts/
# export_zeroshot_model.py` produces it — run once, by a maintainer, outside
# this build (it needs torch + optimum, ~2GB of tooling a serving image has no
# other reason to carry). Until its output is hosted somewhere this build can
# fetch from (a GitHub release on this repo is the obvious default), the
# supported path is `docker-compose.yml`'s `./models:/opt/models:ro` bind
# mount, which reads from the very directory the export script writes to —
# not this stage.
#
# MODEL_URL/MODEL_SHA256/TOKENIZER_URL/TOKENIZER_SHA256 below are therefore
# empty until that hosting step happens. Empty is a supported, tested shape —
# fetch_model.py already treats a missing artifact as a normal, non-fatal
# outcome (Tier 1 falls back to its lexicon scorer, reported as degraded by
# /v1/capabilities) — so the image still builds and still serves correctly
# with these unset; it just does not get the accuracy the model backend buys,
# and relies on the compose mount to get it instead. Fill in all four once the
# artifact is published, and this stage's bake-in becomes the primary path
# again; MODEL_REPO/MODEL_REVISION are recorded for provenance and matter to
# the export script, not to the fetch below.
FROM ${PYTHON_IMAGE} AS model

ARG MODEL_DIR
ARG MODEL_REPO=MoritzLaurer/deberta-v3-base-zeroshot-v1
ARG MODEL_REVISION=c2040e8a2599abd2925fdc716cf40e8052d56ec4
ARG MODEL_URL=
ARG MODEL_SHA256=
ARG TOKENIZER_URL=
ARG TOKENIZER_SHA256=

COPY scripts/fetch_model.py /usr/local/bin/fetch_model.py

# mkdir here as well as in the script: the COPY in the runtime stage must find a
# directory even if the fetch bailed out before creating one.
RUN mkdir -p "${MODEL_DIR}" && \
    python /usr/local/bin/fetch_model.py \
        --target "${MODEL_DIR}" \
        --url "${MODEL_URL}" \
        --sha256 "${MODEL_SHA256}" \
        --tokenizer-url "${TOKENIZER_URL}" \
        --tokenizer-sha256 "${TOKENIZER_SHA256}"


# ── Runtime ───────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ARG VENV
ARG MODEL_DIR

LABEL org.opencontainers.image.title="air-classifier-service" \
      org.opencontainers.image.description="Layered text classification with cost-aware escalation" \
      org.opencontainers.image.source="https://github.com/bookbee/air-classifier-service" \
      org.opencontainers.image.licenses="Proprietary"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="${VENV}/bin:${PATH}" \
    AIR_CLASSIFIER__APP__HOST=0.0.0.0 \
    AIR_CLASSIFIER__APP__PORT=8080 \
    AIR_CLASSIFIER__CLASSIFIER__MODEL_PATH=${MODEL_DIR}/model.onnx \
    AIR_CLASSIFIER__CLASSIFIER__TOKENIZER_PATH=${MODEL_DIR}/tokenizer.json

RUN groupadd --system --gid 1001 air && \
    useradd --system --uid 1001 --gid air --home-dir /app --shell /usr/sbin/nologin air && \
    install -d -o air -g air /app

COPY --from=builder ${VENV} ${VENV}
COPY --from=model ${MODEL_DIR} ${MODEL_DIR}

WORKDIR /app
USER air

EXPOSE 8080

# Liveness only: /v1/health reports the process, not the tiers. A cloud-provider
# outage degrades readiness, and must never restart the container.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=2).read()"]

ENTRYPOINT ["uvicorn", "air_classifier_service.main:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
