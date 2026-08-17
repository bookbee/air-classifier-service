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
RUN mkdir -p src/air_classifier && touch src/air_classifier/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install ".[all]"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --force-reinstall .


# ── Model bake-in ─────────────────────────────────────────────────────────────
# The Tier-1 artifact ships inside the image: reproducible builds and zero
# cold-start network beat image size (docs/00-plan.md §4). The step is
# best-effort by design — see scripts/fetch_model.py.
FROM ${PYTHON_IMAGE} AS model

ARG MODEL_DIR
ARG MODEL_REPO=Xenova/distilbert-base-uncased-finetuned-sst-2-english
ARG MODEL_REVISION=0b6928efcb76139cae2c6881d49cda67fe119f42
ARG MODEL_URL=https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/onnx/model_quantized.onnx
ARG MODEL_SHA256=79b45c7fd8c2673aa53615aba07caeb2de798f63a95a2393a365982429c94f7f
ARG TOKENIZER_URL=https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/tokenizer.json
ARG TOKENIZER_SHA256=d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66

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

LABEL org.opencontainers.image.title="air-classifier" \
      org.opencontainers.image.description="Layered sentiment analysis with cost-aware escalation" \
      org.opencontainers.image.source="https://github.com/bookbee/air-classifier" \
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

ENTRYPOINT ["uvicorn", "air_classifier.main:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
