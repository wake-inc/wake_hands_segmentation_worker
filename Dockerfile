# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS base

ARG APP_VERSION=0.1.0
ARG TARGETARCH
ARG VENV_CACHE_REVISION=3

LABEL org.opencontainers.image.title="WAKE AI Hands Segmentation Worker GPU" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.description="Self-contained persistent CaRe-Ego and CascadePSP GPU inference worker" \
      org.opencontainers.image.weights="CaRe-Ego best-mIoU; CascadePSP v1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp/wake-worker \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    WAKE_DEVICE=cuda \
    WAKE_CHECKPOINT_PATH=/app/weights/care_ego_best_miou_weights.pth \
    WAKE_CASCADEPSP_MODEL_DIR=/app/weights \
    WAKE_CASCADEPSP_ALLOW_DOWNLOAD=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /tmp/wake-worker --uid 10001 worker \
    && mkdir -p /app /tmp/wake-worker \
    && chown -R worker:worker /tmp/wake-worker

FROM base AS dependencies

ARG TARGETARCH
ARG VENV_CACHE_REVISION

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_VENV_RELOCATABLE=1

WORKDIR /build

# Cache dependencies only on the lock/configuration files. The uv binary and
# download cache are build mounts and are absent from the runtime filesystem.
COPY pyproject.toml uv.lock ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=uv-venv-${TARGETARCH}-${VENV_CACHE_REVISION},target=/staged-venv,sharing=locked \
    --mount=type=cache,id=uv-tmp-${TARGETARCH},target=/tmp/uv-tmp,sharing=locked \
    --mount=type=secret,id=github_token,required=false \
    if [ -f /run/secrets/github_token ]; then \
        export GIT_CONFIG_COUNT=1; \
        export GIT_CONFIG_KEY_0="url.https://x-access-token:$(cat /run/secrets/github_token)@github.com/.insteadOf"; \
        export GIT_CONFIG_VALUE_0=https://github.com/; \
    fi; \
    TMPDIR=/tmp/uv-tmp UV_PROJECT_ENVIRONMENT=/staged-venv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE.txt ./
COPY care_ego ./care_ego
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=uv-venv-${TARGETARCH}-${VENV_CACHE_REVISION},target=/staged-venv,sharing=locked \
    --mount=type=cache,id=uv-tmp-${TARGETARCH},target=/tmp/uv-tmp,sharing=locked \
    --mount=type=secret,id=github_token,required=false \
    if [ -f /run/secrets/github_token ]; then \
        export GIT_CONFIG_COUNT=1; \
        export GIT_CONFIG_KEY_0="url.https://x-access-token:$(cat /run/secrets/github_token)@github.com/.insteadOf"; \
        export GIT_CONFIG_VALUE_0=https://github.com/; \
    fi; \
    TMPDIR=/tmp/uv-tmp UV_PROJECT_ENVIRONMENT=/staged-venv \
    uv sync --frozen --no-dev --no-editable \
    && uv venv --relocatable --allow-existing /staged-venv \
    && TMPDIR=/tmp/uv-tmp UV_PROJECT_ENVIRONMENT=/staged-venv \
        uv sync --frozen --no-dev --no-editable --reinstall \
    && { \
        sha256sum pyproject.toml uv.lock /staged-venv/bin/python; \
        find care_ego -type f -print0 | sort -z | xargs -0 sha256sum; \
    } | sha256sum > /venv-ready

FROM base AS runtime

ARG TARGETARCH
ARG VENV_CACHE_REVISION

# Referencing the marker makes the dependency stage complete before the shared
# cache mount is copied. Only the final environment enters the runtime image;
# uv, wheel caches, extraction files, and the project source remain behind.
COPY --from=dependencies /venv-ready /venv-ready

RUN --mount=type=cache,id=uv-venv-${TARGETARCH}-${VENV_CACHE_REVISION},target=/staged-venv,sharing=locked \
    cp --archive /staged-venv /opt/venv \
    && rm /venv-ready

WORKDIR /app
COPY --chmod=0555 scripts/run-worker ./scripts/run-worker
COPY --chown=10001:10001 --chmod=0444 weights/care_ego_best_miou_weights.pth ./weights/care_ego_best_miou_weights.pth
COPY --chown=10001:10001 --chmod=0444 weights/cascadepsp_v1_0.pth ./weights/cascadepsp_v1_0.pth
RUN chmod 0555 ./weights

USER 10001:10001

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/run-worker"]
