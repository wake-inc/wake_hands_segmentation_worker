# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS base

ARG APP_VERSION=""
ARG TARGETARCH

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
    WAKE_BIND=0.0.0.0:8080 \
    WAKE_DEVICE=cuda \
    WAKE_CHECKPOINT_PATH=/app/weights/care_ego_best_miou_weights.pth \
    WAKE_CASCADEPSP_MODEL_DIR=/app/weights \
    WAKE_CASCADEPSP_ALLOW_DOWNLOAD=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /tmp/wake-worker --uid 10001 worker \
    && mkdir -p /app /tmp/wake-worker \
    && chown -R worker:worker /tmp/wake-worker

FROM base AS dependencies

ARG TARGETARCH

ENV UV_LINK_MODE=copy \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=600 \
    UV_PYTHON_DOWNLOADS=never \
    UV_VENV_RELOCATABLE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Cache dependencies only on the lock/configuration files. The uv binary and
# download cache are build mounts and are absent from the runtime filesystem.
COPY pyproject.toml uv.lock ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=uv-tmp-${TARGETARCH},target=/tmp/uv-tmp,sharing=locked \
    TMPDIR=/tmp/uv-tmp \
    uv sync --frozen --no-dev --no-install-project

# This stage changes when application code changes, but inherits the fully
# populated dependency environment above. Installing the local package now
# does not redownload/reinstall CUDA and other locked dependencies.
FROM dependencies AS package

COPY README.md LICENSE.txt ./
COPY care_ego ./care_ego
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=uv-tmp-${TARGETARCH},target=/tmp/uv-tmp,sharing=locked \
    TMPDIR=/tmp/uv-tmp \
    uv sync --frozen --no-dev --no-editable \
    && find /opt/venv -type d -exec chmod 0755 {} + \
    && find /opt/venv -type f -exec chmod a+r {} + \
    && find /opt/venv/bin -type f -exec chmod a+rx {} + \
    && { \
        sha256sum pyproject.toml uv.lock /opt/venv/bin/gunicorn; \
        find care_ego -type f -print0 | sort -z | xargs -0 sha256sum; \
    } | sha256sum > /venv-ready

FROM base AS runtime
# Referencing the marker makes the package stage complete before the runtime
# filesystem is assembled. Only the final environment enters the image.
COPY --from=package /venv-ready /venv-ready
COPY --from=package /opt/venv /opt/venv
RUN rm /venv-ready

WORKDIR /app
COPY --chown=10001:10001 --chmod=0444 gunicorn.conf.py ./gunicorn.conf.py
COPY --chown=10001:10001 --chmod=0444 weights/care_ego_best_miou_weights.pth ./weights/care_ego_best_miou_weights.pth
COPY --chown=10001:10001 --chmod=0444 weights/cascadepsp_v1_0.pth ./weights/cascadepsp_v1_0.pth
RUN chmod 0555 ./weights

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=4).read()"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "-c", "gunicorn.conf.py"]
