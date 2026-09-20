#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE:?Set IMAGE to the complete registry image name}"
: "${TAG:?Set TAG to the immutable image tag}"

docker buildx build \
  --platform "${PLATFORM:-linux/amd64}" \
  --tag "${IMAGE}:${TAG}" \
  --push \
  .
