#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE:?Set IMAGE to the complete registry image name}"
: "${TAG:?Set TAG to the immutable image tag}"

build_args=(
  --platform "${PLATFORM:-linux/amd64}"
  --progress "${BUILD_PROGRESS:-auto}"
  --tag "${IMAGE}:${TAG}"
)
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  build_args+=(--secret id=github_token,env=GITHUB_TOKEN)
fi
build_args+=(--push .)

docker buildx build "${build_args[@]}"
