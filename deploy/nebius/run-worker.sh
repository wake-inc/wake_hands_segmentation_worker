#!/bin/bash
set -Eeuo pipefail

ENVIRONMENT_FILE="/etc/wake-worker.env"
READY_URL="http://127.0.0.1:8080/health/ready"
if [[ ! -r "${ENVIRONMENT_FILE}" ]]; then
  echo "Missing ${ENVIRONMENT_FILE}; install it from worker.env.example" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENVIRONMENT_FILE}"
set +a

: "${WAKE_IMAGE:?WAKE_IMAGE is required}"
: "${WAKE_CONTAINER:?WAKE_CONTAINER is required}"
: "${WAKE_S3_ENDPOINT_URL:?WAKE_S3_ENDPOINT_URL is required}"
: "${WAKE_S3_REGION:?WAKE_S3_REGION is required}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required}"

REGISTRY_HOST="${WAKE_IMAGE%%/*}"
PREVIOUS_IMAGE="$(
  docker inspect --format '{{.Config.Image}}' "${WAKE_CONTAINER}" 2>/dev/null || true
)"

nebius iam get-access-token |
  docker login "${REGISTRY_HOST}" --username iam --password-stdin

nvidia-smi
docker pull "${WAKE_IMAGE}"

start_container() {
  local image="$1"
  docker rm --force "${WAKE_CONTAINER}" 2>/dev/null || true
  docker run --detach \
    --name "${WAKE_CONTAINER}" \
    --restart unless-stopped \
    --gpus all \
    --publish 8080:8080 \
    --env-file "${ENVIRONMENT_FILE}" \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    "${image}"
}

wait_until_ready() {
  local attempts="${1:-90}"
  for _ in $(seq 1 "${attempts}"); do
    if curl --fail --silent --show-error --max-time 5 "${READY_URL}" >/dev/null; then
      return 0
    fi
    if ! docker inspect --format '{{.State.Running}}' "${WAKE_CONTAINER}" 2>/dev/null |
      grep --quiet true; then
      return 1
    fi
    sleep 5
  done
  return 1
}

start_container "${WAKE_IMAGE}"
ACTIVE_IMAGE="${WAKE_IMAGE}"
if ! wait_until_ready 90; then
  echo "Candidate image did not become ready" >&2
  docker logs --tail 500 "${WAKE_CONTAINER}" >&2 || true
  if [[ -z "${PREVIOUS_IMAGE}" || "${PREVIOUS_IMAGE}" == "${WAKE_IMAGE}" ]]; then
    exit 1
  fi
  echo "Rolling back to previous image ${PREVIOUS_IMAGE}" >&2
  start_container "${PREVIOUS_IMAGE}"
  if ! wait_until_ready 90; then
    docker logs --tail 500 "${WAKE_CONTAINER}" >&2 || true
    echo "Rollback image did not become ready" >&2
    exit 1
  fi
  ACTIVE_IMAGE="${PREVIOUS_IMAGE}"
fi

echo "WAKE worker is ready on ${ACTIVE_IMAGE}"

# Docker does not restart a running container merely because its health check
# turns unhealthy. Keep a small systemd-supervised watchdog around the API.
consecutive_failures=0
while sleep 30; do
  if curl --fail --silent --max-time 5 "${READY_URL}" >/dev/null; then
    consecutive_failures=0
    continue
  fi
  consecutive_failures=$((consecutive_failures + 1))
  echo "Worker readiness failure ${consecutive_failures}/3" >&2
  if [[ "${consecutive_failures}" -lt 3 ]]; then
    continue
  fi
  docker logs --tail 200 "${WAKE_CONTAINER}" >&2 || true
  docker restart "${WAKE_CONTAINER}"
  if ! wait_until_ready 90; then
    echo "Worker did not recover after restart" >&2
    exit 1
  fi
  consecutive_failures=0
done
