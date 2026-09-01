#!/bin/bash
set -Eeuo pipefail

REGION="eu-north-1"
ACCOUNT_ID=""
REPOSITORY="wake_hands_segmentation_worker"
IMAGE_DIGEST=""
CONTAINER="wake_hands_segmentation_worker"

if [[ -z "${ACCOUNT_ID}" || -z "${IMAGE_DIGEST}" ]]; then
  echo "Set ACCOUNT_ID and IMAGE_DIGEST before bootstrapping" >&2
  exit 1
fi

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPOSITORY}@${IMAGE_DIGEST}"

systemctl enable --now docker
nvidia-smi

aws ecr get-login-password --region "${REGION}" |
  docker login --username AWS --password-stdin "${REGISTRY}"

docker pull "${IMAGE}"
docker rm --force "${CONTAINER}" 2>/dev/null || true

docker run --detach \
  --name "${CONTAINER}" \
  --restart unless-stopped \
  --gpus all \
  --publish 8080:8080 \
  --env AWS_REGION="${REGION}" \
  --env AWS_DEFAULT_REGION="${REGION}" \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  "${IMAGE}"

for _ in $(seq 1 90); do
  if curl --fail --silent http://127.0.0.1:8080/health/ready; then
    echo
    echo "WAKE worker is ready"
    exit 0
  fi

  if ! docker inspect --format '{{.State.Running}}' "${CONTAINER}" |
    grep --quiet true; then
    docker logs "${CONTAINER}"
    exit 1
  fi

  sleep 5
done

docker logs "${CONTAINER}"
echo "WAKE worker did not become ready in time"
exit 1
