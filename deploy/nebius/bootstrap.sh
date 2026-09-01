#!/bin/bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this bootstrap as root" >&2
  exit 1
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

apt-get update
apt-get install --yes --no-install-recommends \
  ca-certificates \
  curl \
  gnupg

if ! command -v docker >/dev/null 2>&1; then
  apt-get install --yes --no-install-recommends docker.io
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/gpgkey |
    gpg --dearmor --yes --output /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install --yes nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

install --owner=root --group=root --mode=0755 \
  "${SCRIPT_DIRECTORY}/run-worker.sh" /usr/local/sbin/wake-run-worker
install --owner=root --group=root --mode=0644 \
  "${SCRIPT_DIRECTORY}/wake-worker.service" /etc/systemd/system/wake-worker.service
if [[ -f "${SCRIPT_DIRECTORY}/worker.env.example" ]]; then
  install --owner=root --group=root --mode=0600 \
    "${SCRIPT_DIRECTORY}/worker.env.example" /etc/wake-worker.env.example
fi
systemctl daemon-reload

if [[ -r /etc/wake-worker.env ]]; then
  chmod 0600 /etc/wake-worker.env
  systemctl enable --now wake-worker.service
else
  echo "Bootstrap complete. Create /etc/wake-worker.env, then enable wake-worker.service."
fi
