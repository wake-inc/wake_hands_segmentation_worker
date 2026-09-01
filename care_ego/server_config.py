"""Python-only production settings for the Flask/Gunicorn service."""

from __future__ import annotations

import os
from pathlib import Path


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _environment_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


_checkpoint_environment = os.environ.get("WAKE_CHECKPOINT_PATH")
_source_root = Path(__file__).resolve().parents[1]
_checkpoint_candidates = (
    Path.cwd() / "weights" / "care_ego_best_miou_weights.pth",
    _source_root / "weights" / "care_ego_best_miou_weights.pth",
)
CHECKPOINT_PATH = (
    Path(_checkpoint_environment).expanduser().resolve()
    if _checkpoint_environment
    else next(
        (candidate.resolve() for candidate in _checkpoint_candidates if candidate.is_file()),
        _checkpoint_candidates[0].resolve(),
    )
)
DEVICE = os.environ.get("WAKE_DEVICE", "auto")
S3_ENDPOINT_URL = os.environ.get("WAKE_S3_ENDPOINT_URL") or None
S3_REGION = (
    os.environ.get("WAKE_S3_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or None
)

# Probe the requested high-throughput sizes first. Smaller values prevent a
# permanent failure on GPUs where even a batch of 32 does not fit.
BATCH_SIZES = (6, 4, 2, 1)
MIXED_PRECISION = True
GEOMETRY_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
TEMPORAL_STRIDE = _environment_positive_int("WAKE_TEMPORAL_STRIDE", 1)
REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
COMPLETED_REQUEST_CACHE_SIZE = 256
MAX_PENDING_REQUESTS = _environment_positive_int("WAKE_MAX_PENDING_REQUESTS", 8)
SSE_HEARTBEAT_SECONDS = 15.0
REQUEST_MAX_BYTES = _environment_positive_int("WAKE_REQUEST_MAX_BYTES", 1024 * 1024)
MAX_INPUT_BYTES = _environment_positive_int("WAKE_MAX_INPUT_BYTES", 40 * 1024**3)
REFINEMENT_MODE = os.environ.get("WAKE_REFINEMENT_MODE", "low")
_refinement_directory = os.environ.get("WAKE_CASCADEPSP_MODEL_DIR")
_local_refinement_directory = _source_root / "weights"
CASCADEPSP_MODEL_DIRECTORY = (
    Path(_refinement_directory).expanduser().resolve()
    if _refinement_directory
    else (
        _local_refinement_directory
        if (_local_refinement_directory / "cascadepsp_v1_0.pth").is_file()
        else None
    )
)
CASCADEPSP_ALLOW_DOWNLOAD = _environment_bool(
    "WAKE_CASCADEPSP_ALLOW_DOWNLOAD", CASCADEPSP_MODEL_DIRECTORY is None
)

MODEL_METADATA = {
    "id": "care-ego-cascadepsp",
    "version": "1.0",
    "architecture": "CaRe-Ego segmentation followed by CascadePSP refinement",
    "families": ["cnn", "transformer"],
    "framework": "pytorch",
    "attributes": {
        "stages": [
            {
                "id": "care-ego",
                "architecture": "Swin encoder with convolutional decoders",
                "task": "semantic_segmentation",
            },
            {
                "id": "cascadepsp",
                "version": "0.6",
                "task": "mask_refinement",
            },
        ]
    },
}
