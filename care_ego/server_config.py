"""Runtime settings for queue and debug HTTP execution."""

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


def _environment_batch_sizes(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated integers") from error
    if not values or any(value < 1 or value > 4096 for value in values):
        raise ValueError(f"{name} values must be between 1 and 4096")
    return tuple(sorted(set(values), reverse=True))


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

# Probe the requested high-throughput sizes first. Smaller values prevent a
# permanent failure on GPUs where even a batch of 32 does not fit.
BATCH_SIZES = _environment_batch_sizes(
    "WAKE_BATCH_SIZES",
    (128, 64, 32, 16, 8, 4, 2, 1),
)
MIXED_PRECISION = _environment_bool("WAKE_MIXED_PRECISION", True)
GEOMETRY_WORKERS = _environment_positive_int(
    "WAKE_GEOMETRY_WORKERS",
    max(1, min(8, (os.cpu_count() or 2) // 2)),
)
if GEOMETRY_WORKERS > 64:
    raise ValueError("WAKE_GEOMETRY_WORKERS cannot exceed 64")
DEBUG_TASK_OVERRIDES = _environment_bool("WAKE_DEBUG_TASK_OVERRIDES", False)
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
