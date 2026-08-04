"""CascadePSP second-stage refinement for CaRe-Ego masks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Protocol

import numpy as np
import requests
import torch

from .inference import Prediction, derive_contact

RefinementMode = Literal["low", "medium"]


@dataclass(frozen=True)
class RefinementProfile:
    """CascadePSP quality and memory parameters."""

    fast: bool
    max_size: int


REFINEMENT_PROFILES: dict[RefinementMode, RefinementProfile] = {
    # Official fast mode runs only the global refinement stage. A smaller L
    # lowers memory and latency for this throughput-oriented profile.
    "low": RefinementProfile(fast=True, max_size=600),
    # Full global and local refinement with the official default L value.
    "medium": RefinementProfile(fast=False, max_size=900),
}

LOGGER = logging.getLogger(__name__)
CASCADEPSP_MODEL_URL = "https://github.com/hkchengrex/CascadePSP/releases/download/v1.0/model"
CASCADEPSP_MODEL_SHA256 = "60826ace3385961cdf8608abbdf1234cd22379303b97edf96f18d67bba769cfa"
CASCADEPSP_MODEL_FILENAME = "cascadepsp_v1_0.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cascadepsp_model(
    model_directory: str | Path | None,
    *,
    allow_download: bool,
) -> Path:
    """Return a SHA-256-verified official CascadePSP model path."""
    directory = (
        Path(model_directory).expanduser()
        if model_directory is not None
        else Path("~/.segmentation-refinement").expanduser()
    )
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / CASCADEPSP_MODEL_FILENAME
    if model_path.is_file() and _sha256(model_path) == CASCADEPSP_MODEL_SHA256:
        return model_path
    if not allow_download:
        detail = "missing" if not model_path.exists() else "failed SHA-256 verification"
        raise RuntimeError(f"CascadePSP model is {detail}: {model_path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".cascadepsp-", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output:
            with requests.get(
                CASCADEPSP_MODEL_URL,
                stream=True,
                timeout=(10, 300),
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != CASCADEPSP_MODEL_SHA256:
            raise RuntimeError("Downloaded CascadePSP model failed SHA-256 verification")
        os.replace(temporary_path, model_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return model_path


def _load_backend(model_path: Path, device: str) -> Any:
    """Build the pinned upstream backend from an explicitly named checkpoint."""
    from segmentation_refinement import Refiner
    from segmentation_refinement.models.psp.pspnet import RefinementModule
    from torchvision import transforms

    raw_state = torch.load(
        str(model_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw_state, dict):
        raise TypeError("CascadePSP checkpoint must contain a state dictionary")
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in raw_state.items()
    }
    backend = Refiner.__new__(Refiner)
    backend.model = RefinementModule()
    backend.model.load_state_dict(state, strict=True)
    backend.model.eval().to(device)
    backend.device = device
    backend.im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    backend.seg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return backend


class PredictionRefiner(Protocol):
    """Interface consumed by the producer-consumer service."""

    def refine_prediction(
        self,
        image_bgr: np.ndarray,
        prediction: Prediction,
        mode: RefinementMode,
    ) -> Prediction: ...


def refinement_mode(value: str) -> RefinementMode:
    """Validate an external refinement mode string."""
    if value not in REFINEMENT_PROFILES:
        choices = ", ".join(REFINEMENT_PROFILES)
        raise ValueError(f"refinement_mode must be one of: {choices}")
    return value  # type: ignore[return-value]


class CascadePspRefiner:
    """Load CascadePSP once and refine all semantic masks after stage one."""

    def __init__(
        self,
        device: str | torch.device,
        *,
        model_directory: str | Path | None = None,
        allow_download: bool = True,
        backend: Any | None = None,
    ) -> None:
        requested_device = torch.device(device)
        if requested_device.type == "mps":
            LOGGER.warning(
                "CascadePSP does not support its area-resize path on MPS; "
                "using CPU for the refinement stage"
            )
            requested_device = torch.device("cpu")
        self.device = str(requested_device)
        if backend is not None:
            self._backend = backend
            return

        model_path = ensure_cascadepsp_model(
            model_directory,
            allow_download=allow_download,
        )
        self._backend = _load_backend(model_path, self.device)

    def _score(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        profile: RefinementProfile,
    ) -> np.ndarray:
        binary = mask > 0
        if not binary.any():
            return np.zeros(binary.shape, dtype=np.uint8)
        if binary.all():
            return np.full(binary.shape, 255, dtype=np.uint8)
        output = self._backend.refine(
            image_bgr,
            binary.astype(np.uint8) * 255,
            fast=profile.fast,
            L=profile.max_size,
        )
        score = np.asarray(output, dtype=np.uint8)
        if score.shape != binary.shape:
            raise RuntimeError(f"CascadePSP returned shape {score.shape}, expected {binary.shape}")
        return score

    def refine_prediction(
        self,
        image_bgr: np.ndarray,
        prediction: Prediction,
        mode: RefinementMode,
    ) -> Prediction:
        """Refine hand/object masks and rebuild all derived masks."""
        profile = REFINEMENT_PROFILES[mode]
        left_hand = self._score(image_bgr, prediction.hands == 1, profile)
        right_hand = self._score(image_bgr, prediction.hands == 2, profile)
        hand_scores = np.stack((left_hand, right_hand), axis=0)
        best_hand = hand_scores.argmax(axis=0).astype(np.uint8) + 1
        tied = (left_hand == right_hand) & (left_hand > 127)
        best_hand[tied & (prediction.hands == 2)] = 2
        hands = np.where(hand_scores.max(axis=0) > 127, best_hand, 0).astype(np.uint8)

        left_object = (self._score(image_bgr, prediction.left_object > 0, profile) > 127).astype(
            np.uint8
        )
        right_object = (self._score(image_bgr, prediction.right_object > 0, profile) > 127).astype(
            np.uint8
        )
        shared_object = ((left_object > 0) & (right_object > 0)).astype(np.uint8)

        return Prediction(
            hands=hands,
            left_object=left_object,
            right_object=right_object,
            shared_object=shared_object,
            raw_contact=prediction.raw_contact,
            derived_contact=derive_contact(hands, left_object, right_object),
        )
