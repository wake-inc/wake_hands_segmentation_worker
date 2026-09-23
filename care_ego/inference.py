"""Checkpoint loading and frame inference for CaRe-Ego."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from mmseg.registry import MODELS

from . import config
from .registry import register_all_modules

INPUT_SIZE = (448, 448)
MEAN = np.asarray([106.01075, 95.40013, 87.42854], dtype=np.float32)
STD = np.asarray([64.356636, 60.888744, 61.41911], dtype=np.float32)


@dataclass(frozen=True)
class Prediction:
    """Discrete CaRe-Ego masks in the original image resolution."""

    hands: np.ndarray
    left_object: np.ndarray
    right_object: np.ndarray
    shared_object: np.ndarray
    raw_contact: np.ndarray
    derived_contact: np.ndarray


def choose_device(requested: str = "auto") -> torch.device:
    """Choose MPS, CUDA, or CPU in that order unless explicitly requested."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_state_dict(checkpoint: str | Path) -> Mapping[str, torch.Tensor]:
    """Read either an MMEngine checkpoint or a raw/model state dictionary."""
    try:
        raw = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as error:
        if "mmap can only be used with files saved with" not in str(error):
            raise
        try:
            raw = torch.load(str(checkpoint), map_location="cpu", mmap=False, weights_only=True)
        except pickle.UnpicklingError:
            raw = torch.load(str(checkpoint), map_location="cpu", mmap=False, weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping, got {type(raw).__name__}")
    state = raw.get("state_dict", raw.get("model", raw))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return state


def build_model() -> torch.nn.Module:
    """Construct the network from the package's Python configuration."""
    register_all_modules()
    model_config = copy.deepcopy(config.model)
    model_config["pretrained"] = None
    return MODELS.build(model_config)


def load_model(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
    strict: bool = True,
) -> torch.nn.Module:
    """Build CaRe-Ego, load a checkpoint, and switch to evaluation mode."""
    model = build_model()
    model.load_state_dict(extract_state_dict(checkpoint), strict=strict)
    resolved = choose_device(device) if isinstance(device, str) else device
    if resolved.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return model.eval().to(resolved)


def checkpoint_compatibility(checkpoint: str | Path) -> dict:
    """Compare checkpoint keys and shapes with the configured network."""
    model = build_model()
    expected = model.state_dict()
    supplied = extract_state_dict(checkpoint)
    common = expected.keys() & supplied.keys()
    shape_mismatches = {
        key: {
            "expected": list(expected[key].shape),
            "supplied": list(supplied[key].shape),
        }
        for key in sorted(common)
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    }
    missing = sorted(expected.keys() - supplied.keys())
    unexpected = sorted(supplied.keys() - expected.keys())
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "expected_keys": len(expected),
        "supplied_keys": len(supplied),
        "matching_keys": len(common) - len(shape_mismatches),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "strictly_compatible": not missing and not unexpected and not shape_mismatches,
    }


def _preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def _preprocess_batch(frames_bgr: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arrays = []
    for frame in frames_bgr:
        resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        arrays.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=device.type == "cuda")


@torch.inference_mode()
def _predict_logits(
    model: torch.nn.Module, image: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = model.extract_feat(image)
    metadata = [
        {"img_shape": INPUT_SIZE, "ori_shape": INPUT_SIZE} for _ in range(image.shape[0])
    ]
    hand, hand_feature = model.decode_head1.predict(list(features), metadata, model.test_cfg)
    raw_contact = model.decode_head3.predict(list(features), metadata, model.test_cfg)
    left_object = model.decode_head2.predict(list(features), hand_feature, metadata, model.test_cfg)
    right_object = model.decode_head4.predict(
        list(features), hand_feature, metadata, model.test_cfg
    )
    return hand, raw_contact, left_object, right_object


def derive_contact(
    hands: np.ndarray, left_object: np.ndarray, right_object: np.ndarray
) -> np.ndarray:
    hand_mask = (hands > 0).astype(np.uint8)
    object_mask = ((left_object > 0) | (right_object > 0)).astype(np.uint8)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    hand_edge = cv2.morphologyEx(hand_mask, cv2.MORPH_GRADIENT, edge_kernel) > 0
    object_edge = cv2.morphologyEx(object_mask, cv2.MORPH_GRADIENT, edge_kernel) > 0
    near_hand = cv2.dilate(hand_mask, near_kernel) > 0
    near_object = cv2.dilate(object_mask, near_kernel) > 0
    contact = (hand_edge & near_object) | (object_edge & near_hand)
    return cv2.dilate(contact.astype(np.uint8), edge_kernel)


def predict(model: torch.nn.Module, frame_bgr: np.ndarray) -> Prediction:
    """Run inference on an OpenCV BGR image and return original-size masks."""
    return predict_batch(model, [frame_bgr])[0]


def predict_batch(
    model: torch.nn.Module,
    frames_bgr: list[np.ndarray],
    *,
    mixed_precision: bool | None = None,
) -> list[Prediction]:
    """Run one GPU forward pass for a batch of OpenCV BGR frames."""
    if not frames_bgr:
        return []
    if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames_bgr):
        raise ValueError("Expected HxWx3 BGR images")
    device = next(model.parameters()).device
    amp_requested = device.type == "cuda" if mixed_precision is None else mixed_precision
    use_amp = bool(amp_requested) and device.type == "cuda"
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    )
    with amp_context:
        logits = _predict_logits(model, _preprocess_batch(frames_bgr, device))
    batch_masks = [
        output.argmax(1).to("cpu", non_blocking=device.type == "cuda").numpy().astype(np.uint8)
        for output in logits
    ]

    predictions = []
    for index, frame in enumerate(frames_bgr):
        height, width = frame.shape[:2]
        masks = [
            cv2.resize(output[index], (width, height), interpolation=cv2.INTER_NEAREST)
            for output in batch_masks
        ]
        hands, raw_contact, left_object, right_object = masks
        shared = ((left_object > 0) & (right_object > 0)).astype(np.uint8)
        predictions.append(
            Prediction(
                hands=hands,
                left_object=left_object,
                right_object=right_object,
                shared_object=shared,
                raw_contact=raw_contact,
                derived_contact=derive_contact(hands, left_object, right_object),
            )
        )
    return predictions


def visualize(frame_bgr: np.ndarray, prediction: Prediction, alpha: float = 0.58) -> np.ndarray:
    """Overlay hands, objects, and a derived hand-object contact interface."""
    colors = np.zeros_like(frame_bgr)
    active = np.zeros(frame_bgr.shape[:2], dtype=bool)
    shared = prediction.shared_object > 0
    regions = (
        ((prediction.left_object > 0) & ~shared, (220, 40, 220)),
        ((prediction.right_object > 0) & ~shared, (220, 220, 30)),
        (shared, (40, 210, 40)),
        (prediction.hands == 1, (40, 40, 240)),
        (prediction.hands == 2, (240, 80, 20)),
    )
    for region, color in regions:
        colors[region] = color
        active |= region
    result = frame_bgr.copy()
    mixed = cv2.addWeighted(frame_bgr, 1.0 - alpha, colors, alpha, 0)
    result[active] = mixed[active]
    result[prediction.derived_contact > 0] = (0, 240, 255)
    return result
