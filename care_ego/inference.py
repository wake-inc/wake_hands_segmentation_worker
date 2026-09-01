"""Checkpoint loading and frame inference for CaRe-Ego."""

from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict
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
LOGGER = logging.getLogger(__name__)
_MAX_CUDA_GRAPH_CACHE_SIZE = 4
_CUDA_GRAPH_CACHE: OrderedDict[
    tuple[int, tuple[int, ...], tuple[int, int], bool], "_CudaGraphInference"
] = OrderedDict()
_CUDA_GRAPH_DISABLED: set[tuple[int, tuple[int, ...], tuple[int, int], bool]] = set()
_INVALID_GRAPH_BATCH_VALUES: set[str] = set()


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
    raw = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=True)
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
        # NHWC kernels are faster on NVIDIA Tensor Cores for the convolutional
        # parts of the backbone/heads. Keep the model and input layout aligned.
        model = model.to(memory_format=torch.channels_last)
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
    tensor = torch.from_numpy(preprocess_frame(frame_bgr)).unsqueeze(0)
    return tensor.to(device)


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Create the exact CPU input tensor used by the upstream CaRe-Ego pipeline."""
    resized = cv2.resize(frame_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def _preprocess_batch(
    frames_bgr: list[np.ndarray],
    device: torch.device,
    preprocessed_frames: list[np.ndarray] | None = None,
) -> torch.Tensor:
    if preprocessed_frames is not None and len(preprocessed_frames) != len(frames_bgr):
        raise ValueError("preprocessed_frames must match frames_bgr length")
    arrays = preprocessed_frames or [preprocess_frame(frame) for frame in frames_bgr]
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    if device.type == "cuda":
        tensor = tensor.pin_memory()
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor.to(device, non_blocking=device.type == "cuda")


@torch.inference_mode()
def _predict_logits(
    model: torch.nn.Module, image: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = model.extract_feat(image)
    metadata = [{"img_shape": INPUT_SIZE, "ori_shape": INPUT_SIZE} for _ in range(image.shape[0])]
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


def _derive_contact_cuda(
    hands: torch.Tensor,
    left_object: torch.Tensor,
    right_object: torch.Tensor,
) -> torch.Tensor:
    """CUDA implementation of contact morphology for an NCHW mask batch."""
    hand = (hands > 0).float()
    obj = ((left_object > 0) | (right_object > 0)).float()

    def dilate(value: torch.Tensor, size: int) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(value, size, stride=1, padding=size // 2)

    # A 3x3 max-pool is equivalent to binary dilation. Morphological gradient
    # is dilation minus erosion (erosion via max-pool of the complement).
    def gradient(value: torch.Tensor) -> torch.Tensor:
        dilated = dilate(value, 3)
        eroded = 1.0 - dilate(1.0 - value, 3)
        return (dilated - eroded) > 0

    contact = (gradient(hand) & (dilate(obj, 13) > 0)) | (gradient(obj) & (dilate(hand, 13) > 0))
    return dilate(contact.float(), 3).to(torch.uint8)


def _mask_tensor_from_logits(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output_size: tuple[int, int],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.nn.functional.interpolate(
                output,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            .argmax(1)
            .to(torch.uint8)
            for output in logits
        ],
        dim=0,
    )


class _CudaGraphInference:
    """Replay a fixed-shape CaRe-Ego graph with new input pixels."""

    def __init__(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        output_size: tuple[int, int],
        use_amp: bool,
    ) -> None:
        self.static_input = torch.empty_like(sample_input)
        self.graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=sample_input.device)
        capture_stream.wait_stream(torch.cuda.current_stream(sample_input.device))

        def amp_context():
            return (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )

        with torch.cuda.stream(capture_stream):
            self.static_input.copy_(sample_input)
            for _ in range(3):
                with amp_context():
                    logits = _predict_logits(model, self.static_input)
                masks = _mask_tensor_from_logits(logits, output_size)
                _derive_contact_cuda(masks[0], masks[2], masks[3])
        torch.cuda.current_stream(sample_input.device).wait_stream(capture_stream)
        torch.cuda.synchronize(sample_input.device)
        with torch.cuda.graph(self.graph, stream=capture_stream):
            with amp_context():
                logits = _predict_logits(model, self.static_input)
            self.masks = _mask_tensor_from_logits(logits, output_size)
            self.contact = _derive_contact_cuda(self.masks[0], self.masks[2], self.masks[3])

    def replay(self, model_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_input.copy_(model_input)
        self.graph.replay()
        return self.masks, self.contact


def _cuda_graph_batch_size() -> int:
    raw_value = os.environ.get("WAKE_CUDA_GRAPH_BATCH_SIZE", "0")
    try:
        value = int(raw_value)
    except ValueError:
        if raw_value not in _INVALID_GRAPH_BATCH_VALUES:
            LOGGER.warning(
                "Ignoring invalid WAKE_CUDA_GRAPH_BATCH_SIZE=%r; CUDA Graph is disabled",
                raw_value,
            )
            _INVALID_GRAPH_BATCH_VALUES.add(raw_value)
        return 0
    if value < 0:
        if raw_value not in _INVALID_GRAPH_BATCH_VALUES:
            LOGGER.warning(
                "Ignoring negative WAKE_CUDA_GRAPH_BATCH_SIZE=%r; CUDA Graph is disabled",
                raw_value,
            )
            _INVALID_GRAPH_BATCH_VALUES.add(raw_value)
        return 0
    return value


def _is_cuda_out_of_memory(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _is_fatal_cuda_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "device-side assert",
            "illegal memory access",
            "misaligned address",
            "unspecified launch failure",
            "cuda context is destroyed",
            "driver shutting down",
            "cublas_status_execution_failed",
            "cudnn_status_execution_failed",
        )
    )


def _run_eager(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    output_size: tuple[int, int],
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    )
    with amp_context:
        logits = _predict_logits(model, model_input)
    mask_tensor = _mask_tensor_from_logits(logits, output_size)
    cuda_contact = None
    if model_input.device.type == "cuda":
        cuda_contact = _derive_contact_cuda(mask_tensor[0], mask_tensor[2], mask_tensor[3])
    return mask_tensor, cuda_contact


def predict(model: torch.nn.Module, frame_bgr: np.ndarray) -> Prediction:
    """Run inference on an OpenCV BGR image and return original-size masks."""
    return predict_batch(model, [frame_bgr])[0]


def predict_batch(
    model: torch.nn.Module,
    frames_bgr: list[np.ndarray],
    *,
    mixed_precision: bool | None = None,
    preprocessed_frames: list[np.ndarray] | None = None,
) -> list[Prediction]:
    """Run one GPU forward pass for a batch of OpenCV BGR frames."""
    if not frames_bgr:
        return []
    if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames_bgr):
        raise ValueError("Expected HxWx3 BGR images")
    device = next(model.parameters()).device
    original_batch_size = len(frames_bgr)
    graph_batch_size = _cuda_graph_batch_size()
    # Pad only a near-full tail batch. If graph capture at the configured size
    # OOMs, the adaptive consumer can still retry a genuinely smaller eager
    # batch on GPUs with less VRAM.
    graph_enabled = (
        device.type == "cuda" and graph_batch_size >= original_batch_size > graph_batch_size // 2
    )
    inference_frames = frames_bgr
    inference_preprocessed = preprocessed_frames
    if graph_enabled and original_batch_size < graph_batch_size:
        padding = graph_batch_size - original_batch_size
        inference_frames = [*frames_bgr, *([frames_bgr[-1]] * padding)]
        if preprocessed_frames is not None:
            inference_preprocessed = [
                *preprocessed_frames,
                *([preprocessed_frames[-1]] * padding),
            ]
    amp_requested = device.type == "cuda" if mixed_precision is None else mixed_precision
    use_amp = bool(amp_requested) and device.type == "cuda"
    model_input = _preprocess_batch(inference_frames, device, inference_preprocessed)
    # Resize logits (not argmax'ed labels) to each source frame first. This
    # matches MMSeg's postprocess path and avoids spatial edge shifts caused
    # by nearest-neighbour upsampling of a low-resolution class map.
    output_sizes = {frame.shape[:2] for frame in frames_bgr}
    if len(output_sizes) != 1:
        raise ValueError("All frames in an inference batch must have the same dimensions")
    output_size = next(iter(output_sizes))

    # Pack all heads before the blocking device-to-host transfer. Besides
    # reducing synchronization/launch overhead, the blocking copy is required:
    # NumPy has no awareness of CUDA streams and must not observe an incomplete
    # non-blocking destination buffer.
    # One interpolation kernel per output head, regardless of batch size.
    # Convert each resized head to uint8 immediately so full-resolution logits
    # from all four heads are never retained in VRAM at the same time.
    if graph_enabled:
        cache_key = (id(model), tuple(model_input.shape), output_size, use_amp)
        if cache_key in _CUDA_GRAPH_DISABLED:
            graph_enabled = False
            # A smaller adaptive retry may initially have been padded to the
            # failed graph size. Rebuild its true-size input for eager mode.
            if original_batch_size < graph_batch_size:
                model_input = _preprocess_batch(frames_bgr, device, preprocessed_frames)
    if graph_enabled:
        try:
            graph_runner = _CUDA_GRAPH_CACHE.get(cache_key)
            if graph_runner is None:
                graph_runner = _CudaGraphInference(model, model_input, output_size, use_amp)
                _CUDA_GRAPH_CACHE[cache_key] = graph_runner
                while len(_CUDA_GRAPH_CACHE) > _MAX_CUDA_GRAPH_CACHE_SIZE:
                    _CUDA_GRAPH_CACHE.popitem(last=False)
            else:
                _CUDA_GRAPH_CACHE.move_to_end(cache_key)
            mask_tensor, cuda_contact = graph_runner.replay(model_input)
        except RuntimeError as error:
            _CUDA_GRAPH_CACHE.pop(cache_key, None)
            if _is_cuda_out_of_memory(error):
                _CUDA_GRAPH_DISABLED.add(cache_key)
                raise
            if _is_fatal_cuda_error(error):
                raise
            _CUDA_GRAPH_DISABLED.add(cache_key)
            LOGGER.warning(
                "CUDA Graph failed for shape %s and output %s; using eager inference: %s",
                tuple(model_input.shape),
                output_size,
                error,
            )
            mask_tensor, cuda_contact = _run_eager(model, model_input, output_size, use_amp)
    else:
        mask_tensor, cuda_contact = _run_eager(model, model_input, output_size, use_amp)
    batch_masks = mask_tensor.to("cpu").numpy()
    if cuda_contact is not None:
        cuda_contact = cuda_contact.to("cpu").numpy()
    batch_masks = batch_masks.astype(np.uint8, copy=False)

    predictions = []
    for index, frame in enumerate(frames_bgr[:original_batch_size]):
        height, width = frame.shape[:2]
        masks = [output[index] for output in batch_masks]
        hands, raw_contact, left_object, right_object = masks
        shared = ((left_object > 0) & (right_object > 0)).astype(np.uint8)
        derived_contact = (
            cuda_contact[index]
            if cuda_contact is not None
            else derive_contact(hands, left_object, right_object)
        )
        predictions.append(
            Prediction(
                hands=hands,
                left_object=left_object,
                right_object=right_object,
                shared_object=shared,
                raw_contact=raw_contact,
                derived_contact=derived_contact,
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
