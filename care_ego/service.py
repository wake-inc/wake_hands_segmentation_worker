"""Producer-consumer segmentation worker for URI-backed video requests."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from typing_extensions import Self

from .delivery import (
    DeliveryConfig,
    DeliveryError,
    HttpResultDelivery,
    ResultDelivery,
)
from .geometry import prediction_annotations
from .inference import Prediction, load_model, predict_batch, preprocess_frame
from .refinement import (
    CascadePspRefiner,
    PredictionRefiner,
    RefinementMode,
    refinement_mode,
)
from .schema import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION, build_result_document
from .storage import (
    UriStorage,
    file_uri_path,
    result_uri,
    validate_input_uri,
    validate_output_uri,
)

_FRAMES_DONE = object()
_SERVICE_STOP = object()
LOGGER = logging.getLogger(__name__)
_MAX_REQUEST_ID_LENGTH = 128
_MAX_BATCH_SIZE = 4096
_MAX_GEOMETRY_WORKERS = 64
_MAX_REQUEST_ATTEMPTS = 20
_MAX_COMPLETED_REQUEST_CACHE_SIZE = 10_000
_MAX_PENDING_REQUESTS = 1_024
_MAX_RETRY_DELAY_SECONDS = 60.0
_PRODUCER_JOIN_TIMEOUT_SECONDS = 15.0


def _file_uri_path(uri: str) -> Path:
    """Backward-compatible internal alias for local URI resolution."""
    return file_uri_path(uri)


def _safe_request_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("request_id must be a string")
    if not value or len(value) > _MAX_REQUEST_ID_LENGTH:
        raise ValueError(f"request_id must contain 1-{_MAX_REQUEST_ID_LENGTH} characters")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if cleaned != value:
        raise ValueError(
            "request_id may contain only letters, numbers, dots, underscores, and dashes"
        )
    return cleaned


def _json_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _json_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _config_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    result = deepcopy(dict(value))
    if any(not isinstance(key, str) for key in result):
        raise ValueError(f"{name} keys must be strings")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    return result


def _config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return _config_mapping(config.get(name), f"config.{name}")


def _batch_sizes(value: Any, name: str = "batch_sizes") -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ValueError(f"{name} must contain positive integers")
    if any(item > _MAX_BATCH_SIZE for item in value):
        raise ValueError(f"{name} values cannot exceed {_MAX_BATCH_SIZE}")
    return tuple(sorted(set(value), reverse=True))


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true, false, or null")
    return value


def _redact_config(value: Any) -> Any:
    """Remove credentials before embedding request configuration in results."""

    def is_sensitive(key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        return normalized == "authorization" or normalized.endswith(
            ("password", "secret", "token", "apikey")
        )

    if isinstance(value, Mapping):
        return {
            key: (
                "<redacted>" if isinstance(key, str) and is_sensitive(key) else _redact_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


@dataclass(frozen=True)
class SegmentationRequest:
    """Validated service request created from the external input dictionary."""

    video_uri: str
    output_uri: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    simplify_tolerance: float = 2.0
    min_area: int = 64
    max_frames: int | None = None
    refinement_mode: RefinementMode | None = None
    config: dict[str, Any] = field(default_factory=dict)
    batch_sizes: tuple[int, ...] | None = None
    mixed_precision: bool | None = None
    geometry_workers: int | None = None
    request_attempts: int | None = None
    retry_backoff_seconds: float | None = None
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SegmentationRequest:
        if not isinstance(value, Mapping):
            raise ValueError("Request body must be a JSON object")
        allowed = {
            "video_uri",
            "output_uri",
            "request_id",
            "simplify_tolerance",
            "min_area",
            "max_frames",
            "refinement_mode",
            "config",
        }
        unknown = value.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown request fields: {sorted(unknown)}")
        try:
            video_uri = value["video_uri"]
            output_uri = value["output_uri"]
        except KeyError as error:
            raise ValueError(f"Missing request field: {error.args[0]}") from error
        if not isinstance(video_uri, str) or not video_uri:
            raise ValueError("video_uri must be a non-empty string")
        if not isinstance(output_uri, str) or not output_uri:
            raise ValueError("output_uri must be a non-empty string")
        if "\x00" in video_uri or "\x00" in output_uri:
            raise ValueError("URI values cannot contain null bytes")
        request_id = _safe_request_id(value.get("request_id", uuid.uuid4().hex))
        request_config = _config_mapping(value.get("config"), "config")
        input_config = _config_section(request_config, "input")
        inference_config = _config_section(request_config, "inference")
        refinement_config = _config_section(request_config, "refinement")
        geometry_config = _config_section(request_config, "geometry")
        retry_config = _config_section(request_config, "retry")
        delivery_config = _config_section(request_config, "delivery")

        max_frames_value = value.get("max_frames", input_config.get("max_frames"))
        simplify_tolerance = value.get(
            "simplify_tolerance", geometry_config.get("simplify_tolerance", 2.0)
        )
        min_area = value.get("min_area", geometry_config.get("min_area", 64))
        refinement_value = value.get("refinement_mode", refinement_config.get("mode"))
        batch_sizes_value = inference_config.get("batch_sizes")
        geometry_workers_value = geometry_config.get("workers")
        request_attempts_value = retry_config.get("attempts")
        retry_backoff_value = retry_config.get("backoff_seconds")
        request = cls(
            video_uri=video_uri,
            output_uri=output_uri,
            request_id=request_id,
            simplify_tolerance=_json_number(simplify_tolerance, "simplify_tolerance"),
            min_area=_json_integer(min_area, "min_area"),
            max_frames=(
                _json_integer(max_frames_value, "max_frames")
                if max_frames_value is not None
                else None
            ),
            refinement_mode=(
                refinement_mode(str(refinement_value)) if refinement_value is not None else None
            ),
            config=request_config,
            batch_sizes=(
                _batch_sizes(batch_sizes_value, "config.inference.batch_sizes")
                if batch_sizes_value is not None
                else None
            ),
            mixed_precision=_optional_bool(
                inference_config.get("mixed_precision"),
                "config.inference.mixed_precision",
            ),
            geometry_workers=(
                _json_integer(geometry_workers_value, "config.geometry.workers")
                if geometry_workers_value is not None
                else None
            ),
            request_attempts=(
                _json_integer(request_attempts_value, "config.retry.attempts")
                if request_attempts_value is not None
                else None
            ),
            retry_backoff_seconds=(
                _json_number(retry_backoff_value, "config.retry.backoff_seconds")
                if retry_backoff_value is not None
                else None
            ),
            delivery=DeliveryConfig.from_mapping(delivery_config),
        )
        validate_input_uri(request.video_uri)
        validate_output_uri(request.output_uri)
        if request.simplify_tolerance < 0:
            raise ValueError("simplify_tolerance cannot be negative")
        if request.min_area < 1:
            raise ValueError("min_area must be positive")
        if request.max_frames is not None and request.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if request.geometry_workers is not None and not (
            1 <= request.geometry_workers <= _MAX_GEOMETRY_WORKERS
        ):
            raise ValueError(
                f"config.geometry.workers must be between 1 and {_MAX_GEOMETRY_WORKERS}"
            )
        if request.request_attempts is not None and not (
            1 <= request.request_attempts <= _MAX_REQUEST_ATTEMPTS
        ):
            raise ValueError(f"config.retry.attempts must be between 1 and {_MAX_REQUEST_ATTEMPTS}")
        if request.retry_backoff_seconds is not None and not (
            0 <= request.retry_backoff_seconds <= _MAX_RETRY_DELAY_SECONDS
        ):
            raise ValueError(
                f"config.retry.backoff_seconds must be between 0 and {_MAX_RETRY_DELAY_SECONDS:g}"
            )
        return request


@dataclass(frozen=True)
class WorkerConfig:
    """Performance and batching parameters controlled by the worker owner."""

    # Six frames is the measured CUDA Graph throughput optimum on L40S. Retain
    # OOM fallback so the same worker remains portable to smaller GPUs.
    batch_sizes: tuple[int, ...] = (6, 4, 2, 1)
    mixed_precision: bool | None = None
    geometry_workers: int = max(1, min(8, (os.cpu_count() or 2) // 2))
    request_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    completed_request_cache_size: int = 256
    max_pending_requests: int = 8
    refinement_mode: RefinementMode = "low"
    refinement_model_directory: Path | None = None
    refinement_allow_download: bool = True
    temporal_stride: int = 1

    def __post_init__(self) -> None:
        normalized = _batch_sizes(self.batch_sizes)
        object.__setattr__(self, "batch_sizes", normalized)
        if not 1 <= self.geometry_workers <= _MAX_GEOMETRY_WORKERS:
            raise ValueError(f"geometry_workers must be between 1 and {_MAX_GEOMETRY_WORKERS}")
        if not 1 <= self.request_attempts <= _MAX_REQUEST_ATTEMPTS:
            raise ValueError(f"request_attempts must be between 1 and {_MAX_REQUEST_ATTEMPTS}")
        if not 0 <= self.retry_backoff_seconds <= _MAX_RETRY_DELAY_SECONDS:
            raise ValueError(
                f"retry_backoff_seconds must be between 0 and {_MAX_RETRY_DELAY_SECONDS:g}"
            )
        if not 0 <= self.completed_request_cache_size <= _MAX_COMPLETED_REQUEST_CACHE_SIZE:
            raise ValueError(
                "completed_request_cache_size must be between 0 and "
                f"{_MAX_COMPLETED_REQUEST_CACHE_SIZE}"
            )
        if not 1 <= self.max_pending_requests <= _MAX_PENDING_REQUESTS:
            raise ValueError(
                f"max_pending_requests must be between 1 and {_MAX_PENDING_REQUESTS}"
            )
        object.__setattr__(self, "refinement_mode", refinement_mode(self.refinement_mode))
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")


@dataclass(frozen=True)
class SegmentationResult:
    request_id: str
    output_path: Path | None
    frame_count: int
    batch_size: int
    elapsed_seconds: float
    output_uri: str | None = None
    schema_name: str = RESULT_SCHEMA_NAME
    schema_version: str = RESULT_SCHEMA_VERSION
    delivery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Frame:
    index: int
    image: np.ndarray
    model_input: np.ndarray | None = None


class VideoDecodeError(ValueError):
    """Input video cannot be decoded and retrying will not repair it."""


class ServiceOverloadedError(RuntimeError):
    """The bounded request queue cannot accept more work right now."""


class _FrameProducer(threading.Thread):
    """Decode and preprocess video frames into a bounded producer queue."""

    def __init__(
        self,
        video_path: Path,
        frame_queue: queue.Queue,
        max_frames: int | None,
    ) -> None:
        super().__init__(name="care-ego-frame-producer", daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.max_frames = max_frames
        self.error: Exception | None = None
        self.stop_requested = threading.Event()
        self._process_lock = threading.Lock()
        self._nvdec_process: subprocess.Popen | None = None
        self._nvdec_frames_emitted = 0

    def stop(self) -> None:
        self.stop_requested.set()
        with self._process_lock:
            process = self._nvdec_process
        if process is not None:
            self._stop_nvdec_process(process)

    def _put(self, item: object) -> bool:
        while not self.stop_requested.is_set():
            try:
                self.frame_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run(self) -> None:
        try:
            self._run()
        except BaseException as error:  # always wake a waiting consumer
            if isinstance(error, Exception):
                self.error = error
            else:
                self.error = RuntimeError(
                    f"Video producer stopped unexpectedly: {type(error).__name__}"
                )
        finally:
            self._put(_FRAMES_DONE)

    def _run(self) -> None:
        if os.environ.get("WAKE_VIDEO_DECODER", "opencv").lower() == "nvdec":
            self._nvdec_frames_emitted = 0
            try:
                self._run_nvdec()
                return
            except Exception as error:  # noqa: BLE001
                if self.stop_requested.is_set():
                    return
                # Restarting OpenCV after any frame entered the shared queue
                # would mix two timelines beginning at index zero. Let the
                # request retry start from a fresh queue instead.
                if self._nvdec_frames_emitted:
                    raise OSError(
                        "NVDEC failed after emitting "
                        f"{self._nvdec_frames_emitted} frames; refusing an unsafe fallback"
                    ) from error
                LOGGER.warning("NVDEC failed, falling back to OpenCV: %s", error)
        self._run_opencv()

    def _run_opencv(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise VideoDecodeError(f"Cannot open video: {self.video_path}")
            index = 0
            while not self.stop_requested.is_set() and (
                self.max_frames is None or index < self.max_frames
            ):
                ok, image = capture.read()
                if not ok:
                    break
                if not self._put(
                    _Frame(index=index, image=image, model_input=preprocess_frame(image))
                ):
                    break
                index += 1
            if index == 0 and not self.stop_requested.is_set():
                raise VideoDecodeError(f"Video contains no decodable frames: {self.video_path}")
        finally:
            capture.release()

    def _run_nvdec(self) -> None:
        probe = cv2.VideoCapture(str(self.video_path))
        if not probe.isOpened():
            raise VideoDecodeError(f"Cannot open video: {self.video_path}")
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()
        frame_bytes = width * height * 3
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(self.video_path),
                # NVDEC frames must first be downloaded in their native
                # hardware pixel format; converting directly to BGR is not
                # supported by all FFmpeg builds.
                "-vf",
                "hwdownload,format=nv12,format=bgr24",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # Never let an unconsumed diagnostic pipe stall frame production.
            # Request logs retain the decoder exit status and OpenCV fallback.
            stderr=subprocess.DEVNULL,
        )
        with self._process_lock:
            self._nvdec_process = process
        index = 0
        assert process.stdout is not None
        try:
            while not self.stop_requested.is_set() and (
                self.max_frames is None or index < self.max_frames
            ):
                payload = process.stdout.read(frame_bytes)
                if len(payload) != frame_bytes:
                    break
                image = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
                if not self._put(
                    _Frame(index=index, image=image, model_input=preprocess_frame(image))
                ):
                    break
                self._nvdec_frames_emitted += 1
                index += 1
            stopped_early = self.stop_requested.is_set() or (
                self.max_frames is not None and index >= self.max_frames
            )
            if stopped_early:
                self._stop_nvdec_process(process)
            elif process.wait(timeout=30) != 0:
                raise VideoDecodeError("NVDEC ffmpeg decoder failed")
            if index == 0 and not self.stop_requested.is_set():
                raise VideoDecodeError("Video contains no decodable frames")
        finally:
            self._stop_nvdec_process(process)
            with self._process_lock:
                if self._nvdec_process is process:
                    self._nvdec_process = None

    @staticmethod
    def _stop_nvdec_process(process: subprocess.Popen) -> None:
        try:
            running = process.poll() is None
        except OSError:
            running = False
        if running:
            try:
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    LOGGER.error("Unable to stop NVDEC ffmpeg process pid=%s", process.pid)
        try:
            if process.stdout is not None:
                process.stdout.close()
        except (OSError, ValueError):
            pass


def _is_out_of_memory(error: RuntimeError) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _release_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _is_fatal_accelerator_error(error: BaseException) -> bool:
    if not isinstance(error, RuntimeError):
        return False
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


class _AdaptiveConsumer:
    """Consume queued frames with the largest batch that fits the device."""

    def __init__(
        self,
        model: torch.nn.Module,
        refiner: PredictionRefiner,
        config: WorkerConfig,
        known_batch_limit: int | None = None,
    ) -> None:
        self.model = model
        self.refiner = refiner
        self.config = config
        self.device = next(model.parameters()).device
        # Remember the largest successful batch for this model/device. The
        # first request probes the configured candidates; subsequent requests
        # avoid repeating expensive CUDA OOM retries on smaller GPUs.
        self._known_batch_limit = known_batch_limit

    @property
    def known_batch_limit(self) -> int | None:
        return self._known_batch_limit

    def _predict_frames(
        self,
        batch: list[_Frame],
        request: SegmentationRequest,
    ) -> list[Prediction]:
        mixed_precision = (
            request.mixed_precision
            if request.mixed_precision is not None
            else self.config.mixed_precision
        )
        return predict_batch(
            self.model,
            [item.image for item in batch],
            preprocessed_frames=[
                item.model_input if item.model_input is not None else preprocess_frame(item.image)
                for item in batch
            ],
            mixed_precision=mixed_precision,
        )

    def consume(
        self,
        frame_queue: queue.Queue,
        request: SegmentationRequest,
    ) -> tuple[dict[int, list[dict]], int]:
        configured_candidates = request.batch_sizes or self.config.batch_sizes
        eligible_candidates = tuple(
            size
            for size in configured_candidates
            if self._known_batch_limit is None or size <= self._known_batch_limit
        )
        candidates = eligible_candidates or (self._known_batch_limit or min(configured_candidates),)
        candidate_index = 0
        pending: deque[_Frame] = deque()
        producer_done = False
        annotation_jobs: dict[int, Future[list[dict]]] = {}
        annotations: dict[int, list[dict]] = {}
        largest_actual_batch = 0
        temporal_stride = self.config.temporal_stride
        consume_started = time.monotonic()
        inference_seconds = 0.0
        refinement_seconds = 0.0

        geometry_workers = request.geometry_workers or self.config.geometry_workers
        max_geometry_jobs = geometry_workers * 2
        with ThreadPoolExecutor(
            max_workers=geometry_workers,
            thread_name_prefix="care-ego-geometry",
        ) as geometry_pool:
            while pending or not producer_done:
                target_size = candidates[candidate_index]
                while len(pending) < target_size and not producer_done:
                    item = frame_queue.get()
                    if item is _FRAMES_DONE:
                        producer_done = True
                    else:
                        pending.append(item)
                if not pending:
                    continue

                actual_size = min(target_size, len(pending))
                batch = [pending[index] for index in range(actual_size)]
                try:
                    key_positions = [
                        position
                        for position, item in enumerate(batch)
                        if temporal_stride == 1
                        or item.index % temporal_stride == 0
                        or item.index == 0
                    ]
                    key_batch = [batch[position] for position in key_positions]
                    # CaRe-Ego is always evaluated for every frame. Temporal
                    # sparsity applies only to the expensive CascadePSP
                    # refinement stage, never to the base hand segmentation.
                    inference_started = time.monotonic()
                    base_predictions = self._predict_frames(batch, request)
                    inference_seconds += time.monotonic() - inference_started
                    key_predictions = [base_predictions[position] for position in key_positions]
                    selected_mode = request.refinement_mode or self.config.refinement_mode
                    batch_refiner = getattr(self.refiner, "refine_predictions", None)
                    if batch_refiner is not None:
                        refinement_started = time.monotonic()
                        refined_key_predictions = batch_refiner(
                            [item.image for item in key_batch], key_predictions, selected_mode
                        )
                        refinement_seconds += time.monotonic() - refinement_started
                        key_predictions = refined_key_predictions
                    else:
                        refinement_started = time.monotonic()
                        key_predictions = [
                            self.refiner.refine_prediction(item.image, prediction, selected_mode)
                            for item, prediction in zip(key_batch, key_predictions)
                        ]
                        refinement_seconds += time.monotonic() - refinement_started
                    key_by_index = dict(zip((item.index for item in key_batch), key_predictions))
                    predictions: list[Prediction] = []
                    for item in batch:
                        prediction = key_by_index.get(item.index)
                        if prediction is None:
                            # Non-key frames retain their own CaRe-Ego output;
                            # only CascadePSP is sparse.
                            prediction = base_predictions[len(predictions)]
                        predictions.append(prediction)
                except RuntimeError as error:
                    if not _is_out_of_memory(error) or candidate_index == len(candidates) - 1:
                        raise
                    candidate_index += 1
                    _release_device_cache(self.device)
                    continue

                if actual_size == target_size:
                    self._known_batch_limit = max(self._known_batch_limit or 0, target_size)
                largest_actual_batch = max(largest_actual_batch, actual_size)
                for item, prediction in zip(batch, predictions):
                    annotation_jobs[item.index] = geometry_pool.submit(
                        prediction_annotations,
                        prediction,
                        simplify_tolerance=request.simplify_tolerance,
                        min_area=request.min_area,
                    )
                    if len(annotation_jobs) >= max_geometry_jobs:
                        oldest_index = next(iter(annotation_jobs))
                        annotations[oldest_index] = annotation_jobs.pop(oldest_index).result()
                for _ in range(actual_size):
                    pending.popleft()

        geometry_wait_started = time.monotonic()
        for index, future in annotation_jobs.items():
            annotations[index] = future.result()
        annotations = dict(sorted(annotations.items()))
        geometry_wait_seconds = time.monotonic() - geometry_wait_started
        LOGGER.warning(
            "pipeline timing: total=%.3fs inference=%.3fs refinement=%.3fs geometry_wait=%.3fs",
            time.monotonic() - consume_started,
            inference_seconds,
            refinement_seconds,
            geometry_wait_seconds,
        )
        return annotations, largest_actual_batch


@dataclass
class _RequestEnvelope:
    request: SegmentationRequest
    future: Future[SegmentationResult]


class SegmentationService:
    """Single-GPU service that waits for and serially processes requests."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "auto",
        config: WorkerConfig | None = None,
        model_metadata: Mapping[str, Any] | None = None,
        refiner: PredictionRefiner | None = None,
        result_delivery: ResultDelivery | None = None,
        storage: UriStorage | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.config = config or WorkerConfig()
        self.model_metadata = dict(
            model_metadata
            or {
                "id": "care-ego-cascadepsp",
                "architecture": "CaRe-Ego followed by CascadePSP",
                "families": ["cnn", "transformer"],
                "framework": "pytorch",
                "attributes": {
                    "stages": [
                        {"id": "care-ego", "task": "semantic_segmentation"},
                        {"id": "cascadepsp", "task": "mask_refinement"},
                    ]
                },
            }
        )
        self.model_metadata.setdefault("checkpoint", self.checkpoint.name)
        self._requests: queue.Queue = queue.Queue(maxsize=self.config.max_pending_requests)
        self._model: torch.nn.Module | None = None
        self._refiner = refiner
        self._result_delivery = result_delivery or HttpResultDelivery()
        self._storage = storage or UriStorage()
        self._thread: threading.Thread | None = None
        self._model_lock = threading.RLock()
        self._processing_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._accepting_requests = False
        self._inflight: dict[str, tuple[SegmentationRequest, Future[SegmentationResult]]] = {}
        self._completed: OrderedDict[
            str, tuple[SegmentationRequest, Future[SegmentationResult]]
        ] = OrderedDict()
        self._known_batch_limit: int | None = None

    def _ensure_model(self) -> torch.nn.Module:
        with self._model_lock:
            if self._model is None:
                self._model = load_model(self.checkpoint, self.device)
            return self._model

    def _ensure_refiner(self) -> PredictionRefiner:
        with self._model_lock:
            if self._refiner is None:
                model = self._ensure_model()
                device = next(model.parameters()).device
                self._refiner = CascadePspRefiner(
                    device,
                    model_directory=self.config.refinement_model_directory,
                    allow_download=self.config.refinement_allow_download,
                )
            return self._refiner

    def start(self) -> SegmentationService:
        """Load the model once and start the request consumer."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._accepting_requests:
                    return self
                raise RuntimeError("Service is stopping and cannot be restarted yet")
            self._ensure_model()
            self._ensure_refiner()
            self._accepting_requests = True
            self._thread = threading.Thread(
                target=self._serve,
                name="care-ego-request-consumer",
                daemon=True,
            )
            self._thread.start()
        return self

    def submit(self, value: Mapping[str, Any]) -> Future[SegmentationResult]:
        """Queue one request dictionary and return its completion future."""
        request = SegmentationRequest.from_dict(value)
        with self._lifecycle_lock:
            if not self._accepting_requests or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("Service is not running; call start() first")
            existing = self._inflight.get(request.request_id)
            if existing is None:
                existing = self._completed.get(request.request_id)
            if existing is not None:
                existing_request, existing_future = existing
                if existing_request != request:
                    raise ValueError(
                        f"request_id {request.request_id!r} is already in use by a different request"
                    )
                return existing_future

            future: Future[SegmentationResult] = Future()
            self._inflight[request.request_id] = (request, future)

            def forget(_future: Future[SegmentationResult]) -> None:
                completed_error = None if _future.cancelled() else _future.exception()
                with self._lifecycle_lock:
                    if self._inflight.get(request.request_id) == (request, future):
                        self._inflight.pop(request.request_id, None)
                    if (
                        self.config.completed_request_cache_size
                        and not _future.cancelled()
                        and (completed_error is None or isinstance(completed_error, DeliveryError))
                    ):
                        self._completed[request.request_id] = (request, future)
                        self._completed.move_to_end(request.request_id)
                        while len(self._completed) > self.config.completed_request_cache_size:
                            self._completed.popitem(last=False)

            future.add_done_callback(forget)
            try:
                self._requests.put_nowait(_RequestEnvelope(request=request, future=future))
            except queue.Full as error:
                # Trigger the registered cleanup callback before rejecting the
                # request, otherwise its id would remain reserved.
                future.cancel()
                raise ServiceOverloadedError(
                    "Worker queue is full; retry after an active request completes"
                ) from error
            return future

    def process(self, value: Mapping[str, Any]) -> SegmentationResult:
        """Process one request synchronously using the same worker pipeline."""
        with self._processing_lock:
            return self._process_with_retries(SegmentationRequest.from_dict(value))

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return bool(
                self._accepting_requests and self._thread is not None and self._thread.is_alive()
            )

    @property
    def pending_requests(self) -> int:
        return self._requests.qsize()

    def stop(self, wait: bool = True) -> None:
        """Stop accepting work and drain requests already accepted."""
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._accepting_requests = False
                return
            if self._accepting_requests:
                self._accepting_requests = False
                self._requests.put(_SERVICE_STOP)
        if wait:
            thread.join()

    def _serve(self) -> None:
        fatal_error: BaseException | None = None
        try:
            while True:
                envelope = self._requests.get()
                if envelope is _SERVICE_STOP:
                    return
                try:
                    with self._processing_lock:
                        result = self._process_with_retries(envelope.request)
                    envelope.future.set_result(result)
                except Exception as error:  # noqa: BLE001 - delivered through the Future
                    if not envelope.future.done():
                        envelope.future.set_exception(error)
                    if _is_fatal_accelerator_error(error):
                        fatal_error = error
                        LOGGER.critical(
                            "Fatal accelerator state detected; terminating the worker so "
                            "Gunicorn can start it with a fresh CUDA context: %s",
                            error,
                        )
                        os._exit(70)
                except BaseException as error:
                    fatal_error = error
                    if not envelope.future.done():
                        envelope.future.set_exception(
                            RuntimeError(
                                f"Request consumer stopped unexpectedly: {type(error).__name__}"
                            )
                        )
                    raise
        except BaseException as error:
            fatal_error = fatal_error or error
            raise
        finally:
            with self._lifecycle_lock:
                if threading.current_thread() is self._thread:
                    self._accepting_requests = False
            if fatal_error is not None:
                while True:
                    try:
                        queued = self._requests.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(queued, _RequestEnvelope) and not queued.future.done():
                        queued.future.set_exception(
                            RuntimeError("Request consumer stopped before processing the request")
                        )

    def _process_with_retries(self, request: SegmentationRequest) -> SegmentationResult:
        last_error: Exception | None = None
        attempts = request.request_attempts or self.config.request_attempts
        backoff = (
            request.retry_backoff_seconds
            if request.retry_backoff_seconds is not None
            else self.config.retry_backoff_seconds
        )
        for attempt in range(1, attempts + 1):
            try:
                return self._process(request)
            except (FileNotFoundError, VideoDecodeError):
                raise
            except (OSError, RuntimeError) as error:
                if _is_fatal_accelerator_error(error):
                    raise
                last_error = error
                device = next(self._ensure_model().parameters()).device
                _release_device_cache(device)
                if attempt == attempts:
                    break
                delay = min(_MAX_RETRY_DELAY_SECONDS, backoff * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "Request %s failed on attempt %d/%d; retrying in %.1fs: %s",
                    request.request_id,
                    attempt,
                    attempts,
                    delay,
                    error,
                )
                if delay:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _process(self, request: SegmentationRequest) -> SegmentationResult:
        started = time.monotonic()
        completed_result_uri = result_uri(request.output_uri, request.request_id)

        with tempfile.TemporaryDirectory(prefix="care-ego-video-") as temporary:
            temporary_directory = Path(temporary)
            local_video = self._storage.stage_input(request.video_uri, temporary_directory)
            requested_batches = request.batch_sizes or self.config.batch_sizes
            # Three batches keep decode/preprocessing ahead of CUDA without
            # buffering an entire long video in host RAM.
            frame_queue: queue.Queue = queue.Queue(maxsize=max(requested_batches) * 3)
            producer = _FrameProducer(local_video, frame_queue, request.max_frames)
            producer.start()
            consumer = _AdaptiveConsumer(
                self._ensure_model(),
                self._ensure_refiner(),
                self.config,
                known_batch_limit=(
                    self._known_batch_limit if request.batch_sizes is None else None
                ),
            )
            try:
                annotations, batch_size = consumer.consume(frame_queue, request)
            finally:
                if request.batch_sizes is None:
                    self._known_batch_limit = consumer.known_batch_limit
                producer.stop()
                producer.join(timeout=_PRODUCER_JOIN_TIMEOUT_SECONDS)
            if producer.is_alive():
                raise RuntimeError(
                    "Video decoder did not stop within "
                    f"{_PRODUCER_JOIN_TIMEOUT_SECONDS:.0f} seconds"
                )
            if producer.error is not None:
                raise producer.error

            document = build_result_document(
                request_id=request.request_id,
                request_config=_redact_config(request.config),
                model=self.model_metadata,
                inputs=[
                    {
                        "id": "video",
                        "modality": "video",
                        "source": {"type": "uri", "value": request.video_uri},
                    }
                ],
                outputs=[
                    {
                        "id": "frame_predictions",
                        "modality": "vision",
                        "unit": "frame",
                        "coordinate_system": {
                            "type": "pixel",
                            "origin": "top_left",
                            "axes": ["x", "y"],
                        },
                        "items": annotations,
                    }
                ],
                runtime={
                    "frame_count": len(annotations),
                    "largest_batch_size": batch_size,
                    "elapsed_seconds": 0.0,
                    "refinement_mode": request.refinement_mode or self.config.refinement_mode,
                    "effective_config": {
                        "input": {"max_frames": request.max_frames},
                        "inference": {
                            "batch_sizes": list(request.batch_sizes or self.config.batch_sizes),
                            "mixed_precision": (
                                request.mixed_precision
                                if request.mixed_precision is not None
                                else self.config.mixed_precision
                            ),
                        },
                        "refinement": {
                            "mode": request.refinement_mode or self.config.refinement_mode
                        },
                        "geometry": {
                            "simplify_tolerance": request.simplify_tolerance,
                            "min_area": request.min_area,
                            "workers": request.geometry_workers or self.config.geometry_workers,
                        },
                        "retry": {
                            "attempts": request.request_attempts or self.config.request_attempts,
                            "backoff_seconds": (
                                request.retry_backoff_seconds
                                if request.retry_backoff_seconds is not None
                                else self.config.retry_backoff_seconds
                            ),
                        },
                    },
                },
            )
            elapsed_seconds = time.monotonic() - started
            document["runtime"]["elapsed_seconds"] = elapsed_seconds
            output_path = self._storage.result_path(completed_result_uri, temporary_directory)
            # Persist before delivery so reference-mode receivers never observe
            # a callback for a missing output object.
            self._write_json_atomic(output_path, document)
            self._storage.publish_result(output_path, completed_result_uri)
            delivery = self._result_delivery.deliver(
                request.delivery,
                document,
                completed_result_uri,
                request.request_id,
            )
            delivery_payload = delivery.to_dict()
            document["runtime"]["delivery"] = delivery_payload
            self._write_json_atomic(output_path, document)
            self._storage.publish_result(output_path, completed_result_uri)
            if delivery.status == "failed" and request.delivery.required:
                raise DeliveryError(
                    "Result was saved to "
                    f"{completed_result_uri}, but required delivery failed: {delivery.error}"
                )
            return SegmentationResult(
                request_id=request.request_id,
                output_path=(
                    file_uri_path(completed_result_uri)
                    if urlparse(completed_result_uri).scheme == "file"
                    else None
                ),
                output_uri=completed_result_uri,
                frame_count=len(annotations),
                batch_size=batch_size,
                elapsed_seconds=elapsed_seconds,
                delivery=delivery_payload,
            )

    @staticmethod
    def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()
