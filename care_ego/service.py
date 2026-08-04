"""Producer-consumer segmentation worker for file-backed video requests."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import shutil
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
from urllib.parse import unquote, urlparse

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
from .inference import load_model, predict_batch
from .refinement import (
    CascadePspRefiner,
    PredictionRefiner,
    RefinementMode,
    refinement_mode,
)
from .schema import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION, build_result_document

_FRAMES_DONE = object()
_SERVICE_STOP = object()
LOGGER = logging.getLogger(__name__)
_MAX_REQUEST_ID_LENGTH = 128
_MAX_BATCH_SIZE = 4096
_MAX_GEOMETRY_WORKERS = 64
_MAX_REQUEST_ATTEMPTS = 20
_MAX_COMPLETED_REQUEST_CACHE_SIZE = 10_000


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Only file:// URIs are supported, got: {uri}")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"Remote file URI authorities are unsupported: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _safe_request_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("request_id must be a string")
    if not value or len(value) > _MAX_REQUEST_ID_LENGTH:
        raise ValueError(f"request_id must contain 1-{_MAX_REQUEST_ID_LENGTH} characters")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if cleaned != value:
        raise ValueError("request_id may contain only letters, numbers, dots, underscores, and dashes")
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
                "<redacted>"
                if isinstance(key, str) and is_sensitive(key)
                else _redact_config(item)
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
        _file_uri_path(request.video_uri)
        _file_uri_path(request.output_uri)
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
            raise ValueError(
                f"config.retry.attempts must be between 1 and {_MAX_REQUEST_ATTEMPTS}"
            )
        if request.retry_backoff_seconds is not None and request.retry_backoff_seconds < 0:
            raise ValueError("config.retry.backoff_seconds cannot be negative")
        return request


@dataclass(frozen=True)
class WorkerConfig:
    """Performance and batching parameters controlled by the worker owner."""

    batch_sizes: tuple[int, ...] = (128, 64, 32, 16, 8, 4, 2, 1)
    mixed_precision: bool | None = None
    geometry_workers: int = max(1, min(8, (os.cpu_count() or 2) // 2))
    request_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    completed_request_cache_size: int = 256
    refinement_mode: RefinementMode = "low"
    refinement_model_directory: Path | None = None
    refinement_allow_download: bool = True

    def __post_init__(self) -> None:
        normalized = _batch_sizes(self.batch_sizes)
        object.__setattr__(self, "batch_sizes", normalized)
        if not 1 <= self.geometry_workers <= _MAX_GEOMETRY_WORKERS:
            raise ValueError(f"geometry_workers must be between 1 and {_MAX_GEOMETRY_WORKERS}")
        if not 1 <= self.request_attempts <= _MAX_REQUEST_ATTEMPTS:
            raise ValueError(f"request_attempts must be between 1 and {_MAX_REQUEST_ATTEMPTS}")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if not 0 <= self.completed_request_cache_size <= _MAX_COMPLETED_REQUEST_CACHE_SIZE:
            raise ValueError(
                "completed_request_cache_size must be between 0 and "
                f"{_MAX_COMPLETED_REQUEST_CACHE_SIZE}"
            )
        object.__setattr__(self, "refinement_mode", refinement_mode(self.refinement_mode))


@dataclass(frozen=True)
class SegmentationResult:
    request_id: str
    output_path: Path
    frame_count: int
    batch_size: int
    elapsed_seconds: float
    schema_name: str = RESULT_SCHEMA_NAME
    schema_version: str = RESULT_SCHEMA_VERSION
    delivery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Frame:
    index: int
    image: np.ndarray


class VideoDecodeError(ValueError):
    """Input video cannot be decoded and retrying will not repair it."""


class _FrameProducer(threading.Thread):
    """Decode a video into an unbounded in-memory frame queue."""

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

    def stop(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise VideoDecodeError(f"Cannot open video: {self.video_path}")
            index = 0
            while (
                not self.stop_requested.is_set()
                and (self.max_frames is None or index < self.max_frames)
            ):
                ok, image = capture.read()
                if not ok:
                    break
                self.frame_queue.put(_Frame(index=index, image=image))
                index += 1
            if index == 0 and not self.stop_requested.is_set():
                raise VideoDecodeError(f"Video contains no decodable frames: {self.video_path}")
        except Exception as error:  # noqa: BLE001 - forwarded to the consumer thread
            self.error = error
        finally:
            capture.release()
            self.frame_queue.put(_FRAMES_DONE)


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


class _AdaptiveConsumer:
    """Consume queued frames with the largest batch that fits the device."""

    def __init__(
        self,
        model: torch.nn.Module,
        refiner: PredictionRefiner,
        config: WorkerConfig,
    ) -> None:
        self.model = model
        self.refiner = refiner
        self.config = config
        self.device = next(model.parameters()).device

    def consume(
        self,
        frame_queue: queue.Queue,
        request: SegmentationRequest,
    ) -> tuple[dict[int, list[dict]], int]:
        candidates = request.batch_sizes or self.config.batch_sizes
        candidate_index = 0
        pending: deque[_Frame] = deque()
        producer_done = False
        annotation_jobs: dict[int, Future[list[dict]]] = {}
        largest_actual_batch = 0

        with ThreadPoolExecutor(
            max_workers=request.geometry_workers or self.config.geometry_workers,
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
                    predictions = predict_batch(
                        self.model,
                        [item.image for item in batch],
                        mixed_precision=(
                            request.mixed_precision
                            if request.mixed_precision is not None
                            else self.config.mixed_precision
                        ),
                    )
                    selected_mode = request.refinement_mode or self.config.refinement_mode
                    predictions = [
                        self.refiner.refine_prediction(item.image, prediction, selected_mode)
                        for item, prediction in zip(batch, predictions)
                    ]
                except RuntimeError as error:
                    if not _is_out_of_memory(error) or candidate_index == len(candidates) - 1:
                        raise
                    candidate_index += 1
                    _release_device_cache(self.device)
                    continue

                largest_actual_batch = max(largest_actual_batch, actual_size)
                for item, prediction in zip(batch, predictions):
                    annotation_jobs[item.index] = geometry_pool.submit(
                        prediction_annotations,
                        prediction,
                        simplify_tolerance=request.simplify_tolerance,
                        min_area=request.min_area,
                    )
                for _ in range(actual_size):
                    pending.popleft()

        annotations = {index: annotation_jobs[index].result() for index in sorted(annotation_jobs)}
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
        self._requests: queue.Queue = queue.Queue()
        self._model: torch.nn.Module | None = None
        self._refiner = refiner
        self._result_delivery = result_delivery or HttpResultDelivery()
        self._thread: threading.Thread | None = None
        self._model_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._accepting_requests = False
        self._inflight: dict[str, tuple[SegmentationRequest, Future[SegmentationResult]]] = {}
        self._completed: OrderedDict[
            str, tuple[SegmentationRequest, Future[SegmentationResult]]
        ] = OrderedDict()

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
            if (
                not self._accepting_requests
                or self._thread is None
                or not self._thread.is_alive()
            ):
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
                        and (
                            completed_error is None
                            or isinstance(completed_error, DeliveryError)
                        )
                    ):
                        self._completed[request.request_id] = (request, future)
                        self._completed.move_to_end(request.request_id)
                        while (
                            len(self._completed) > self.config.completed_request_cache_size
                        ):
                            self._completed.popitem(last=False)

            future.add_done_callback(forget)
            self._requests.put(_RequestEnvelope(request=request, future=future))
            return future

    def process(self, value: Mapping[str, Any]) -> SegmentationResult:
        """Process one request synchronously using the same worker pipeline."""
        return self._process_with_retries(SegmentationRequest.from_dict(value))

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return bool(
                self._accepting_requests
                and self._thread is not None
                and self._thread.is_alive()
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
        try:
            while True:
                envelope = self._requests.get()
                if envelope is _SERVICE_STOP:
                    return
                try:
                    envelope.future.set_result(self._process_with_retries(envelope.request))
                except Exception as error:  # noqa: BLE001 - delivered through the Future
                    envelope.future.set_exception(error)
        finally:
            with self._lifecycle_lock:
                if threading.current_thread() is self._thread:
                    self._accepting_requests = False

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
                last_error = error
                device = next(self._ensure_model().parameters()).device
                _release_device_cache(device)
                if attempt == attempts:
                    break
                delay = backoff * (2 ** (attempt - 1))
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
        source = _file_uri_path(request.video_uri)
        if not source.is_file():
            raise FileNotFoundError(source)
        output_directory = _file_uri_path(request.output_uri)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"{request.request_id}.json"

        with tempfile.TemporaryDirectory(prefix="care-ego-video-") as temporary:
            local_video = Path(temporary) / source.name
            shutil.copy2(source, local_video)
            frame_queue: queue.Queue = queue.Queue(maxsize=0)
            producer = _FrameProducer(local_video, frame_queue, request.max_frames)
            producer.start()
            try:
                annotations, batch_size = _AdaptiveConsumer(
                    self._ensure_model(), self._ensure_refiner(), self.config
                ).consume(frame_queue, request)
            finally:
                producer.stop()
                producer.join()
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
                    "refinement": {"mode": request.refinement_mode or self.config.refinement_mode},
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
        # Persist before delivery so reference-mode receivers never observe a
        # callback for a missing output file.
        self._write_json_atomic(output_path, document)
        delivery = self._result_delivery.deliver(
            request.delivery,
            document,
            output_path,
            request.request_id,
        )
        delivery_payload = delivery.to_dict()
        document["runtime"]["delivery"] = delivery_payload
        self._write_json_atomic(output_path, document)
        if delivery.status == "failed" and request.delivery.required:
            raise DeliveryError(
                f"Result was saved to {output_path}, but required delivery failed: {delivery.error}"
            )
        return SegmentationResult(
            request_id=request.request_id,
            output_path=output_path,
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
