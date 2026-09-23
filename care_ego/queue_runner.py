"""RabbitMQ and local HTTP entry point for GPU hand segmentation."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import signal
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import requests
from wake_job_queue.broker import Broker
from wake_job_queue.config import QueueConfig
from wake_job_queue.consumer import JobConsumer, PermanentJobError, RetryableJobError
from wake_job_queue.contracts import JobEvent, JobInput, JobTelemetry
from wake_job_queue.state import JobState, JobStateUpdate
from wake_job_queue.workspace import JobWorkspace

from . import server_config
from .service import SegmentationService, WorkerConfig

LOGGER = logging.getLogger("care_ego.queue_runner")
_SHUTDOWN = threading.Event()


class _ProgressPublisher:
    """Publish durable safe-boundary progress without blocking inference."""

    def __init__(self, config: QueueConfig, *, buffer_size: int = 64) -> None:
        self.config = config
        self._pending: queue.Queue[tuple[str, JobStateUpdate]] = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-runner-progress",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def set_active(self, _active: bool) -> None:
        pass

    def observe_duration(self, _kind: str, _outcome: str, _seconds: float) -> None:
        pass

    def state_updated(self, kind: str, update: JobStateUpdate) -> None:
        if update.phase != "heartbeat":
            return
        try:
            self._pending.put_nowait((kind, update))
        except queue.Full:
            LOGGER.warning(
                "job_progress_dropped job_id=%s kind=%s attempt=%s sequence=%s",
                update.job_id,
                kind,
                update.attempt,
                update.sequence,
            )

    def _run(self) -> None:
        broker = Broker(self.config)
        current: tuple[str, JobStateUpdate] | None = None
        connected = False
        try:
            while not self._stop.is_set() or current is not None or not self._pending.empty():
                if current is None:
                    try:
                        current = self._pending.get(timeout=0.2)
                    except queue.Empty:
                        continue
                try:
                    if not connected:
                        broker.connect()
                        connected = True
                    kind, update = current
                    runner_id = update.owner_id.rsplit(":", 1)[0]
                    broker.publish_event(
                        JobEvent(
                            type="PROGRESS",
                            job_id=update.job_id,
                            attempt=update.attempt,
                            runner_id=runner_id,
                            sequence=update.sequence,
                            elapsed_seconds=update.elapsed_seconds,
                            estimated_next_batch_seconds=update.estimated_next_batch_seconds,
                            progress=update.progress,
                        ).to_bytes()
                    )
                    if update.batch_processing_seconds > 0:
                        broker.publish_telemetry(
                            JobTelemetry(
                                job_id=update.job_id,
                                attempt=update.attempt,
                                kind=kind,
                                runner_id=runner_id,
                                sequence=update.sequence,
                                elapsed_seconds=update.elapsed_seconds,
                                batch_processing_seconds=update.batch_processing_seconds,
                                estimated_next_batch_seconds=update.estimated_next_batch_seconds,
                            ).to_bytes()
                        )
                    current = None
                except Exception as error:  # noqa: BLE001
                    LOGGER.warning("job_progress_publish_failed error=%s", error)
                    broker.close()
                    connected = False
                    if self._stop.wait(1):
                        current = None
        finally:
            broker.close()


@dataclass(slots=True)
class _RunnerLifecycle:
    max_jobs: int
    idle_exit_seconds: float
    last_delivery_at: float
    handled_jobs: int = 0

    def record_delivery(self, *, now: float | None = None) -> None:
        self.handled_jobs += 1
        self.last_delivery_at = time.monotonic() if now is None else now

    def should_stop(self, *, now: float | None = None) -> bool:
        if self.max_jobs > 0 and self.handled_jobs >= self.max_jobs:
            return True
        current = time.monotonic() if now is None else now
        return current - self.last_delivery_at >= self.idle_exit_seconds


@dataclass(frozen=True, slots=True)
class _DebugOptions:
    service_lifetime: Literal["persistent", "perJob"] = "persistent"
    request_config: dict[str, Any] = field(default_factory=dict)


def _debug_options(payload: dict[str, Any], *, enabled: bool) -> _DebugOptions:
    value = payload.get("debug")
    if value is None:
        return _DebugOptions()
    if not enabled:
        raise PermanentJobError("track_hands debug overrides are disabled")
    if not isinstance(value, dict):
        raise PermanentJobError("track_hands debug must be an object")
    if value.keys() - {"serviceLifetime", "inference"}:
        raise PermanentJobError("track_hands debug has unknown fields")

    lifetime = value.get("serviceLifetime", "persistent")
    inference = value.get("inference", {})
    if lifetime not in {"persistent", "perJob"}:
        raise PermanentJobError("track_hands debug serviceLifetime is invalid")
    if not isinstance(inference, dict):
        raise PermanentJobError("track_hands debug inference must be an object")
    if inference.keys() - {"batchSizes", "mixedPrecision", "geometryWorkers", "refinementMode"}:
        raise PermanentJobError("track_hands debug inference has unknown fields")

    request_config: dict[str, Any] = {}
    if "batchSizes" in inference:
        values = inference["batchSizes"]
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 4096
                for item in values
            )
        ):
            raise PermanentJobError("track_hands debug batchSizes is invalid")
        request_config.setdefault("inference", {})["batch_sizes"] = values
    if "mixedPrecision" in inference:
        mixed_precision = inference["mixedPrecision"]
        if not isinstance(mixed_precision, bool):
            raise PermanentJobError("track_hands debug mixedPrecision is invalid")
        request_config.setdefault("inference", {})["mixed_precision"] = mixed_precision
    if "geometryWorkers" in inference:
        geometry_workers = inference["geometryWorkers"]
        if (
            isinstance(geometry_workers, bool)
            or not isinstance(geometry_workers, int)
            or not 1 <= geometry_workers <= 64
        ):
            raise PermanentJobError("track_hands debug geometryWorkers is invalid")
        request_config.setdefault("geometry", {})["workers"] = geometry_workers
    if "refinementMode" in inference:
        refinement_mode = inference["refinementMode"]
        if refinement_mode not in {"low", "medium"}:
            raise PermanentJobError("track_hands debug refinementMode is invalid")
        request_config.setdefault("refinement", {})["mode"] = refinement_mode
    return _DebugOptions(lifetime, request_config)


def _download(url: str, expected_sha256: str, destination: Path) -> None:
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        digest.update(chunk)
                        output.write(chunk)
    except requests.RequestException as error:
        raise RetryableJobError("source-video download failed") from error
    if digest.hexdigest() != expected_sha256:
        raise PermanentJobError("source-video sha256 does not match the Job payload")


def _batch_progress_callback(state: JobState, total_frames: int | None) -> Callable[..., None]:
    started = time.monotonic()

    def on_batch_complete(processed_frames: int, batch_size: int, batch_seconds: float) -> None:
        progress: dict[str, Any] = {
            "phase": "inference",
            "processedFrames": processed_frames,
            "batchSize": batch_size,
        }
        if total_frames is not None:
            progress["totalFrames"] = total_frames
        state.heartbeat(
            elapsed_seconds=time.monotonic() - started,
            estimated_next_batch_seconds=batch_seconds,
            progress=progress,
        )

    return on_batch_complete


def _segmentation_service() -> SegmentationService:
    return SegmentationService(
        server_config.CHECKPOINT_PATH,
        device=server_config.DEVICE,
        config=WorkerConfig(
            batch_sizes=server_config.BATCH_SIZES,
            mixed_precision=server_config.MIXED_PRECISION,
            geometry_workers=server_config.GEOMETRY_WORKERS,
            refinement_mode=server_config.REFINEMENT_MODE,
            refinement_model_directory=server_config.CASCADEPSP_MODEL_DIRECTORY,
            refinement_allow_download=server_config.CASCADEPSP_ALLOW_DOWNLOAD,
        ),
        model_metadata=server_config.MODEL_METADATA,
    )


class _ServiceProvider:
    def __init__(self, *, debug_task_overrides: bool) -> None:
        self.debug_task_overrides = debug_task_overrides
        self._service: SegmentationService | None = None

    def process(
        self,
        request: dict[str, Any],
        debug: _DebugOptions,
        progress: Callable[..., None],
    ) -> Any:
        if debug.service_lifetime == "perJob":
            return _segmentation_service().process(request, on_batch_complete=progress)
        if self._service is None:
            self._service = _segmentation_service()
        return self._service.process(request, on_batch_complete=progress)

    def close(self) -> None:
        if self._service is not None:
            self._service.stop()


def _processor(
    provider: _ServiceProvider,
    payload: dict[str, Any],
    inputs: dict[str, JobInput] | None,
    state: JobState,
) -> dict[str, Any]:
    debug = _debug_options(payload, enabled=provider.debug_task_overrides)
    source = payload.get("sourceVideo")
    if not isinstance(source, dict):
        raise PermanentJobError("track_hands payload has no sourceVideo")
    url = source.get("downloadUrl")
    sha256 = source.get("sha256")
    if not isinstance(url, str) or not isinstance(sha256, str):
        raise PermanentJobError("track_hands sourceVideo is incomplete")

    total_frames = None
    decode = inputs.get("decode") if inputs else None
    if decode is not None:
        output = decode.output
        decoded = state.workspace.read_json(output.key)
        if decoded is None or decoded.etag != output.etag or decoded.sha256 != output.sha256:
            raise RetryableJobError("decode dependency output is unavailable or changed")
        frames = decoded.value.get("frames")
        if isinstance(frames, int) and not isinstance(frames, bool) and frames > 0:
            total_frames = frames

    scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="wake-hands-") as directory:
        root = Path(directory)
        video = root / "source-video"
        output = root / "output"
        _download(url, sha256, video)
        request_config = {"delivery": {"enabled": False}, **debug.request_config}
        result = provider.process(
            {
                "request_id": f"job-{state.job_id}-attempt-{state.attempt}",
                "video_uri": video.as_uri(),
                "output_uri": output.resolve().as_uri(),
                "config": request_config,
            },
            debug,
            _batch_progress_callback(state, total_frames),
        )
        artifact = state.workspace.put_file(
            state.workspace.key("track_hands", state.job_id, "frame_predictions.json"),
            str(result.output_path),
            content_type="application/json",
        )

    summary = {
        "engine": "care_ego_cascadepsp_gpu",
        "videoSha256": sha256,
        "framesProcessed": result.frame_count,
        "batchSize": result.batch_size,
        "framesArtifact": artifact,
    }
    summary["workspaceOutput"] = state.write_output(summary)
    return summary


def _processor_resolver(provider: _ServiceProvider) -> Callable[..., Any]:
    def resolve(kind: str, kind_version: int = 1) -> Callable[..., Any]:
        if (kind, kind_version) != ("track_hands", 1):
            raise PermanentJobError(f"unsupported GPU job {kind}:{kind_version}")
        return partial(_processor, provider)

    return resolve


def _handle_delivery(
    consumer: JobConsumer,
    lifecycle: _RunnerLifecycle,
    body: bytes,
    redelivered: bool,
) -> Any:
    try:
        return consumer.handle(body, redelivered)
    finally:
        lifecycle.record_delivery()


def _run_queue() -> None:
    config = QueueConfig.from_env()
    workspace = JobWorkspace.from_env()
    runner_id = os.environ.get("RUNNER_ID") or os.environ.get("HOSTNAME") or socket.gethostname()
    max_jobs = int(os.environ.get("RUNNER_MAX_JOBS", "0"))
    idle_exit_seconds = float(os.environ.get("RUNNER_IDLE_EXIT_S", "30"))
    if max_jobs < 0:
        raise ValueError("RUNNER_MAX_JOBS must be zero or a positive integer")
    if idle_exit_seconds <= 0:
        raise ValueError("RUNNER_IDLE_EXIT_S must be positive")

    lifecycle = _RunnerLifecycle(max_jobs, idle_exit_seconds, time.monotonic())
    broker = Broker(config)
    progress = _ProgressPublisher(config)
    provider = _ServiceProvider(debug_task_overrides=False)
    try:
        progress.start()
        broker.connect()
        consumer = JobConsumer(
            config=config,
            broker=broker,
            workspace=workspace,
            runner_id=runner_id,
            resolve_processor=_processor_resolver(provider),
            shutdown_requested=_SHUTDOWN.is_set,
            observer=progress,
        )
        LOGGER.info(
            "gpu_runner_started runner_id=%s max_jobs=%s idle_exit_seconds=%s",
            runner_id,
            max_jobs,
            idle_exit_seconds,
        )
        broker.consume(
            config.work_queue,
            partial(_handle_delivery, consumer, lifecycle),
            should_stop=lambda: _SHUTDOWN.is_set() or lifecycle.should_stop(),
        )
    finally:
        provider.close()
        progress.close()
        broker.close()


def _run_http() -> None:
    from wake_job_queue.http import serve_http_from_env

    provider = _ServiceProvider(debug_task_overrides=server_config.DEBUG_TASK_OVERRIDES)
    try:
        serve_http_from_env(resolve_processor=_processor_resolver(provider))
    finally:
        provider.close()


def _request_shutdown(_signum: int, _frame: object) -> None:
    _SHUTDOWN.set()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)
    mode = os.environ.get("WAKE_RUNNER_MODE", "queue")
    if mode == "queue":
        _run_queue()
        return
    if mode == "http":
        _run_http()
        return
    raise ValueError("WAKE_RUNNER_MODE must be queue or http")


if __name__ == "__main__":
    main()
