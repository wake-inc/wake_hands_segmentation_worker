from pathlib import Path
import queue
import threading

import pytest

from care_ego.delivery import DeliveryError
from care_ego.service import (
    SegmentationRequest,
    SegmentationService,
    ServiceOverloadedError,
    WorkerConfig,
    _FRAMES_DONE,
    _FrameProducer,
    _RequestEnvelope,
    _redact_config,
)


def test_request_accepts_local_file_uris(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.touch()
    output = tmp_path / "output"

    request = SegmentationRequest.from_dict(
        {
            "video_uri": video.as_uri(),
            "output_uri": output.as_uri(),
            "request_id": "job-42",
        }
    )

    assert request.request_id == "job-42"
    assert request.video_uri == video.as_uri()
    assert request.refinement_mode is None


def test_request_accepts_s3_uris() -> None:
    request = SegmentationRequest.from_dict(
        {
            "video_uri": "s3://wake-test/jobs/job-42/input.mp4",
            "output_uri": "s3://wake-test/jobs/job-42/results/",
            "request_id": "job-42",
        }
    )

    assert request.video_uri == "s3://wake-test/jobs/job-42/input.mp4"
    assert request.output_uri == "s3://wake-test/jobs/job-42/results/"


def test_request_accepts_medium_refinement(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.touch()

    request = SegmentationRequest.from_dict(
        {
            "video_uri": video.as_uri(),
            "output_uri": tmp_path.as_uri(),
            "refinement_mode": "medium",
        }
    )

    assert request.refinement_mode == "medium"


def test_request_rejects_unknown_refinement_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="low, medium"):
        SegmentationRequest.from_dict(
            {
                "video_uri": (tmp_path / "input.mp4").as_uri(),
                "output_uri": tmp_path.as_uri(),
                "refinement_mode": "high",
            }
        )


def test_request_applies_namespaced_config_and_preserves_extensions(tmp_path: Path) -> None:
    request = SegmentationRequest.from_dict(
        {
            "video_uri": (tmp_path / "input.mp4").as_uri(),
            "output_uri": tmp_path.as_uri(),
            "config": {
                "input": {"max_frames": 12},
                "inference": {
                    "batch_sizes": [32, 128, 64, 32],
                    "mixed_precision": False,
                },
                "refinement": {"mode": "medium"},
                "geometry": {
                    "simplify_tolerance": 1.25,
                    "min_area": 20,
                    "workers": 3,
                },
                "retry": {"attempts": 5, "backoff_seconds": 0.25},
                "delivery": {
                    "url": "http://next-worker.internal/results",
                    "payload": "reference",
                    "required": False,
                },
                "pipeline": {"next_task": "caption"},
            },
        }
    )

    assert request.max_frames == 12
    assert request.batch_sizes == (128, 64, 32)
    assert request.mixed_precision is False
    assert request.refinement_mode == "medium"
    assert request.simplify_tolerance == 1.25
    assert request.min_area == 20
    assert request.geometry_workers == 3
    assert request.request_attempts == 5
    assert request.retry_backoff_seconds == 0.25
    assert request.delivery.enabled
    assert request.delivery.payload == "reference"
    assert not request.delivery.required
    assert request.config["pipeline"] == {"next_task": "caption"}


def test_legacy_top_level_fields_override_nested_config(tmp_path: Path) -> None:
    request = SegmentationRequest.from_dict(
        {
            "video_uri": (tmp_path / "input.mp4").as_uri(),
            "output_uri": tmp_path.as_uri(),
            "min_area": 99,
            "refinement_mode": "low",
            "config": {
                "geometry": {"min_area": 10},
                "refinement": {"mode": "medium"},
            },
        }
    )

    assert request.min_area == 99
    assert request.refinement_mode == "low"


def test_persisted_config_redacts_delivery_credentials() -> None:
    redacted = _redact_config(
        {
            "delivery": {
                "headers": {
                    "Authorization": "Bearer secret",
                    "X-Tenant": "tenant-1",
                }
            }
        }
    )

    assert redacted["delivery"]["headers"]["Authorization"] == "<redacted>"
    assert redacted["delivery"]["headers"]["X-Tenant"] == "tenant-1"


def test_delivery_failure_does_not_repeat_gpu_processing(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService.__new__(SegmentationService)
    service.config = WorkerConfig(request_attempts=3, retry_backoff_seconds=0)
    calls = 0

    def fail_delivery(_request):
        nonlocal calls
        calls += 1
        raise DeliveryError("downstream unavailable")

    monkeypatch.setattr(service, "_process", fail_delivery)
    request = SegmentationRequest.from_dict(
        {
            "video_uri": (tmp_path / "input.mp4").as_uri(),
            "output_uri": tmp_path.as_uri(),
        }
    )

    with pytest.raises(DeliveryError, match="downstream unavailable"):
        service._process_with_retries(request)

    assert calls == 1


def test_request_rejects_unsupported_uri(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file:// or s3://"):
        SegmentationRequest.from_dict(
            {
                "video_uri": "https://example.com/video.mp4",
                "output_uri": tmp_path.as_uri(),
            }
        )


def test_batch_sizes_are_unique_and_descending() -> None:
    assert WorkerConfig(batch_sizes=(1, 8, 4, 8)).batch_sizes == (8, 4, 1)


def test_default_batches_probe_requested_large_sizes_first() -> None:
    assert WorkerConfig().batch_sizes == (6, 4, 2, 1)


def test_worker_config_rejects_an_unbounded_or_invalid_request_queue() -> None:
    with pytest.raises(ValueError, match="max_pending_requests"):
        WorkerConfig(max_pending_requests=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_id", "unsafe/id", "may contain only"),
        ("min_area", True, "must be an integer"),
        ("max_frames", "12", "must be an integer"),
        ("simplify_tolerance", float("inf"), "must be finite"),
    ],
)
def test_request_rejects_ambiguous_or_unsafe_values(
    tmp_path: Path, field: str, value, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SegmentationRequest.from_dict(
            {
                "video_uri": (tmp_path / "input.mp4").as_uri(),
                "output_uri": tmp_path.as_uri(),
                field: value,
            }
        )


def test_duplicate_request_id_reuses_completed_result(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService("unused.pth", config=WorkerConfig())
    started = threading.Event()
    release = threading.Event()
    calls = 0

    monkeypatch.setattr(service, "_ensure_model", lambda: object())
    monkeypatch.setattr(service, "_ensure_refiner", lambda: object())

    def process(request):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=2)
        from care_ego.service import SegmentationResult

        return SegmentationResult(request.request_id, tmp_path / "job.json", 1, 1, 0.1)

    monkeypatch.setattr(service, "_process_with_retries", process)
    payload = {
        "request_id": "idempotent-job",
        "video_uri": (tmp_path / "input.mp4").as_uri(),
        "output_uri": tmp_path.as_uri(),
    }
    service.start()
    try:
        first = service.submit(payload)
        assert started.wait(timeout=2)
        assert service.submit(payload) is first
        release.set()
        assert first.result(timeout=2).request_id == "idempotent-job"
        assert service.submit(payload) is first
        assert calls == 1

        with pytest.raises(ValueError, match="already in use"):
            service.submit({**payload, "max_frames": 2})
    finally:
        release.set()
        service.stop()


def test_stop_without_wait_rejects_new_submissions(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService("unused.pth")
    monkeypatch.setattr(service, "_ensure_model", lambda: object())
    monkeypatch.setattr(service, "_ensure_refiner", lambda: object())
    service.start()
    service.stop(wait=False)

    with pytest.raises(RuntimeError, match="not running"):
        service.submit(
            {
                "video_uri": (tmp_path / "input.mp4").as_uri(),
                "output_uri": tmp_path.as_uri(),
            }
        )
    service.stop()


def test_required_delivery_failure_is_idempotently_cached(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService("unused.pth", config=WorkerConfig())
    calls = 0
    monkeypatch.setattr(service, "_ensure_model", lambda: object())
    monkeypatch.setattr(service, "_ensure_refiner", lambda: object())

    def process(_request):
        nonlocal calls
        calls += 1
        raise DeliveryError("downstream unavailable")

    monkeypatch.setattr(service, "_process_with_retries", process)
    payload = {
        "request_id": "delivery-failed-job",
        "video_uri": (tmp_path / "input.mp4").as_uri(),
        "output_uri": tmp_path.as_uri(),
    }
    service.start()
    try:
        first = service.submit(payload)
        with pytest.raises(DeliveryError):
            first.result(timeout=2)
        assert service.submit(payload) is first
        assert calls == 1
    finally:
        service.stop()


def test_frame_producer_always_wakes_consumer_after_unexpected_error(
    tmp_path: Path, monkeypatch
) -> None:
    frames: queue.Queue = queue.Queue()
    producer = _FrameProducer(tmp_path / "input.mp4", frames, None)

    def fail() -> None:
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(producer, "_run", fail)
    producer.run()

    assert isinstance(producer.error, RuntimeError)
    assert frames.get_nowait() is _FRAMES_DONE


def test_nvdec_partial_failure_does_not_fall_back_into_the_same_queue(
    tmp_path: Path, monkeypatch
) -> None:
    producer = _FrameProducer(tmp_path / "input.mp4", queue.Queue(), None)
    used_opencv = False

    def fail_after_output() -> None:
        producer._nvdec_frames_emitted = 1
        raise RuntimeError("decoder failed")

    def opencv_fallback() -> None:
        nonlocal used_opencv
        used_opencv = True

    monkeypatch.setenv("WAKE_VIDEO_DECODER", "nvdec")
    monkeypatch.setattr(producer, "_run_nvdec", fail_after_output)
    monkeypatch.setattr(producer, "_run_opencv", opencv_fallback)

    with pytest.raises(OSError, match="refusing an unsafe fallback"):
        producer._run()

    assert not used_opencv


def test_request_queue_rejects_overload_without_leaking_request_id(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService("unused.pth", config=WorkerConfig(max_pending_requests=1))
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(service, "_ensure_model", lambda: object())
    monkeypatch.setattr(service, "_ensure_refiner", lambda: object())

    def process(request):
        started.set()
        release.wait(timeout=2)
        from care_ego.service import SegmentationResult

        return SegmentationResult(request.request_id, tmp_path / "job.json", 1, 1, 0.1)

    monkeypatch.setattr(service, "_process_with_retries", process)
    payload = {
        "video_uri": (tmp_path / "input.mp4").as_uri(),
        "output_uri": tmp_path.as_uri(),
    }
    service.start()
    try:
        first = service.submit({**payload, "request_id": "first"})
        assert started.wait(timeout=2)
        second = service.submit({**payload, "request_id": "second"})
        with pytest.raises(ServiceOverloadedError, match="queue is full"):
            service.submit({**payload, "request_id": "third"})
        release.set()
        assert first.result(timeout=2).request_id == "first"
        assert second.result(timeout=2).request_id == "second"
        # The rejected id is not retained and may be submitted later.
        assert service.submit({**payload, "request_id": "third"}).result(timeout=2).request_id == "third"
    finally:
        release.set()
        service.stop()


def test_consumer_failure_resolves_queued_futures(tmp_path: Path, monkeypatch) -> None:
    service = SegmentationService("unused.pth")
    monkeypatch.setattr(service, "_ensure_model", lambda: object())
    monkeypatch.setattr(service, "_ensure_refiner", lambda: object())
    first_request = SegmentationRequest.from_dict(
        {
            "request_id": "first",
            "video_uri": (tmp_path / "first.mp4").as_uri(),
            "output_uri": tmp_path.as_uri(),
        }
    )
    second_request = SegmentationRequest.from_dict(
        {
            "request_id": "second",
            "video_uri": (tmp_path / "second.mp4").as_uri(),
            "output_uri": tmp_path.as_uri(),
        }
    )
    from concurrent.futures import Future

    first = Future()
    second = Future()
    service._requests.put(_RequestEnvelope(first_request, first))
    service._requests.put(_RequestEnvelope(second_request, second))
    monkeypatch.setattr(
        service,
        "_process_with_retries",
        lambda _request: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    service._thread = threading.current_thread()
    service._accepting_requests = True

    with pytest.raises(KeyboardInterrupt):
        service._serve()

    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        first.result()
    with pytest.raises(RuntimeError, match="stopped before processing"):
        second.result()
    assert not service.is_running
