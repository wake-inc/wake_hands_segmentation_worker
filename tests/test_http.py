from concurrent.futures import Future
from pathlib import Path

from care_ego.http import create_app
from care_ego.service import SegmentationResult


class FakeService:
    is_running = True
    pending_requests = 0

    def start(self):
        return self

    def stop(self, _wait=True):
        return None

    def submit(self, _value):
        future = Future()
        future.set_result(
            SegmentationResult(
                request_id="job-1",
                output_path=Path("/tmp/job-1.json"),
                frame_count=10,
                batch_size=128,
                elapsed_seconds=1.5,
            )
        )
        return future


def test_blocking_endpoint_returns_completed_result() -> None:
    client = create_app(FakeService()).test_client()

    response = client.post("/v1/segment", json={"video_uri": "ignored"})

    assert response.status_code == 200
    assert response.json["request_id"] == "job-1"
    assert response.json["batch_size"] == 128
    assert response.json["output_uri"] == Path("/tmp/job-1.json").resolve().as_uri()


def test_blocking_endpoint_returns_s3_output_without_temporary_path() -> None:
    class FakeS3Service(FakeService):
        def submit(self, _value):
            future = Future()
            future.set_result(
                SegmentationResult(
                    request_id="job-1",
                    output_path=None,
                    output_uri="s3://wake-test/jobs/job-1/results/job-1.json",
                    frame_count=10,
                    batch_size=8,
                    elapsed_seconds=1.5,
                )
            )
            return future

    response = (
        create_app(FakeS3Service()).test_client().post("/v1/segment", json={"video_uri": "ignored"})
    )

    assert response.status_code == 200
    assert "output_path" not in response.json
    assert response.json["output_uri"] == "s3://wake-test/jobs/job-1/results/job-1.json"


def test_stream_endpoint_returns_sse_result() -> None:
    client = create_app(FakeService()).test_client()

    response = client.post("/v1/segment/stream", json={"video_uri": "ignored"})

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert b"event: queued" in response.data
    assert b"event: result" in response.data


def test_schema_endpoint_returns_versioned_contract() -> None:
    client = create_app(FakeService()).test_client()

    response = client.get("/v1/schema")

    assert response.status_code == 200
    assert response.json["$id"] == "urn:wake-ai:schema:inference-result:1.0.0"
    assert "prediction" in response.json["$defs"]


def test_malformed_json_returns_bad_request() -> None:
    client = create_app(FakeService()).test_client()

    response = client.post("/v1/segment", data="{", content_type="application/json")

    assert response.status_code == 400
    assert response.json["type"] == "BadRequest"


def test_non_json_body_returns_unsupported_media_type() -> None:
    client = create_app(FakeService()).test_client()

    response = client.post("/v1/segment", data="hello", content_type="text/plain")

    assert response.status_code == 415
    assert response.json["type"] == "UnsupportedMediaType"


def test_oversized_json_body_returns_payload_too_large() -> None:
    client = create_app(FakeService()).test_client()

    response = client.post(
        "/v1/segment",
        data='{"padding":"' + ("x" * (1024 * 1024)) + '"}',
        content_type="application/json",
    )

    assert response.status_code == 413
    assert response.json["type"] == "RequestEntityTooLarge"
