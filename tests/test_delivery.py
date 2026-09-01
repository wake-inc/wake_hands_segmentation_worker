from pathlib import Path

import pytest
import requests

from care_ego.delivery import DeliveryConfig, HttpResultDelivery


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = iter(statuses)
        self.calls: list[dict] = []
        self.responses: list[FakeResponse] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = FakeResponse(next(self.statuses))
        self.responses.append(response)
        return response


def _document() -> dict:
    return {
        "schema": {"name": "wake-ai/inference-result", "version": "1.0.0"},
        "request": {"id": "job-1"},
        "outputs": [],
    }


def test_delivery_is_disabled_by_default(tmp_path: Path) -> None:
    session = FakeSession([])

    receipt = HttpResultDelivery(session).deliver(
        DeliveryConfig(), _document(), (tmp_path / "result.json").as_uri(), "job-1"
    )

    assert receipt.status == "disabled"
    assert session.calls == []


def test_document_delivery_retries_without_changing_payload(tmp_path: Path) -> None:
    session = FakeSession([503, 200])
    config = DeliveryConfig.from_mapping(
        {
            "url": "https://next.internal/results",
            "headers": {"Authorization": "Bearer secret"},
            "attempts": 2,
            "backoff_seconds": 0,
        }
    )

    receipt = HttpResultDelivery(session).deliver(
        config, _document(), (tmp_path / "result.json").as_uri(), "job-1"
    )

    assert receipt.status == "delivered"
    assert receipt.attempts == 2
    assert session.calls[0]["json"] == _document()
    assert session.calls[0]["headers"]["Idempotency-Key"] == "job-1"
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert all(response.closed for response in session.responses)


def test_reference_delivery_sends_file_uri(tmp_path: Path) -> None:
    session = FakeSession([202])
    output = tmp_path / "result.json"
    config = DeliveryConfig.from_mapping(
        {
            "url": "http://next.internal/events",
            "payload": "reference",
        }
    )

    receipt = HttpResultDelivery(session).deliver(
        config, _document(), output.resolve().as_uri(), "job-1"
    )

    assert receipt.status_code == 202
    assert session.calls[0]["json"]["event"] == "inference.completed"
    assert session.calls[0]["json"]["result"]["uri"] == output.resolve().as_uri()


def test_reference_delivery_preserves_s3_uri() -> None:
    session = FakeSession([202])
    output_uri = "s3://wake-test/jobs/job-1/result.json"
    config = DeliveryConfig.from_mapping(
        {
            "url": "http://next.internal/events",
            "payload": "reference",
        }
    )

    HttpResultDelivery(session).deliver(config, _document(), output_uri, "job-1")

    assert session.calls[0]["json"]["result"]["uri"] == output_uri


def test_delivery_config_supports_explicit_off_switch() -> None:
    config = DeliveryConfig.from_mapping({"enabled": False})

    assert not config.enabled
    assert config.url is None


def test_delivery_does_not_retry_permanent_client_error(tmp_path: Path) -> None:
    session = FakeSession([400, 200])
    config = DeliveryConfig.from_mapping(
        {"url": "https://next.internal/results", "attempts": 2, "backoff_seconds": 0}
    )

    receipt = HttpResultDelivery(session).deliver(
        config, _document(), (tmp_path / "result.json").as_uri(), "job-1"
    )

    assert receipt.status == "failed"
    assert receipt.status_code == 400
    assert receipt.attempts == 1
    assert len(session.calls) == 1


@pytest.mark.parametrize("field", ["attempts", "timeout_seconds", "backoff_seconds"])
def test_delivery_rejects_boolean_numeric_values(field: str) -> None:
    with pytest.raises(ValueError):
        DeliveryConfig.from_mapping({"url": "https://next.internal/results", field: True})


@pytest.mark.parametrize(
    ("field", "value"),
    [("timeout_seconds", 301), ("backoff_seconds", 61)],
)
def test_delivery_rejects_unbounded_waits(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be between"):
        DeliveryConfig.from_mapping({"url": "https://next.internal/results", field: value})
