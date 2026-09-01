from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from care_ego.storage import (
    UriStorage,
    result_uri,
    s3_bucket_key,
    validate_input_uri,
    validate_output_uri,
)


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[dict] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.uploads.append({"bucket": bucket, "key": key, "extra_args": ExtraArgs})


def test_s3_input_is_staged_and_result_is_uploaded(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.objects[("wake-test", "jobs/job-1/input.mp4")] = b"video"
    storage = UriStorage(client)

    staged = storage.stage_input("s3://wake-test/jobs/job-1/input.mp4", tmp_path / "stage")
    destination_uri = result_uri("s3://wake-test/jobs/job-1/results/", "job-1")
    output = storage.result_path(destination_uri, tmp_path / "result")
    output.parent.mkdir(parents=True)
    output.write_text('{"ok":true}', encoding="utf-8")
    storage.publish_result(output, destination_uri)

    assert staged.read_bytes() == b"video"
    assert destination_uri == "s3://wake-test/jobs/job-1/results/job-1.json"
    assert client.objects[("wake-test", "jobs/job-1/results/job-1.json")] == b'{"ok":true}'
    assert client.uploads[0]["extra_args"] == {"ContentType": "application/json"}


def test_file_storage_preserves_local_behavior(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output_directory = tmp_path / "results"
    storage = UriStorage()

    staged = storage.stage_input(source.as_uri(), tmp_path / "stage")
    destination_uri = result_uri(output_directory.as_uri(), "job-1")
    destination = storage.result_path(destination_uri, tmp_path / "unused")
    destination.write_text("{}", encoding="utf-8")
    storage.publish_result(destination, destination_uri)

    assert staged.read_bytes() == b"video"
    assert destination == output_directory / "job-1.json"
    assert destination.read_text(encoding="utf-8") == "{}"


def test_input_size_limit_is_checked_before_s3_download(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.objects[("wake-test", "jobs/input.mp4")] = b"12345"
    storage = UriStorage(client, max_input_bytes=4)

    with pytest.raises(ValueError, match="exceeding the configured limit"):
        storage.stage_input("s3://wake-test/jobs/input.mp4", tmp_path)


@pytest.mark.parametrize(
    "uri",
    [
        "s3:///missing-bucket.mp4",
        "s3://bucket/path/",
        "https://bucket.s3.amazonaws.com/input.mp4",
    ],
)
def test_invalid_input_uris_are_rejected(uri: str) -> None:
    with pytest.raises(ValueError):
        validate_input_uri(uri)


def test_s3_output_allows_bucket_root_and_escapes_key() -> None:
    validate_output_uri("s3://wake-test")

    uri = result_uri("s3://wake-test/results with spaces", "job-1")

    assert uri == "s3://wake-test/results%20with%20spaces/job-1.json"
    assert s3_bucket_key(uri) == ("wake-test", "results with spaces/job-1.json")


def test_s3_client_uses_configured_compatible_endpoint(monkeypatch) -> None:
    calls = []
    client = object()

    def create_client(service_name: str, **kwargs):
        calls.append((service_name, kwargs))
        return client

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=create_client))
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=FakeConfig))
    storage = UriStorage(
        s3_endpoint_url="https://storage.eu-north1.nebius.cloud",
        s3_region_name="eu-north1",
    )

    assert storage._s3() is client
    assert calls[0][0] == "s3"
    assert calls[0][1]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert calls[0][1]["region_name"] == "eu-north1"
    assert calls[0][1]["config"].kwargs == {
        "connect_timeout": 10,
        "read_timeout": 300,
        "tcp_keepalive": True,
        "retries": {"max_attempts": 10, "mode": "adaptive"},
    }


def test_s3_client_preserves_aws_sdk_defaults(monkeypatch) -> None:
    calls = []
    client = object()

    def create_client(service_name: str, **kwargs):
        calls.append((service_name, kwargs))
        return client

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=create_client))
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=FakeConfig))

    assert UriStorage()._s3() is client
    assert calls[0][0] == "s3"
    assert calls[0][1]["endpoint_url"] is None
    assert calls[0][1]["region_name"] is None
    assert calls[0][1]["config"].kwargs["retries"] == {
        "max_attempts": 10,
        "mode": "adaptive",
    }
