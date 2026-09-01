"""URI-backed input and result storage for inference requests."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlparse

_S3_SAFE_KEY_CHARACTERS = "/!$&'()*+,;=:@-._~"


class S3Client(Protocol):
    """Subset of the boto3 S3 client used by the worker."""

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

    def download_file(self, bucket: str, key: str, filename: str) -> Any: ...

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, str] | None = None,
    ) -> Any: ...


def file_uri_path(uri: str) -> Path:
    """Resolve a local file URI after rejecting remote authorities."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected a file:// URI, got: {uri}")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"Remote file URI authorities are unsupported: {uri}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"file:// URIs cannot contain a query or fragment: {uri}")
    return Path(unquote(parsed.path)).resolve()


def s3_bucket_key(uri: str, *, require_key: bool = True) -> tuple[str, str]:
    """Parse a canonical s3://bucket/key URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    if not parsed.netloc:
        raise ValueError(f"S3 URI must include a bucket: {uri}")
    if parsed.username is not None or parsed.password is not None or parsed.port is not None:
        raise ValueError(f"S3 URI cannot contain credentials or a port: {uri}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"S3 URI cannot contain a query or fragment: {uri}")
    key = unquote(parsed.path.lstrip("/"))
    if require_key and (not key or key.endswith("/")):
        raise ValueError(f"S3 input URI must identify an object: {uri}")
    return parsed.netloc, key


def validate_input_uri(uri: str) -> None:
    """Validate a supported input URI without accessing its object."""
    scheme = urlparse(uri).scheme
    if scheme == "file":
        file_uri_path(uri)
    elif scheme == "s3":
        s3_bucket_key(uri)
    else:
        raise ValueError(f"Input URI must use file:// or s3://, got: {uri}")


def validate_output_uri(uri: str) -> None:
    """Validate a supported result-directory URI."""
    scheme = urlparse(uri).scheme
    if scheme == "file":
        file_uri_path(uri)
    elif scheme == "s3":
        s3_bucket_key(uri, require_key=False)
    else:
        raise ValueError(f"Output URI must use file:// or s3://, got: {uri}")


def result_uri(output_uri: str, request_id: str) -> str:
    """Build the result object URI below a request's output directory."""
    scheme = urlparse(output_uri).scheme
    filename = f"{request_id}.json"
    if scheme == "file":
        return (file_uri_path(output_uri) / filename).resolve().as_uri()
    if scheme == "s3":
        bucket, prefix = s3_bucket_key(output_uri, require_key=False)
        key = f"{prefix.rstrip('/')}/{filename}" if prefix else filename
        encoded_key = quote(key, safe=_S3_SAFE_KEY_CHARACTERS)
        return f"s3://{bucket}/{encoded_key}"
    raise ValueError(f"Output URI must use file:// or s3://, got: {output_uri}")


class UriStorage:
    """Stage file:// or s3:// inputs and publish result objects."""

    def __init__(
        self,
        s3_client: S3Client | None = None,
        *,
        s3_endpoint_url: str | None = None,
        s3_region_name: str | None = None,
        max_input_bytes: int | None = None,
    ) -> None:
        if max_input_bytes is not None and max_input_bytes < 1:
            raise ValueError("max_input_bytes must be positive when configured")
        self._s3_client = s3_client
        self._s3_endpoint_url = s3_endpoint_url
        self._s3_region_name = s3_region_name
        self._max_input_bytes = max_input_bytes

    def _check_input_size(self, size: int, uri: str) -> None:
        if self._max_input_bytes is not None and size > self._max_input_bytes:
            raise ValueError(
                f"Input {uri} is {size} bytes, exceeding the configured "
                f"limit of {self._max_input_bytes} bytes"
            )

    def _s3(self) -> S3Client:
        if self._s3_client is None:
            import boto3
            from botocore.config import Config

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self._s3_endpoint_url,
                region_name=self._s3_region_name,
                config=Config(
                    connect_timeout=10,
                    read_timeout=300,
                    tcp_keepalive=True,
                    retries={"max_attempts": 10, "mode": "adaptive"},
                ),
            )
        return self._s3_client

    @staticmethod
    def _storage_error(operation: str, uri: str, error: Exception) -> OSError:
        response = getattr(error, "response", {})
        code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
        if code in {"404", "NoSuchBucket", "NoSuchKey", "NotFound"}:
            return FileNotFoundError(uri)
        return OSError(f"Could not {operation} {uri}: {error}")

    def stage_input(self, uri: str, directory: Path) -> Path:
        """Copy or download one input object into an isolated directory."""
        directory.mkdir(parents=True, exist_ok=True)
        scheme = urlparse(uri).scheme
        if scheme == "file":
            source = file_uri_path(uri)
            if not source.is_file():
                raise FileNotFoundError(source)
            self._check_input_size(source.stat().st_size, uri)
            destination = directory / source.name
            shutil.copy2(source, destination)
            return destination
        if scheme == "s3":
            bucket, key = s3_bucket_key(uri)
            destination = directory / PurePosixPath(key).name
            try:
                metadata = self._s3().head_object(Bucket=bucket, Key=key)
            except Exception as error:  # noqa: BLE001 - normalized for request retries
                raise self._storage_error("inspect", uri, error) from error
            self._check_input_size(int(metadata["ContentLength"]), uri)
            try:
                self._s3().download_file(bucket, key, str(destination))
            except Exception as error:  # noqa: BLE001 - normalized for request retries
                raise self._storage_error("download", uri, error) from error
            return destination
        raise ValueError(f"Input URI must use file:// or s3://, got: {uri}")

    @staticmethod
    def result_path(uri: str, temporary_directory: Path) -> Path:
        """Return the path used to assemble a result before publication."""
        scheme = urlparse(uri).scheme
        if scheme == "file":
            path = file_uri_path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        if scheme == "s3":
            _, key = s3_bucket_key(uri)
            return temporary_directory / PurePosixPath(key).name
        raise ValueError(f"Result URI must use file:// or s3://, got: {uri}")

    def publish_result(self, path: Path, uri: str) -> None:
        """Upload an assembled result; local results are already in place."""
        scheme = urlparse(uri).scheme
        if scheme == "file":
            if path.resolve() != file_uri_path(uri):
                raise ValueError("Local result path does not match its file URI")
            return
        if scheme == "s3":
            bucket, key = s3_bucket_key(uri)
            try:
                self._s3().upload_file(
                    str(path),
                    bucket,
                    key,
                    ExtraArgs={"ContentType": "application/json"},
                )
            except Exception as error:  # noqa: BLE001 - normalized for request retries
                raise self._storage_error("upload", uri, error) from error
            return
        raise ValueError(f"Result URI must use file:// or s3://, got: {uri}")
