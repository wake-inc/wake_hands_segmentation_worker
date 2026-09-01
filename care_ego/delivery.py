"""Optional service-to-service result delivery."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import requests

DeliveryPayload = Literal["document", "reference"]
_MAX_DELIVERY_ATTEMPTS = 20
_MAX_DELIVERY_TIMEOUT_SECONDS = 300.0
_MAX_DELIVERY_BACKOFF_SECONDS = 60.0
_MAX_DELIVERY_RETRY_DELAY_SECONDS = 60.0


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class DeliveryConfig:
    """Validated downstream delivery settings supplied with a request."""

    enabled: bool = False
    url: str | None = None
    payload: DeliveryPayload = "document"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    attempts: int = 3
    backoff_seconds: float = 1.0
    required: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> DeliveryConfig:
        if not value:
            return cls()
        allowed = {
            "enabled",
            "url",
            "payload",
            "headers",
            "timeout_seconds",
            "attempts",
            "backoff_seconds",
            "required",
        }
        unknown = value.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown config.delivery fields: {sorted(unknown)}")

        enabled = value.get("enabled", True)
        required = value.get("required", True)
        if not isinstance(enabled, bool):
            raise ValueError("config.delivery.enabled must be a boolean")
        if not isinstance(required, bool):
            raise ValueError("config.delivery.required must be a boolean")

        url_value = value.get("url")
        if url_value is not None and not isinstance(url_value, str):
            raise ValueError("config.delivery.url must be a string")
        url = url_value
        if enabled:
            if not url:
                raise ValueError("config.delivery.url is required when delivery is enabled")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("config.delivery.url must be an http:// or https:// URL")
            if parsed.username or parsed.password:
                raise ValueError("config.delivery.url cannot contain embedded credentials")

        payload = value.get("payload", "document")
        if not isinstance(payload, str):
            raise ValueError("config.delivery.payload must be a string")
        if payload not in {"document", "reference"}:
            raise ValueError("config.delivery.payload must be document or reference")

        headers_value = value.get("headers", {})
        if not isinstance(headers_value, Mapping) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in headers_value.items()
        ):
            raise ValueError("config.delivery.headers must map strings to strings")

        timeout_seconds = _number(
            value.get("timeout_seconds", 30.0), "config.delivery.timeout_seconds"
        )
        attempts = _integer(value.get("attempts", 3), "config.delivery.attempts")
        backoff_seconds = _number(
            value.get("backoff_seconds", 1.0), "config.delivery.backoff_seconds"
        )
        if not 0 < timeout_seconds <= _MAX_DELIVERY_TIMEOUT_SECONDS:
            raise ValueError(
                "config.delivery.timeout_seconds must be between 0 and "
                f"{_MAX_DELIVERY_TIMEOUT_SECONDS:g}"
            )
        if not 1 <= attempts <= _MAX_DELIVERY_ATTEMPTS:
            raise ValueError(
                f"config.delivery.attempts must be between 1 and {_MAX_DELIVERY_ATTEMPTS}"
            )
        if not 0 <= backoff_seconds <= _MAX_DELIVERY_BACKOFF_SECONDS:
            raise ValueError(
                "config.delivery.backoff_seconds must be between 0 and "
                f"{_MAX_DELIVERY_BACKOFF_SECONDS:g}"
            )

        return cls(
            enabled=enabled,
            url=url,
            payload=payload,  # type: ignore[arg-type]
            headers=dict(headers_value),
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            required=required,
        )


@dataclass(frozen=True)
class DeliveryReceipt:
    enabled: bool
    status: Literal["disabled", "delivered", "failed"]
    attempts: int = 0
    destination: str | None = None
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class ResultDelivery(Protocol):
    def deliver(
        self,
        config: DeliveryConfig,
        document: Mapping[str, Any],
        output_uri: str,
        request_id: str,
    ) -> DeliveryReceipt: ...


class HttpResultDelivery:
    """POST completed results with bounded delivery-only retries."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def deliver(
        self,
        config: DeliveryConfig,
        document: Mapping[str, Any],
        output_uri: str,
        request_id: str,
    ) -> DeliveryReceipt:
        if not config.enabled:
            return DeliveryReceipt(enabled=False, status="disabled")
        if not config.url:
            raise ValueError("Enabled result delivery requires a destination URL")

        payload: Mapping[str, Any]
        if config.payload == "document":
            payload = document
        else:
            payload = {
                "event": "inference.completed",
                "schema": document["schema"],
                "request": document["request"],
                "result": {"uri": output_uri},
            }

        headers = {
            "Accept": "application/json",
            "Idempotency-Key": request_id,
            **config.headers,
        }
        last_error: Exception | None = None
        last_status: int | None = None
        attempts_made = 0
        for attempt in range(1, config.attempts + 1):
            attempts_made = attempt
            retryable = True
            response = None
            try:
                response = self.session.post(
                    config.url,
                    json=payload,
                    headers=headers,
                    timeout=config.timeout_seconds,
                    allow_redirects=False,
                )
                last_status = response.status_code
                if 200 <= response.status_code < 300:
                    return DeliveryReceipt(
                        enabled=True,
                        status="delivered",
                        attempts=attempt,
                        destination=config.url,
                        status_code=response.status_code,
                    )
                last_error = requests.HTTPError(f"HTTP {response.status_code}")
                retryable = response.status_code in {408, 425, 429} or response.status_code >= 500
            except requests.RequestException as error:
                last_error = error
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if close is not None:
                        close()
            if not retryable or attempt == config.attempts:
                break
            if config.backoff_seconds:
                time.sleep(
                    min(
                        _MAX_DELIVERY_RETRY_DELAY_SECONDS,
                        config.backoff_seconds * (2 ** (attempt - 1)),
                    )
                )

        return DeliveryReceipt(
            enabled=True,
            status="failed",
            attempts=attempts_made,
            destination=config.url,
            status_code=last_status,
            error=str(last_error),
        )


class DeliveryError(Exception):
    """Required downstream delivery failed after inference was persisted."""
