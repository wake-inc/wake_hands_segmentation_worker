"""Flask application for the long-running segmentation worker."""

from __future__ import annotations

import atexit
import json
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException, ServiceUnavailable

from . import server_config
from .schema import json_schema
from .service import SegmentationResult, SegmentationService, WorkerConfig

LOGGER = logging.getLogger(__name__)


def _result_payload(result: SegmentationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["output_path"] = str(result.output_path)
    return payload


def _sse(event: str, value: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(value, separators=(',', ':'))}\n\n"


def create_app(
    service: SegmentationService | None = None,
) -> Flask:
    """Create the application and start its persistent GPU service."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = server_config.REQUEST_MAX_BYTES
    runtime = service or SegmentationService(
        server_config.CHECKPOINT_PATH,
        device=server_config.DEVICE,
        config=WorkerConfig(
            batch_sizes=server_config.BATCH_SIZES,
            mixed_precision=server_config.MIXED_PRECISION,
            geometry_workers=server_config.GEOMETRY_WORKERS,
            request_attempts=server_config.REQUEST_ATTEMPTS,
            retry_backoff_seconds=server_config.RETRY_BACKOFF_SECONDS,
            completed_request_cache_size=server_config.COMPLETED_REQUEST_CACHE_SIZE,
            refinement_mode=server_config.REFINEMENT_MODE,
            refinement_model_directory=server_config.CASCADEPSP_MODEL_DIRECTORY,
            refinement_allow_download=server_config.CASCADEPSP_ALLOW_DOWNLOAD,
        ),
        model_metadata=server_config.MODEL_METADATA,
    )
    runtime.start()
    app.extensions["segmentation_service"] = runtime
    atexit.register(runtime.stop, False)

    def ensure_running() -> None:
        if not runtime.is_running:
            LOGGER.error("Request consumer stopped unexpectedly; restarting it")
            try:
                runtime.start()
            except RuntimeError as error:
                raise ServiceUnavailable("Segmentation service is restarting") from error

    def submit_request():
        value = request.get_json(silent=False)
        if not isinstance(value, dict):
            raise TypeError("Request body must be a JSON object")
        ensure_running()
        return runtime.submit(value)

    @app.get("/health/live")
    def live():
        return jsonify({"status": "alive"})

    @app.get("/health/ready")
    def ready():
        status = 200 if runtime.is_running else 503
        return (
            jsonify(
                {
                    "status": "ready" if runtime.is_running else "not-ready",
                    "pending_requests": runtime.pending_requests,
                }
            ),
            status,
        )

    @app.get("/v1/schema")
    def result_schema():
        return jsonify(json_schema())

    @app.post("/v1/segment")
    def segment():
        """Keep the HTTP request open without a service-side timeout."""
        try:
            result = submit_request().result(timeout=None)
            return jsonify(_result_payload(result))
        except HTTPException as error:
            return jsonify({"error": error.description, "type": type(error).__name__}), error.code
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error), "type": type(error).__name__}), 400
        except Exception as error:
            LOGGER.exception("Segmentation request failed")
            return jsonify({"error": str(error), "type": type(error).__name__}), 500

    @app.post("/v1/segment/stream")
    def segment_stream():
        """Return SSE heartbeats until the queued request completes."""
        try:
            future = submit_request()
        except HTTPException as error:
            return jsonify({"error": error.description, "type": type(error).__name__}), error.code
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error), "type": type(error).__name__}), 400

        @stream_with_context
        def events():
            yield _sse("queued", {"status": "queued"})
            while True:
                try:
                    result = future.result(timeout=server_config.SSE_HEARTBEAT_SECONDS)
                except FutureTimeout:
                    yield _sse(
                        "heartbeat",
                        {
                            "status": "processing",
                            "pending_requests": runtime.pending_requests,
                        },
                    )
                    continue
                except Exception as error:
                    LOGGER.exception("Streaming segmentation request failed")
                    yield _sse(
                        "error",
                        {"error": str(error), "type": type(error).__name__},
                    )
                    return
                yield _sse("result", _result_payload(result))
                return

        return Response(
            events(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
