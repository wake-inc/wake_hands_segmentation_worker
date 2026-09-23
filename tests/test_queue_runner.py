from __future__ import annotations

from typing import Any

import pytest

from care_ego.queue_runner import (
    _RunnerLifecycle,
    _ServiceProvider,
    _batch_progress_callback,
    _debug_options,
    _processor_resolver,
)
from wake_job_queue.consumer import PermanentJobError


def test_debug_overrides_are_disabled_by_default() -> None:
    with pytest.raises(PermanentJobError, match="disabled"):
        _debug_options({"debug": {}}, enabled=False)


def test_debug_overrides_only_allow_bounded_inference_settings() -> None:
    options = _debug_options(
        {
            "debug": {
                "serviceLifetime": "perJob",
                "inference": {
                    "batchSizes": [64, 32],
                    "mixedPrecision": False,
                    "geometryWorkers": 4,
                    "refinementMode": "medium",
                },
            }
        },
        enabled=True,
    )

    assert options.service_lifetime == "perJob"
    assert options.request_config == {
        "inference": {"batch_sizes": [64, 32], "mixed_precision": False},
        "geometry": {"workers": 4},
        "refinement": {"mode": "medium"},
    }


@pytest.mark.parametrize(
    "debug",
    [
        {"checkpoint": "/untrusted.pth"},
        {"inference": {"device": "cpu"}},
        {"inference": {"geometryWorkers": 65}},
    ],
)
def test_debug_overrides_reject_unbounded_or_unsupported_values(debug: dict[str, Any]) -> None:
    with pytest.raises(PermanentJobError):
        _debug_options({"debug": debug}, enabled=True)


def test_runner_lifecycle_waits_for_idle_timeout_or_job_limit() -> None:
    lifecycle = _RunnerLifecycle(max_jobs=2, idle_exit_seconds=10, last_delivery_at=100)

    assert not lifecycle.should_stop(now=109)
    assert lifecycle.should_stop(now=110)

    lifecycle.record_delivery(now=111)
    assert not lifecycle.should_stop(now=112)
    lifecycle.record_delivery(now=113)
    assert lifecycle.should_stop(now=113)


def test_processor_resolver_only_exposes_track_hands() -> None:
    resolver = _processor_resolver(_ServiceProvider(debug_task_overrides=False))

    assert callable(resolver("track_hands", 1))
    with pytest.raises(PermanentJobError, match="unsupported GPU job"):
        resolver("track_hands", 2)


def test_batch_progress_renews_state_after_each_completed_batch() -> None:
    calls: list[dict[str, Any]] = []

    class State:
        def heartbeat(self, **value: Any) -> None:
            calls.append(value)

    callback = _batch_progress_callback(State(), total_frames=900)  # type: ignore[arg-type]
    callback(128, 64, 4.5)

    assert calls[0]["estimated_next_batch_seconds"] == 4.5
    assert calls[0]["progress"] == {
        "phase": "inference",
        "processedFrames": 128,
        "batchSize": 64,
        "totalFrames": 900,
    }
