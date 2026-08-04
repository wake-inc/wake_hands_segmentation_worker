# Request configuration and service chaining

Every segmentation request may include a namespaced `config` object. Known
sections change worker execution. Unknown sections are accepted and preserved
in the universal result document, allowing orchestration metadata to travel
through heterogeneous CNN, VLM, LLM, and transformer pipelines.

## Complete example

```json
{
  "request_id": "segment-job-001",
  "video_uri": "file:///data/input.mp4",
  "output_uri": "file:///data/results",
  "config": {
    "input": {"max_frames": 300},
    "inference": {
      "batch_sizes": [128, 64, 32],
      "mixed_precision": true
    },
    "refinement": {"mode": "low"},
    "geometry": {
      "simplify_tolerance": 2.0,
      "min_area": 64,
      "workers": 8
    },
    "retry": {"attempts": 3, "backoff_seconds": 1.0},
    "delivery": {
      "enabled": true,
      "url": "http://next-worker:8080/v1/results",
      "payload": "document",
      "headers": {"Authorization": "Bearer service-token"},
      "timeout_seconds": 30,
      "attempts": 3,
      "backoff_seconds": 1.0,
      "required": true
    },
    "pipeline": {
      "trace_id": "pipeline-42",
      "step": 1,
      "next_task": "caption_hands"
    }
  }
}
```

The custom `pipeline` section is not interpreted by this worker. It is copied
into `request.config` in the saved result and forwarded to the next service.

## Active configuration sections

### `input`

| Field | Type | Meaning |
| --- | --- | --- |
| `max_frames` | positive integer or null | Limit decoded frames; null processes the whole video. |

### `inference`

| Field | Type | Meaning |
| --- | --- | --- |
| `batch_sizes` | positive integer array | Request-specific adaptive GPU batch candidates. Values are deduplicated and sorted descending. |
| `mixed_precision` | boolean or null | Enable or disable CUDA AMP; null uses the worker default. |

### `refinement`

| Field | Type | Meaning |
| --- | --- | --- |
| `mode` | `low` or `medium` | Select the CascadePSP refinement profile. |

### `geometry`

| Field | Type | Meaning |
| --- | --- | --- |
| `simplify_tolerance` | non-negative number | Shapely polygon simplification tolerance. |
| `min_area` | positive integer | Minimum connected-component area. |
| `workers` | positive integer | Request-specific CPU geometry thread count. |

### `retry`

| Field | Type | Meaning |
| --- | --- | --- |
| `attempts` | positive integer | Whole-request attempts for transient I/O or inference errors. |
| `backoff_seconds` | non-negative number | Initial exponential retry delay. |

Legacy top-level fields—`max_frames`, `refinement_mode`, `min_area`, and
`simplify_tolerance`—remain supported. If a value appears at both levels, the
top-level value wins.

The result contains both the redacted original configuration at
`request.config` and resolved values at `runtime.effective_config`.

## Optional downstream delivery

Delivery is off when the section is absent or explicitly disabled. A non-empty
delivery section defaults `enabled` to `true`, so providing only `url` is
sufficient to turn it on. With delivery off, no network request is made:

```json
{"config": {"delivery": {"enabled": false}}}
```

When enabled, `url` must be an `http://` or `https://` endpoint.

### Full-document mode

`payload: "document"` sends the complete universal result document as the POST
body. This is the default and is suitable when the next service needs geometry
or model output immediately.

```json
{
  "config": {
    "delivery": {
      "url": "http://next-worker:8080/v1/results",
      "payload": "document"
    }
  }
}
```

### Reference mode

`payload: "reference"` sends a small completion event containing the atomic
output file URI. Use this when workers share a mounted filesystem and the full
result is large.

```json
{
  "event": "inference.completed",
  "schema": {"name": "wake-ai/inference-result", "version": "1.0.0"},
  "request": {"id": "segment-job-001", "config": {}},
  "result": {"uri": "file:///data/results/segment-job-001.json"}
}
```

### Reliability behavior

1. Inference and refinement complete first.
2. The universal result is written atomically.
3. Delivery is attempted independently with exponential backoff for network
   errors, HTTP 408/425/429, and HTTP 5xx responses.
4. Every POST includes `Idempotency-Key: <request_id>`.
5. The delivery receipt is saved under `runtime.delivery`.
6. A delivery failure never repeats GPU inference.

Other HTTP 4xx responses and redirects are permanent delivery failures and are
not retried. Redirects are deliberately not followed, so the configured URL is
the actual destination.

`required: true` makes a final delivery failure fail the HTTP/SSE request while
leaving the completed result file in place. `required: false` is best-effort:
the segmentation request succeeds and returns a failed delivery receipt.
Required delivery failure returns HTTP 500 from the blocking endpoint or an
SSE `error` event from the streaming endpoint.

Successful HTTP completion metadata includes:

```json
{
  "delivery": {
    "enabled": true,
    "status": "delivered",
    "attempts": 1,
    "destination": "http://next-worker:8080/v1/results",
    "status_code": 202
  }
}
```

### Credentials

Custom delivery headers are used for the outgoing request. Values under common
credential keys such as `Authorization`, `token`, `password`, `secret`, and
`api_key` are replaced with `<redacted>` before request configuration is saved
or forwarded. URLs containing embedded username/password credentials are
rejected.

The receiving endpoint should treat `Idempotency-Key` as the stable delivery
identity and return a successful HTTP status only after it has safely accepted
the document or reference. Pipeline loop detection and incrementing custom
fields such as `pipeline.step` remain the orchestrator's responsibility.
