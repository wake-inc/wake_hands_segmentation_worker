# Universal result format

The worker writes every completed inference result as a JSON document using the
`wake-ai/inference-result` contract. The current schema version is `1.0.0`.
The canonical machine-readable JSON Schema is defined in
[`care_ego/schema.py`](../care_ego/schema.py) and is also returned by:

```http
GET /v1/schema
```

The envelope is architecture-neutral. CNN, VLM, LLM, transformer, and hybrid
workers use the same structure; their modalities, tasks, prediction types, and
values differ.

`request.config` preserves orchestration and execution configuration so results
can move between services without losing pipeline context. Credential values
are redacted before persistence or forwarding.

## Top-level document

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema` | object | yes | Contract name and semantic version. |
| `request` | object | yes | Stable request identity and redacted request configuration. |
| `created_at` | string | yes | UTC ISO 8601 envelope-creation timestamp. |
| `model` | object | yes | Model and checkpoint provenance. |
| `inputs` | array | yes | One or more model inputs. |
| `outputs` | array | yes | One or more typed output streams. |
| `runtime` | object | no | Worker timing, usage, and batching data. |

### Model

```json
{
  "id": "care-ego-cascadepsp",
  "version": "1.0",
  "architecture": "CaRe-Ego segmentation followed by CascadePSP refinement",
  "families": ["cnn", "transformer"],
  "framework": "pytorch",
  "checkpoint": "care_ego_best_miou_weights.pth",
  "attributes": {
    "stages": [
      {"id": "care-ego", "task": "semantic_segmentation"},
      {"id": "cascadepsp", "task": "mask_refinement"}
    ]
  }
}
```

`families` is a non-exclusive list. For example, a vision-language model may
declare `["vlm", "llm", "transformer"]`.

### Inputs

Every input has an ID, modality, and source. URI-backed and inline values share
the same shape, allowing VLM requests to carry both image and text inputs.

```json
[
  {
    "id": "image",
    "modality": "image",
    "source": {"type": "uri", "value": "file:///data/image.jpg"}
  },
  {
    "id": "prompt",
    "modality": "text",
    "source": {"type": "inline", "value": "Describe the image"}
  }
]
```

### Outputs and units

Each output declares its modality and indexing unit. `items` maps a unit ID to
a list of predictions. The hand worker uses video frame indexes:

```json
{
  "id": "frame_predictions",
  "modality": "vision",
  "unit": "frame",
  "coordinate_system": {
    "type": "pixel",
    "origin": "top_left",
    "axes": ["x", "y"]
  },
  "items": {
    "0": [],
    "1": []
  }
}
```

Frame keys are integers inside Python and strings after JSON serialization,
because JSON object keys are always strings. Language workers can instead use
units such as `message`, `token`, `document`, or `chunk`.

## Prediction record

Every prediction has four required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Unique within the result document. |
| `task` | string | Operation such as `semantic_segmentation` or `generation`. |
| `type` | string | Representation such as `polygon`, `text`, or `embedding`. |
| `value` | any JSON value | Payload interpreted according to `type`. |

Common optional fields are:

| Field | Type | Meaning |
| --- | --- | --- |
| `label` | string | Semantic class or category. |
| `score` | number or null | Confidence, probability, logit, or task-specific score. |
| `track_id` | integer or string | Identity shared across frames or related records. |
| `source_id` | string | Input or upstream prediction that produced this result. |
| `span` | object | Start, end, and unit for text, audio, or temporal spans. |
| `attributes` | object | Namespaced task-specific extension data. |

Prediction types are extensible strings. Recommended value conventions are:

| `type` | `value` format |
| --- | --- |
| `polygon` | Closed `[[x, y], ...]` exterior ring. |
| `bbox` | `[[min_x, min_y], [max_x, max_y]]`. |
| `point` | `[x, y]`. |
| `class` | String or integer class value. |
| `text` | Generated or extracted string. |
| `token` | Token string or token object. |
| `embedding` | Array of numbers. |
| `scalar` | Number or boolean. |
| `json` | Arbitrary JSON object or array. |

For this worker, track IDs identify stable semantic roles (`left_hand`,
`right_hand`, left/right/shared held objects, and contact). Multiple disconnected
regions of one role share that ID. This is intentionally not a general-purpose
multi-instance temporal tracker; downstream consumers must not interpret these
IDs as independently tracked arbitrary objects.

## Hand-segmentation example

```json
{
  "schema": {"name": "wake-ai/inference-result", "version": "1.0.0"},
  "request": {
    "id": "job-001",
    "config": {
      "refinement": {"mode": "low"},
      "pipeline": {"trace_id": "pipeline-42", "step": 1},
      "delivery": {"enabled": false}
    }
  },
  "created_at": "2026-08-02T14:00:00+00:00",
  "model": {
    "id": "care-ego-cascadepsp",
    "families": ["cnn", "transformer"],
    "framework": "pytorch",
    "checkpoint": "care_ego_best_miou_weights.pth"
  },
  "inputs": [
    {
      "id": "video",
      "modality": "video",
      "source": {"type": "uri", "value": "file:///data/input.mp4"}
    }
  ],
  "outputs": [
    {
      "id": "frame_predictions",
      "modality": "vision",
      "unit": "frame",
      "coordinate_system": {
        "type": "pixel",
        "origin": "top_left",
        "axes": ["x", "y"]
      },
      "items": {
        "0": [
          {
            "id": "frame_predictions:0:0",
            "task": "semantic_segmentation",
            "source_id": "cascadepsp",
            "type": "polygon",
            "value": [[120, 400], [132, 397], [120, 400]],
            "label": "left_hand",
            "track_id": 1
          },
          {
            "id": "frame_predictions:0:1",
            "task": "semantic_segmentation",
            "source_id": "cascadepsp",
            "type": "bbox",
            "value": [[120, 397], [220, 510]],
            "label": "left_hand",
            "track_id": 1
          },
          {
            "id": "frame_predictions:0:2",
            "task": "semantic_segmentation",
            "source_id": "cascadepsp",
            "type": "point",
            "value": [170, 454],
            "label": "left_hand",
            "track_id": 1
          }
        ]
      }
    }
  ],
  "runtime": {
    "frame_count": 1,
    "largest_batch_size": 1,
    "elapsed_seconds": 0.74,
    "refinement_mode": "low",
    "effective_config": {
      "input": {"max_frames": null},
      "inference": {
        "batch_sizes": [128, 64, 32, 16, 8, 4, 2, 1],
        "mixed_precision": null
      },
      "refinement": {"mode": "low"},
      "geometry": {
        "simplify_tolerance": 2.0,
        "min_area": 64,
        "workers": 8
      },
      "retry": {"attempts": 3, "backoff_seconds": 1.0}
    },
    "delivery": {
      "enabled": false,
      "status": "disabled",
      "attempts": 0
    }
  }
}
```

`runtime.elapsed_seconds` covers video acquisition, decoding, first-stage
inference, CascadePSP refinement, geometry conversion, and envelope assembly.
Output fsync and optional downstream delivery occur afterward and are not
included in that value.

## Language-model example

```json
{
  "id": "completion:assistant:0",
  "task": "generation",
  "type": "text",
  "value": "A hand is holding an object.",
  "score": null,
  "span": {"start": 0, "end": 29, "unit": "character"}
}
```

This record can live under an output with `modality: "text"` and
`unit: "message"`. No segmentation-specific fields are required.

## Compatibility policy

- Major versions may rename or remove fields.
- Minor versions may add optional fields or prediction conventions.
- Patch versions clarify behavior without changing valid document structure.
- Consumers should dispatch on `schema.name` and the major component of
  `schema.version`.
- Architecture-specific data belongs in `attributes`, keeping shared fields
  portable across workers.
