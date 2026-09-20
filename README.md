# WAKE AI Hands Segmentation Worker GPU

Contact-aware Relationship Modeling for Egocentric Interactive Hand-object
Segmentation, based on the upstream CaRe-Ego
[paper](https://arxiv.org/abs/2407.05576) and
[project page](https://yuggiehk.github.io/CaRe-Ego/).

This repository is packaged for direct use. Custom modules no longer need to
be copied into an MMSegmentation checkout. Model, data, optimizer, and runtime
configuration live in [`care_ego/config.py`](care_ego/config.py); dependency
resolution is captured by `uv.lock`.

Detailed documentation:

- [Universal result format](docs/result-format.md)
- [Service workflow](docs/workflow.md)
- [Request configuration and service chaining](docs/request-configuration.md)

## Install

Install [Git LFS](https://git-lfs.com/) and
[uv](https://docs.astral.sh/uv/), then initialize LFS and create the locked
environment:

```bash
git lfs install
git lfs pull
uv sync --frozen
```

The runtime weights are versioned with Git LFS under `weights/` and are
included automatically in a normal clone when Git LFS is installed. Run
`git lfs pull` after cloning if the files are still LFS pointer files:

```text
wake_hands_segmentation_worker/
└── weights/
    ├── care_ego_best_miou_weights.pth
    ├── care_ego_best_miou_mmengine_checkpoint.pth
    ├── cascadepsp_v1_0.pth
    └── upstream_pretraining_checkpoint.pth
```

The service discovers the first-stage and CascadePSP checkpoint locations
automatically. `WAKE_CHECKPOINT_PATH` and `WAKE_CASCADEPSP_MODEL_DIR` remain
available for deployments that mount weights elsewhere.

The default lock uses `mmcv-lite`, which is sufficient for inference on macOS,
Linux, and Windows. Training losses that call MMCV native operators require a
platform-appropriate full MMCV build.

## Python API

Validate a checkpoint before loading it:

```python
from care_ego.inference import checkpoint_compatibility

report = checkpoint_compatibility("weights/care_ego_best_miou_weights.pth")
assert report["strictly_compatible"]
```

A compatible released checkpoint has 1,779 matching keys with no missing,
unexpected, or shape-mismatched tensors.

Run image inference:

```python
import cv2

from care_ego.inference import load_model, predict, visualize

model = load_model("weights/care_ego_best_miou_weights.pth")
image = cv2.imread("input.jpg")
prediction = predict(model, image)
overlay = visualize(image, prediction)
cv2.imwrite("output.jpg", overlay)
```

This direct image API returns the first-stage CaRe-Ego prediction. The video
service runs CascadePSP as its second stage before converting masks to geometry.

`prediction.raw_contact` exposes the released auxiliary contact head. Its loss
is disabled by the original segmentor implementation, so visualizations use
`prediction.derived_contact`, the local interface between predicted hand and
object boundaries.

## Producer-consumer service

`SegmentationService` is an API-only, single-GPU worker. It loads the model
once, waits on a request queue, decodes each requested video on a producer
thread, and consumes frames in the largest configured batch that fits GPU
memory. A persistent CascadePSP model refines the predicted masks before
polygon conversion. Geometry conversion runs concurrently on CPU workers so
the next GPU batch can start immediately.

```mermaid
flowchart LR
    R[Request queue] --> D[file:// download]
    D --> P[Frame producer]
    P --> Q[In-memory frame queue]
    Q --> G[Adaptive GPU batch consumer]
    G --> F[GPU CascadePSP refinement]
    F --> C[CPU Shapely worker pool]
    C --> J[Atomic JSON output]
```

```python
from pathlib import Path

from care_ego.service import SegmentationService, WorkerConfig

checkpoint = Path("weights/care_ego_best_miou_weights.pth")
video = Path("input.mp4").resolve()
output = Path("outputs").resolve()

with SegmentationService(
    checkpoint,
    config=WorkerConfig(batch_sizes=(128, 64, 32, 16, 8, 4, 2, 1)),
) as service:
    future = service.submit(
        {
            "request_id": "example-job",
            "video_uri": video.as_uri(),
            "output_uri": output.as_uri(),
            "simplify_tolerance": 2.0,
            "min_area": 64,
            "refinement_mode": "low",
            "config": {
                "pipeline": {"trace_id": "pipeline-42", "step": 1},
                "delivery": {"enabled": False},
            },
        }
    )
    result = future.result()
    print(result.output_path, result.batch_size)
```

For now, `video_uri` and `output_uri` must use `file://`; the output URI names
a directory. Each request produces `<request_id>.json` using the versioned,
architecture-neutral `wake-ai/inference-result` schema. Frame indexes are
integers in memory and strings after JSON serialization.

Requests may override batching, precision, refinement, geometry, input limits,
and retry policy through a namespaced `config` object. Optional downstream
delivery can POST the completed universal document or its `file://` reference
to another service. See
[Request configuration and service chaining](docs/request-configuration.md).

The hand worker stores its frame-indexed geometry in
`outputs[0].items`. A shortened result looks like this:

```json
{
  "schema": {"name": "wake-ai/inference-result", "version": "1.0.0"},
  "request": {
    "id": "job-001",
    "config": {
      "refinement": {"mode": "low"},
      "delivery": {"enabled": false}
    }
  },
  "created_at": "2026-08-02T14:00:00+00:00",
  "model": {
    "id": "care-ego-cascadepsp",
    "families": ["cnn", "transformer"],
    "framework": "pytorch"
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
      "coordinate_system": {"type": "pixel", "origin": "top_left"},
      "items": {
        "0": [
          {
            "id": "frame_predictions:0:0",
            "task": "semantic_segmentation",
            "source_id": "cascadepsp",
            "track_id": 1,
            "label": "left_hand",
            "type": "polygon",
            "value": [[120, 400], [132, 397], [120, 400]]
          }
        ]
      }
    }
  ],
  "runtime": {
    "frame_count": 1,
    "largest_batch_size": 1,
    "refinement_mode": "low",
    "effective_config": {
      "inference": {"batch_sizes": [128, 64, 32, 16, 8, 4, 2, 1]},
      "refinement": {"mode": "low"}
    },
    "delivery": {"enabled": false, "status": "disabled", "attempts": 0}
  }
}
```

Masks are converted to Shapely polygons and use
[topology-preserving simplification](https://shapely.readthedocs.io/en/stable/reference/shapely.simplify.html).
Bboxes contain two corner points; point annotations contain one `[x, y]`
coordinate. Track IDs are stable semantic-role IDs for the left/right hand,
left/right/shared object, and hand-object contact. They identify those roles
across frames; they are not a general multi-instance object tracker.

The envelope is not tied to segmentation or PyTorch. `model.families`, input
modalities, output modalities, tasks, unit names, and prediction types are
extensible strings. A CNN can emit classes or geometry; a VLM can accept image
and text inputs; an LLM can emit `type: "text"`; and transformer workers can
emit tokens, spans, embeddings, or structured JSON through the same prediction
record. Every prediction has `id`, `task`, `type`, and `value`; optional common
fields include `label`, `score`, `track_id`, `source_id`, `span`, and
`attributes`. The canonical JSON Schema lives in
[`care_ego/schema.py`](care_ego/schema.py) and is served by `GET /v1/schema`.

## CascadePSP refinement

The second stage uses the official
[CascadePSP](https://github.com/hkchengrex/CascadePSP) implementation. It
refines four binary semantic masks per frame—left/right hands and left/right
objects—then reconstructs mutually exclusive hand labels, shared objects, and
the derived contact boundary.

- `low` is the throughput default and uses CascadePSP's global-only fast path
  with `L=600`.
- `medium` uses global plus local refinement with the official `L=900` default.

Set the default with `WAKE_REFINEMENT_MODE`; a request can override it with
`refinement_mode`. The expected local artifact is
`weights/cascadepsp_v1_0.pth`. Git LFS supplies this artifact with the other
repository weights, and every startup verifies its SHA-256 digest. Set
`WAKE_CASCADEPSP_MODEL_DIR` to use another model directory.

Performance defaults favor GPU throughput: CUDA convolution benchmarking and
TF32 are enabled, CUDA inference uses automatic mixed precision, frame decode
overlaps inference, geometry work is threaded, and first-stage batches are
attempted from 128 down to 1 on out-of-memory failures. CascadePSP then refines
four binary masks per frame with one persistent second-stage model.

## Wake runner

The image starts `python -m care_ego.queue_runner`. `WAKE_RUNNER_MODE=queue`
(the default) consumes durable Wake jobs from RabbitMQ and reports progress at
each completed inference batch. It accepts only `track_hands:1`, downloads the
payload's checksum-verified `sourceVideo` once to scratch, and writes processor
output through the Wake workspace.

`WAKE_RUNNER_MODE=http` starts the generic Wake HTTP runner for a temporary,
local debug session. It exposes `/health/live`, `/health/ready`, `POST
/v1/jobs`, job status/result/artifact endpoints, and SSE events. The model
remains loaded between debug jobs by default. Set `WAKE_DEBUG_TASK_OVERRIDES=true`
only for an isolated debug session to allow bounded payload overrides for batch
sizes, mixed precision, geometry workers, refinement mode, or per-job model
lifetime. See [the workflow guide](docs/workflow.md).

Build and publish with BuildKit; the image fetches the pinned Wake queue packages:

```bash
IMAGE=registry.example/wake-hands-gpu TAG=debug-$(git rev-parse --short HEAD) \
  ./scripts/build-and-push.sh
```

For service chaining, enable `config.delivery` and provide the next service
URL. Full-document and shared-file reference modes, authentication headers,
idempotency, delivery-only retries, and required/best-effort behavior are
documented in
[Request configuration and service chaining](docs/request-configuration.md).

## Docker GPU service

The production image is Linux/x86-64 and uses the CUDA 12.1 dependencies
selected by `uv.lock`. It embeds the CaRe-Ego inference checkpoint and
CascadePSP checkpoint, so it needs no configuration or weight mount.

```bash
docker build --platform linux/amd64 \
  --tag wake_hands_segmentation_worker:0.1.0 .

docker run --detach \
  --name wake_hands_segmentation_worker \
  --restart unless-stopped \
  --gpus all \
  --publish 8080:8080 \
  wake_hands_segmentation_worker:0.1.0
```

The host needs an NVIDIA driver and NVIDIA Container Toolkit configured for
Docker. The image defaults to `WAKE_DEVICE=cuda`, runs as UID/GID 10001, uses
Tini for signal forwarding, and reports readiness through its Docker health
check. `--gpus all` and `--publish` grant host resources and therefore cannot be
encoded in an image. If the Docker daemon already uses the NVIDIA runtime by
default and the API is consumed from the container network, the literal command
`docker run wake_hands_segmentation_worker:0.1.0` is sufficient.

Input and output `file://` URIs still refer to the container filesystem. A data
volume is optional and is needed only when a job must exchange persistent files
with the host; it is not needed for model startup.

For packaging tests on a machine without NVIDIA hardware, override the device:

```bash
docker run --rm --env WAKE_DEVICE=cpu \
  wake_hands_segmentation_worker:0.1.0
```

See [Service workflow](docs/workflow.md#docker-deployment) for runtime and
verification details.

## Training configuration

The MMSeg Python configuration is importable as `care_ego.config` and includes
`custom_imports` for automatic registry setup. Set `data_root` and, when
training from backbone initialization, `model["pretrained"]` in Python before
launching an MMEngine runner.

Dataset layout follows the original EgoHOS-derived structure:

```text
train/
  image/
  label/
  label_hand/
  lbl_obj_left/
  lbl_obj_right/
  lbl_obj_two/
  label_contact_first/
```

`weights/upstream_pretraining_checkpoint.pth` is retained for possible future
training work, but it is not a CaRe-Ego resume checkpoint. It contains 760
`backbone`/`sem_seg_head` tensors and is not directly compatible with the
1,779-key CaRe-Ego model. Keep `model["pretrained"] = None` unless a dedicated
conversion/mapping is implemented and validated.

`care_ego_best_miou_weights.pth` is the tensor-only, restricted-loader-safe
inference artifact. `care_ego_best_miou_mmengine_checkpoint.pth` preserves the
original MMEngine metadata for future training experiments; it contains the
same 1,779 model tensors but no optimizer state.

## License

Apache-2.0. See [`LICENSE.txt`](LICENSE.txt).
