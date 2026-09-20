# Worker workflow

`python -m care_ego.queue_runner` is the only worker entry point. It runs the
same `track_hands` processor in two transports selected by `WAKE_RUNNER_MODE`.

## Queue mode

`WAKE_RUNNER_MODE=queue` is the production path. One process loads CaRe-Ego
and CascadePSP once, consumes `jobs.gpu` serially from RabbitMQ, downloads the
coordinator-provided source video to scratch, verifies its SHA-256, and writes
the output artifact plus final result to the job workspace. The generic queue
library owns delivery fencing, finish markers, STARTED/PROGRESS/FINISHED events,
and acknowledgement.

After each completed inference batch, the processor renews the attempt lock and
publishes total processed frames, batch size, and the decode frame count when a
decode dependency was supplied. It does not persist partial hand predictions,
so a retry starts inference again. The worker exits after `RUNNER_IDLE_EXIT_S`
without another delivery; KEDA can then remove the empty GPU node.

## Local HTTP mode

`WAKE_RUNNER_MODE=http` starts the generic `wake-job-queue` HTTP runner. It
registers the identical `track_hands` processor, executes one submitted job at
a time, stores state and artifacts in `HTTP_WORKSPACE_DIR`, and returns status,
results, artifacts, and SSE progress through the generic endpoints:

```text
GET  /health/live
GET  /health/ready
POST /v1/jobs
GET  /v1/jobs/:id
GET  /v1/jobs/:id/events
GET  /v1/jobs/:id/result
GET  /v1/jobs/:id/artifacts/:key
```

The mode is for private temporary diagnostic Jobs and is accessed only through
`kubectl port-forward`. It exits after `RUNNER_IDLE_EXIT_S` with no active work.

The expected submission is the normal queue processor payload:

```json
{
  "kind": "track_hands",
  "kindVersion": 1,
  "payload": {
    "sourceVideo": {
      "downloadUrl": "https://short-lived-source-url",
      "sha256": "lowercase-sha256"
    }
  }
}
```

`WAKE_DEBUG_TASK_OVERRIDES=true` is honored only in HTTP mode. A job may then
select `serviceLifetime` (`persistent` or `perJob`) and bounded batch-size,
precision, geometry-worker, or refinement-mode settings under `payload.debug`.
It cannot change the model, checkpoint, device, credentials, or source URL.

## Processing

The verified scratch video is processed directly; the service does not make a
second temporary copy. OpenCV decoding overlaps GPU inference. CascadePSP runs
after CaRe-Ego on the selected device, and a CPU thread pool converts masks to
polygon, bounding-box, representative-point, and contact annotations. Results
are written atomically to JSON before the queue processor uploads them as an
artifact.

## Runtime configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `WAKE_RUNNER_MODE` | `queue` | `queue` or local `http` transport. |
| `RUNNER_IDLE_EXIT_S` | queue deployment value | Exit when idle in either mode. |
| `WAKE_CHECKPOINT_PATH` | bundled weight | CaRe-Ego checkpoint. |
| `WAKE_DEVICE` | `auto` | Torch device selection. |
| `WAKE_BATCH_SIZES` | `128,64,32,16,8,4,2,1` | Adaptive GPU batch candidates. |
| `WAKE_MIXED_PRECISION` | `true` | Default AMP setting. |
| `WAKE_GEOMETRY_WORKERS` | bounded CPU count | Geometry thread count, maximum 64. |
| `WAKE_REFINEMENT_MODE` | `low` | CascadePSP profile. |
| `WAKE_DEBUG_TASK_OVERRIDES` | `false` | Permit bounded HTTP-only task tuning. |
| `WAKE_BIND` | `0.0.0.0:8080` | Generic HTTP runner bind in HTTP mode. |
| `HTTP_WORKSPACE_DIR` | `/scratch/http-workspace` | Local HTTP state/artifact root. |

## Container build

The image uses Python 3.12 and CUDA-enabled PyTorch 2.7.1. It downloads the
version-pinned generic queue packages from the Wake app repository during the
locked Docker build, so build with SSH forwarding:

```bash
docker buildx build --ssh default --platform linux/amd64 --tag wake-hands:dev .
```
