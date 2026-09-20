# Project memory

## Purpose

`wake_hands_segmentation_worker` packages CaRe-Ego as a Wake GPU worker. Its
single entry point consumes durable `track_hands` jobs from RabbitMQ by default;
an isolated operator debug Job may instead expose the same registered processor
through the generic local Wake HTTP runner. Both paths run adaptive batched
CaRe-Ego inference, refine masks with CascadePSP, and write a versioned
architecture-neutral JSON result.

## Runtime pipeline

1. A queue job, or generic local HTTP job, enters the persistent GPU worker.
2. A producer decodes video frames into an in-memory queue.
3. One GPU consumer attempts batches from 128 down to 1 on OOM.
4. CascadePSP refines left/right hand and object masks.
5. CPU geometry workers create simplified Shapely geometry and atomically save
   `<request_id>.json`.

The worker receives a short-lived source URL and expected SHA-256 in its Wake
payload, downloads one verified scratch copy, and processes that copy directly.
The segmentation service itself continues to accept local `file://` paths.

## Result contract and verified behavior

Results use `wake-ai/inference-result` schema version `1.0.0`, under
`outputs[0].items[frame_index]`.

Per refined component that passes `min_area`, the worker emits:

| Role | Track ID | Emitted geometry |
| --- | ---: | --- |
| `left_hand` | 1 | polygon, bbox, point |
| `right_hand` | 2 | polygon, bbox, point |
| `left_object` | 3 | polygon, bbox, point |
| `right_object` | 4 | polygon, bbox, point |
| `shared_object` | 5 | polygon, bbox, point (only when object masks overlap) |
| `hand_object_contact` | 6 | one or more points only |

Object labels are semantic roles, not an explicit `held_by` relation. Contact
is derived from refined hand/object boundaries and is emitted as point-only
`contact_detection` records; no contact polygon or bbox exists today.

Native one-frame full-pipeline smoke test on `../IMG_4350_5s.mp4` passed:
14 annotations, including two hand polygons, two object polygons, and two
contact points. It used CascadePSP low refinement and emitted polygon, bbox,
and point geometry under the expected schema.

## Weights

Required runtime artifacts in `weights/`:

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `care_ego_best_miou_weights.pth` | CaRe-Ego inference checkpoint | `21dd052d6abf914cd4a3aad67e3ed2576a0138abfac3ed92bf08835fbde42afd` |
| `cascadepsp_v1_0.pth` | CascadePSP refinement checkpoint | `60826ace3385961cdf8608abbdf1234cd22379303b97edf96f18d67bba769cfa` |

The CaRe-Ego inference checkpoint was checked against the packaged model:
1,779 expected/supplied/matching tensors, with no missing, unexpected, or
shape-mismatched keys.

Training-only artifacts are deliberately excluded from Docker:
`care_ego_best_miou_mmengine_checkpoint.pth` and
`upstream_pretraining_checkpoint.pth`.

## Docker image

The Dockerfile builds a Linux/amd64 CUDA 12.8-compatible image that embeds exactly the two
runtime weights at `/app/weights`, owned by UID/GID 10001, files mode `0444`,
directory mode `0555`.

Last verified image:

- tag: `wake_hands_segmentation_worker:0.1.0`
- digest: `sha256:a4a768a03506d76ed6db2cdafd4df0ff5a4dc7cb7081a5d4396245436049b6e8`
- architecture: `linux/amd64`
- size: about 7.49 GB
- runtime: PyTorch `2.7.1`, CUDA build `12.8`

The image must be built and smoke-tested on a Linux/NVIDIA runner before
publishing. Queue mode needs RabbitMQ and processing-workspace configuration.
HTTP debug mode listens only inside its temporary Kubernetes Job and is reached
through `kubectl port-forward`; it is not a public production service.

```bash
docker run --rm --gpus all \
  -e WAKE_RUNNER_MODE=http \
  -e RUNNER_IDLE_EXIT_S=900 \
  wake_hands_segmentation_worker:0.1.0
```

`--gpus all` is a Docker host setting and cannot be encoded in the image.

## Quality checks completed

- `uv lock --check` passed.
- `uv run --frozen ruff check .` passed.
- `uv run --frozen pytest -q` passed: 40 tests.
- `uv build` produced a valid wheel and source distribution for `0.1.0`.
- `twine check` passed for both artifacts.
- Native direct CaRe-Ego inference and full service/CascadePSP smoke tests
  passed on the supplied short MP4.

Non-failing upstream runtime warnings occur during model construction from
MMSeg/PyTorch (`build_loss`, `torch.meshgrid`, and binary segmentation output
channel guidance).

## Documentation findings

Documentation is generally aligned with the implementation. Before release,
make the output contract explicit in README/result-format examples:

- hands and object roles emit polygon + bbox + point;
- contacts emit points only;
- outputs are omitted when masks are empty or below `min_area`;
- left/right object labels are roles, not explicit hand-object relation links.

The README diagram currently says `file:// download`; it should say local copy
or local staging because remote download is not supported.

## Release checklist / open items

1. The worktree contains the uncommitted packaging migration. Review, stage,
   commit, and create the release tag before publishing.
2. Update `pyproject.toml` repository metadata if this package should point to
   the WAKE fork rather than upstream CaRe-Ego; retain appropriate attribution.
3. Confirm redistribution rights for both embedded checkpoint artifacts before
   publishing the image.
4. Build and smoke-test the Docker image on Linux/amd64 with NVIDIA GPU and
   push the approved digest to the target registry.
5. Add CI/release automation (lock check, Ruff, pytest, build, Twine check,
   Docker build) before recurring releases.
