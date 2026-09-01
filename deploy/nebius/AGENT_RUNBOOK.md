# Agent runbook: Nebius test worker

Use this runbook to continue or repeat the deployment without rediscovering
the project, credentials model, or two-stage image workflow.

## Fixed target

- Project: `test-workers`
- Nebius profile, project ID, and tenant ID: supply them locally; they are
  intentionally absent from the release tree
- Region: `eu-north1`
- Terraform directory: `deploy/nebius/terraform`
- Registry repository: obtain it with `terraform output -raw registry_repository`
- Job bucket: obtain it with `terraform output -raw bucket_name`

Never copy, commit, or print `terraform.tfstate`: it contains the Object
Storage access-key secret that cloud-init writes to `/etc/wake-worker.env`.
The directory `.gitignore` excludes state, local tfvars, and plans. Do not run
`terraform destroy`, rotate the access key, or replace the VM unless the user
has explicitly authorized that lifecycle action.

## Authenticate and inspect

```bash
nebius iam get-access-token >/dev/null
cd deploy/nebius/terraform
terraform init
terraform validate
terraform plan
```

If the operator's public IP changed, update both CIDR lists in the ignored
`terraform.tfvars` before applying. Port 8080 is unauthenticated and must never
be exposed as `0.0.0.0/0`.

```hcl
api_allowed_cidrs = ["CURRENT.PUBLIC.IP/32"]
ssh_allowed_cidrs = ["CURRENT.PUBLIC.IP/32"]
```

## First deployment or a new image

Terraform intentionally uses two stages. With `worker_image = ""`, the first
apply creates IAM, registry, bucket, and network resources but no GPU VM:

```bash
terraform apply
REPOSITORY="$(terraform output -raw registry_repository)"
```

Build the exact workspace for `linux/amd64`, authenticate Docker, and push it:

```bash
nebius registry configure-helper
docker build --platform linux/amd64 -t "${REPOSITORY}:candidate" ../../..
docker push "${REPOSITORY}:candidate"
IMAGE="$(docker image inspect "${REPOSITORY}:candidate" \
  --format '{{index .RepoDigests 0}}')"
```

Confirm that `IMAGE` belongs to `REPOSITORY` and ends in `@sha256:...`. Put it
in the ignored `terraform.tfvars`, then create/update the VM:

```hcl
worker_image = "cr.eu-north1.nebius.cloud/.../wake_hands_segmentation_worker@sha256:..."
```

```bash
terraform plan
terraform apply
terraform output
```

Do not deploy a mutable tag. A changed digest updates cloud-init; if Nebius
cannot apply that safely to the existing VM, inspect the plan and get approval
before accepting replacement of the GPU VM.

Keep `preemptible = false` for a continuously available service. Enabling it
is an explicit cost/reliability trade-off for disposable benchmark machines:
Nebius may stop them and Terraform configures no automatic recovery.

## Verify and debug

Cloud-init can take several minutes while Docker downloads the 7.5 GB image.

```bash
WORKER_URL="$(terraform output -raw worker_url)"
curl --fail "${WORKER_URL}/health/live"
curl --fail "${WORKER_URL}/health/ready"
```

If SSH was configured, inspect the host without exposing the environment file:

```bash
sudo cloud-init status --wait
sudo systemctl status wake-worker.service
sudo journalctl -u wake-worker.service -n 200 --no-pager
sudo docker ps --all
sudo docker logs --tail 200 wake_hands_segmentation_worker
nvidia-smi
```

Never run `cat /etc/wake-worker.env` in captured agent output. Test jobs use
`s3://<bucket>/jobs/<request-id>/input.mp4` and write to
`s3://<bucket>/jobs/<request-id>/results/<request-id>.json`.

## Current measured baseline

No environment-specific image reference is committed. Resolve the release
digest from the target registry and record it only in the ignored local
`terraform.tfvars`.

Use `WAKE_CUDA_GRAPH_BATCH_SIZE=6`, `WAKE_TEMPORAL_STRIDE=60`, and
`WAKE_VIDEO_DECODER=nvdec`. CaRe-Ego must run on every frame; the temporal
stride applies only to CascadePSP. The CUDA graph pads only near-full tail
batches. If graph capture OOMs, adaptive batching retries eagerly at 4, 2,
then 1, so this image remains usable on smaller GPUs.

On the 150-frame 960x1280 regression clip, three steady-state runs were
4.696, 4.711, and 4.748 seconds (31.8 FPS median). GPU sampling reported 80.0%
average utilization, 100% peak, and 8.4 GiB peak memory. Both hands were
present in 150/150 frames. Against the corrected eager baseline, hand raster
mean IoU was 0.999839/0.999807 and minimum IoU was 0.997581/0.994208. Generated
benchmark outputs are intentionally not committed; reproduce them before a
new performance release.

Do not re-enable the rejected experiments without a new isolated benchmark:

- two model replicas: 7.40-7.54 seconds and higher VRAM;
- parallel decode-head streams: 6.54-6.84 seconds;
- explicit FP16 weights: no material speedup and larger small-object drift;
- runtime `torch.compile`: the old path compiled no executed forward; the real
  graph requires a compiler toolchain and exceeded four minutes of cold start.

For every performance change, run at least one warmup plus three measured
requests, rasterize polygons, and require both hands in every frame. Compare
all semantic labels to a freshly generated eager baseline; do not accept
throughput alone. Keep the previous digest in `/etc/wake-worker.env` until
these quality checks pass, then update the digest-pinned Terraform value. The
deployment service automatically rolls back a candidate that never becomes
ready and restarts an unhealthy container after three failed readiness checks.
