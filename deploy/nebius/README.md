# Nebius GPU VM deployment

This deployment runs one worker container on one Nebius Compute GPU VM. It
preserves the existing single-GPU/FIFO service topology and replaces AWS ECR,
EC2, and S3 with Nebius Container Registry, Compute, and Object Storage.

The supported fast path is the Terraform configuration in [`terraform`](terraform).
It is already parameterized for the existing `test-workers` project; copy
`terraform/terraform.tfvars.example` to the ignored `terraform.tfvars`, replace
the allowed CIDRs, and follow the two-stage apply below. Agents continuing an
existing deployment must first read [`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md).

```bash
cd deploy/nebius/terraform
terraform init
terraform apply                 # IAM, registry, bucket, and network; no VM yet
REPOSITORY="$(terraform output -raw registry_repository)"

nebius registry configure-helper
docker build --platform linux/amd64 -t "${REPOSITORY}:candidate" ../../..
docker push "${REPOSITORY}:candidate"
docker image inspect "${REPOSITORY}:candidate" --format '{{index .RepoDigests 0}}'
```

Put that digest-pinned reference in `worker_image` in `terraform.tfvars`, then
run `terraform apply` again. This creates the GPU VM only after its image
exists. The state contains an Object Storage secret and must not be committed
or shared as a normal artifact.

## Target resources

- One regular Nebius Compute VM with one GPU. `gpu-l40s-a` is an
  inference-oriented starting point in `eu-north1`; use an H100/H200 preset if
  measured throughput requires it.
- An Ubuntu 24.04 GPU boot disk. Query current compatible images with
  `nebius compute image list-public`; the current quickstart uses the
  `ubuntu24.04-cuda13.0` image family.
- One Container Registry repository containing a digest-pinned worker image.
- One Object Storage bucket with separate input and result prefixes.
- A VM-attached service account permitted to pull from Container Registry.
- A Nebius Object Storage access key belonging to a service account whose IAM
  group has read access to inputs and upload access to results.

The VM-attached identity authenticates Nebius API and registry operations. The
S3-compatible Object Storage API uses `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`; those are separate credentials.

## Build and publish

Configure the Nebius Docker credential helper, create a registry, and publish
the image:

```bash
export NB_REGION_ID=eu-north1
export NB_REGISTRY_PATH="$(nebius registry create \
  --name wake-workers --format json | jq -r '.metadata.id' | cut -d- -f 2)"
nebius registry configure-helper

docker build --platform linux/amd64 \
  --tag "cr.${NB_REGION_ID}.nebius.cloud/${NB_REGISTRY_PATH}/wake_hands_segmentation_worker:candidate" \
  ../..
docker push \
  "cr.${NB_REGION_ID}.nebius.cloud/${NB_REGISTRY_PATH}/wake_hands_segmentation_worker:candidate"
```

Resolve the pushed digest and put the digest-pinned reference in
`/etc/wake-worker.env` rather than deploying a mutable tag.

## VM setup

Create the VM with a current Ubuntu 24.04 CUDA image, one GPU, a sufficiently
large boot disk, the VM service account, and a network security rule allowing
TCP 8080 only from the calling service or private network. A GPU cluster is not
needed for this single-GPU inference worker.

Copy this directory to the VM, then run:

```bash
sudo ./bootstrap.sh
sudo install --owner=root --group=root --mode=0600 \
  worker.env.example /etc/wake-worker.env
sudoedit /etc/wake-worker.env
sudo systemctl enable --now wake-worker.service
```

The systemd service stays active as a readiness watchdog. It rolls back to the
previous container image when a candidate cannot start, and restarts a running
container after three consecutive readiness failures.

Do not put the Object Storage secret in cloud-init user data, the container
image, source control, or an HTTP request. Provision `/etc/wake-worker.env`
through the deployment system's secret channel. Rotate the access key if the
file or VM is exposed.

## Object Storage access

Set these runtime values for `eu-north1`:

```text
WAKE_S3_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
WAKE_S3_REGION=eu-north1
AWS_ACCESS_KEY_ID=<Nebius access key AWS ID>
AWS_SECRET_ACCESS_KEY=<Nebius access key secret>
```

Use a bucket policy or access permits to grant only the required paths:

- `storage.object-viewer` for `jobs/*/input/*`;
- `storage.uploader` for `jobs/*/results/*`.

The worker does not need bucket listing or delete permission.

## Verify

```bash
sudo systemctl status wake-worker.service
sudo docker logs wake_hands_segmentation_worker
curl --fail http://127.0.0.1:8080/health/ready
```

Submit a Nebius Object Storage job with the existing API:

```json
{
  "request_id": "nebius-job-001",
  "video_uri": "s3://wake-inference/jobs/nebius-job-001/input/input.mp4",
  "output_uri": "s3://wake-inference/jobs/nebius-job-001/results/",
  "max_frames": 10
}
```

The completed object is
`s3://wake-inference/jobs/nebius-job-001/results/nebius-job-001.json`.
