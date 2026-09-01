output "registry_repository" {
  description = "Repository to which the local Docker image must be pushed."
  value       = local.image_repo
}

output "bucket_name" {
  value = nebius_storage_v1_bucket.jobs.name
}

output "worker_public_ip" {
  value = try(split("/", nebius_compute_v1_instance.worker[0].status.network_interfaces[0].public_ip_address.address)[0], null)
}

output "worker_url" {
  value = try("http://${split("/", nebius_compute_v1_instance.worker[0].status.network_interfaces[0].public_ip_address.address)[0]}:8080", null)
}

output "next_step" {
  value = var.worker_image == "" ? "Push the image to ${local.image_repo}, resolve its digest, then apply with -var worker_image=<repo>@sha256:<digest>." : "Wait for cloud-init, then GET /health/ready on the worker URL."
}
