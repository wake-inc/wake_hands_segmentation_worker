variable "project_id" {
  description = "ID of the existing test-workers Nebius project."
  type        = string
}

variable "tenant_id" {
  description = "Tenant that owns test-workers; needed to create the worker IAM group."
  type        = string
}

variable "nebius_profile" {
  description = "Local Nebius CLI profile used by Terraform."
  type        = string
  default     = null
  nullable    = true
}

variable "region" {
  description = "Nebius region containing the project."
  type        = string
  default     = "eu-north1"
}

variable "name_prefix" {
  description = "Prefix used for all worker resources."
  type        = string
  default     = "wake-hands-test"
}

variable "worker_image" {
  description = "Digest-pinned image reference. Leave empty during the first apply that creates the registry."
  type        = string
  default     = ""

  validation {
    condition     = var.worker_image == "" || can(regex("@sha256:[0-9a-f]{64}$", var.worker_image))
    error_message = "worker_image must be empty or pinned by @sha256:<64 lowercase hex characters>."
  }
}

variable "api_allowed_cidrs" {
  description = "CIDRs allowed to call unauthenticated port 8080. Use only trusted callers/VPN ranges."
  type        = list(string)

  validation {
    condition     = length(var.api_allowed_cidrs) > 0 && length(var.api_allowed_cidrs) <= 8
    error_message = "api_allowed_cidrs must contain between one and eight trusted CIDRs."
  }
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to SSH to the VM. Empty disables SSH ingress."
  type        = list(string)
  default     = []
}

variable "ssh_authorized_key" {
  description = "Optional public SSH key installed for the ubuntu user."
  type        = string
  default     = ""
}

variable "gpu_platform" {
  type    = string
  default = "gpu-l40s-a"
}

variable "gpu_preset" {
  type    = string
  default = "1gpu-8vcpu-32gb"
}

variable "boot_disk_size_gib" {
  type    = number
  default = 100
}

variable "preemptible" {
  description = "Use a cheaper interruptible VM for disposable test workloads. Keep false for a continuously available worker."
  type        = bool
  default     = false
}

variable "refinement_mode" {
  type    = string
  default = "low"

  validation {
    condition     = contains(["low", "medium"], var.refinement_mode)
    error_message = "refinement_mode must be low or medium."
  }
}

variable "temporal_stride" {
  description = "Run CascadePSP every Nth frame. CaRe-Ego still runs on every frame."
  type        = number
  default     = 1

  validation {
    condition     = var.temporal_stride >= 1 && floor(var.temporal_stride) == var.temporal_stride
    error_message = "temporal_stride must be a positive integer."
  }
}

variable "video_decoder" {
  description = "Video decoder backend: opencv or nvdec"
  type        = string
  default     = "opencv"
  validation {
    condition     = contains(["opencv", "nvdec"], var.video_decoder)
    error_message = "video_decoder must be opencv or nvdec."
  }
}

variable "cuda_graph_batch_size" {
  description = "Capture and replay the fixed CaRe-Ego CUDA graph at this batch size; 0 disables it. Smaller OOM fallback batches run eagerly."
  type        = number
  default     = 6

  validation {
    condition     = var.cuda_graph_batch_size >= 0 && floor(var.cuda_graph_batch_size) == var.cuda_graph_batch_size
    error_message = "cuda_graph_batch_size must be a non-negative integer."
  }
}
