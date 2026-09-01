locals {
  labels = {
    application = "wake_hands_segmentation_worker"
    environment = "test"
    managed-by  = "terraform"
  }

  registry_path = trimprefix(nebius_registry_v1_registry.worker.id, "registry-")
  image_repo    = "cr.${var.region}.nebius.cloud/${local.registry_path}/wake_hands_segmentation_worker"
}

resource "nebius_registry_v1_registry" "worker" {
  parent_id   = var.project_id
  name        = "${var.name_prefix}-registry"
  description = "Images for the WAKE hands segmentation test worker"
  labels      = local.labels
}

resource "nebius_storage_v1_bucket" "jobs" {
  parent_id             = var.project_id
  name                  = "${var.name_prefix}-jobs"
  default_storage_class = "STANDARD"
  versioning_policy     = "ENABLED"
  labels                = local.labels

  bucket_policy = {
    rules = [{
      group_id = nebius_iam_v1_group.worker.id
      paths    = ["jobs/*"]
      roles    = ["storage.object-viewer", "storage.uploader"]
    }]
  }
}

resource "nebius_iam_v1_service_account" "worker" {
  parent_id   = var.project_id
  name        = "${var.name_prefix}-sa"
  description = "Runtime identity for the WAKE hands segmentation worker"
  labels      = local.labels
}

resource "nebius_iam_v1_group" "worker" {
  parent_id = var.tenant_id
  name      = "${var.name_prefix}-runtime"
  labels    = local.labels
}

resource "nebius_iam_v1_group_membership" "worker" {
  parent_id = nebius_iam_v1_group.worker.id
  member_id = nebius_iam_v1_service_account.worker.id
  labels    = local.labels
}

resource "nebius_iam_v1_access_permit" "project_viewer" {
  parent_id   = nebius_iam_v1_group.worker.id
  resource_id = var.project_id
  role        = "viewer"
  labels      = local.labels
}

resource "nebius_iam_v2_access_key" "storage" {
  parent_id            = var.project_id
  name                 = "${var.name_prefix}-storage-key"
  description          = "S3-compatible access key used only by the test worker"
  secret_delivery_mode = "INLINE"
  labels               = local.labels

  account = {
    service_account = {
      id = nebius_iam_v1_service_account.worker.id
    }
  }
}

resource "nebius_vpc_v1_network" "worker" {
  parent_id = var.project_id
  name      = "${var.name_prefix}-network"
  labels    = local.labels
}

resource "nebius_vpc_v1_subnet" "worker" {
  parent_id  = var.project_id
  name       = "${var.name_prefix}-subnet"
  network_id = nebius_vpc_v1_network.worker.id
  labels     = local.labels

  ipv4_private_pools = {
    use_network_pools = true
  }
  ipv4_public_pools = {
    use_network_pools = true
  }
}

resource "nebius_vpc_v1_security_group" "worker" {
  parent_id  = var.project_id
  name       = "${var.name_prefix}-sg"
  network_id = nebius_vpc_v1_network.worker.id
  labels     = local.labels
}

resource "nebius_vpc_v1_security_rule" "api" {
  parent_id = nebius_vpc_v1_security_group.worker.id
  name      = "allow-worker-api"
  access    = "ALLOW"
  protocol  = "TCP"
  priority  = 100
  type      = "STATEFUL"

  ingress = {
    source_cidrs      = var.api_allowed_cidrs
    destination_ports = [8080]
  }
}

resource "nebius_vpc_v1_security_rule" "ssh" {
  count = length(var.ssh_allowed_cidrs) == 0 ? 0 : 1

  parent_id = nebius_vpc_v1_security_group.worker.id
  name      = "allow-ssh"
  access    = "ALLOW"
  protocol  = "TCP"
  priority  = 110
  type      = "STATEFUL"

  ingress = {
    source_cidrs      = var.ssh_allowed_cidrs
    destination_ports = [22]
  }
}

resource "nebius_vpc_v1_security_rule" "egress" {
  parent_id = nebius_vpc_v1_security_group.worker.id
  name      = "allow-egress"
  access    = "ALLOW"
  protocol  = "ANY"
  priority  = 100
  type      = "STATEFUL"

  egress = {
    destination_cidrs = ["0.0.0.0/0"]
  }
}

resource "nebius_compute_v1_instance" "worker" {
  count = var.worker_image == "" ? 0 : 1

  parent_id          = var.project_id
  name               = "${var.name_prefix}-vm"
  hostname           = "wake-hands-worker"
  service_account_id = nebius_iam_v1_service_account.worker.id
  recovery_policy    = var.preemptible ? "FAIL" : "RECOVER"
  labels             = local.labels

  resources = {
    platform = var.gpu_platform
    preset   = var.gpu_preset
  }

  preemptible = var.preemptible ? {
    on_preemption = "STOP"
  } : null

  boot_disk = {
    attach_mode = "READ_WRITE"
    managed_disk = {
      name = "${var.name_prefix}-boot"
      spec = {
        type           = "NETWORK_SSD"
        size_gibibytes = var.boot_disk_size_gib
        source_image_family = {
          image_family = "ubuntu24.04-cuda13.0"
        }
      }
    }
  }

  network_interfaces = [{
    name       = "eth0"
    subnet_id  = nebius_vpc_v1_subnet.worker.id
    ip_address = {}
    public_ip_address = {
      static = true
    }
    security_groups = [{ id = nebius_vpc_v1_security_group.worker.id }]
  }]

  cloud_init_user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    ssh_authorized_key = var.ssh_authorized_key
    bootstrap_b64      = base64encode(file("${path.module}/../bootstrap.sh"))
    run_worker_b64     = base64encode(file("${path.module}/../run-worker.sh"))
    systemd_unit_b64   = base64encode(file("${path.module}/../wake-worker.service"))
    worker_env_b64 = base64encode(templatefile("${path.module}/worker.env.tftpl", {
      image                 = var.worker_image
      region                = var.region
      access_key_id         = nebius_iam_v2_access_key.storage.status.aws_access_key_id
      secret_access_key     = nebius_iam_v2_access_key.storage.status.secret
      refinement_mode       = var.refinement_mode
      temporal_stride       = var.temporal_stride
      video_decoder         = var.video_decoder
      cuda_graph_batch_size = var.cuda_graph_batch_size
    }))
  })

  depends_on = [
    nebius_iam_v1_access_permit.project_viewer,
    nebius_iam_v1_group_membership.worker,
    nebius_vpc_v1_security_rule.api,
    nebius_vpc_v1_security_rule.egress,
  ]
}
