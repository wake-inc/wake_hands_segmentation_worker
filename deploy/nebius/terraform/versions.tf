terraform {
  required_version = ">= 1.10.0"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = "~> 0.6"
    }
  }
}

provider "nebius" {
  parent_id = var.project_id
  profile   = var.nebius_profile == null ? null : { name = var.nebius_profile }
}
