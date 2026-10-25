# Terraform module — Azure-emulated / local K3s infrastructure for ai-copilot
#
# Provisions the minimal cloud resources needed to run the Enterprise Analytics
# Copilot stack on a K3s cluster (Azure emulator, local K3s, or a real AKS cluster).
#
# Usage:
#
#   terraform init
#   terraform plan -var="environment=dev"
#   terraform apply -var="environment=dev"

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = null
}

variable "k3s_cluster_name" {
  description = "Name of the K3s cluster (for tagging)"
  type        = string
  default     = "ai-copilot-k3s"
}

variable "vm_size" {
  description = "Azure VM size for the K3s node pool"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "postgres_enabled" {
  description = "Whether to provision an Azure Database for PostgreSQL"
  type        = bool
  default     = false
}

variable "neo4j_enabled" {
  description = "Whether to provision a managed Neo4j (Azure Marketplace offer)"
  type        = bool
  default     = false
}

# ---------------------------------------------------------------------------
# Resource group
# ---------------------------------------------------------------------------

locals {
  resource_group_name = var.resource_group_name != null ? var.resource_group_name : "rg-ai-copilot-${var.environment}"
  common_tags = {
    environment = var.environment
    managed-by  = "terraform"
    project     = "ai-copilot"
  }
}

resource "azurerm_resource_group" "rg" {
  name     = local.resource_group_name
  location = var.location
  tags     = local.common_tags
}

# ---------------------------------------------------------------------------
# Optional: Azure Database for PostgreSQL Flexible Server
# ---------------------------------------------------------------------------

resource "azurerm_postgresql_flexible_server" "postgres" {
  count = var.postgres_enabled ? 1 : 0

  name                = "pg-ai-copilot-${var.environment}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = var.location
  administrator_login = "aiocopilot"

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "resource_group_name" {
  value       = local.resource_group_name
  description = "Name of the resource group"
}

output "postgres_enabled" {
  value       = var.postgres_enabled
  description = "Whether PostgreSQL is enabled"
}

output "postgres_fqdn" {
  value       = var.postgres_enabled ? azurerm_postgresql_flexible_server.postgres[0].fqdn : null
  description = "Fully qualified domain name of the PostgreSQL server (if enabled)"
}

output "postgres_connection_string" {
  value       = var.postgres_enabled ? "postgresql://aiocopilot@${azurerm_postgresql_flexible_server.postgres[0].fqdn}:5432/postgres" : null
  description = "Connection string for the PostgreSQL server (if enabled)"
  sensitive   = true
}
