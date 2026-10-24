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
  description = "Name of the resource group (auto-generated if null)"
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
  description = "Whether to provision a managed Neo4j"
  type        = bool
  default     = false
}
