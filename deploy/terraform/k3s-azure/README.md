# Terraform module — K3s/Azure emulator for ai-copilot
#
# This module provisions the minimal Azure infrastructure needed to run the
# Enterprise Analytics Copilot stack on a K3s cluster.
#
# For local K3s (no Azure), skip Terraform entirely and use Helm directly:
#
#   helm upgrade --install ai-copilot ./deploy/helm/ai-copilot
#
# For Azure AKS:
#
#   terraform init
#   terraform apply -var="environment=prod"
#
# Then deploy Helm onto the AKS cluster:
#
#   az aks get-credentials --resource-group <rg> --name <cluster>
#   helm upgrade --install ai-copilot ./deploy/helm/ai-copilot \
#     --values values-prod.yaml

## Example terraform.tfvars

environment = "dev"
location    = "eastus"

# Enable managed PostgreSQL (optional; by default Postgres runs in-cluster)
postgres_enabled = false

# Example for production:
# environment = "prod"
# location    = "eastus"
# vm_size     = "Standard_D4s_v3"
# postgres_enabled = true
