.PHONY: help up down logs seed ingest test lint api frontend clean helm-template kubeval terraform-validate helm-upgrade helm-uninstall

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## Start the full stack (postgres, neo4j, prometheus, grafana, api, frontend)
	docker compose up -d --build

down:  ## Stop the stack
	docker compose down

logs:  ## Tail logs from all services
	docker compose logs -f --tail=100

seed:  ## Seed Postgres (dim/fact tables) using the local venv
	.venv/bin/python db/seed.py

ingest:  ## Re-ingest policy PDFs into Neo4j (needs the LLM embed endpoint up)
	.venv/bin/python ingest/graphrag_ingest.py

batch:  ## Run the batch KPI aggregation
	.venv/bin/python jobs/batch_agg.py

test:  ## Run the API test suite
	.venv/bin/python -m pytest tests/ -v

lint:  ## Ruff lint
	ruff check .

api:  ## Run the API locally (uvicorn, reload)
	.venv/bin/uvicorn api.main:app --reload --port 8000

frontend:  ## Run the Streamlit app locally
	API_BASE_URL=http://localhost:8000 .venv/bin/streamlit run frontend/app.py

clean:  ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .ruff_cache .pytest_cache

helm-template:  ## Render Helm chart templates to stdout
	helm template ai-copilot ./deploy/helm/ai-copilot

kubeval:  ## Validate rendered Helm manifests with kubeval
	helm template ai-copilot ./deploy/helm/ai-copilot > /tmp/ai-copilot-rendered.yaml
	kubeval /tmp/ai-copilot-rendered.yaml --strict
	kubeval deploy/k8s/*.yaml --strict

terraform-validate:  ## Validate Terraform configuration
	cd deploy/terraform/k3s-azure && terraform init && terraform validate

helm-upgrade:  ## Upgrade/install Helm chart (local K3s)
	helm upgrade --install ai-copilot ./deploy/helm/ai-copilot --namespace default --create-namespace

helm-uninstall:  ## Uninstall Helm chart
	helm uninstall ai-copilot --namespace default
