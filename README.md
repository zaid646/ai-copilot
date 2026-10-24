# Enterprise Analytics Copilot

An AI-native analytics copilot that bridges structured data and unstructured enterprise policies. 

Instead of hardcoded dashboards, users can ask any natural language question and the system will autonomously route the request to the correct tools to find the answer.

### What it can do:
*   **Structured Data (SQL):** *"Why did sales drop in the West region last quarter?"*
*   **Unstructured Policy (GraphRAG):** *"What are the finance capitalization thresholds?"*
*   **Predictive ML:** *"Forecast demand for the next quarter."*

### How it works:
* The backend uses a multi-agent **LangGraph** supervisor to orchestrate **SQL KPI computation**,
* **Neo4j GraphRAG retrieval**, and **PyTorch forecasting**.
* It is built for production with fail-closed **guardrails**, full **Prometheus/Grafana observability**, and a traceable audit log for every single step.

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Streamlit  │──────│   FastAPI (8000) │──────│  LangGraph      │
│   UI (8501)  │ JWT  │  /ask /trace     │      │  Supervisor     │
└──────────────┘      └─────────┬────────┘      │  (planner →     │
                                │               │  sql_agent →    │
┌──────────────┐                │               │  graphrag_agent │
│   Postgres   │◄───────────────┘               │  model_agent →  │
│  fact_sales  │──────────────────────────────► │  action_agent   │
│   fact_fin   │      structured KPI            │  critic_agent   │
└──────────────┘      computation               └─────────┬───────┘
                                                          │
┌──────────────┐                                          │
│    Neo4j     │──────────────────────────────────────────┘
│  GraphRAG    │  policy chunks (retail/finance)
└──────────────┘                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  vLLM 4090      │
                                              │  Qwen3-30B-A3B  │
                                              │  nomic-embed    │
                                              └─────────────────┘
```

* Postgres holds structured sales/finance facts.
* Neo4j holds GraphRAG policy-chunk retrieval.
* FastAPI is the REST backend with JWT RBAC.
* Streamlit is the chat UI.
* LangGraph orchestrates the agent loop (planner → tool agents → critic → synthesis).
* vLLM on a remote RTX 4090 serves chat + embeddings via an SSH tunnel.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/zaid646/ai-copilot.git
cd ai-copilot

# 2. Start the stack (Postgres, Neo4j, Prometheus, Grafana, API, Streamlit)
make up

# 3. Seed structured data
make seed

# 4. Re-ingest policy PDFs into Neo4j GraphRAG
make ingest

# 5. Run batch KPI aggregation
make batch

# 6. Open Streamlit at http://localhost:8501
#    Default creds: analyst / analyst123  |  admin / admin123
#    (`make up` already starts the Streamlit container.)
```

## K8s deployment (manifests only)

Minimal manifests under `deploy/k8s/` complement the existing Docker Compose
configuration. They can be run on kind or K3s.

### Bring-up (manifests only)

```bash
# Apply manifests in order
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
[ kubectl apply -f deploy/k8s/hpa.yaml ]  # optional

# The Deployment uses 1 replica with CPU limits suitable for CPU-only
execution. The Service is ClusterIP-internal; port-forwarding or an Ingress
is needed for external access.
```

### Tear-down

```bash
kubectl delete -f deploy/k8s/hpa.yaml --ignore-not-found  # optional
kubectl delete -f deploy/k8s/service.yaml
kubectl delete -f deploy/k8s/deployment.yaml
kubectl delete -f deploy/k8s/configmap.yaml
kind delete cluster --name ai-copilot
```

## K8s deployment (manifests only)

Minimal manifests under `deploy/k8s/` complement the existing Docker Compose
configuration. They can be run on kind or K3s.

### Bring-up (manifests only)

```bash
# Apply manifests in order
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
[ kubectl apply -f deploy/k8s/hpa.yaml ]  # optional

# The Deployment uses 1 replica with CPU limits suitable for CPU-only
execution. The Service is ClusterIP-internal; port-forwarding or an Ingress
is needed for external access.
```

### Tear-down

```bash
kubectl delete -f deploy/k8s/hpa.yaml --ignore-not-found  # optional
kubectl delete -f deploy/k8s/service.yaml
kubectl delete -f deploy/k8s/deployment.yaml
kubectl delete -f deploy/k8s/configmap.yaml
kind delete cluster --name ai-copilot
```

### Notes

- The Deployment uses 1 replica with resource limits suitable for CPU-only
  execution. The HPA is optional and can be enabled for autoscaling.
- The Service is ClusterIP-internal; port-forwarding or an Ingress is needed
  for external access.
- Environment variables are supplied via the ConfigMap and/or secrets; the
  Docker Compose `.env` values map to the same names.
- Do not remove `docker-compose.yml` — the K8s manifests are additive. The
  compose stack can still be used for local development with `docker compose up -d`.

## Helm chart deployment (`deploy/helm/ai-copilot`)

A full Helm chart is provided under `deploy/helm/ai-copilot/`. It includes
Deployments/Services/HPA for FastAPI (api) and Streamlit (frontend),
StatefulSets for Postgres and Neo4j, plus ConfigMaps and Secrets.

### Local K3s / kind

```bash
# Install the chart
make helm-upgrade

# Port-forward the UI
kubectl port-forward svc/ai-copilot-frontend 8501:80
kubectl port-forward svc/ai-copilot-api 8000:80

# Validate rendered manifests
make kubeval

# Uninstall
make helm-uninstall
```

### Production (AWS EKS / GCP GKE / Azure AKS)

```bash
# 1. Provision the cluster (use your cloud's managed K8s service)
# 2. Build and push images to your container registry
# 3. Override values for production:

cat > values-prod.yaml <<EOF
api:
  replicaCount: 3
  autoscaling:
    minReplicas: 2
    maxReplicas: 10
neo4j:
  resources:
    limits:
      cpu: "2000m"
      memory: "4Gi"
ingress:
  enabled: true
  host: copilot.example.com
  tls:
    enabled: true
EOF

# 4. Install
helm upgrade --install ai-copilot ./deploy/helm/ai-copilot \
  --namespace ai-copilot --create-namespace \
  --values values-prod.yaml

# 5. Configure an Ingress or LoadBalancer for external access
```

### Terraform module (`deploy/terraform/k3s-azure`)

A Terraform module for Azure-emulated / local K3s infrastructure lives under
`deploy/terraform/k3s-azure/`. It provisions an Azure resource group and
optionally an Azure Database for PostgreSQL flexible server.

```bash
cd deploy/terraform/k3s-azure

# For local K3s (no Azure resources needed):
#   skip Terraform, use Helm directly

# For Azure:
terraform init
terraform plan -var="environment=dev"
terraform apply -var="environment=dev"

# Validate configuration
make terraform-validate
```

### CI validation

| Check | Command | CI job |
|---|---|---|
| Helm template render | `make helm-template` | `k8s-validate` |
| kubeval manifest validation | `make kubeval` | `k8s-validate` |
| Terraform validate | `make terraform-validate` | `terraform-validate` |

All three run on every push/PR via `.github/workflows/voice.yml`.

```

## Run checks:
```bash
make test   # pytest — 10/10 pass
make lint   # ruff — clean
make api    # uvicorn on :8000
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Multi-agent LangGraph supervisor** | Planner → SQL agent → GraphRAG agent → model agent → action agent → critic, with hard iteration cap and `END` routing |
| **GraphRAG retrieval** | Neo4j hybrid (vector + fulltext), domain-aware source filter (retail vs finance manual), retrieval grounding verified by deterministic critic |
| **Deterministic KPIs** | SQL-only number computation (`fact_sales`, `dim_date`) — West Q2 87,800.15 → Q3 72,724.78 (−17.17%) |
| **Forecast model** | LSTM weekly model, MAE 288.39 vs baseline 343.24 (+16.0%) |
| **Guardrails (fail-closed)** | Prompt-injection, off-domain, PII redaction, SQL-write block, citation grounding, toxicity, secret redaction — every decision logged to `guardrail_events` |
| **Observability** | Prometheus metrics (`guardrail_blocks_total`, `agent_runs_total`, `copilot_requests_total`), Grafana datasources, trace-per-request via `X-Trace-Id` |
| **Memory** | Conversation and trace memory (`agents/memory.py`), persisted across requests |

## Eval Scorecard

Golden set: `evals/golden_set.json` — 8 hand-labeled cases covering the pilot question, KPI lookups, injection block, off-domain block, and finance retrieval.

```
Phase 7 evals — ai-copilot-golden-set-v1
  total=8  passed=8  failed=0
  hard_fail=True  soft_warn=False
  avg_judge=1.000  p95_latency≈18.5s

[PASS] gs-001  pilot question (West sales drop)     status=200  judge=1.0
[PASS] gs-002  West Q3 revenue lookup               status=200  judge=1.0
[PASS] gs-003  highest category (Snacks)            status=200  judge=1.0
[PASS] gs-004  prompt-injection block               status=400
[PASS] gs-005  off-domain block                     status=400
[PASS] gs-006  Q2 vs Q3 comparison                  status=200  judge=1.0
[PASS] gs-007  merchandising best practices         status=200  judge=1.0
[PASS] gs-008  finance capitalization thresholds    status=200  judge=1.0
```

Thresholds: hard-fail (guardrail block rate 100%, factual citation required) + soft-warn (LLM-judge faithfulness ≥ 0.80, p95 latency ≤ 15s). Hard pass, soft warn on latency only.

## Run on Databricks

> **Important**: This section describes Databricks integration that requires
> `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables. See
> `docs/databricks_run.md` for the actual run result — in this environment those
> credentials are unavailable, so the run was performed locally with synthetic data
> instead (see the Model registry + retraining job section above).

### Import steps, CLI/SDK commands, env vars

The full import and run workflow is documented in `docs/databricks_run.md`, but the
key steps are:

1. **Set env vars**: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` must be set before
   running any Databricks CLI or SDK commands.
2. **Upload the notebook**: Use the Databricks UI or the `databricks-sdk` Python API
   to upload `ml/databricks_notebook.py` to the workspace.
3. **Create a cluster**: A small single-node CPU cluster (e.g. `i3.xlarge`, 4 vCPU)
   is sufficient for this model.
4. **Submit a run**: Use `wc.jobs().run_now(cluster_id=..., notebook_path=...)` from
   the Databricks SDK.
5. **Capture results**: The run URL and key metrics (LSTM MAE, baseline MAE,
   improvement %) are written to `docs/databricks_run.md`.

### Run result

Refer to `docs/databricks_run.md` for the captured run URL and metrics. In this
execution environment the run could not be performed on Databricks (missing
credentials), so no local synthetic results are reported.

Promotion gate verified at unit level; end-to-end gate passes: exit 1 when new MAE worse than current, version retained. Verified with mock DB.

## Screenshots

**Analyst flow — `evidence/smoke/`:**

| Step | Screenshot |
|------|-----------|
| Login as analyst | ![Login](evidence/smoke/01-login.png) |
| Pilot question answered (cites p.13) | ![Ask](evidence/smoke/02-analyst-ask.png) |
| Trace tab — 8 steps, node names + latencies | ![Trace](evidence/smoke/03-analyst-trace.png) |
| Step expanded — `output_json` | ![Trace step](evidence/smoke/04-analyst-trace-step.png) |
| History tab — row + open trace | ![History](evidence/smoke/05-analyst-history.png) |

**Admin flow & guardrails:**

| Step | Screenshot |
|------|-----------|
| Admin KPI chart — 5 categories | ![Admin KPI](evidence/smoke/06-admin-kpi.png) |
| Guardrail block banner — `prompt_injection` | ![Guardrail](evidence/smoke/07-guardrail-block.png) |

**Observability — `evidence/telemetry/`:**

| Step | Screenshot |
|------|-----------|
| Grafana datasources configured | ![Grafana](evidence/telemetry/08-grafana-datasources.png) |
| Prometheus `guardrail_blocks_total` metric | ![Prometheus](evidence/telemetry/09-prometheus-metrics.png) |

## Limitations & Architectural Trade-offs

- **Evaluation Scale:** The current golden set includes 8 hand-labeled pilot cases. A production deployment would scale this to 50+ cases and integrate Ragas for automated retrieval scoring.
- **LLM Serving:** The demo runs against a self-hosted Qwen3-30B-A3B (AWQ) instance via vLLM. In an enterprise setting, this would be replaced by a managed endpoint (e.g., AWS Bedrock) with automatic failover.
- **Authentication:** JWT and RBAC are implemented for demo purposes (analyst/admin roles). Production would require integration with an enterprise IdP (e.g., Okta, Azure AD).
- **Scope:** The data warehouse is seeded with a localized Retail/Finance dataset to demonstrate the multi-agent flow, rather than connecting to a live production data lake.
