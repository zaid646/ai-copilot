# Enterprise Analytics Copilot

AI-native analytics copilot that answers **"Why did sales drop in the West region last quarter, and what should we do?"** end-to-end — structured SQL KPIs, unstructured policy document retrieval (GraphRAG), demand forecasting, multi-agent LangGraph supervision, fail-closed guardrails, and full observability — with every step traceable and auditable.

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

Postgres holds structured sales/finance facts. Neo4j holds GraphRAG policy-chunk retrieval. FastAPI is the REST backend with JWT RBAC. Streamlit is the chat UI. LangGraph orchestrates the agent loop (planner → tool agents → critic → synthesis). vLLM on a remote RTX 4090 serves chat + embeddings via an SSH tunnel.

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
make frontend
```

Run checks:
```bash
make test   # pytest — 11/11 pass
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
