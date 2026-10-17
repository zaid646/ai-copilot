# Enterprise Analytics Copilot

AI-native analytics copilot that answers **"Why did sales drop in the West region last quarter, and what should we do?"** end-to-end — structured SQL KPIs, unstructured policy document retrieval (GraphRAG), demand forecasting, multi-agent LangGraph supervision, fail-closed guardrails, and full observability — with every step traceable and auditable.

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Streamlit  │──────│   FastAPI (8000) │──────│  LangGraph      │
│   UI (8501)  │ JWT  │  /ask /trace     │      │  Supervisor     │
└──────────────┘      └─────────┬────────┘      │  (planner →     │
                                │               │   sql_agent →   │
┌──────────────┐               │               │   graphrag_agent │
│   Postgres   │◄──────────────┘               │   model_agent → │
│  fact_sales  │──────────────────────────────►│   action_agent  │
│   fact_fin   │      structured KPI           │   critic_agent  │
└──────────────┘      computation             └─────────┬─────────┘
                                                        │
┌──────────────┐                                        │
│    Neo4j     │────────────────────────────────────────┘
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

## Demo GIF

Add `docs/demo.gif` here — a 30–60 second screen recording of:
1. Login as analyst → ask the pilot question
2. Answer with citations + KPI cards populate
3. Trace tab: node-by-node latencies
4. History tab → open trace
5. Logout → login as admin → KPI chart renders
6. Guardrail block banner on an injection prompt

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

Run checks:
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

## Limitations

- **8-case golden set** — pilot-scoped evaluation; production would expand to ~25+ hand-labeled questions
- **Pilot scope** — optimized for the West-region sales-drop question; other regions/quarters use the same paths but may need threshold tuning
- **LLM endpoint** — remote vLLM on RTX 4090 via SSH tunnel; local LM Studio fallback configured but not loaded
- **Authentication** — JWT + RBAC demo roles only (analyst/admin), no enterprise SSO
- **Free-tier LLM** — Gemini quota exhausted; remote GPU instance incurs hourly cost

## JD Mapping (Apple)

- **Full-stack AI/LLM application development** — FastAPI backend + Streamlit frontend + LangGraph agent loop + Docker compose orchestration
- **RAG / retrieval systems** — Neo4j GraphRAG with hybrid vector + fulltext, domain-aware source filtering, retrieval-grounded answers with citations
- **SQL / data engineering** — Postgres warehouse schema, seed jobs, batch KPI aggregation, deterministic number computation (no LLM-hallucinated figures)
- **ML / forecasting** — LSTM weekly demand model with train/evaluate/serve pipeline and model card
- **AI safety / guardrails** — fail-closed ingress/egress policy engine, prompt-injection detection, PII redaction, citation grounding, full audit log
- **Observability / evals** — Prometheus + Grafana, LLM-as-judge golden set at temperature=0, hard-fail/soft-warn thresholds, per-request trace IDs
- **CI / shipping** — ruff lint, pytest suite (10/10), Makefile targets, Docker images for API + frontend, GitHub Actions ready
