# PRD — Enterprise Analytics Copilot ("Why did sales drop in West region last quarter, and what should we do?")

- **Status:** Approved (simulated cross-functional sign-off — see `docs/partner-review.md`)
- **Author:** Data & AI Platform
- **Version:** 1.0

## 1. Problem statement

Regional leaders and finance analysts can see *that* a KPI moved, but answering **why** requires stitching together structured data (sales, finance), unstructured documents (policy, playbooks), and forecasts. Today this takes 2–4 analyst days per question and the reasoning is not auditable.

**Pilot question:** *"Why did sales drop in the West region last quarter, and what should we do?"*

## 2. Goals

1. Answer the pilot question end-to-end in < 60 seconds with citations.
2. Every answer traceable node-by-node (planner → SQL → docs → model → critic).
3. Exact numbers come from deterministic KPI computation, never from the LLM's memory.
4. Policy/playbook recommendations cite source documents (GraphRAG).
5. Safety: prompt-injection, PII and unsafe-output handling is fail-closed and audited.

## 3. Non-goals (out of scope)

- Real-time transactional writes from external ERP/CRM systems.
- Production SSO / enterprise identity (JWT + RBAC demo roles only).
- Forecasting SKUs below region×category×month granularity.
- Fine-tuning foundation models.

## 4. Personas

| Persona | Need | Success signal |
|---|---|---|
| Regional Sales Manager | Understand the dip, get an action plan | Answer cites numbers + policy steps |
| Finance Analyst | Verify the numbers independently | KPI cards + trace show SQL and sources |
| Platform Admin | Trust & operate the system | Metrics, guardrail audit log, RBAC |

## 5. KPIs of the copilot itself

| Metric | Target |
|---|---|
| End-to-end p95 latency | < 60 s |
| Answer groundedness (LLM-judge) | ≥ 0.80 |
| Retrieval hit-rate (Ragas context precision) | ≥ 0.70 |
| Golden-set accuracy (~25 questions) | ≥ 70% |
| Guardrail block recall (injection corpus) | 100% on test corpus |

## 6. Data sources

| Source | Type | Path |
|---|---|---|
| Sales & finance facts | Structured | Postgres (`fact_sales`, `fact_finance`, …) |
| Live sales events | Near-real-time stream | `raw_sales_events` → consumer → `fact_sales` |
| Policy & playbooks | Unstructured (PDF) | Neo4j GraphRAG (`samples/policies/*.pdf`) |
| Demand forecast model | ML service | FastAPI `/predict` |

## 7. User experience

1. Analyst logs in (JWT), asks the pilot question in a chat UI.
2. UI streams: answer, KPI cards, agent trace timeline, citations.
3. Guardrail-blocked queries show a red banner with the policy reason.
4. Query history persisted and re-openable.

## 8. Success criteria (acceptance)

- `docker compose up` brings the full stack online.
- One command seeds data; batch job produces `agg_daily_kpi`; stream writes live rows; PDFs queryable via GraphRAG.
- The pilot question produces: root-cause numbers (SQL), policy-backed actions (GraphRAG), forecast (model), critic verification — with trace IDs.
- CI runs lint + tests + eval gate.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM hallucinated numbers | Deterministic KPI tool is the only number source; egress schema check |
| Free-tier LLM rate limits | Retry/backoff + provider-agnostic layer (Gemini primary) |
| Graph extraction cost on large PDFs | Page-capped ingestion, chunk-level vector + fulltext index |
| Local RAM pressure | Container memory limits; Neo4j heap capped at 1 GB |
