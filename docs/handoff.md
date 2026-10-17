# Handoff — Enterprise Analytics Copilot

Brief operational runbook for restarting the stack, rotating the LLM endpoint, and viewing logs.

## 1. Restart the full stack

```bash
cd /Users/zaid/Documents/ai-copilot

# Start all containers (Postgres, Neo4j, Prometheus, Grafana, API, Streamlit)
make up

# Verify services
curl http://localhost:8000/health          # API
curl http://localhost:8501/                # Streamlit UI
curl http://localhost:9090/                # Prometheus
curl http://localhost:3000/                # Grafana (admin/admin)
```

Default credentials:
- Analyst: `analyst` / `analyst123`
- Admin: `admin` / `admin123`

## 2. Seed data + ingest policy docs

```bash
# Structured sales/finance facts into Postgres
make seed

# Policy PDFs into Neo4j GraphRAG (requires LLM embed endpoint up)
make ingest

# Batch KPI aggregation (West Q2 vs Q3 comparison)
make batch
```

## 3. Rotate the LLM endpoint

The active chat/embed provider is set in `.env`:

```
CHAT_PROVIDER=openai_compat
EMBED_PROVIDER=openai_compat
OPENAI_CHAT_BASE_URL=http://localhost:8100/v1
OPENAI_EMBED_BASE_URL=http://localhost:8101/v1
OPENAI_CHAT_MODEL=qwen3-30b-a3b
OPENAI_EMBED_MODEL=nomic-embed-text-v1.5
OPENAI_API_KEY=<vllm-api-key>
```

To switch providers, edit `.env` and restart the API:
- **LM Studio (local fallback):** `CHAT_PROVIDER=lmstudio`, `EMBED_PROVIDER=lmstudio`, `LMSTUDIO_BASE_URL=http://localhost:1234/v1`
- **Gemini (free tier, quota-exhausted):** `CHAT_PROVIDER=gemini`, `EMBED_PROVIDER=gemini`, `GEMINI_API_KEY=<key>`
- **Remote vLLM (current):** keep `openai_compat` and point the SSH tunnel at the new instance host/port

Restart API after `.env` changes:
```bash
pkill -f "uvicorn api.main:app"
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## 4. Restart the Vast.ai tunnel (remote LLM)

Current tunnel: local `8100→remote 8000` (chat), `8101→remote 8001` (embed).

```bash
# Kill existing tunnel (find by PID, not by pattern)
ps -eo pid,args | awk '/[u]vicorn api.main:app|ssh.*vast_ai/{print $1}' | xargs -r kill

# Start new tunnel (replace host/port/key as needed)
ssh -i ~/.ssh/vast_ai -p <SSH_PORT> -N \
  -L 8100:127.0.0.1:8000 -L 8101:127.0.0.1:8001 \
  root@<INSTANCE_HOST>

# Verify
curl -s http://localhost:8100/v1/models -H "Authorization: Bearer <VLLM_API_KEY>"
curl -s http://localhost:8101/v1/models -H "Authorization: Bearer <VLLM_API_KEY>"
```

If spinning up a fresh Vast.ai RTX 4090 instance:
1. Install torch 2.11.0+cu128 and vLLM 0.21.0+cu129 (GitHub release wheel — PyPI vLLM pulls CUDA 13 which breaks on driver ≤12.7)
2. Download models to `/workspace/.hf_home`: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`, `nomic-ai/nomic-embed-text-v1.5`
3. Run supervisor services: `vllm.entrypoints.openai.api_server` on ports 8000/8001 with `VLLM_API_KEY` env var

## 5. View logs

```bash
# All docker containers
make logs

# API (local uvicorn)
tail -f /tmp/aicopilot_api.log

# Streamlit (local)
tail -f /tmp/aicopilot_frontend.log

# vLLM on remote instance (via SSH)
ssh <instance> "tail -f /var/log/portal/vllm-chat.log"
ssh <instance> "tail -f /var/log/portal/vllm-embed.log"

# Supervisor status on remote
ssh <instance> "supervisorctl status vllm-chat vllm-embed"
```

## 6. Run checks

```bash
make test    # pytest — 11/11 pass
make lint    # ruff — clean
make api     # uvicorn reload on :8000
make frontend # Streamlit on :8501
```

Eval suite (Phase 7):
```bash
.venv/bin/python -m evals.evaluator   # 8/8 pass, hard_fail=True, avg_judge=1.0
```

## 7. Key files

| Path | Purpose |
|------|---------|
| `agents/graph.py` | LangGraph supervisor, `run_agent()`, `recursion_limit` |
| `agents/domain.py` | Domain/intent classification, `source_filter_for` |
| `agents/nodes/` | Planner, SQL agent, GraphRAG agent, model agent, action agent, critic |
| `guardrails/checks.py` | Fail-closed ingress/egress policy engine |
| `api/main.py` | FastAPI: `/ask`, `/trace`, `/history`, `/kpi/*`, `/metrics`, `/auth/login` |
| `frontend/app.py` | Streamlit chat UI |
| `evals/golden_set.json` | 8-case hand-labeled golden set |
| `evals/evaluator.py` | Temp=0 LLM-as-judge runner with disk cache |
| `db/schema.sql`, `db/seed.py` | Postgres schema + seed |
| `ingest/graphrag_ingest.py` | Neo4j GraphRAG ingestion |
| `jobs/batch_agg.py`, `jobs/stream_simulator.py` | KPI aggregation + live event simulation |
| `docker-compose.yml` | Full stack orchestration |
| `Makefile` | up/down/seed/ingest/batch/test/lint/api/frontend |

## 8. Evidence

Screenshots committed for Phase 8 demo/README:
- `evidence/smoke/01-login.png` → `07-guardrail-block.png`
- `evidence/telemetry/08-grafana-datasources.png`, `09-prometheus-metrics.png`
