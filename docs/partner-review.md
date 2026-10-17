# Partner review note (simulated cross-functional sign-off)

**Meeting:** West-region sales dip analysis — copilot pilot
**Attendees:** Regional Sales (business owner), Finance Analytics, Data & AI Platform (engineering), Security & Governance (reviewer)
**Date:** 2026-10-17

## Business framing (Regional Sales)

- Agreed pilot question: *"Why did sales drop in West region last quarter, and what should we do?"*
- Root causes the business cares about, in priority order:
  1. Volume vs price decomposition by category.
  2. Regional promo calendar effects (documented in playbooks).
  3. Customer churn / account losses.
- **Ask:** recommendations must map to an existing playbook step where one exists; otherwise clearly flag "no policy on file".

## Finance (analyst)

- Requires a **deterministic KPI layer** — finance will not accept numbers re-derived by an LLM.
- Requires the ability to open the trace and see the exact SQL executed (read-only).
- Accepted definition of the drop: `revenue(Q-1) - revenue(Q-2)` on `fact_sales` at region×category×month.

## Engineering (platform)

- Design accepted per ADRs 0001–0003: LangGraph supervisor, Postgres warehouse, GraphRAG-as-tool.
- Definition of done for the pilot: `docker compose up` + seeded data + the pilot question answered with a persisted trace.
- Non-functional: p95 < 60 s; runs within a 16 GB laptop budget (container limits enforced).

## Security & governance

- **Approved with conditions:**
  1. Read-only SQL enforced (keyword guard + DB role + statement timeout).
  2. Guardrails fail **closed**: on a guardrail breach the request is blocked, logged with a trace ID, and surfaced in the UI.
  3. PII redaction on ingress before any prompt is built.
  4. Every guardrail decision is queryable by trace ID (`guardrail_events` table).

## Sign-off

| Area | Decision |
|---|---|
| Business | ✅ Approved — pilot question scoped |
| Finance | ✅ Approved — deterministic KPI requirement recorded |
| Engineering | ✅ Approved — ADRs 0001–0003 accepted |
| Security | ✅ Approved with conditions 1–4 (tracked as acceptance tests) |

**Open follow-ups (post-pilot):** SSO integration, source-system write-backs, cost dashboard for paid LLM tiers.
