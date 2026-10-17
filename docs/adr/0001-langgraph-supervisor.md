# ADR 0001 — LangGraph supervisor over a single monolithic agent

- **Status:** Accepted
- **Date:** 2026-10-17
- **Deciders:** Data & AI Platform, Business stakeholders

## Context

The pilot question requires four distinct capability classes: deterministic
numbers (SQL/KPI), document retrieval (GraphRAG), prediction (ML model), and
action synthesis. A single agent with a large tool list tends to (a) skip
deterministic KPI checks in favour of eyeballed SQL, (b) conflate policy text
with numbers, and (c) provide no auditable plan.

## Decision

Adopt a **LangGraph supervisor** that routes to specialized nodes:
`planner → sql_agent → graphrag_agent → model_agent → action_agent → critic`,
with a bounded execution loop, shared scratchpad memory, and trace IDs on
every transition.

## Consequences

**Positive**
- Each node has a narrow prompt and a narrow tool contract → testable in isolation.
- The critic node can enforce "numbers only from KPI tool" and guardrail policy.
- Trace IDs map 1:1 to nodes → the UI can render a node-by-node timeline.
- A failed node has a fallback path without poisoning the whole run.

**Negative / trade-offs**
- More LLM calls per question than a single agent (mitigated by cheap, fast models on free tier).
- State schema is a contract that must be versioned.

## Alternatives considered

1. **Single ReAct agent with all tools.** Rejected: weak auditability, tool conflation, no planner artifact.
2. **Hard-coded DAG (no LLM routing).** Rejected: cannot handle ambiguous follow-ups.
3. **Two-agent (router + worker).** Rejected: agent handoff between retrieval and action is part of the demo value.
