# ADR 0003 — GraphRAG as an agent tool, not a standalone service

- **Status:** Accepted
- **Date:** 2026-10-17

## Context

Policy and playbook PDFs must be retrievable with citations, and ideally linked
to entities mentioned in the structured data (regions, categories, programs).
Options were: (a) a standalone RAG microservice, (b) a vector store inside
Postgres (pgvector), or (c) the Neo4j GraphRAG engine exposed to agents purely
as a tool.

## Decision

Ingest PDFs into **Neo4j** using the **Neo4j GraphRAG** library (chunk vector
index + full-text index + entity/relationship graph derived by LLM extraction).
Expose it to the supervisor as a single `graphrag_search(query, filters)` tool
implemented in-process. Neo4j runs as a container; no separate RAG service is deployed.

## Consequences

**Positive**
- One tool contract for the agent; embeddings and graph stay behind it.
- Graph neighbourhood expansion gives citations ("policy → section → entity")
  that a flat vector store cannot.
- No extra network hop / service to operate; less RAM than a second app container.

**Negative**
- Neo4j adds ~1.5–2 GB RAM footprint (heap capped at 1 GB).
- Entity extraction costs LLM calls at ingest; mitigated by page caps and chunk reuse.

## Alternatives considered

1. **Standalone RAG service.** Rejected: operational overhead, no graph benefit for this phase.
2. **pgvector in Postgres.** Simpler, but loses the knowledge-graph relationships that make "why" answers explainable.
3. **LangChain VectorStore-only.** Rejected: no entity graph, weaker multi-hop story.
