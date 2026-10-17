# ADR 0002 — Postgres as warehouse (not DuckDB)

- **Status:** Accepted
- **Date:** 2026-10-17

## Context

The warehouse must serve (a) batch KPI aggregation, (b) a near-real-time event
consumer writing concurrently, and (c) an API layer with connection pooling and
RBAC'd reads. DuckDB is excellent for single-writer local analytics but is a
process-embedded OLAP engine: concurrent writers and a long-running server
process with pooled connections are not its sweet spot.

## Decision

Use **PostgreSQL 16** (Docker) as the single warehouse for facts, dimensions,
aggregates, agent traces and conversations. The read-only SQL tool is enforced
against this database with a `SELECT`-only guard + statement timeout.

## Consequences

**Positive**
- Concurrent batch job + stream consumer + API reads without file locking games.
- Real production shape: migrations, pooling (`psycopg_pool`), permissions.
- Traces and conversations live next to the data they explain.

**Negative**
- Slightly heavier than DuckDB locally (~ed 512 MB container limit).
- No columnar vectorised scans; acceptable at demo data scale.

## Alternatives considered

1. **DuckDB.** Great locally, but single-writer + embedded model fights the stream consumer and the API server.
2. **SQLite.** Even more constrained concurrency; no server RBAC.
3. **Snowflake/BigQuery.** Out of scope for a local, reproducible demo.
