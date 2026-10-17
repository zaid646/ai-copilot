"""Tool: graphrag_search — hybrid retrieval over ingested policy PDFs in Neo4j.

Combines vector (semantic) and full-text (keyword) search with Reciprocal Rank
Fusion (RRF). Chunks and queries are embedded locally via LM Studio
(`nomic-embed`, 768-dim, quota-free) using nomic's `search_document`/`search_query`
task prefixes. Keyword search is retained because queries that hinge on specific
terms ("capitalization threshold", "budget transfer") benefit from exact matching.

Returns grounded chunks with source + page so the answer layer can cite them.
"""
from __future__ import annotations

import re
import time
from typing import Any

import neo4j

from core.config import settings
from core.llm import LLMError, embed_texts
from core.logging import get_logger
from observability.metrics import TOOL_CALLS, TOOL_LATENCY
from tools.contract import ErrorCode, err, ok

log = get_logger("tools.graphrag")

VECTOR_INDEX = "chunk_embeddings"
FULLTEXT_INDEX = "chunk_fulltext"
MAX_TOP_K = 20
RRF_K = 60
_CANDIDATES = 30

_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS id, node.text AS text, node.page AS page,
       d.title AS document, score
"""

_FULLTEXT_CYPHER = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS id, node.text AS text, node.page AS page,
       d.title AS document, score
"""

_LUCENE_SPECIAL = re.compile(r'[^0-9a-zA-Z ]+')


def _driver() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def _lucene_query(text: str) -> str:
    cleaned = _LUCENE_SPECIAL.sub(" ", text.lower()).strip()
    tokens = [t for t in cleaned.split() if len(t) > 1]
    return " OR ".join(tokens)


def _filter_by_source(hits: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    """Keep only hits whose source document title contains `needle` (case-insensitive)."""
    return [h for h in hits if needle in str(h.get("document") or "").lower()]


def _rrf(rank_lists: list[list[dict[str, Any]]], k: int = RRF_K) -> list[dict[str, Any]]:
    fused: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks):
            cid = item["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(cid, item)
    out = []
    for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(by_id[cid])
        item["rrf_score"] = round(score, 5)
        out.append(item)
    return out


def graphrag_search(
    query: str, top_k: int = 5, source_filter: str | None = None
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        q = (query or "").strip()
        if not q:
            return err(ErrorCode.INPUT_INVALID, "query is required")
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        source_needle = (source_filter or "").strip().lower() or None

        try:
            vec = embed_texts([q], kind="query")[0]
        except LLMError as exc:
            TOOL_CALLS.labels(tool="graphrag_search", status="embed_error").inc()
            return err(ErrorCode.UPSTREAM_TIMEOUT, f"embedding failed: {exc}")

        driver = _driver()
        try:
            with driver.session() as session:
                vector_hits = session.run(
                    _VECTOR_CYPHER, index=VECTOR_INDEX, k=_CANDIDATES, vec=vec
                ).data()
                fulltext_hits: list[dict[str, Any]] = []
                lq = _lucene_query(q)
                if lq:
                    try:
                        fulltext_hits = session.run(
                            _FULLTEXT_CYPHER, index=FULLTEXT_INDEX, q=lq
                        ).data()
                    except neo4j.exceptions.Neo4jError:
                        fulltext_hits = []
        finally:
            driver.close()
    except neo4j.exceptions.ServiceUnavailable as exc:
        TOOL_CALLS.labels(tool="graphrag_search", status="error").inc()
        return err(ErrorCode.UPSTREAM_TIMEOUT, f"neo4j unavailable: {exc}")
    except neo4j.exceptions.Neo4jError as exc:
        TOOL_CALLS.labels(tool="graphrag_search", status="error").inc()
        return err(ErrorCode.TOOL_ERROR, f"retrieval failed: {exc}")
    finally:
        TOOL_LATENCY.labels(tool="graphrag_search").observe(time.perf_counter() - start)

    if source_needle:
        vector_hits = _filter_by_source(vector_hits, source_needle)
        fulltext_hits = _filter_by_source(fulltext_hits, source_needle)

    if not vector_hits and not fulltext_hits:
        TOOL_CALLS.labels(tool="graphrag_search", status="empty").inc()
        msg = "no matching policy chunks"
        if source_needle:
            msg = f"no matching policy chunks for source_filter={source_filter!r}"
        return err(ErrorCode.NO_RESULTS, msg)

    fused = _rrf([vector_hits, fulltext_hits])[:top_k]
    vector_score = {h["id"]: h["score"] for h in vector_hits}
    fulltext_score = {h["id"]: h["score"] for h in fulltext_hits}
    results = [
        {
            "text": h.get("text", ""),
            "document": h.get("document"),
            "page": h.get("page"),
            "rrf_score": h["rrf_score"],
            "vector_score": round(float(vector_score.get(h["id"], 0.0)), 4),
            "fulltext_score": round(float(fulltext_score.get(h["id"], 0.0)), 4),
        }
        for h in fused
    ]
    TOOL_CALLS.labels(tool="graphrag_search", status="success").inc()
    return ok({
        "query": q,
        "source_filter": source_filter,
        "count": len(results),
        "strategy": "hybrid_rrf",
        "vector_hits": len(vector_hits),
        "fulltext_hits": len(fulltext_hits),
        "results": results,
    })
