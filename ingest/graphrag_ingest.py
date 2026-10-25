"""Ingest policy/playbook PDFs into the Neo4j GraphRAG store.

Builds, per document:
  (:Document)-[:HAS_CHUNK]->(:Chunk {text, page_span})  with 768-dim embeddings
and creates a vector index (`chunk_embeddings`) + full-text index
(`chunk_fulltext`) so retrieval can run hybrid (vector + keyword) search.

Run:  uv run python -m ingest.graphrag_ingest [--force]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import neo4j
from pypdf import PdfReader

from core.config import ROOT, settings
from core.llm import embed_texts
from core.logging import configure_logging, get_logger

log = get_logger("ingest.graphrag")

POLICY_DIR = ROOT / "samples" / "policies"
VECTOR_INDEX = "chunk_embeddings"
FULLTEXT_INDEX = "chunk_fulltext"
EMBED_DIM = 768


def get_driver() -> neo4j.Driver:
    return neo4j.GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def ensure_indexes(driver: neo4j.Driver) -> None:
    stmts = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    ]
    with driver.session() as s:
        for stmt in stmts:
            s.run(stmt)
    log.info("neo4j indexes ensured")


def _clean_text(text: str) -> str:
    """Repair common PDF-extraction artifacts before chunking."""
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # join "sum-\nmarize" -> "summarize"
    text = re.sub(r"\u00ad", "", text)                 # soft hyphens
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newlines -> space
    text = re.sub(r"[ \t]+", " ", text)                # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str) -> list[str]:
    """Sentence-aware chunker: pack sentences up to chunk_size, with a
    character overlap carried from the tail of the previous chunk."""
    size = settings.ingest_chunk_size
    overlap = settings.ingest_chunk_overlap
    text = _clean_text(text)

    # split on sentence boundaries, keeping paragraphs as soft units
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= size:
            cur = f"{cur} {sent}"
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = f"{tail} {sent}".strip() if tail else sent
    if cur:
        chunks.append(cur)

    return [c for c in chunks if len(c) > 50]


def extract_pdf_chunks(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    max_pages = settings.ingest_max_pages_per_doc or len(reader.pages)
    pages = reader.pages[:max_pages]
    chunks: list[dict] = []
    for page_no, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            log.warning("page extract failed", path=str(path), page=page_no, error=str(exc))
            continue
        for ch in _split_text(text):
            chunks.append({"text": ch, "page": page_no})
    return chunks


def ingest_document(
    driver: neo4j.Driver, path: Path, force: bool = False, tenant_id: str = "default"
) -> int:
    doc_id = path.stem
    with driver.session() as s:
        existing = s.run(
            "MATCH (d:Document {id:$id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n", id=doc_id
        ).single()["n"]
        if existing and not force:
            log.info("document already ingested (use --force to redo)", doc=doc_id, chunks=existing)
            return existing

    chunks = extract_pdf_chunks(path)
    if not chunks:
        log.warning("no text extracted", doc=doc_id)
        return 0

    log.info("embedding chunks", doc=doc_id, n=len(chunks))
    vectors = embed_texts([c["text"] for c in chunks], dim=EMBED_DIM)

    rows = [
        {"id": f"{doc_id}::{i}", "text": c["text"], "page": c["page"], "embedding": v}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]

    with driver.session() as s:
        s.run("MATCH (c:Chunk) WHERE c.id STARTS WITH $prefix DETACH DELETE c", prefix=f"{doc_id}::")
        s.run("MATCH (d:Document {id:$id}) DETACH DELETE d", id=doc_id)
        s.run(
            """
            MERGE (d:Document {id:$id})
            SET d.title=$title, d.path=$path, d.chunks=$n, d.tenant_id=$tenant_id
            WITH d
            UNWIND $rows AS row
            CREATE (c:Chunk {id:row.id, text:row.text, page:row.page,
                             embedding:row.embedding, tenant_id:$tenant_id})
            CREATE (d)-[:HAS_CHUNK]->(c)
            """,
            id=doc_id,
            title=path.stem,
            tenant_id=tenant_id,
            path=str(path),
            n=len(rows),
            rows=rows,
        )
    log.info("document ingested", doc=doc_id, chunks=len(rows))
    return len(rows)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-ingest even if present")
    parser.add_argument("--tenant", default="default", help="tenant_id stamped on chunks")
    args = parser.parse_args()

    pdfs = sorted(POLICY_DIR.glob("*.pdf"))
    if not pdfs:
        log.error("no PDFs found", dir=str(POLICY_DIR))
        return

    driver = get_driver()
    try:
        ensure_indexes(driver)
        total = 0
        for pdf in pdfs:
            total += ingest_document(driver, pdf, force=args.force, tenant_id=args.tenant)
        log.info("ingest complete", documents=len(pdfs), total_chunks=total)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
