"""FastAPI application for the Enterprise Analytics Copilot.

Endpoints
---------
POST /auth/login            public   -> JWT (carries role)
POST /ask                   analyst+ -> run the LangGraph supervisor, persist conversation
GET  /history               analyst+ -> the caller's conversations/messages (all for admins)
GET  /trace/{trace_id}      analyst+ -> step-by-step trace (scoped)
GET  /kpi/west-compare      analyst+ -> West Q2-vs-Q3 revenue comparison
GET  /kpi/category-breakdown analyst+ -> West Q3 revenue by category
GET  /health                public   -> DB + Neo4j liveness
GET  /metrics               admin    -> Prometheus exposition

Every error response uses the shared tool-result envelope:
{ok, data, error_code, error_msg, trace_id}.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agents.graph import run_agent
from api.auth import authenticate, issue_token, require_role
from api.models import (
    AskRequest,
    AskResponse,
    CategoryRow,
    HealthResponse,
    KpiCard,
    LoginRequest,
    MessageOut,
    TokenResponse,
    TraceResponse,
    TraceStep,
)
from core.config import settings
from core.db import close_pool, get_pool, query, query_one
from core.logging import configure_logging, get_logger, get_trace_id, set_trace_id
from guardrails.checks import egress_check, ingress_check
from observability.metrics import (
    AGENT_LOOP_ITERATIONS,
    AGENT_RUNS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_response,
)

configure_logging()
log = get_logger("api.main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_pool()
    except Exception as exc:  # pragma: no cover - startup best effort
        log.warning("startup_pool_failed", error=str(exc))
    yield
    close_pool()


app = FastAPI(title="Enterprise Analytics Copilot", version="0.1.0", lifespan=lifespan)


def _envelope(
    status_code: int, error_code: str, error_msg: str, data: Any = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "data": data,
            "error_code": error_code,
            "error_msg": error_msg,
            "trace_id": get_trace_id(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_req: Request, exc: HTTPException) -> JSONResponse:
    code = {
        401: "UNAUTHENTICATED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        400: "INPUT_INVALID",
    }.get(exc.status_code, "HTTP_ERROR")
    return _envelope(exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_req: Request, exc: Exception) -> JSONResponse:  # noqa: BLE001
    log.error("unhandled_error", error=f"{type(exc).__name__}: {exc}")
    return _envelope(500, "TOOL_ERROR", "internal server error")


def _create_conversation(user_id: int | None, title: str) -> int | None:
    try:
        row = query_one(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING conversation_id",
            (user_id, (title or "")[:120]),
        )
        return row["conversation_id"] if row else None
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("conversation_create_failed", error=str(exc))
        return None


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # give every request a trace id so errors/logs are correlatable
    if not get_trace_id():
        set_trace_id(uuid4().hex[:16])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    route = request.url.path
    REQUEST_COUNT.labels(endpoint=route, method=request.method, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(endpoint=route).observe(elapsed)
    response.headers["X-Trace-Id"] = get_trace_id() or ""
    return response


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
@app.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    user = authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = issue_token(user)
    return TokenResponse(
        access_token=token,
        username=user["username"],
        role=user["role"],
        expires_in=settings.jwt_expire_minutes * 60,
    )


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    user: dict[str, Any] = Depends(require_role("analyst", "admin")),
) -> AskResponse:
    trace_id = uuid4().hex
    set_trace_id(trace_id)

    # --- ingress guardrail ---
    verdict = ingress_check(body.question, trace_id)
    if verdict["action"] == "blocked":
        raise HTTPException(status_code=400, detail=f"blocked by guardrail: {verdict['rule']}")
    question = verdict.get("redacted_question", body.question)

    # Ensure a conversation exists so the turn is persisted.
    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = _create_conversation(user.get("user_id"), question)

    start = time.perf_counter()
    final = run_agent(
        question,
        user_id=user.get("user_id"),
        conversation_id=conversation_id,
        max_iterations=body.max_iterations,
        trace_id=trace_id,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    answer = final.get("answer", "") or ""
    domain = final.get("domain")
    # Determine evidence source from the final state so the egress guardrail
    # only enforces citations for document-grounded answers (SQL answers have
    # no chunks to cite).
    evidence = final.get("evidence", {}) or {}
    evidence_keys = set(evidence.get("evidence_keys", []) or [])
    sql_keys = {"kpi", "breakdown", "forecast", "model"}
    docs = evidence.get("docs") or {}
    docs_results = docs.get("results", []) if isinstance(docs, dict) else []
    if evidence_keys.intersection(sql_keys):
        answer_source = "sql"
    elif evidence_keys or docs_results:
        answer_source = "retrieval"
    else:
        answer_source = "sql"
    # --- egress guardrail ---
    # Pass retrieved doc pages (not just the LLM's citation list) so the
    # groundedness check compares against what was actually retrieved.
    egress = egress_check(answer, trace_id, citations=docs_results, domain=domain, source=answer_source)
    if egress.get("action") == "blocked":
        raise HTTPException(status_code=400, detail=f"blocked by guardrail: {egress['rule']}")
    if egress.get("action") == "redacted":
        answer = egress.get("redacted_answer", answer)

    status = final.get("status") or "completed"
    AGENT_RUNS.labels(status=status).inc()
    try:
        AGENT_LOOP_ITERATIONS.observe(int(final.get("iteration", 0)))
    except Exception:  # pragma: no cover
        pass

    the_steps = final.get("steps", []) or []
    return AskResponse(
        trace_id=trace_id,
        conversation_id=conversation_id,
        answer=answer,
        citations=final.get("citations", []) or [],
        steps=[
            {"seq": s.get("seq", i), "node": s.get("node", ""), "latency_ms": s.get("latency_ms")}
            for i, s in enumerate(the_steps)
        ],
        status=status,
        domain=final.get("domain"),
        guardrail_blocked=bool(final.get("guardrail_blocked", False)),
        latency_ms=elapsed_ms,
    )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------
@app.get("/history", response_model=list[MessageOut])
async def history(
    user: dict[str, Any] = Depends(require_role("analyst", "admin")),
    limit: int = 50,
) -> list[MessageOut]:
    limit = max(1, min(limit, 200))
    if user.get("role") == "admin":
        rows = query(
            "SELECT m.conversation_id, m.role, m.content, m.trace_id, m.created_at "
            "FROM messages m ORDER BY m.message_id DESC LIMIT %s",
            (limit,),
        )
    else:
        rows = query(
            "SELECT m.conversation_id, m.role, m.content, m.trace_id, m.created_at "
            "FROM messages m JOIN conversations c ON c.conversation_id = m.conversation_id "
            "WHERE c.user_id = %s ORDER BY m.message_id DESC LIMIT %s",
            (user.get("user_id"), limit),
        )
    return [
        MessageOut(
            conversation_id=r["conversation_id"],
            role=r["role"],
            content=r["content"],
            trace_id=r["trace_id"],
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------
@app.get("/trace/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    user: dict[str, Any] = Depends(require_role("analyst", "admin")),
) -> TraceResponse:
    trace = query_one("SELECT * FROM agent_traces WHERE trace_id=%s", (trace_id,))
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    if user.get("role") != "admin" and trace.get("user_id") not in (None, user.get("user_id")):
        raise HTTPException(status_code=403, detail="not your trace")
    steps = query(
        "SELECT seq, node, latency_ms, output_json FROM agent_steps "
        "WHERE trace_id=%s ORDER BY seq",
        (trace_id,),
    )
    return TraceResponse(
        trace_id=trace["trace_id"],
        user_id=trace.get("user_id"),
        question=trace["question"],
        answer=trace.get("answer"),
        status=trace["status"],
        total_steps=trace.get("total_steps") or 0,
        latency_ms=trace.get("latency_ms"),
        guardrail_blocked=bool(trace.get("guardrail_blocked")),
        created_at=str(trace["created_at"]) if trace.get("created_at") else None,
        finished_at=str(trace["finished_at"]) if trace.get("finished_at") else None,
        steps=[
            TraceStep(
                seq=s["seq"],
                node=s["node"],
                latency_ms=s.get("latency_ms"),
                output_json=s.get("output_json"),
            )
            for s in steps
        ],
    )


# --------------------------------------------------------------------------
# KPI endpoints
# --------------------------------------------------------------------------
@app.get("/kpi/west-compare", response_model=KpiCard)
async def kpi_west_compare(
    _user: dict[str, Any] = Depends(require_role("analyst", "admin")),
) -> KpiCard:
    row = query_one(
        """
        SELECT
          SUM(f.revenue) AS revenue
        FROM fact_sales f JOIN dim_date d ON f.date_id = d.date_id
        WHERE f.region_id='WEST' AND d.year=2026 AND d.quarter=3
        """
    )
    prev = query_one(
        """
        SELECT SUM(f.revenue) AS revenue
        FROM fact_sales f JOIN dim_date d ON f.date_id = d.date_id
        WHERE f.region_id='WEST' AND d.year=2026 AND d.quarter=2
        """
    )
    rev = float(row["revenue"] or 0) if row else 0.0
    prev_rev = float(prev["revenue"] or 0) if prev else 0.0
    delta = rev - prev_rev
    pct = (delta / prev_rev * 100.0) if prev_rev else 0.0
    return KpiCard(
        region="WEST",
        quarter="2026Q3",
        previous_quarter="2026Q2",
        revenue=round(rev, 2),
        previous_revenue=round(prev_rev, 2),
        delta_abs=round(delta, 2),
        delta_pct=round(pct, 2),
    )


@app.get("/kpi/category-breakdown", response_model=list[CategoryRow])
async def kpi_category_breakdown(
    _user: dict[str, Any] = Depends(require_role("analyst", "admin")),
) -> list[CategoryRow]:
    rows = query(
        """
        SELECT dp.category AS category,
               SUM(f.revenue) AS revenue,
               SUM(f.quantity) AS units
        FROM fact_sales f
        JOIN dim_date d ON f.date_id = d.date_id
        JOIN dim_product dp ON f.product_id = dp.product_id
        WHERE f.region_id='WEST' AND d.year=2026 AND d.quarter=3
        GROUP BY dp.category
        ORDER BY revenue DESC
        """
    )
    return [
        CategoryRow(
            category=r["category"],
            revenue=round(float(r["revenue"] or 0), 2),
            units=int(r["units"] or 0),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# Health + metrics
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    pg_ok = True
    try:
        query_one("SELECT 1 AS ok")
    except Exception:  # pragma: no cover
        pg_ok = False
    neo_ok = True
    try:
        import neo4j

        driver = neo4j.GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        try:
            with driver.session() as s:
                s.run("RETURN 1 AS ok").consume()
        finally:
            driver.close()
    except Exception:  # pragma: no cover
        neo_ok = False
    status = "ok" if (pg_ok and neo_ok) else "degraded"
    return HealthResponse(status=status, postgres=pg_ok, neo4j=neo_ok)


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "ai-copilot", "status": "ok", "docs": "/docs"}
