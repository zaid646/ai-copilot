"""Phase 7 eval runner.

Runs the golden set against the running API, checks deterministic guardrails
and factual claims, then uses an LLM-as-judge (temperature=0, cache-backed) for
faithfulness. Produces hard-fail and soft-warn verdicts per the golden-set
thresholds.

Usage:
    python -m evals.evaluator           # full run, prints JSON summary
    python -m evals.evaluator --json    # machine-readable summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

from core.cache import get as cache_get
from core.cache import set as cache_set
from core.logging import get_logger

log = get_logger("evals.evaluator")

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
BASE_URL = "http://127.0.0.1:8000"
ANALYST_CREDS = {"username": "analyst", "password": "analyst123"}

_JUDGE_CACHE_TTL = 86400  # 24h


def _auth_token() -> str:
    resp = httpx.post(f"{BASE_URL}/auth/login", json=ANALYST_CREDS, timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _cache_key(judge_prompt: str) -> str:
    return "judge:" + hashlib.sha256(judge_prompt.encode()).hexdigest()[:32]


def _llm_judge(question: str, answer: str, expected_contains: list[str]) -> dict:
    """LLM-as-judge faithfulness check at temperature=0, cache-backed."""
    prompt = (
        "You are a strict, impartial evaluator at temperature=0. "
        "Given a question, an answer, and a list of facts the answer must contain, "
        "score faithfulness from 0.0 to 1.0. "
        "1.0 = every required fact is present and correct; "
        "0.0 = answer contradicts or omits all required facts. "
        "Respond with a single valid JSON object: {\"score\": <float>, \"reason\": \"...\"}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Required facts: {json.dumps(expected_contains)}\n"
    )
    key = _cache_key(prompt)
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        from core.llm import chat_json

        out = chat_json("Strict evaluator, temperature=0.", prompt, temperature=0.0, think=False)
        cache_set(key, "judge", "evals", out)
        return out
    except Exception as exc:  # pragma: no cover - dependency failure
        log.warning("judge_call_failed", error=str(exc))
        return {"score": 0.0, "reason": f"judge error: {exc}"}


def _check_case(token: str, case: dict) -> dict:
    """Run one golden case and evaluate against expectations."""
    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{BASE_URL}/ask",
            json={"question": case["question"], "max_iterations": 4},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )
    except Exception as exc:
        return {
            "id": case["id"],
            "passed": False,
            "error": f"request_failed: {exc}",
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
    latency_ms = int((time.perf_counter() - start) * 1000)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

    expected_status = case["expected_status"]
    status_ok = (resp.status_code == 200) == (expected_status == "completed")
    if expected_status == "blocked":
        status_ok = resp.status_code == 400

    answer = body.get("answer", "") or ""
    citations = body.get("citations", []) or []
    citation_pages = [str(c.get("page")) for c in citations if c.get("page") is not None]

    # Deterministic checks
    contains_ok = all(fact.lower() in answer.lower() for fact in case.get("expected_answer_contains", []))
    cites_ok = all(pg in citation_pages for pg in case.get("expected_citations", []))
    guardrail_ok = body.get("guardrail_blocked") is not None  # field present

    # LLM-as-judge faithfulness (only for completed answers with required facts)
    judge_score = None
    judge_reason = "n/a"
    if expected_status == "completed" and case.get("expected_answer_contains"):
        judgment = _llm_judge(case["question"], answer, case["expected_answer_contains"])
        judge_score = float(judgment.get("score", 0.0))
        judge_reason = str(judgment.get("reason", ""))

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": status_ok and contains_ok and cites_ok,
        "status_code": resp.status_code,
        "expected_status": expected_status,
        "status_ok": status_ok,
        "contains_ok": contains_ok,
        "cites_ok": cites_ok,
        "guardrail_ok": guardrail_ok,
        "judge_score": judge_score,
        "judge_reason": judge_reason,
        "latency_ms": latency_ms,
        "answer_preview": answer[:200],
        "citations": citation_pages,
    }


def run_evals(json_out: bool = False) -> dict:
    """Run the full golden set. Returns a summary dict."""
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"golden set not found: {GOLDEN_PATH}")
    golden = json.loads(GOLDEN_PATH.read_text())
    thresholds = golden["thresholds"]
    cases = golden["cases"]

    token = _auth_token()
    results = [_check_case(token, c) for c in cases]

    # Aggregate
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    blocked_expected = [r for r in results if r["expected_status"] == "blocked"]
    blocked_actual = [r for r in blocked_expected if r["status_code"] == 400]
    completed = [r for r in results if r["expected_status"] == "completed"]

    judge_scores = [r["judge_score"] for r in completed if r["judge_score"] is not None]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
    p95_latency = (
        sorted([r["latency_ms"] for r in results])[int(total * 0.95) - 1] if total else 0
    )

    hard_fail = {
        "guardrail_block_rate_max": len(blocked_actual) == len(blocked_expected),
        "factual_citation_required": all(r["cites_ok"] for r in completed if r.get("expected_citations")),
    }
    soft_warn = {
        "answer_contains_key_facts_min": all(r["contains_ok"] for r in completed),
        "citation_page_match_min": all(r["cites_ok"] for r in completed),
        "llm_judge_faithfulness_min": avg_judge >= thresholds["soft_warn"]["llm_judge_faithfulness_min"],
        "latency_p95_ms_max": p95_latency <= thresholds["soft_warn"]["latency_p95_ms_max"],
    }

    hard_all = all(hard_fail.values())
    soft_all = all(soft_warn.values())

    summary = {
        "golden_set": golden["name"],
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hard_fail": hard_fail,
        "hard_pass": hard_all,
        "soft_warn": soft_warn,
        "soft_pass": soft_all,
        "avg_judge_score": round(avg_judge, 3),
        "p95_latency_ms": p95_latency,
        "results": results,
    }

    if json_out:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Phase 7 evals — {golden['name']}")
        print(f"  total={total} passed={passed} failed={total-passed}")
        print(f"  hard_fail={hard_all} soft_warn={soft_all}")
        print(f"  avg_judge={avg_judge:.3f} p95_latency={p95_latency}ms")
        for r in results:
            flag = "PASS" if r["passed"] else "FAIL"
            print(f"  [{flag}] {r['id']} status={r['status_code']} "
                  f"contains={r['contains_ok']} cites={r['cites_ok']} "
                  f"judge={r['judge_score']} latency={r['latency_ms']}ms")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = parser.parse_args()
    summary = run_evals(json_out=args.json)
    sys.exit(0 if (summary["hard_pass"] and summary["soft_pass"]) else 1)
