"""Trajectory evaluation over the 8-case golden set.

Two modes:

``recorded`` (default, CI-safe)
    Read planner outputs committed under ``evals/trajectories/planner_traces.json``
    and score the derived tool trajectory. No network, no DB, no LLM key.

``llm``
    Re-plan each golden question with the configured local/remote LLM and write
    the planner outputs + metrics. Used to refresh the recorded file.

The planner is the real LangGraph planner node; tool execution is mapped from
the plan by ``plan_to_tool_calls`` (the same deterministic mapping the supervisor
uses: sql_agent -> compute_kpi/run_sql, graphrag_agent -> graphrag_search,
model_agent -> forecast_demand).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.trajectory import score_trajectory

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden_set.json"
RECORDED = ROOT / "evals" / "trajectories" / "planner_traces.json"
OUT = ROOT / "evals" / "trajectory_results.json"


def load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text())


def plan_to_tool_calls(plan: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in plan:
        agent = step.get("agent")
        args = step.get("args") or {}
        task = str(step.get("task") or "")
        if agent == "sql_agent":
            if args.get("region") and args.get("quarter"):
                kpi_args = {
                    "region": str(args["region"]).upper(),
                    "quarter": args["quarter"],
                }
                if args.get("category"):
                    kpi_args["category"] = args["category"]
                calls.append({"tool": "compute_kpi", "args": kpi_args})
            if any(w in task.lower() for w in ("breakdown", "by ", "driver", "category", "channel", "week")):
                calls.append(
                    {
                        "tool": "run_sql",
                        "args": {"query": task or question},
                    }
                )
        elif agent == "graphrag_agent":
            calls.append(
                {
                    "tool": "graphrag_search",
                    "args": {
                        "query": str(args.get("query") or task or question),
                        "source_filter": args.get("source_filter") or "Retail",
                    },
                }
            )
        elif agent == "model_agent":
            calls.append(
                {
                    "tool": "forecast_demand",
                    "args": {
                        "region": str(args.get("region") or "WEST").upper(),
                        "category": args.get("category") or "Beverages",
                        "horizon_weeks": int(args.get("horizon_weeks") or 4),
                    },
                }
            )
    return calls


def _recorded_plans() -> dict[str, dict[str, Any]]:
    if not RECORDED.exists():
        return {}
    return {row["id"]: row for row in json.loads(RECORDED.read_text())["cases"]}


def run(mode: str = "recorded") -> dict[str, Any]:
    golden = load_golden()
    recorded = _recorded_plans()
    cases_out = []
    for case in golden["cases"]:
        if case.get("expected_status") != "completed":
            continue
        expected = case.get("expected_trajectory", [])
        if mode == "llm":
            import os

            os.environ.setdefault("CHAT_PROVIDER", "lmstudio")
            from agents.nodes.planner import planner_node

            state = planner_node(
                {
                    "question": case["question"],
                    "history": [],
                    "steps": [],
                    "scratchpad": [],
                    "voice_mode": False,
                }
            )
            plan = state.get("plan", [])
            actual = plan_to_tool_calls(plan, case["question"])
        else:
            row = recorded.get(case["id"], {})
            plan = row.get("plan", [])
            actual = row.get("tool_calls") or plan_to_tool_calls(plan, case["question"])
        score = score_trajectory(expected, actual)
        cases_out.append(
            {
                "id": case["id"],
                "question": case["question"],
                "plan": plan,
                "expected_trajectory": expected,
                "actual_trajectory": actual,
                **score,
            }
        )
        print(
            f"[{case['id']}] tool={score['tool_choice_accuracy']:.2f} "
            f"args={score['argument_validity']:.2f} order={score['order_correctness']:.2f}"
        )

    n = len(cases_out) or 1
    summary = {
        "mode": mode,
        "n_cases": len(cases_out),
        "tool_choice_accuracy": round(
            sum(c["tool_choice_accuracy"] for c in cases_out) / n, 4
        ),
        "argument_validity": round(sum(c["argument_validity"] for c in cases_out) / n, 4),
        "order_correctness": round(sum(c["order_correctness"] for c in cases_out) / n, 4),
        "cases": cases_out,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"TOTAL tool={summary['tool_choice_accuracy']:.3f} "
        f"args={summary['argument_validity']:.3f} order={summary['order_correctness']:.3f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["recorded", "llm"], default="recorded")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
