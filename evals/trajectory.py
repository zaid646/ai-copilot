"""Trajectory scoring: tool choice, argument validity and call order per step."""
from __future__ import annotations

import re
from typing import Any


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _text_matches(want: str, got: str) -> bool:
    """Free-text args are valid if every expected token appears in the actual value.

    Recorded planners write full sentences (e.g. "Sum revenue by product
    category ...") while labels use a shorter intent phrase. Contiguous
    substring matching wrongly rejected those, so compare token sets instead.
    """
    want_norm = _norm(want)
    got_norm = _norm(got)
    if want_norm in got_norm:
        return True
    want_tokens = set(re.findall(r"[a-z0-9$']+", want_norm))
    got_tokens = set(re.findall(r"[a-z0-9$']+", got_norm))
    return bool(want_tokens) and want_tokens <= got_tokens


def _arg_matches(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, str) and key in {"query", "task", "subject"}:
            if not _text_matches(want, str(got or "")):
                missing.append(key)
        elif isinstance(want, int) and isinstance(got, (int, float)):
            if int(got) != want:
                missing.append(key)
        elif _norm(got) != _norm(want):
            missing.append(key)
    return (not missing, missing)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def score_trajectory(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score an observed tool trajectory against a hand-labelled one.

    * tool-choice accuracy: correct tool at the same position / expected steps
    * argument validity: matching steps whose expected args are present
    * order correctness: LCS of tool names / expected steps
    """
    expected_tools = [str(s.get("tool")) for s in expected]
    actual_tools = [str(s.get("tool")) for s in actual]
    if not expected_tools:
        ok = not actual_tools
        return {
            "tool_choice_accuracy": 1.0 if ok else 0.0,
            "argument_validity": 1.0 if ok else 0.0,
            "order_correctness": 1.0 if ok else 0.0,
            "extra_calls": len(actual_tools),
            "details": [],
        }

    position_matches = 0
    arg_valid = 0
    details: list[dict[str, Any]] = []
    for i, step in enumerate(expected):
        want_tool = expected_tools[i]
        got = actual[i] if i < len(actual) else {}
        got_tool = str(got.get("tool")) if got else None
        tool_ok = got_tool == want_tool
        if tool_ok:
            position_matches += 1
        arg_ok, missing = _arg_matches(
            step.get("args", {}) or {}, (got.get("args", {}) if got else {}) or {}
        )
        if tool_ok and arg_ok:
            arg_valid += 1
        details.append(
            {
                "step": i,
                "expected_tool": want_tool,
                "actual_tool": got_tool,
                "tool_ok": tool_ok,
                "args_ok": arg_ok if tool_ok else False,
                "missing_args": missing,
            }
        )
    denom = len(expected) or 1
    order = _lcs(expected_tools, actual_tools)
    return {
        "tool_choice_accuracy": round(position_matches / denom, 4),
        "argument_validity": round(arg_valid / denom, 4),
        "order_correctness": round(order / denom, 4),
        "extra_calls": max(0, len(actual) - len(expected)),
        "details": details,
    }
