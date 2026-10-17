"""Smoke test for the tool layer — run after every tool change.

Each check tries to import and call a tool with a known input and asserts a
sane output. Tools that are not yet implemented (Phase 3 still in progress) are
SKIPPED rather than failed, so this script is useful from day one and only gets
stricter as tools land.

Run:  uv run python -m tools.smoke_test
Exit code 0 = all present checks passed; 1 = a present check failed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from core.config import settings
from core.logging import configure_logging, get_logger, set_trace_id
from tools.contract import ErrorCode, err, ok

configure_logging()
log = get_logger("tools.smoke")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    assert_fn: Callable[[dict[str, Any]], None]


class _AssertionError(AssertionError):
    pass


class _SkipCheck(Exception):
    """Raised by a check to report SKIP (e.g. optional dependency not running)."""


def _must(cond: bool, msg: str) -> None:
    if not cond:
        raise _AssertionError(msg)


def _registry_tool(name: str) -> Callable[..., Any] | None:
    """Return a dispatch callable for a registered tool, or None if absent."""
    from tools.registry import TOOLS, dispatch

    if name not in TOOLS:
        return None

    def _call(**kwargs: Any) -> dict[str, Any]:
        return dispatch(name, kwargs)

    return _call


def _envelope_shape_assert(result: dict[str, Any]) -> None:
    _must(isinstance(result, dict), "result is not a dict")
    for k in ("ok", "data", "error_code", "error_msg", "trace_id"):
        _must(k in result, f"missing key {k}")


def _data_not_empty(result: dict[str, Any]) -> None:
    _must(
        result.get("ok") is True,
        f"tool returned ok=False [{result.get('error_code')}]: {result.get('error_msg')}",
    )
    data = result.get("data")
    _must(data is not None, "data is None")
    if isinstance(data, list):
        _must(len(data) > 0, "data is an empty list")
    if isinstance(data, dict):
        _must(len(data) > 0, "data is an empty dict")


def run_checks() -> list[tuple[str, str, str]]:
    set_trace_id("smoke-test")
    checks: list[Check] = [
        Check(
            name="contract:ok shape",
            fn=ok,
            kwargs={"data": {"n": 1}},
            assert_fn=_envelope_shape_assert,
        ),
        Check(
            name="contract:err shape",
            fn=err,
            kwargs={"code": ErrorCode.NO_RESULTS, "msg": "x"},
            assert_fn=_envelope_shape_assert,
        ),
        Check(
            name="tool:run_sql",
            fn=_registry_tool("run_sql"),
            kwargs={"sql": "SELECT 1 AS n"},
            assert_fn=_data_not_empty,
        ),
        Check(
            name="tool:compute_kpi",
            fn=_registry_tool("compute_kpi"),
            kwargs={"region": "WEST", "quarter": "2026Q2"},
            assert_fn=_data_not_empty,
        ),
        Check(
            name="tool:graphrag_search",
            fn=_registry_tool("graphrag_search"),
            kwargs={"query": "What is the return policy for electronics?"},
            assert_fn=_data_not_empty,
        ),
        Check(
            name="tool:forecast_demand (service)",
            fn=_check_model_service,
            kwargs={},
            assert_fn=lambda r: _must(r.get("ok"), f"model service unhealthy: {r}"),
        ),
    ]

    results: list[tuple[str, str, str]] = []
    for check in checks:
        if check.fn is None:
            results.append((check.name, SKIP, "tool not implemented yet"))
            continue
        try:
            out = check.fn(**check.kwargs)
            check.assert_fn(out)
            results.append((check.name, PASS, "ok"))
        except _SkipCheck as exc:
            results.append((check.name, SKIP, str(exc)))
        except _AssertionError as exc:
            results.append((check.name, FAIL, str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface any tool crash as a FAIL
            results.append((check.name, FAIL, f"{type(exc).__name__}: {exc}"))
    return results


def _check_model_service() -> dict[str, Any]:
    url = f"{settings.model_service_url}/predict"
    body = {"region": "WEST", "category": "Beverages", "horizon": 2}
    try:
        r = httpx.post(url, json=body, timeout=15.0)
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as exc:
        raise _SkipCheck(f"model service not running ({settings.model_service_url})") from exc
    _must(
        isinstance(payload.get("forecast"), list) and len(payload["forecast"]) == 2,
        f"unexpected forecast payload: {payload}",
    )
    return ok(payload)


def main() -> None:
    results = run_checks()
    width = max(len(name) for name, _, _ in results)
    print()
    for name, status, detail in results:
        mark = {PASS: "✔", FAIL: "✖", SKIP: "○"}[status]
        print(f"  {mark} {status:4}  {name:<{width}}  {detail}")
    fails = [r for r in results if r[1] == FAIL]
    skips = [r for r in results if r[1] == SKIP]
    print(f"\n  {len(results) - len(fails) - len(skips)} passed, {len(skips)} skipped, {len(fails)} failed\n")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
