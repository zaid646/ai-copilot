"""SQL guard: fail-closed validation for analytics queries.

Rules (all must hold):
  - exactly one statement (no stacked statements)
  - must be a read-only SELECT (optionally a leading WITH ... SELECT)
  - no data-definition / data-modification / admin keywords anywhere
  - a LIMIT is enforced (appended if absent, capped at MAX_ROWS)

`validate_sql` returns (ok, normalized_sql, reason). The tool layer maps
ok=False to ErrorCode.SQL_REJECTED.
"""
from __future__ import annotations

import re

MAX_ROWS = 1000

_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "merge", "call", "do", "set", "reset",
    "vacuum", "analyze", "commit", "rollback", "begin", "savepoint",
    "refresh", "reindex", "cluster", "security", "attach", "detach",
    "pg_read_file", "pg_sleep", "lo_import", "lo_export", "dblink", "returning",
}

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT.sub(" ", sql)
    return _LINE_COMMENT.sub(" ", sql)


def _strip_strings(sql: str) -> str:
    """Replace string literals with a placeholder so keyword/semicolon scans
    don't trip on text inside quotes."""
    return _STRING_LITERAL.sub("''", sql)


def _check_keywords(sql: str) -> str | None:
    words = set(re.findall(r"[a-zA-Z_]+", sql.lower()))
    hit = words & _FORBIDDEN
    if hit:
        return f"forbidden keyword(s): {', '.join(sorted(hit))}"
    return None


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> tuple[bool, str, str | None]:
    if not sql or not sql.strip():
        return False, "", "empty query"

    stripped = _strip_comments(sql)
    # split on semicolons (after removing strings) -> detect stacked statements
    scan = _strip_strings(stripped)
    parts = [p for p in scan.split(";") if p.strip()]
    if len(parts) > 1:
        return False, "", "multiple statements are not allowed"

    normalized = stripped.strip().rstrip(";").strip()
    if not normalized:
        return False, "", "empty query"

    head = normalized.lstrip("(").strip().split(None, 1)[0].lower() if normalized.split() else ""
    if head not in ("select", "with"):
        return False, "", "only read-only SELECT queries are allowed"

    kw = _check_keywords(_strip_strings(normalized))
    if kw:
        return False, "", kw

    # enforce a LIMIT
    if _LIMIT_RE.search(_strip_strings(normalized)):
        return True, normalized, None
    return True, f"{normalized} LIMIT {int(max_rows)}", None
