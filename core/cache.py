"""Disk-backed response cache for LLM calls (protects free-tier quota).

Keys are sha256(model + temperature + system + user + namespace), so identical
requests hit the cache across processes and restarts. SQLite (WAL) is used so
the API, agent loop, and eval harness can share one cache file safely.

Modes (via `cache_mode` context manager or env/config):
  - "on"      read + write (default)
  - "read"    read only (no new writes)  -> reproducible eval replays
  - "write"   write only (ignore existing, refill)
  - "off"     bypass entirely
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.config import settings
from core.logging import get_logger

log = get_logger("core.cache")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_mode: str = "on"

_INSERT = (
    "INSERT OR REPLACE INTO llm_cache (key, model, namespace, value_json, created_at, hits) "
    "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hits FROM llm_cache WHERE key = ?), 0))"
)


class CacheMode:
    ON = "on"
    READ = "read"
    WRITE = "write"
    OFF = "off"


@contextmanager
def cache_mode(mode: str) -> Iterator[None]:
    global _mode
    prev = _mode
    _mode = mode
    try:
        yield
    finally:
        _mode = prev


def _db_path() -> Path:
    return Path(settings.llm_cache_path)


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache ("
        " key TEXT PRIMARY KEY, model TEXT, namespace TEXT, value_json TEXT,"
        " created_at REAL, hits INTEGER DEFAULT 0)"
    )
    conn.commit()
    _conn = conn
    return conn


def make_key(model: str, temperature: float, system: str, user: str, namespace: str = "chat") -> str:
    h = hashlib.sha256()
    for part in (model, f"{temperature:.4f}", namespace, system, user):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get(key: str) -> dict[str, Any] | None:
    if not settings.llm_cache_enabled or _mode in (CacheMode.OFF, CacheMode.WRITE):
        return None
    with _lock:
        conn = _get_conn()
        row = conn.execute("SELECT value_json FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE llm_cache SET hits = hits + 1 WHERE key = ?", (key,))
        conn.commit()
    return json.loads(row[0])


def set(key: str, model: str, namespace: str, value: dict[str, Any]) -> None:
    if not settings.llm_cache_enabled or _mode in (CacheMode.OFF, CacheMode.READ):
        return
    with _lock:
        conn = _get_conn()
        conn.execute(
            _INSERT,
            (key, model, namespace, json.dumps(value), time.time(), key),
        )
        conn.commit()


def key_for(model: str, temperature: float, system: str, user: str, namespace: str = "chat") -> str:
    return make_key(model, temperature, system, user, namespace)


def stats() -> dict[str, Any]:
    with _lock:
        conn = _get_conn()
        n, hits = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM llm_cache"
        ).fetchone()
    return {"entries": int(n), "hits": int(hits), "path": str(_db_path())}


def clear() -> int:
    with _lock:
        conn = _get_conn()
        n = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
    log.info("cache cleared", removed=int(n))
    return int(n)
