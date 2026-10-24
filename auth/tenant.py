"""Tenant isolation: JWT claims + scoped reads for SQL/vector/graph stores.

The production request path puts ``tenant_id`` in the JWT and every persistence
call carries it. The in-memory ``TenantStore`` below is the deterministic unit
under test: a read with another tenant's id returns an empty result rather than
raising, so cross-tenant probes cannot distinguish "empty" from "not found".

For Postgres, ``db/migrations/0002_tenant_isolation.sql`` adds ``tenant_id``
columns, indexes and row-level-security policies. For Neo4j, chunk nodes carry
``tenant_id`` and ``tools.graphrag`` filters on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jwt

from core.config import settings

DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class TenantScope:
    tenant_id: str
    user_id: int | None = None
    username: str | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id or not str(self.tenant_id).strip():
            raise ValueError("tenant_id is required")

    @property
    def key(self) -> str:
        return str(self.tenant_id)


def issue_tenant_token(
    username: str,
    role: str,
    tenant_id: str = DEFAULT_TENANT,
    user_id: int | None = None,
    expires_minutes: int | None = None,
) -> str:
    from core.security import create_access_token

    return create_access_token(
        username=username,
        role=role,
        user_id=user_id,
        tenant_id=tenant_id,
        expires_minutes=expires_minutes,
    )


def tenant_from_token(token: str) -> TenantScope:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return TenantScope(
        tenant_id=str(payload.get("tenant_id") or DEFAULT_TENANT),
        user_id=payload.get("uid"),
        username=payload.get("sub"),
        role=payload.get("role"),
    )


def scope_rows(rows: list[dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("tenant_id") or DEFAULT_TENANT) == str(tenant_id)
    ]


def tenant_where(tenant_id: str, column: str = "tenant_id") -> tuple[str, tuple[str]]:
    """Return a SQL predicate + params for parameterised queries.

    Callers must append this to their own WHERE clause. It exists so ad-hoc
    helpers cannot forget the tenant predicate.
    """
    if not tenant_id:
        raise ValueError("tenant_id is required")
    return f"{column}=%s", (str(tenant_id),)


@dataclass
class TenantStore[T]:
    """A minimal tenant-scoped key/value store used by the isolation test."""

    _rows: dict[str, dict[str, T]] = field(default_factory=dict)

    def put(self, tenant_id: str, row_id: str, value: T) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._rows.setdefault(str(tenant_id), {})[row_id] = value

    def get(self, tenant_id: str, row_id: str) -> T | None:
        return self._rows.get(str(tenant_id), {}).get(row_id)

    def list(self, tenant_id: str) -> list[T]:
        return list(self._rows.get(str(tenant_id), {}).values())

    def count(self, tenant_id: str) -> int:
        return len(self._rows.get(str(tenant_id), {}))


def filter_graph_hits(hits: list[dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
    """Apply the tenant predicate to Neo4j result rows (defence in depth)."""
    return [hit for hit in hits if str(hit.get("tenant_id", DEFAULT_TENANT)) == str(tenant_id)]
