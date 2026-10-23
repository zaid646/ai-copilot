"""Tenant isolation: JWT claims + cross-tenant reads return empty."""
from __future__ import annotations

from auth.tenant import (
    TenantScope,
    TenantStore,
    issue_tenant_token,
    scope_rows,
    tenant_from_token,
    tenant_where,
)


def test_tenant_id_is_carried_in_jwt():
    token = issue_tenant_token("analyst", "analyst", tenant_id="tenant-a", user_id=7)
    scope = tenant_from_token(token)
    assert scope.tenant_id == "tenant-a"
    assert scope.user_id == 7


def test_cross_tenant_reads_return_empty():
    store: TenantStore[dict] = TenantStore()
    store.put("tenant-a", "conv-1", {"tenant_id": "tenant-a", "text": "a"})
    store.put("tenant-b", "conv-2", {"tenant_id": "tenant-b", "text": "b"})

    assert store.count("tenant-a") == 1
    assert store.count("tenant-b") == 1
    # A tenant-b principal must not see tenant-a rows.
    assert store.get("tenant-b", "conv-1") is None
    assert store.list("tenant-b") == [{"tenant_id": "tenant-b", "text": "b"}]
    assert store.list("tenant-c") == []


def test_scope_rows_and_graph_filter_are_strict():
    rows = [
        {"tenant_id": "tenant-a", "id": 1},
        {"tenant_id": "tenant-b", "id": 2},
        {"id": 3},  # legacy row defaults to the default tenant
    ]
    assert [r["id"] for r in scope_rows(rows, "tenant-a")] == [1]
    assert [r["id"] for r in scope_rows(rows, "default")] == [3]
    assert scope_rows(rows, "tenant-z") == []


def test_sql_predicate_requires_tenant():
    clause, params = tenant_where("tenant-a")
    assert clause == "tenant_id=%s"
    assert params == ("tenant-a",)
    try:
        tenant_where("")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty tenant must raise")


def test_tenant_scope_rejects_blank_tenant():
    try:
        TenantScope(tenant_id="")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("blank tenant must raise")
