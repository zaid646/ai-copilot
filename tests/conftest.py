"""Shared pytest fixtures for the API tests.

These are integration tests: they exercise the real FastAPI app against the
local Postgres (seeded) and the configured LLM endpoint. If the LLM endpoint is
unreachable, `test_ask_as_analyst` will be skipped rather than failed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def analyst_token(client: TestClient) -> str:
    return _login(client, "analyst", "analyst123")


@pytest.fixture(scope="session")
def admin_token(client: TestClient) -> str:
    return _login(client, "admin", "admin123")


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
