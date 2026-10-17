"""Pydantic request/response models for the copilot API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    expires_in: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    max_iterations: int = Field(default=8, ge=1, le=12)


class Citation(BaseModel):
    document: str | None = None
    page: int | None = None
    reason: str | None = None


class StepSummary(BaseModel):
    seq: int
    node: str
    latency_ms: int | None = None


class AskResponse(BaseModel):
    trace_id: str
    conversation_id: int | None = None
    answer: str
    citations: list[Citation] = []
    steps: list[StepSummary] = []
    status: str
    domain: str | None = None
    guardrail_blocked: bool = False
    latency_ms: int | None = None


class MessageOut(BaseModel):
    conversation_id: int
    role: str
    content: str
    trace_id: str | None = None
    created_at: str | None = None


class TraceStep(BaseModel):
    seq: int
    node: str
    latency_ms: int | None = None
    output_json: Any = None


class TraceResponse(BaseModel):
    trace_id: str
    user_id: int | None = None
    question: str
    answer: str | None = None
    status: str
    total_steps: int
    latency_ms: int | None = None
    guardrail_blocked: bool = False
    created_at: str | None = None
    finished_at: str | None = None
    steps: list[TraceStep] = []


class KpiCard(BaseModel):
    region: str
    quarter: str
    previous_quarter: str
    revenue: float
    previous_revenue: float
    delta_abs: float
    delta_pct: float


class CategoryRow(BaseModel):
    category: str
    revenue: float
    units: int | None = None


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    neo4j: bool
    version: str = "0.1.0"
