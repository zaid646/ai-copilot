"""Node input/output helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.llm import chat_json

_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS = {
    "planner": "agents/prompts/planner.md",
    "sql_agent": "agents/prompts/sql_agent.md",
    "graphrag_agent": "agents/prompts/graphrag_agent.md",
    "model_agent": "agents/prompts/model_agent.md",
    "action_agent": "agents/prompts/action_agent.md",
    "critic_agent": "agents/prompts/critic_agent.md",
}


def _load_prompt(name: str, context: dict[str, Any]) -> str:
    path = str(_ROOT / _PROMPTS[name])
    try:
        with open(path, encoding="utf-8") as f:
            tmpl = f.read()
        for k, v in context.items():
            if isinstance(v, dict):
                tmpl = tmpl.replace(f"{{{k}}}", json.dumps(v, indent=0, default=str).strip('"').replace('"', '\\"') if False else json.dumps(v, default=str))
            else:
                tmpl = tmpl.replace(f"{{{k}}}", str(v))
        return tmpl
    except Exception:
        return json.dumps(context, default=str)


def llm_json(name: str, context: dict[str, Any]) -> dict[str, Any]:
    prompt = _load_prompt(name, context)
    try:
        return chat_json("You are a careful agent. Follow output rules strictly.", prompt)
    except Exception as exc:
        raise RuntimeError(f"{name}_json_failed: {exc}") from exc
