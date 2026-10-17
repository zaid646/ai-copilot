"""Backwards-compatible dispatcher alias for registry.dispatch."""
from tools.registry import dispatch, get_tool, tool_names, tool_schemas

__all__ = ["dispatch", "tool_names", "tool_schemas", "get_tool"]
