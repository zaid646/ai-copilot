"""Domain routing + query reformulation for retrieval grounding.

`classify_domain` maps a question to "retail" or "finance" using a simple,
deterministic keyword check (no LLM call, zero latency, demo-safe). The result
drives the GraphRAG `source_filter` so a sales-drop question retrieves from the
retail manual instead of the finance approvals manual.

`reformulate_query` appends domain-appropriate grounding terms when the intent
is a sales-drop/driver investigation, so hybrid retrieval surfaces merchandising
and price-adjustment pages rather than generic approval thresholds.
"""
from __future__ import annotations

# Finance-domain terms: budget, accounting, controls, approvals, capitalization.
_FINANCE_TERMS = (
    "budget",
    "accounting",
    "control",
    "approval",
    "capitalization",
    "appropriation",
    "encumbrance",
    "disbursement",
    "voucher",
    "purchase order",
    "procurement",
    "auditor",
    "gastax",
    "federal funds",
)

# Retail-domain terms: sales, merchandising, store, product, promotion.
_RETAIL_TERMS = (
    "sales",
    "revenue",
    "region",
    "store",
    "retail",
    "merchandising",
    "display",
    "promotion",
    "markdown",
    "discount",
    "price",
    "customer",
    "category",
    "produce",
    "stocking",
    "drop",
    "demand",
)

# Intent signals for a "why did X drop / what should we do" investigation.
_DROP_INTENT_TERMS = (
    "drop",
    "decline",
    "fall",
    "decrease",
    "down",
    "droped",
    "dip",
    "why did",
    "what should we do",
    "driver",
    "cause",
)

# Terms appended to a sales-drop query to steer retrieval to retail grounding.
_DROP_QUERY_TERMS = (
    "merchandising display price adjustment stocking promotion retail"
)


def classify_domain(question: str) -> str:
    """Return "retail" or "finance" for a question (deterministic keyword check)."""
    q = (question or "").lower()
    retail = sum(1 for t in _RETAIL_TERMS if t in q)
    finance = sum(1 for t in _FINANCE_TERMS if t in q)
    if finance > retail:
        return "finance"
    return "retail"


def classify_intent(question: str) -> str:
    """Return a coarse intent label (e.g. "sales_drop", "forecast", "general")."""
    q = (question or "").lower()
    if any(t in q for t in ("forecast", "predict", "projection", "next week", "project")):
        return "forecast"
    if any(t in q for t in _DROP_INTENT_TERMS):
        return "sales_drop"
    return "general"


def source_filter_for(domain: str) -> str | None:
    """Map a domain label to the GraphRAG source_filter substring."""
    return "Retail" if domain == "retail" else "Finance"


def reformulate_query(query: str, intent: str) -> str:
    """Append grounding terms for a sales-drop query so retrieval hits retail pages."""
    base = (query or "").strip()
    if intent != "sales_drop":
        return base
    if any(t in base.lower() for t in _DROP_QUERY_TERMS.split()):
        return base
    return f"{base} {_DROP_QUERY_TERMS}".strip()
