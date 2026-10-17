"""Streamlit front end for the Enterprise Analytics Copilot.

Single-file app. Flow: login -> Ask (answer + KPI cards + guardrail banner) ->
Trace (per-step expander) -> History (table + view-trace) -> admin KPI chart.

The API base URL is read from API_BASE_URL (default http://localhost:8000).
All calls go through `_request`, which injects the JWT bearer token and clears
the session on 401 so the user is bounced back to the login screen.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise Analytics Copilot", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    for k, v in {
        "token": None,
        "role": None,
        "username": None,
        "answer": None,
        "trace_id": None,
        "conversation_id": None,
    }.items():
        st.session_state.setdefault(k, v)


def _logout() -> None:
    for k in ("token", "role", "username", "answer", "trace_id", "conversation_id"):
        st.session_state[k] = None


def _headers() -> dict[str, str]:
    tok = st.session_state.get("token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _request(method: str, path: str, **kwargs) -> requests.Response | None:
    """Call the API; on 401 clear the session and return None."""
    try:
        resp = requests.request(method, f"{API_BASE}{path}", headers=_headers(), timeout=300, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {API_BASE}: {exc}")
        return None
    if resp.status_code == 401:
        st.warning("Session expired — please log in again.")
        _logout()
        st.rerun()
    return resp


def _error_of(resp: requests.Response) -> str:
    try:
        return resp.json().get("error_msg") or resp.text
    except Exception:
        return resp.text


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_login() -> None:
    st.title("📊 Enterprise Analytics Copilot")
    st.caption("Sign in to ask questions grounded in your warehouse and policy docs.")
    with st.form("login"):
        username = st.text_input("Username", value="analyst")
        password = st.text_input("Password", type="password", value="analyst123")
        submitted = st.form_submit_button("Log in")
    if submitted:
        resp = _request("POST", "/auth/login", json={"username": username, "password": password})
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            st.session_state.token = data["access_token"]
            st.session_state.role = data["role"]
            st.session_state.username = data["username"]
            st.rerun()
        elif resp is not None:
            st.error("Invalid username or password.")


def _render_kpi_cards() -> None:
    resp = _request("GET", "/kpi/west-compare")
    if resp is None or resp.status_code != 200:
        return
    k = resp.json()
    c1, c2, c3 = st.columns(3)
    c1.metric("WEST revenue (2026Q3)", f"${k['revenue']:,.0f}", f"{k['delta_pct']:.1f}%")
    c2.metric("Previous (2026Q2)", f"${k['previous_revenue']:,.0f}")
    c3.metric("Change", f"${k['delta_abs']:,.0f}")


def page_ask() -> None:
    st.header("Ask")
    with st.form("ask"):
        question = st.text_area(
            "Your question",
            value="Why did sales drop in West region last quarter, and what should we do?",
            height=90,
        )
        submitted = st.form_submit_button("Ask the copilot")

    if submitted and question.strip():
        with st.spinner("Thinking — planning, querying the warehouse, retrieving policy…"):
            resp = _request(
                "POST",
                "/ask",
                json={"question": question, "conversation_id": st.session_state.conversation_id},
            )
        if resp is None:
            return
        if resp.status_code != 200:
            st.error(f"Request failed ({resp.status_code}): {_error_of(resp)}")
            return
        data = resp.json()
        st.session_state.answer = data
        st.session_state.trace_id = data.get("trace_id")
        st.session_state.conversation_id = data.get("conversation_id")

    answer = st.session_state.get("answer")
    if answer:
        if answer.get("guardrail_blocked"):
            st.warning("This response was flagged by a guardrail.")
        st.markdown("### Answer")
        st.write(answer.get("answer", ""))
        cits = answer.get("citations") or []
        if cits:
            st.markdown("**Citations**")
            for c in cits:
                st.markdown(f"- {c.get('document')} — p.{c.get('page')}")
        st.caption(
            f"trace `{answer.get('trace_id')}` · domain `{answer.get('domain')}` · "
            f"status `{answer.get('status')}` · {answer.get('latency_ms')} ms"
        )
        _render_kpi_cards()


def page_trace() -> None:
    st.header("Trace")
    default = st.session_state.get("trace_id") or ""
    trace_id = st.text_input("Trace ID", value=default)
    if not trace_id:
        st.info("Run a question on the Ask tab, or paste a trace ID.")
        return
    resp = _request("GET", f"/trace/{trace_id}")
    if resp is None:
        return
    if resp.status_code == 404:
        st.error("Trace not found.")
        return
    if resp.status_code == 403:
        st.error("You do not have access to this trace.")
        return
    if resp.status_code != 200:
        st.error(_error_of(resp))
        return
    t = resp.json()
    st.markdown(f"**Question:** {t['question']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", t["status"])
    c2.metric("Steps", t["total_steps"])
    c3.metric("Latency", f"{t.get('latency_ms') or 0} ms")
    if t.get("answer"):
        st.markdown("**Answer**")
        st.write(t["answer"])
    st.markdown("**Steps**")
    for s in t.get("steps", []):
        with st.expander(f"{s['seq']:>2}. {s['node']}  ·  {s.get('latency_ms') or 0} ms"):
            st.json(s.get("output_json"))


def page_history() -> None:
    st.header("History")
    resp = _request("GET", "/history")
    if resp is None or resp.status_code != 200:
        if resp is not None:
            st.error(_error_of(resp))
        return
    rows = resp.json()
    if not rows:
        st.info("No history yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    tids = [r["trace_id"] for r in rows if r.get("trace_id")]
    if tids:
        pick = st.selectbox("View trace for", options=tids)
        if st.button("Open trace") and pick:
            st.session_state.trace_id = pick
            st.success(f"Loaded trace `{pick}` — open the Trace tab.")


def page_admin() -> None:
    st.header("Admin — KPI overview")
    resp = _request("GET", "/kpi/category-breakdown")
    if resp is None or resp.status_code != 200:
        if resp is not None:
            st.error(_error_of(resp))
        return
    rows = resp.json()
    if not rows:
        st.info("No data.")
        return
    df = pd.DataFrame(rows).set_index("category")
    st.bar_chart(df["revenue"])
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------
def main() -> None:
    _init_state()
    if not st.session_state.get("token"):
        page_login()
        return

    st.sidebar.title("📊 Copilot")
    st.sidebar.caption(f"{st.session_state.username} · {st.session_state.role}")
    pages = ["Ask", "Trace", "History"]
    if st.session_state.role == "admin":
        pages.append("Admin")
    choice = st.sidebar.radio("Navigate", pages)
    if st.sidebar.button("Log out"):
        _logout()
        st.rerun()

    if choice == "Ask":
        page_ask()
    elif choice == "Trace":
        page_trace()
    elif choice == "History":
        page_history()
    elif choice == "Admin":
        page_admin()


if __name__ == "__main__":
    main()
