"""
UI-facing context builder for Options panels.
Pulls values from Streamlit session state and delegates to the pure model helper.
"""

from __future__ import annotations

import time

import streamlit as st

from app.controller.options_controller import build_option_context
from app.vue.components.options.ui_helpers import _k

# The router and every family/leaf panel call get_option_context() on each rerun
# (~32 calls). The context is a pure function of the ticker for the life of a rerun,
# and a stale spot for a few minutes is harmless, so one load per ticker is kept in
# session state. Failed lookups are memoised too: an unknown ticker used to fire a
# fresh batch of network attempts on every rerun.
_CTX_MEMO_KEY = "_opt_ctx_memo"
_CTX_MEMO_TTL_SEC = 300.0


def _resolve_ticker() -> str:
    tk = st.session_state.get("common_underlying") or st.session_state.get("tkr_common") or ""
    return str(tk or "").strip().upper()


def get_option_context():
    tk = _resolve_ticker()
    memo = st.session_state.get(_CTX_MEMO_KEY)
    if (
        isinstance(memo, dict)
        and memo.get("ticker") == tk
        and (time.monotonic() - float(memo.get("ts", 0.0))) < _CTX_MEMO_TTL_SEC
    ):
        return dict(memo["ctx"])

    # The spot is derived from the ticker's own close history — never from the
    # shared `common_spot_value` key, which the router seeds at 100.0 and which
    # would otherwise pin every panel to that default.
    state = {
        "common_underlying": st.session_state.get("common_underlying"),
        "tkr_common": st.session_state.get("tkr_common"),
        "_k": _k,
    }
    ctx = build_option_context(state)
    st.session_state[_CTX_MEMO_KEY] = {"ticker": tk, "ts": time.monotonic(), "ctx": dict(ctx)}
    return ctx


__all__ = ["get_option_context"]
