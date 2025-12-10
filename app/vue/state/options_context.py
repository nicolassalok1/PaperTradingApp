"""
UI-facing context builder for Options panels.
Pulls values from Streamlit session state and delegates to the pure model helper.
"""

from __future__ import annotations

import streamlit as st

from app.controller.options_controller import build_option_context
from app.vue.components.options.ui_helpers import _k


def get_option_context():
    state = {
        "common_spot_value": st.session_state.get("common_spot_value", 100.0),
        "common_underlying": st.session_state.get("common_underlying"),
        "tkr_common": st.session_state.get("tkr_common"),
        "heston_cboe_ticker": st.session_state.get("heston_cboe_ticker"),
        "_k": _k,
    }
    return build_option_context(state)


__all__ = ["get_option_context"]
