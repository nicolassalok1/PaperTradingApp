import streamlit as st
from app.vue.components.options.panels.calendar.tab_calendar import render_tab_calendar
from app.vue.components.options.panels.calendar.tab_diagonal import render_tab_diagonal
from app.vue.components.options.controller_bridge import *


def render_panel_calendar():
    st.subheader("Options Calendriers & Diagonals")
    ctx = get_option_context()
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not ensure_close_history(ctx):
        return

    calendar_labels = [
        "Calendar Spread",
        "Diagonal Spread",
    ]
    calendar_tabs = st.tabs(calendar_labels)

    for label, tab in zip(calendar_labels, calendar_tabs):
        if hasattr(tab, "__enter__"):
            with tab:
                if label == "Calendar Spread":
                    render_tab_calendar()
                elif label == "Diagonal Spread":
                    render_tab_diagonal()
                else:
                    st.error("Type inconnu.")
        else:
            if label == "Calendar Spread":
                render_tab_calendar()
            elif label == "Diagonal Spread":
                render_tab_diagonal()
            else:
                st.error("Type inconnu.")


def render_calendar_panel():
    """Alias for tests."""
    render_panel_calendar()
