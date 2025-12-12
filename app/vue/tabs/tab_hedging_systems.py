import streamlit as st

from app.vue.components.page_utils import render_page_header
from app.vue.tabs import tab_hedger_v2 as hedger_v2_tab
from app.vue.tabs import tab_hedger_rl_live_v2 as rl_live_tab

TAB_LABEL = "??? Hedging Systems"


def render_tab() -> None:
    render_page_header(
        "Hedging Systems",
        "Manual hedging (v2) and RL-based live hedging in one place.",
        icon="???",
        badge="Hedging",
    )

    st.markdown("### Hedger v2")
    if hasattr(hedger_v2_tab, "render_tab"):
        hedger_v2_tab.render_tab()
    else:
        hedger_v2_tab.render()

    st.divider()

    st.markdown("### RL Live v2")
    if hasattr(rl_live_tab, "render_tab"):
        rl_live_tab.render_tab()
    else:
        rl_live_tab.render()


def render() -> None:
    render_tab()

