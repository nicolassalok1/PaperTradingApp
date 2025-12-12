import streamlit as st

from app.vue.tabs import tab_hedger_v2 as hedger_v2_tab
from app.vue.tabs import tab_hedger_rl_live_v2 as rl_live_tab

TAB_LABEL = "🛡️ Hedging Systems"


def render_tab() -> None:
    st.title("🛡️ Hedging Systems")
    mode = st.radio(
        "Hedging mode",
        options=["Hedger v2", "RL Live v2"],
        horizontal=True,
        key="hedging_systems_mode",
    )
    st.divider()

    if mode == "Hedger v2":
        if hasattr(hedger_v2_tab, "render_tab"):
            hedger_v2_tab.render_tab()
        else:
            hedger_v2_tab.render()
    else:
        if hasattr(rl_live_tab, "render_tab"):
            rl_live_tab.render_tab()
        else:
            rl_live_tab.render()


def render() -> None:
    render_tab()

