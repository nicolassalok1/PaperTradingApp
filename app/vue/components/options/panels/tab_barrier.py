import streamlit as st
from app.vue.components.options.panels.barrier.tab_grp_barrier import render_tab_grp_barrier
from app.vue.components.options.panels.barrier.tab_digital import render_tab_digital
from app.vue.components.options.panels.barrier.tab_asset_on import render_tab_asset_on
from app.vue.components.options.controller_bridge import *


def render_panel_barrier():
    st.subheader("Options à barrière / Digitals / Asset-on")
    ctx = get_option_context()
    available = ensure_close_history(ctx)
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not available:
        return

    barrier_labels = [
        "Vanilla Barrières",
        "Option Digitale",
        "Asset-on Option",
    ]
    barrier_tabs = st.tabs(barrier_labels)

    for label, tab in zip(barrier_labels, barrier_tabs):
        with tab:
            if label == "Vanilla Barrières":
                render_tab_grp_barrier()
            elif label == "Option Digitale":
                render_tab_digital()
            elif label == "Asset-on Option":
                render_tab_asset_on()
            else:
                st.error("Type inconnu.")
