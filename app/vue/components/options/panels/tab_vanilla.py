import streamlit as st
from app.vue.components.options.panels.vanilla.tab_heston import render_tab_heston
from app.vue.components.options.panels.vanilla.tab_american import render_tab_american
from app.vue.components.options.panels.vanilla.tab_bermudan import render_tab_bermudan
from app.vue.components.options.controller_bridge import *


def render_panel_vanilla():
    st.subheader("Options Vanilla / Early Exercise")
    ctx = get_option_context()
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    close_series = ctx.get("close_series")
    if close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        st.line_chart(close_series)

    models = [
        "Européenne",
        "Américaine",
        "Bermudan",
    ]
    model_tabs = st.tabs(models)

    for label, tab in zip(models, model_tabs):
        with tab:
            if label == "Européenne":
                render_tab_heston()
            elif label == "Américaine":
                render_tab_american()
            elif label == "Bermudan":
                render_tab_bermudan()
            else:
                st.error("Option inconnue.")
