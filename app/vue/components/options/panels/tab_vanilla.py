import streamlit as st
from app.vue.components.options.panels.vanilla.tab_american import render_tab_american
from app.vue.components.options.panels.vanilla.tab_bermudan import render_tab_bermudan
from app.vue.components.options.panels.vanilla.tab_european import render_tab_european
from app.vue.components.options.controller_bridge import *


def render_panel_vanilla():
    st.subheader("Options Vanilla / Early Exercise")
    ctx = get_option_context()
    available = ensure_close_history(ctx)
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not available:
        return
    st.caption(f"Spot actuel ({current_ticker(ctx) or ctx.get('ticker') or 'Ticker'}) : {current_spot(ctx):.2f}")
    models = [
        "Européenne",
        "Américaine",
        "Bermudan",
    ]
    model_tabs = st.tabs(models)

    for label, tab in zip(models, model_tabs):
        with tab:
            if label == "Européenne":
                render_tab_european()
            elif label == "Américaine":
                render_tab_american()
            elif label == "Bermudan":
                render_tab_bermudan()
            else:
                st.error("Option inconnue.")
