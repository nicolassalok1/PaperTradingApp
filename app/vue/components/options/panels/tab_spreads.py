import streamlit as st
from app.vue.components.options.panels.spreads.tab_straddle import render_tab_straddle
from app.vue.components.options.panels.spreads.tab_strangle import render_tab_strangle
from app.vue.components.options.panels.spreads.tab_call_spread import render_tab_call_spread
from app.vue.components.options.panels.spreads.tab_put_spread import render_tab_put_spread
from app.vue.components.options.panels.spreads.tab_butterfly import render_tab_butterfly
from app.vue.components.options.panels.spreads.tab_condor import render_tab_condor
from app.vue.components.options.panels.spreads.tab_iron_bfly import render_tab_iron_bfly
from app.vue.components.options.panels.spreads.tab_iron_condor import render_tab_iron_condor
from app.vue.components.options.controller_bridge import *


def render_panel_spreads():
    st.subheader("Options Spreads & Wings")
    ctx = get_option_context()
    available = ensure_close_history(ctx)
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not available:
        return

    spread_labels = [
        "Straddle",
        "Strangle",
        "Call Spread",
        "Put Spread",
        "Butterfly",
        "Condor",
        "Iron Butterfly",
        "Iron Condor",
    ]
    spread_tabs = st.tabs(spread_labels)

    for label, tab in zip(spread_labels, spread_tabs):
        with tab:
            if label == "Straddle":
                render_tab_straddle()
            elif label == "Strangle":
                render_tab_strangle()
            elif label == "Call Spread":
                render_tab_call_spread()
            elif label == "Put Spread":
                render_tab_put_spread()
            elif label == "Butterfly":
                render_tab_butterfly()
            elif label == "Condor":
                render_tab_condor()
            elif label == "Iron Butterfly":
                render_tab_iron_bfly()
            elif label == "Iron Condor":
                render_tab_iron_condor()
            else:
                st.error("Type inconnu.")
