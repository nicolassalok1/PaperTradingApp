import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.panels.path.tab_asian import render_tab_asian
from app.vue.components.options.panels.path.tab_asian_geo import render_tab_asian_geo
from app.vue.components.options.panels.path.tab_lookback import render_tab_lookback
from app.vue.components.options.panels.path.tab_lookback_fixed import render_tab_lookback_fixed
from app.vue.components.options.panels.path.tab_forward_start import render_tab_forward_start
from app.vue.components.options.panels.path.tab_cliquet import render_tab_cliquet
from app.vue.components.options.controller_bridge import *


def render_panel_path():
    st.subheader("Path-dependent options")
    ctx = get_option_context()
    available = ensure_close_history(ctx)
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not available:
        return

    path_labels = [
        "Asian",
        "Asian géométrique",
        "Lookback",
        "Lookback (fixed strike)",
        "Forward-start",
        "Cliquet / Ratchet",
    ]
    path_tabs = st.tabs(path_labels)

    for label, tab in zip(path_labels, path_tabs):
        with tab:
            if label == "Asian":
                render_tab_asian()
            elif label == "Asian géométrique":
                render_tab_asian_geo()
            elif label == "Lookback":
                render_tab_lookback()
            elif label == "Lookback (fixed strike)":
                render_tab_lookback_fixed()
            elif label == "Forward-start":
                render_tab_forward_start()
            elif label == "Cliquet / Ratchet":
                render_tab_cliquet()
            else:
                st.error("Option inconnue.")
