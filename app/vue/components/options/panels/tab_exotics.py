import streamlit as st
from app.vue.components.options.panels.exotics.tab_quanto import render_tab_quanto
from app.vue.components.options.panels.exotics.tab_rainbow import render_tab_rainbow
from app.vue.components.options.panels.exotics.tab_chooser import render_tab_chooser
from app.vue.components.options.controller_bridge import *


def render_panel_exotics():
    st.subheader("Options Exotiques (Advanced)")
    ctx = get_option_context()
    available = ensure_close_history(ctx)
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    if not available:
        return

    exotic_labels = [
        "Quanto",
        "Rainbow",
        "Chooser",
    ]
    exotic_tabs = st.tabs(exotic_labels)

    for label, tab in zip(exotic_labels, exotic_tabs):
        with tab:
            if label == "Quanto":
                render_tab_quanto()
            elif label == "Rainbow":
                render_tab_rainbow()
            elif label == "Chooser":
                render_tab_chooser()
            else:
                st.error("Structure inconnue.")
