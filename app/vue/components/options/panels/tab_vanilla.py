import streamlit as st
from app.vue.components.options.panels.vanilla.tab_heston import render_tab_heston
from app.vue.components.options.panels.vanilla.tab_american import render_tab_american
from app.vue.components.options.panels.vanilla.tab_bermudan import render_tab_bermudan
from app.vue.components.options.controller_bridge import *


def render_panel_vanilla():
    st.subheader("Options Vanilla / Early Exercise")

    models = [
        "Européenne (Heston)",
        "Américaine (CRR)",
        "Bermudan (Longstaff-Schwartz)",
    ]
    model_tabs = st.tabs(models)

    for label, tab in zip(models, model_tabs):
        with tab:
            if label == "Européenne (Heston)":
                render_tab_heston()
            elif label == "Américaine (CRR)":
                render_tab_american()
            elif label == "Bermudan (Longstaff-Schwartz)":
                render_tab_bermudan()
            else:
                st.error("Option inconnue.")
