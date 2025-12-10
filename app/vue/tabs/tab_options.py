import streamlit as st

from app.vue.components.options.router import render_options_router


def render_tab_options():
    st.markdown("### 📈 Mode Options")
    render_options_router()


def render():
    render_tab_options()
