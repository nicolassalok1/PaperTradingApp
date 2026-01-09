import streamlit as st

from app.vue.components.options.router import render_options_router
from app.vue.components.page_utils import render_page_header


TAB_LABEL = "📈 Options"


def render_tab_options():
    render_page_header(
        "Options",
        "Pricing, grecs, surfaces IV et stratégies — avec paramètres globaux partagés.",
        icon="📈",
        badge="Pricing",
    )
    render_options_router()


def render():
    render_tab_options()
