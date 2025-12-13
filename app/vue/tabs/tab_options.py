import streamlit as st

from app.vue.components.options.router import render_options_router
from app.vue.components.page_utils import render_page_header
from app.vue.components.ui_helpers import render_quickstart


TAB_LABEL = "📈 Options"


def render_tab_options():
    render_page_header(
        "Options",
        "Pricing, grecs, surfaces IV et stratégies — avec paramètres globaux partagés.",
        icon="📈",
        badge="Pricing",
    )
    render_quickstart(
        "Guide rapide",
        [
            "Règle d’abord les paramètres globaux (`r`, `q`, `sigma`).",
            "Charge une surface IV (Yahoo) ou envoie une surface depuis `🧪 Calibration avancée`.",
            "Chaque famille d’options est dans un sous-onglet (vanilla, barrières, spreads…).",
        ],
        expanded=False,
    )
    render_options_router()


def render():
    render_tab_options()
