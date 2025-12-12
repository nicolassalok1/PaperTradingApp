import importlib
import streamlit as st

from app.vue.components.selector import choose_option_select
from app.vue.components.options.layout import (
    render_heatmap_diagnostics,
    render_options_history_block,
    render_payoff_text,
)
from app.vue.components.options.shared import refresh_underlying_cache, show_cache_status
from app.vue.pages.shared_ui import render_page_header

# Mapping of categories to their sub-tabs
CATEGORY_SUB_TABS = {
    "Vanilla / Early exercise": ["Americaine", "Bermuda"],
    "Path-dependent": [
        "Asian",
        "Asian geometrique",
        "Lookback",
        "Lookback fixed",
        "Forward-start",
        "Cliquet / Ratchet",
    ],
    "Barrieres": ["Vanilla", "Binaire"],
    "Spreads & Wings": [
        "Straddle",
        "Strangle",
        "Call spread",
        "Put spread",
        "Butterfly",
        "Condor",
        "Iron Condor",
        "Iron Butterfly",
    ],
    "Calendriers": ["Calendar spread", "Diagonal spread"],
    "Exotiques": ["Digital", "Asset-or-nothing", "Chooser", "Quanto", "Rainbow"],
    "Basket": ["Call", "Put"],
}

CATEGORY_TO_FOLDER = {
    "Vanilla / Early exercise": "vanilla",
    "Path-dependent": "path",
    "Barrieres": "barrier",
    "Spreads & Wings": "spreads",
    "Calendriers": "calendars",
    "Exotiques": "exotic",
    "Basket": "basket",
}

SUBTAB_TO_MODULE = {
    "Vanilla / Early exercise": {
        "Americaine": "american",
        "Bermuda": "bermuda",
    },
    "Path-dependent": {
        "Asian": "asian",
        "Asian geometrique": "asian_geo",
        "Lookback": "lookback",
        "Lookback fixed": "lookback_fixed",
        "Forward-start": "forward_start",
        "Cliquet / Ratchet": "cliquet",
    },
    "Barrieres": {
        "Vanilla": "vanilla_barrier",
        "Binaire": "digital_barrier",
    },
    "Spreads & Wings": {
        "Straddle": "straddle",
        "Strangle": "strangle",
        "Call spread": "call_spread",
        "Put spread": "put_spread",
        "Butterfly": "butterfly",
        "Condor": "condor",
        "Iron Condor": "iron_condor",
        "Iron Butterfly": "iron_butterfly",
    },
    "Calendriers": {
        "Calendar spread": "calendar",
        "Diagonal spread": "diagonal",
    },
    "Exotiques": {
        "Digital": "digital",
        "Asset-or-nothing": "asset_or_nothing",
        "Chooser": "chooser",
        "Quanto": "quanto",
        "Rainbow": "rainbow",
    },
    "Basket": {
        "Call": "call",
        "Put": "put",
    },
}


def render_options_root():
    """Top-level entry point for the Options section."""
    st.markdown(
        """
        <style>
            .stTabs [data-baseweb="tab"] {
                font-weight: 500;
                padding-top: 0px;
                padding-bottom: 0px;
                margin-right: 16px;
            }
            .stTabs [aria-selected="true"] {
                border-bottom: 2px solid #E63946 !important;
                color: #E63946 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_page_header(
        "Options",
        "Suite complete de pricing d'options : vanilla, exotiques, spreads et structures.",
        icon="🧮",
        badge="Pricing",
    )
    ticker = st.text_input("Ticker (sous-jacent)", value="AAPL")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Refresh"):
            refresh_underlying_cache(ticker)
    with col2:
        show_cache_status(ticker)

    render_options_history_block()
    st.markdown("---")

    st.subheader("Analyse générale (GPT-style)")
    with st.expander("Analyse de structure de l’option sélectionnée", expanded=False):
        option_label, option_char = choose_option_select("Call")
        st.markdown(f"Type sélectionné : **{option_label}**")
        render_payoff_text(option_label, "payoff")
        render_heatmap_diagnostics(
            S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2, n_steps=25, option_char=option_char
        )

    category_tabs = st.tabs(list(CATEGORY_SUB_TABS.keys()))
    for tab, category in zip(category_tabs, CATEGORY_SUB_TABS.keys()):
        with tab:
            _render_category(category)


def _render_category(category: str):
    """Render the sub-tabs for a given category."""
    sub_tabs = CATEGORY_SUB_TABS.get(category, [])
    tabs = st.tabs(sub_tabs)
    for tab, sub in zip(tabs, sub_tabs):
        with tab:
            render_option_panel(category, sub)


def render_option_panel(category: str, sub: str):
    """Render a specific option panel."""
    st.markdown(f"#### {sub}")
    module = load_pricing_module(category, sub)
    if hasattr(module, "render"):
        module.render()


def _resolve_module_target(category: str, sub: str):
    """Return the target folder and module name for a given category/sub-tab pair."""
    folder = CATEGORY_TO_FOLDER.get(category)
    if not folder:
        raise ValueError(f"Unknown category: {category}")
    module_name = SUBTAB_TO_MODULE.get(category, {}).get(sub)
    if not module_name:
        module_name = sub.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return folder, module_name


def load_pricing_module(category: str, sub: str):
    """
    Dynamically import the pricing module for the given category and sub-tab.
    Assumes that the module exists and exposes a render() function.
    """
    try:
        folder, module_name = _resolve_module_target(category, sub)
        module_path = f"app.vue.components.options.{folder}.{module_name}"
        return importlib.import_module(module_path)
    except Exception as exc:
        st.error(f"Impossible de charger le module pour {category} / {sub}: {exc}")
        raise
