import streamlit as st

from app.vue.components.page_utils import render_page_header
from app.vue.components.ui_helpers import render_quickstart
from app.vue.tabs import tab_alpaca_spot as spot_tab
from app.vue.tabs import tab_alpaca_orders as orders_tab
from app.vue.tabs import tab_alpaca_options as options_tab

TAB_LABEL = "💹 Trading"


def render_tab() -> None:
    render_page_header(
        "Trading (Alpaca)",
        "Spot, options et ordres avancés — le tout via Alpaca.",
        icon="💹",
        badge="Trading",
    )
    render_quickstart(
        "Guide rapide",
        [
            "‘Spot’ = actions/ETF + ordres simples + état du compte.",
            "‘Ordres avancés’ = stop/limit/bracket (attention aux envois live).",
            "‘Options’ = chaîne d’options Alpaca + envoi d’ordres options.",
        ],
        expanded=False,
    )

    spot_tab_obj, orders_tab_obj, options_tab_obj = st.tabs(
        ["Spot (compte & ordres)", "Ordres avancés", "Options (Alpaca)"]
    )

    with spot_tab_obj:
        if hasattr(spot_tab, "render_tab"):
            spot_tab.render_tab()
        else:
            spot_tab.render()

    with orders_tab_obj:
        if hasattr(orders_tab, "render_tab"):
            orders_tab.render_tab()
        else:
            orders_tab.render()

    with options_tab_obj:
        if hasattr(options_tab, "render_tab"):
            options_tab.render_tab()
        else:
            options_tab.render()


def render() -> None:
    render_tab()
