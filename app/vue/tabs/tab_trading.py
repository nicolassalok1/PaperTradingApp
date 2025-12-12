import streamlit as st

from app.vue.components.page_utils import render_page_header
from app.vue.tabs import tab_alpaca_spot as spot_tab
from app.vue.tabs import tab_alpaca_orders as orders_tab

TAB_LABEL = "💹 Trading"


def render_tab() -> None:
    render_page_header(
        "Trading",
        "Spot account view, open positions, and advanced order entry via Alpaca.",
        icon="💹",
        badge="Trading",
    )

    spot_tab_obj, orders_tab_obj = st.tabs(["Spot (account & orders)", "Advanced Orders"])

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


def render() -> None:
    render_tab()
