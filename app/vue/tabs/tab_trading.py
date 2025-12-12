import streamlit as st

from app.vue.tabs import tab_alpaca_spot as spot_tab
from app.vue.tabs import tab_alpaca_orders as orders_tab

TAB_LABEL = "💹 Trading"


def render_tab() -> None:
    st.title("💹 Trading - Spot & Advanced Orders")
    mode = st.radio(
        "Trading view",
        options=["Spot", "Advanced Orders"],
        horizontal=True,
        key="trading_mode",
    )
    st.divider()

    if mode == "Spot":
        if hasattr(spot_tab, "render_tab"):
            spot_tab.render_tab()
        else:
            spot_tab.render()
    else:
        if hasattr(orders_tab, "render_tab"):
            orders_tab.render_tab()
        else:
            orders_tab.render()


def render() -> None:
    render_tab()

