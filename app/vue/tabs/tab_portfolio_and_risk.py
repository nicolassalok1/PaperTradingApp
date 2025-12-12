import streamlit as st

from app.vue.components.page_utils import render_page_header
from app.vue.tabs import tab_portfolio_allocation as alloc_tab
from app.vue.tabs import tab_risk_management as risk_tab

TAB_LABEL = "📊 Portfolio & Risk"


def render_tab() -> None:
    render_page_header(
        "Portfolio & Risk",
        "Alpaca portfolio allocation, rebalancing, and live risk metrics.",
        icon="📊",
        badge="Portfolio",
    )

    alloc_tab_obj, risk_tab_obj = st.tabs(["Allocation & Rebalance", "Risk & Exposure"])

    with alloc_tab_obj:
        if hasattr(alloc_tab, "render_tab"):
            alloc_tab.render_tab()
        else:
            alloc_tab.render()

    with risk_tab_obj:
        if hasattr(risk_tab, "render_tab"):
            risk_tab.render_tab()
        else:
            risk_tab.render()


def render() -> None:
    render_tab()
