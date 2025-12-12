import streamlit as st

from app.vue.tabs import tab_portfolio_allocation as alloc_tab
from app.vue.tabs import tab_risk_management as risk_tab

TAB_LABEL = "📊 Portfolio & Risk"


def render_tab() -> None:
    st.title("📊 Portfolio & Risk")
    section = st.radio(
        "Section",
        options=["Allocation", "Risk"],
        horizontal=True,
        key="portfolio_risk_section",
    )
    st.divider()

    if section == "Allocation":
        if hasattr(alloc_tab, "render_tab"):
            alloc_tab.render_tab()
        else:
            alloc_tab.render()
    else:
        if hasattr(risk_tab, "render_tab"):
            risk_tab.render_tab()
        else:
            risk_tab.render()


def render() -> None:
    render_tab()

