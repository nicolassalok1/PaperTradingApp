import streamlit as st

from app.vue.components.options.panels.tab_vanilla import render_panel_vanilla
from app.vue.components.options.panels.tab_path import render_panel_path
from app.vue.components.options.panels.tab_barrier import render_panel_barrier
from app.vue.components.options.panels.tab_spreads import render_panel_spreads
from app.vue.components.options.panels.tab_calendar import render_panel_calendar
from app.vue.components.options.panels.tab_exotics import render_panel_exotics
from app.vue.components.options.controller_bridge import *


def render_options_router():
    st.header("?? Options - Interface Professionnelle")

    # Compact tab styling
    st.markdown(
        """
        <style>
        [data-testid="stTabs"] button[role="tab"] {
            padding: 0.35rem 0.65rem !important;
            font-size: 0.85rem !important;
            min-height: 2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tkr_common = st.text_input(
        "Ticker commun pour les historiques IV/cl“tures (optionnel)",
        value=st.session_state.get("tkr_common", ""),
        placeholder="ex: AAPL",
    )
    tkr_common_norm = (tkr_common or "").strip().upper()
    prev_tkr = st.session_state.get("_prev_tkr_common")
    if tkr_common_norm and tkr_common_norm != prev_tkr:
        try:
            clear_closing_history_cache(tkr_common_norm, period="2y", interval="1d")
            clear_closing_history_cache(tkr_common_norm, period="1y", interval="1d")
        except Exception:
            pass
    st.session_state["_prev_tkr_common"] = tkr_common_norm
    st.session_state["tkr_common"] = tkr_common_norm
    st.session_state["common_underlying"] = tkr_common_norm

    # Close prices chart displayed directly under the ticker input
    ctx = get_option_context()
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]
    close_series = ctx.get("close_series")
    if close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        tkr_label = ctx.get("ticker") or tkr_common_norm or "Ticker"
        render_static_line_chart(
            close_series,
            title=f"{tkr_label} - Clotures (cache)",
            y_label="Prix de cloture",
        )

    families = [
        "Vanilla / Early Exercise",
        "Path-dependent",
        "BarriŠres",
        "Spreads & Wings",
        "Calendriers",
        "Exotiques avanc‚es",
    ]

    family_tabs = st.tabs(families)
    for fam_label, fam_tab in zip(families, family_tabs):
        with fam_tab:
            if fam_label == "Vanilla / Early Exercise":
                render_panel_vanilla()
            elif fam_label == "Path-dependent":
                render_panel_path()
            elif fam_label == "BarriŠres":
                render_panel_barrier()
            elif fam_label == "Spreads & Wings":
                render_panel_spreads()
            elif fam_label == "Calendriers":
                render_panel_calendar()
            elif fam_label == "Exotiques avanc‚es":
                render_panel_exotics()
            else:
                st.error("Famille inconnue.")
