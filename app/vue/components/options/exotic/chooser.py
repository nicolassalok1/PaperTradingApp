import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_chooser
from app.vue.components.options.controller_bridge import *


def render():
    option_panel("Chooser")
    base = "chooser_"
    with params_expander():
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input(
                "Sous-jacent (S0)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            K = st.number_input(
                "Strike K",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
        with col2:
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
            )
            r = st.number_input(
                "Taux sans risque (r)",
                value=0.02,
                step=0.005,
                format="%.4f",
                key=f"{base}r",
            )
            q = st.number_input(
                "Dividend yield (q)",
                value=0.0,
                step=0.005,
                format="%.4f",
                key=f"{base}q",
            )
        with col3:
            sigma = st.number_input(
                "Volatilite (sigma)",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma",
            )

    if compute_button("Pricer Chooser"):
        try:
            price = price_chooser(S0, K, r, q, T, sigma)
            st.success(f"Prix Chooser ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Chooser : {exc}")
