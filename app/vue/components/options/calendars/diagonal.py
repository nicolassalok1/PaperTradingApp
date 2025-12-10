import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_diagonal_spread
from app.vue.components.options.controller_bridge import *


def render():
    option_panel("Diagonal spread")
    base = "diagonal_"
    with params_expander():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            S0 = st.number_input(
                "Sous-jacent (S0)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            K_short = st.number_input(
                "Strike court",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k_short",
            )
        with col2:
            K_long = st.number_input(
                "Strike long",
                value=105.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k_long",
            )
        with col3:
            T_short = st.number_input(
                "Maturite courte",
                value=0.5,
                min_value=0.01,
                step=0.05,
                key=f"{base}t_short",
            )
            T_long = st.number_input(
                "Maturite longue",
                value=1.0,
                min_value=0.05,
                step=0.05,
                key=f"{base}t_long",
            )
        with col4:
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
            sigma = st.number_input(
                "Volatilite (sigma)",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma",
            )

    if compute_button("Pricer Diagonal"):
        try:
            price = price_diagonal_spread(S0, K_short, K_long, r, q, T_short, T_long, sigma)
            st.success(f"Prix Diagonal ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Diagonal : {exc}")
