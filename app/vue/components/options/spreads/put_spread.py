import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
)
from app.vue.components.options.pricing import price_put_spread


def render():
    option_panel("Put spread")
    base = "put_spread_"
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
            K_long = st.number_input(
                "Strike Long",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k_long",
            )
        with col2:
            K_short = st.number_input(
                "Strike Short",
                value=90.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k_short",
            )
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
            )
        with col3:
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

    if compute_button("Calculer le prix"):
        try:
            price = price_put_spread(S0, K_long, K_short, r, q, T, sigma)
            st.success(f"Prix : {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Put spread : {exc}")

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=S0,
            K=K_long,
            T=T,
            r=r,
            sigma=sigma,
            option_char="p",
        )
