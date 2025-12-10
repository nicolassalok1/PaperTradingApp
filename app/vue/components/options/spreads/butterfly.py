import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
)
from app.vue.components.options.pricing import price_butterfly


def render():
    option_panel("Butterfly (calls)")
    base = "butterfly_"
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
            K1 = st.number_input(
                "Strike bas (K1)",
                value=90.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k1",
            )
        with col2:
            K2 = st.number_input(
                "Strike centre (K2)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k2",
            )
        with col3:
            K3 = st.number_input(
                "Strike haut (K3)",
                value=110.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k3",
            )
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
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

    if compute_button("Calculer le prix"):
        try:
            price = price_butterfly(S0, K1, K2, K3, r, q, T, sigma)
            st.success(f"Prix : {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Butterfly : {exc}")

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=S0,
            K=K2,
            T=T,
            r=r,
            sigma=sigma,
            option_char="c",
        )
