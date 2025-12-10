import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
)
from app.vue.components.options.pricing import price_iron_condor


def render():
    option_panel("Iron Condor")
    base = "iron_condor_"
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
            K_put_long = st.number_input(
                "Put long (K1)",
                value=90.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k1",
            )
        with col2:
            K_put_short = st.number_input(
                "Put short (K2)",
                value=95.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k2",
            )
        with col3:
            K_call_short = st.number_input(
                "Call short (K3)",
                value=105.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k3",
            )
        with col4:
            K_call_long = st.number_input(
                "Call long (K4)",
                value=110.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k4",
            )

    r = st.number_input(
        "Taux sans risque (r)", value=0.02, step=0.005, format="%.4f", key=f"{base}r"
    )
    q = st.number_input("Dividend yield (q)", value=0.0, step=0.005, format="%.4f", key=f"{base}q")
    T = st.number_input("Maturite", value=1.0, min_value=0.01, step=0.05, key=f"{base}t")
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
            price = price_iron_condor(
                S0, K_put_long, K_put_short, K_call_short, K_call_long, r, q, T, sigma
            )
            st.success(f"Prix : {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Iron Condor : {exc}")

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=S0,
            K=K_put_short,
            T=T,
            r=r,
            sigma=sigma,
            option_char="c",
        )
