import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
)
from app.vue.components.options.pricing import price_american_crr


def render():
    option_panel("Option americaine (CRR)")
    with params_expander():
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input(
                "Sous-jacent (S0)", value=100.0, min_value=0.01, step=1.0, key="am_s0"
            )
            K = st.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, key="am_k")
            T = st.number_input(
                "Maturite T (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key="am_t",
            )
        with col2:
            r = st.number_input(
                "Taux sans risque (r)", value=0.02, step=0.005, format="%.4f", key="am_r"
            )
            q = st.number_input(
                "Dividend yield (q)", value=0.00, step=0.005, format="%.4f", key="am_q"
            )
            sigma = st.number_input(
                "Volatilite (sigma)",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key="am_sigma",
            )
        with col3:
            steps = st.number_input(
                "Nombre d'etapes (n_steps)", value=50, min_value=5, step=5, key="am_steps"
            )
            option_type = st.selectbox("Type d'option", ["Call", "Put"], index=0, key="am_type")

    if compute_button():
        try:
            price = price_american_crr(
                S0=S0,
                K=K,
                r=r,
                q=q,
                T=T,
                sigma=sigma,
                steps=int(steps),
                option_type="call" if option_type.lower().startswith("c") else "put",
            )
            st.success(f"Prix americain {option_type} ≈ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing CRR : {exc}")

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_char="c" if option_type.lower().startswith("c") else "p",
            n_steps=int(steps),
        )
