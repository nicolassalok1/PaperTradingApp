import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_digital


def render():
    option_panel("Digital (cash-or-nothing)")
    st.caption("Payoff binaire conditionne au franchissement du seuil.")
    base = "digital_exo_"
    with params_expander():
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input(
                "Sous-jacent S0",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            K = st.number_input(
                "Seuil (K)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
            payout = st.number_input(
                "Payout", value=1.0, min_value=0.0, step=0.1, key=f"{base}payout"
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
            option_type = st.selectbox("Type", ["Call", "Put"], index=0, key=f"{base}type")

    if compute_button("Calculer le prix"):
        try:
            price = price_digital(
                S0, K, r, q, T, sigma, payout=payout, option_type=option_type.lower()
            )
            st.success(f"Prix : {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Digital : {exc}")
