import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_forward_start


def render():
    option_panel("Forward-start (approx BS)")
    base = "forward_start_"
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
            T_start = st.number_input(
                "Maturite de depart T_start (annees)",
                value=0.5,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key=f"{base}tstart",
            )
            T_end = st.number_input(
                "Maturite finale T_end (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key=f"{base}tend",
            )
        with col2:
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
        with col3:
            option_type = st.selectbox(
                "Type d'option (Call / Put)", ["Call", "Put"], index=0, key=f"{base}type"
            )

    if compute_button("Pricer Forward-start"):
        try:
            price = price_forward_start(
                S0=S0,
                r=r,
                q=q,
                T_start=T_start,
                T_end=T_end,
                sigma=sigma,
                option_type=option_type.lower(),
            )
            st.success(f"Prix Forward-start {option_type} ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Forward-start : {exc}")
