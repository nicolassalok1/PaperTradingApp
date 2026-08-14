import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_quanto


def render():
    option_panel("Quanto")
    base = "quanto_"
    with params_expander():
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input(
                "Sous-jacent etranger S0",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            fx_rate = st.number_input(
                "Taux FX (dom/for)",
                value=1.0,
                min_value=0.0001,
                step=0.01,
                key=f"{base}fx",
            )
            K = st.number_input(
                "Strike K",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
        with col2:
            r_dom = st.number_input(
                "Taux domestique r_dom",
                value=0.02,
                step=0.005,
                format="%.4f",
                key=f"{base}rdom",
            )
            r_for = st.number_input(
                "Taux etranger r_for",
                value=0.01,
                step=0.005,
                format="%.4f",
                key=f"{base}rfor",
            )
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
            )
        with col3:
            sigma_dom = st.number_input(
                "Vol sous-jacent",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma_dom",
            )
            sigma_fx = st.number_input(
                "Vol FX",
                value=0.1,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma_fx",
            )
            rho = st.number_input(
                "Correlation (eq/FX)",
                value=0.0,
                min_value=-1.0,
                max_value=1.0,
                step=0.05,
                key=f"{base}rho",
            )
            option_type = st.selectbox("Type", ["Call", "Put"], index=0, key=f"{base}type")

    if compute_button("Pricer Quanto"):
        try:
            price = price_quanto(
                S0_domestic=S0,
                fx_rate=fx_rate,
                K=K,
                r_dom=r_dom,
                r_for=r_for,
                sigma_dom=sigma_dom,
                sigma_fx=sigma_fx,
                rho=rho,
                T=T,
                option_type=option_type.lower(),
            )
            st.success(f"Prix Quanto ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Quanto : {exc}")
