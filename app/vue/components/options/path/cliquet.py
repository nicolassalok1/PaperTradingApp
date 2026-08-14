import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_cliquet


def render():
    option_panel("Cliquet / Ratchet (MC)")
    base = "cliquet_"
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
            T = st.number_input(
                "Maturite T (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key=f"{base}t",
            )
            n_periods = st.number_input(
                "Nombre de periodes",
                value=4,
                min_value=1,
                step=1,
                key=f"{base}periods",
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
            cap = st.number_input(
                "Cap par periode",
                value=0.1,
                step=0.01,
                format="%.4f",
                key=f"{base}cap",
            )
            floor = st.number_input(
                "Floor par periode",
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"{base}floor",
            )

    if compute_button("Pricer Cliquet"):
        try:
            price = price_cliquet(
                S0=S0,
                r=r,
                q=q,
                T=T,
                sigma=sigma,
                n_periods=int(n_periods),
                cap=cap,
                floor=floor,
            )
            st.success(f"Prix Cliquet ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Cliquet : {exc}")
