import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_asian_geo_mc
from app.vue.components.options.controller_bridge import *


def render():
    option_panel("Option asiatique (geometrique)", "Monte Carlo")
    base = "path_asian_geo_mc_"
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
                "Strike (K)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
            T = st.number_input(
                "Maturite T (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key=f"{base}t",
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
            steps = st.number_input(
                "Nombre d'etapes (n_steps)",
                value=50,
                min_value=5,
                step=5,
                key=f"{base}steps",
            )
            n_paths = st.number_input(
                "Nombre de trajectoires (n_paths)",
                value=2000,
                min_value=100,
                step=500,
                key=f"{base}paths",
            )
            option_type = st.selectbox(
                "Type d'option (Call / Put)", ["Call", "Put"], index=0, key=f"{base}type"
            )

    if compute_button("Pricer (MC geometrique)"):
        try:
            price = price_asian_geo_mc(
                S0=S0,
                K=K,
                r=r,
                q=q,
                T=T,
                sigma=sigma,
                steps=int(steps),
                n_paths=int(n_paths),
                option_type=option_type.lower(),
            )
            st.success(f"Prix Asian geometrique {option_type} ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Asian geometrique : {exc}")
