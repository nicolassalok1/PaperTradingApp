import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_barrier_digital
from app.vue.components.options.controller_bridge import *


def render():
    option_panel("Option barriere binaire (MC)")
    base = "bar_digital_"
    with params_expander():
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input(
                "Sous-jacent (S0)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            K = st.number_input(
                "Seuil payoff (K)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
            barrier = st.number_input(
                "Niveau barriere",
                value=110.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}barrier",
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
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
            )
        with col2:
            sigma = st.number_input(
                "Volatilite (sigma)",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma",
            )
            payout = st.number_input(
                "Payout binaire",
                value=1.0,
                min_value=0.0,
                step=0.1,
                key=f"{base}payout",
            )
            option_type = st.selectbox("Type d'option", ["Call", "Put"], index=0, key=f"{base}type")
            barrier_type = st.selectbox("Direction", ["up", "down"], index=0, key=f"{base}dir")
            knock = st.selectbox("Knock", ["out", "in"], index=0, key=f"{base}knock")
        steps = st.number_input("Pas temps", value=100, min_value=10, step=10, key=f"{base}steps")
        n_paths = st.number_input(
            "Trajectoires", value=5000, min_value=500, step=500, key=f"{base}paths"
        )

    if compute_button():
        try:
            price = price_barrier_digital(
                S0=S0,
                K=K,
                r=r,
                q=q,
                T=T,
                sigma=sigma,
                barrier=barrier,
                barrier_type=barrier_type,
                knock=knock,
                payout=payout,
                option_type=option_type.lower(),
                steps=int(steps),
                n_paths=int(n_paths),
            )
            st.success(f"Prix binaire {option_type} ({barrier_type}/{knock}) ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing barriere binaire : {exc}")
