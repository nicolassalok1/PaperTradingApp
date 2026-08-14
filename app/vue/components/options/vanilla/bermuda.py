import numpy as np
import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_bermuda_crr


def render():
    option_panel("Option bermudienne (CRR discret)")
    base = "bermuda_"
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
                value=0.00,
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
            n_ex_dates = st.number_input(
                "Nb dates d'exercice",
                value=5,
                min_value=1,
                step=1,
                key=f"{base}exdates",
            )
            option_type = st.selectbox(
                "Type d'option (Call / Put)", ["Call", "Put"], index=0, key=f"{base}type"
            )

    if compute_button("Pricer (Bermuda CRR)"):
        try:
            ex_dates = list(
                sorted(
                    set(
                        max(1, int(i))
                        for i in np.linspace(1, int(steps), num=int(n_ex_dates), endpoint=True)
                    )
                )
            )
            price = price_bermuda_crr(
                S0=S0,
                K=K,
                r=r,
                q=q,
                T=T,
                sigma=sigma,
                steps=int(steps),
                exercise_dates=ex_dates,
                option_type="call" if option_type.lower().startswith("c") else "put",
            )
            st.success(f"Prix bermudien {option_type} ~ {price:.4f} (ex dates: {ex_dates})")
        except Exception as exc:
            st.error(f"Echec du pricing Bermudan : {exc}")
