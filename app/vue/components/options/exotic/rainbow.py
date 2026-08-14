import streamlit as st
from app.vue.components.options.layout import compute_button, option_panel, params_expander
from app.vue.components.options.pricing import price_rainbow


def render():
    option_panel("Rainbow (max de 2 actifs)")
    base = "rainbow_"
    with params_expander():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            S1 = st.number_input(
                "Sous-jacent 1 (S1)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s1",
            )
            sigma1 = st.number_input(
                "Vol S1",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma1",
            )
        with col2:
            S2 = st.number_input(
                "Sous-jacent 2 (S2)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s2",
            )
            sigma2 = st.number_input(
                "Vol S2",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma2",
            )
        with col3:
            K = st.number_input(
                "Strike K",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
            corr = st.number_input(
                "Correlation",
                value=0.0,
                min_value=-1.0,
                max_value=1.0,
                step=0.05,
                key=f"{base}corr",
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
            T = st.number_input(
                "Maturite (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                key=f"{base}t",
            )
    option_type = st.selectbox("Type", ["Call (max)", "Put (min)"], index=0, key=f"{base}type")
    n_paths = st.number_input(
        "Trajectoires", value=5000, min_value=500, step=500, key=f"{base}paths"
    )

    if compute_button("Pricer Rainbow"):
        try:
            price = price_rainbow(
                S1=S1,
                S2=S2,
                K=K,
                r=r,
                q=q,
                T=T,
                sigma1=sigma1,
                sigma2=sigma2,
                corr=corr,
                option_type="call" if option_type.lower().startswith("call") else "put",
                n_paths=int(n_paths),
            )
            st.success(f"Prix Rainbow ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Rainbow : {exc}")
