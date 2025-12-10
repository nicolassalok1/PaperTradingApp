import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
)
from app.vue.components.options.pricing import price_basket_call


def render():
    option_panel("Basket Call (MC)")
    base = "basket_call_"
    with params_expander():
        weights_raw = st.text_input("Poids (separes par ,)", value="0.5,0.5", key=f"{base}weights")
        spots_raw = st.text_input("Spots (separes par ,)", value="100,100", key=f"{base}spots")
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
            "Maturite T (annees)",
            value=1.0,
            min_value=0.01,
            step=0.05,
            key=f"{base}t",
        )
        sigma = st.number_input(
            "Volatilite (sigma) commune",
            value=0.2,
            min_value=0.0001,
            step=0.01,
            format="%.4f",
            key=f"{base}sigma",
        )
        n_paths = st.number_input(
            "Trajectoires", value=5000, min_value=500, step=500, key=f"{base}paths"
        )

    if compute_button():
        try:
            weights = [float(x) for x in weights_raw.split(",") if x.strip()]
            spots = [float(x) for x in spots_raw.split(",") if x.strip()]
            price = price_basket_call(weights, spots, r, q, T, sigma, n_paths=int(n_paths))
            st.success(f"Prix Basket Call ≈ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Basket Call : {exc}")

    try:
        weights = [float(x) for x in weights_raw.split(",") if x.strip()]
        spots = [float(x) for x in spots_raw.split(",") if x.strip()]
        basket_spot = (
            sum(w * s for w, s in zip(weights, spots)) / sum(weights)
            if weights and spots and len(weights) == len(spots)
            else 100.0
        )
    except Exception:
        basket_spot = 100.0

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=basket_spot,
            K=basket_spot,
            T=T,
            r=r,
            sigma=sigma,
            option_char="c",
        )
