from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from app.controller import heston_controller


def _hero(title: str, subtitle: str, icon: str = "??", badge: str | None = None):
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero__icon">{icon}</div>
            <div class="page-hero__titles">
                <div class="page-hero__title">{title}</div>
                <div class="page-hero__subtitle">{subtitle}</div>
            </div>
            {'<div class="page-hero__badge">'+badge+'</div>' if badge else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _params_inputs():
    cols = st.columns(5)
    kappa = cols[0].slider("kappa", 0.01, 5.0, 1.0, 0.01)
    theta = cols[1].slider("theta (long-run var)", 0.0001, 1.0, 0.04, 0.0001)
    sigma = cols[2].slider("sigma (vol of vol)", 0.001, 3.0, 0.5, 0.001)
    rho = cols[3].slider("rho", -0.99, 0.99, -0.5, 0.01)
    v0 = cols[4].slider("v0 (init var)", 0.0001, 2.0, 0.04, 0.0001)
    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "v0": v0,
    }


def _pricing_section():
    st.subheader("Heston Pricing")
    col1, col2, col3 = st.columns(3)
    S0 = col1.number_input(
        "Spot S0", value=100.0, min_value=0.0, format="%.4f", key="heston_price_s0"
    )
    r = col2.number_input("Rate r", value=0.01, format="%.4f", key="heston_price_r")
    q = col3.number_input("Dividend q", value=0.0, format="%.4f", key="heston_price_q")
    K = st.number_input("Strike K", value=100.0, min_value=0.0, format="%.4f", key="heston_price_k")
    T = st.number_input(
        "Maturity T (years)", value=1.0, min_value=0.0, format="%.4f", key="heston_price_t"
    )
    params = _params_inputs()
    params["r"] = r
    params["q"] = q

    if st.button("Compute Heston Price", type="primary"):
        res = heston_controller.compute_heston_price({"S0": S0, "K": K, "T": T, "params": params})
        st.metric("Call Price", f"{res.get('price', float('nan')):.6f}")


def _load_surface_from_upload():
    uploaded = st.file_uploader("Upload IV surface CSV (columns: K,T,iv)", type=["csv"])
    if not uploaded:
        return None
    try:
        content = uploaded.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        return df
    except Exception:
        st.error("Impossible de lire le CSV.")
        return None


def _calibration_section():
    st.subheader("Calibration (IV surface)")
    col1, col2, col3 = st.columns(3)
    S0 = col1.number_input(
        "Spot S0 (calibration)", value=100.0, min_value=0.0, format="%.4f", key="hcal_s0"
    )
    r = col2.number_input("Rate r (calibration)", value=0.01, format="%.4f", key="hcal_r")
    q = col3.number_input("Dividend q (calibration)", value=0.0, format="%.4f", key="hcal_q")

    df = _load_surface_from_upload()
    st.caption("Alternatively, paste rows (K,T,iv) below:")
    pasted = st.text_area("K,T,iv lines", height=120, placeholder="100,0.5,0.25\n110,0.5,0.24")
    if df is None and pasted.strip():
        try:
            parsed = [line.split(",") for line in pasted.strip().splitlines()]
            df = pd.DataFrame(parsed, columns=["K", "T", "iv"]).astype(float)
        except Exception:
            st.error("Format invalide pour les donnÈes collees.")
            df = None

    if df is not None and not df.empty:
        st.dataframe(df, hide_index=True)

    if st.button("Calibrate Heston", type="primary", disabled=df is None or df.empty):
        surface = df.to_dict(orient="list") if df is not None else {}
        res = heston_controller.calibrate_heston_from_market(
            {"market_iv_surface": surface, "S0": S0, "r": r, "q": q}
        )
        params = res.get("params", {})
        st.json(params)


def render():
    _hero("Heston Model", "Pricing and calibration pipeline", icon="📈", badge="Heston")
    _pricing_section()
    st.markdown("---")
    _calibration_section()


__all__ = ["render"]
