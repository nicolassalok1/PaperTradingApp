from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.vue.components.options.controller_bridge import (
    get_common_div_yield,
    get_common_maturity_value,
    get_common_sigma_value,
    get_rate_for_ttm,
    load_close_series_for_ticker,
)
from app.vue.components.options.pricing import price_basket_call, price_basket_put


def _norm_ticker(t: str) -> str:
    return (t or "").strip().upper()


def _default_assets() -> list[dict]:
    return [
        {"Ticker": "AAPL", "Weight": 0.5, "Spot": 100.0},
        {"Ticker": "MSFT", "Weight": 0.5, "Spot": 100.0},
    ]


def _load_spots_from_cache(rows: list[dict]) -> list[dict]:
    updated: list[dict] = []
    for row in rows:
        rec = dict(row or {})
        tkr = _norm_ticker(str(rec.get("Ticker", "")))
        rec["Ticker"] = tkr
        if not tkr:
            updated.append(rec)
            continue
        series = load_close_series_for_ticker(tkr, fallback_value=None)
        try:
            spot = float(series.dropna().iloc[-1]) if series is not None and not series.empty else None
        except Exception:
            spot = None
        if spot is not None and np.isfinite(spot) and spot > 0:
            rec["Spot"] = float(spot)
        updated.append(rec)
    return updated


def _sanitize_assets(df: pd.DataFrame) -> tuple[list[float], list[float], list[str]]:
    if df is None or df.empty:
        return [], [], []
    tickers: list[str] = []
    weights: list[float] = []
    spots: list[float] = []

    for _, row in df.iterrows():
        tkr = _norm_ticker(str(row.get("Ticker", "")))
        w = row.get("Weight", 0.0)
        s = row.get("Spot", 0.0)
        try:
            w = float(w)
        except Exception:
            w = 0.0
        try:
            s = float(s)
        except Exception:
            s = 0.0
        if not np.isfinite(w) or not np.isfinite(s) or s <= 0:
            continue
        tickers.append(tkr)
        weights.append(float(w))
        spots.append(float(s))

    if not weights or not spots:
        return [], [], []

    w_sum = float(np.sum(weights))
    if not np.isfinite(w_sum) or abs(w_sum) <= 1e-12:
        return [], [], []

    weights_norm = [float(w) / w_sum for w in weights]
    return weights_norm, spots, tickers


def render_panel_basket() -> None:
    st.subheader("Basket options (multi-sous-jacents)")
    st.caption("MC pricer (log-normal, corrélation constante ρ). Poids normalisés automatiquement.")

    if "basket_assets_rows" not in st.session_state:
        st.session_state["basket_assets_rows"] = _default_assets()

    df_assets = pd.DataFrame(st.session_state["basket_assets_rows"])
    df_assets = st.data_editor(
        df_assets,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=False),
            "Weight": st.column_config.NumberColumn("Weight", min_value=-10.0, max_value=10.0, step=0.01),
            "Spot": st.column_config.NumberColumn("Spot", min_value=0.0, max_value=1e9, step=0.01),
        },
        key="basket_assets_editor",
    )
    st.session_state["basket_assets_rows"] = df_assets.to_dict(orient="records")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Charger les spots (dernier close)", use_container_width=True, type="secondary"):
            st.session_state["basket_assets_rows"] = _load_spots_from_cache(
                st.session_state.get("basket_assets_rows") or []
            )
            st.rerun()
    with col_b:
        rho = st.slider("Corrélation ρ (pairwise)", min_value=-0.50, max_value=0.90, value=0.20, step=0.05)
    with col_c:
        n_paths = st.number_input(
            "Trajectoires (MC)",
            min_value=1000,
            max_value=200000,
            value=20000,
            step=1000,
        )

    weights, spots, tickers = _sanitize_assets(df_assets)
    if not weights:
        st.info("Renseigne au moins un sous-jacent avec (Spot > 0) et des poids non nuls.")
        return

    basket_spot0 = float(np.sum(np.array(weights) * np.array(spots)))

    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns([1, 1, 1, 1, 1])
    with col_p1:
        option_type = st.selectbox("Type", ["call", "put"], index=0)
    with col_p2:
        T = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=float(get_common_maturity_value()),
            step=0.05,
        )
    with col_p3:
        sigma = st.number_input(
            "σ (commune)",
            min_value=0.0001,
            max_value=5.0,
            value=float(get_common_sigma_value()),
            step=0.01,
        )
    with col_p4:
        q = st.number_input(
            "q (div yield)",
            min_value=-0.50,
            max_value=1.00,
            value=float(get_common_div_yield()),
            step=0.001,
            format="%.6f",
        )
    with col_p5:
        r = float(get_rate_for_ttm(T))
        st.metric("r(T) (YC)", f"{r * 100:.2f}%", f"T={T:.2f}y")

    strike = st.number_input(
        "Strike K (sur le basket)",
        min_value=0.01,
        value=float(basket_spot0),
        step=0.5,
    )

    compute = st.button("Pricer (MC)", type="primary", use_container_width=True)
    if compute:
        try:
            if option_type == "call":
                price = float(
                    price_basket_call(
                        weights,
                        spots,
                        r=r,
                        q=float(q),
                        T=float(T),
                        sigma=float(sigma),
                        strike=float(strike),
                        rho=float(rho),
                        n_paths=int(n_paths),
                    )
                )
            else:
                price = float(
                    price_basket_put(
                        weights,
                        spots,
                        r=r,
                        q=float(q),
                        T=float(T),
                        sigma=float(sigma),
                        strike=float(strike),
                        rho=float(rho),
                        n_paths=int(n_paths),
                    )
                )
            st.success(f"Prix {option_type} ~ {price:.6f}")
        except Exception as exc:
            st.error(f"Echec pricing basket: {exc}")
            return

    st.markdown("#### Payoff (à maturité)")
    span = 0.6
    b_grid = np.linspace(max(0.01, basket_spot0 * (1 - span)), basket_spot0 * (1 + span), 200)
    payoff = np.maximum(b_grid - float(strike), 0.0) if option_type == "call" else np.maximum(float(strike) - b_grid, 0.0)
    chart_df = pd.DataFrame({"Basket": b_grid, "Payoff": payoff})
    st.line_chart(chart_df.set_index("Basket"))
