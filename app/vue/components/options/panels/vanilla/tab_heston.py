from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.vue.components.options.controller_bridge import *


def render_tab_heston():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    if not ensure_close_history(ctx):
        return

    spot_ref = float(S0) if S0 is not None else None
    if spot_ref is None or not np.isfinite(spot_ref) or spot_ref <= 0:
        try:
            fallback = float(st.session_state.get("common_spot_value", 100.0))
        except Exception:
            fallback = 100.0
        spot_ref = fallback if np.isfinite(fallback) and fallback > 0 else 100.0

    if not isinstance(close_series, pd.Series) or close_series.empty:
        close_series = pd.Series([spot_ref], index=pd.Index([pd.Timestamp.today()]))

    if not ticker:
        st.info("Choisis un ticker global dans l'entete Options pour activer ce panneau.")
        return

    st.subheader("Option europeenne (CBOE IV + Black-Scholes)")

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Type d'option", ["call", "put"], key=_k("heston_type"))
        strike_val = st.slider(
            "Strike",
            min_value=0.6 * spot_ref,
            max_value=1.4 * spot_ref,
            value=spot_ref,
            step=max(spot_ref * 0.01, 0.25),
            key=_k("heston_k"),
        )
    with col2:
        maturity_val = st.slider(
            "T (annees)",
            min_value=0.05,
            max_value=2.0,
            value=0.5,
            step=0.05,
            key=_k("heston_T"),
        )
        qty = st.number_input("Quantite", min_value=1, value=1, step=1, key=_k("heston_qty"))

    try:
        price_data = price_european_from_cboe(ticker, float(strike_val), float(maturity_val), option_type)
    except Exception as exc:
        st.error(f"Pricing indisponible : {exc}")
        return

    unit_price = float(price_data.get("price", 0.0))
    K_used = float(price_data.get("K", strike_val))
    T_used = float(price_data.get("T", maturity_val))
    iv_used = price_data.get("iv", float("nan"))
    try:
        S_used = float(price_data.get("S0", spot_ref))
    except Exception:
        S_used = spot_ref
    if not np.isfinite(S_used) or S_used <= 0:
        S_used = spot_ref

    price_cols = st.columns(3)
    price_cols[0].metric("Prix unitaire", f"${unit_price:.4f}")
    price_cols[1].metric("Prix total", f"${unit_price * qty:.4f}")
    iv_label = f"{iv_used:.4f}" if np.isfinite(iv_used) else "N/A"
    price_cols[2].metric("IV utilisee", iv_label)
    st.caption(f"K utilise: {K_used:.4f} | T utilise: {T_used:.4f} an(s)")

    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{ticker} close")
    ax_ts.axhline(S_used, color="crimson", linestyle="-.", label=f"S0 = {S_used:.2f}")
    ax_ts.axhline(K_used, color="gray", linestyle="--", label=f"K = {K_used:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clotures {ticker}")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()
    show_and_close(fig_ts)

    s_grid = np.linspace(0.4 * spot_ref, 1.6 * spot_ref, 200)
    if option_type == "call":
        payoff = np.maximum(s_grid - K_used, 0.0)
    else:
        payoff = np.maximum(K_used - s_grid, 0.0)
    pnl = payoff - unit_price

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff, label="Payoff")
    ax_pay.plot(s_grid, pnl, label="P&L net", color="darkorange")
    ax_pay.axvline(K_used, color="gray", linestyle="--", label=f"K = {K_used:.2f}")
    ax_pay.axvline(S_used, color="crimson", linestyle="-.", label=f"S0 = {S_used:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Payoff {option_type} (CBOE + BS)")
    ax_pay.legend(loc="best")
    show_and_close(fig_pay)

    if st.button("Ajouter au dashboard", key=_k("heston_add_clean"), type="primary"):
        payload = {
            "underlying": ticker,
            "S0": float(S_used),
            "K": float(K_used),
            "T": float(T_used),
            "price": float(unit_price),
            "qty": int(qty),
            "option_type": option_type,
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutee au dashboard (id={oid})")
