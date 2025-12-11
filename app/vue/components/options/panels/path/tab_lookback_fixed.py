import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_lookback_fixed():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series([S0], index=pd.Index([pd.Timestamp.today()]))
    # --------------------------------
    hist_tkr = ticker
    st.subheader("Lookback fixed – vue Notebook")
    col1, col2 = st.columns(2)
    with col1:
        option_type_lbf = st.selectbox("Type", ["call", "put"], key=_k("lbf_type"))
        min_lbf = st.slider(
            "Min path",
            min_value=0.8 * S0,
            max_value=1.0 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lbf_min"),
        )
        max_lbf = st.slider(
            "Max path",
            min_value=1.0 * S0,
            max_value=1.2 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lbf_max"),
        )
    with col2:
        strike_lbf = st.slider(
            "Strike",
            min_value=0.8 * S0,
            max_value=1.2 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lbf_k"),
        )
        span_lbf = st.slider(
            "Span payoff (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=_k("lbf_span"),
        )
        T_lbf = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=float(common_maturity_value),
            step=0.05,
            key=_k("lbf_T"),
        )

    view_dyn = view_lookback_fixed(
        S0,
        min_lbf,
        max_lbf,
        strike_lbf,
        option_type=option_type_lbf,
        span=span_lbf,
        T=float(T_lbf),
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(S0, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
    ax_ts.axhline(min_lbf, color="teal", linestyle=":", label=f"Min = {min_lbf:.2f}")
    ax_ts.axhline(max_lbf, color="gray", linestyle="--", label=f"Max = {max_lbf:.2f}")
    ax_ts.axhline(strike_lbf, color="firebrick", linestyle="-.", label=f"K = {strike_lbf:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {hist_tkr} (Lookback fixed)")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(S0, color="crimson", linestyle="-.", label=f"S0 = {S0:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Lookback fixed ({option_type_lbf})")
    render_figures_grid([fig_ts, fig_pay])

    price = float(premium)
    st.markdown("### Ajouter au dashboard")
    st.metric("Prix calculé", f"${price:.6f}")
    underlying = (
        (
            st.session_state.get("heston_cboe_ticker")
            or st.session_state.get("tkr_common")
            or st.session_state.get("common_underlying")
            or st.session_state.get("ticker_default")
            or ""
        )
        .strip()
        .upper()
    )
    st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
    today = datetime.date.today()
    expiration_dt = today + datetime.timedelta(days=int((T_lbf or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("lbf_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("lbf_side"))

    if st.button("Ajouter au dashboard", key=_k("lbf_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_type_lbf,
            "product_type": "Lookback fixed",
            "type": "Lookback fixed",
            "strike": float(strike_lbf),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(S0),
            "maturity_years": float(T_lbf),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "min_path": float(min_lbf),
                "max_path": float(max_lbf),
                "span": float(span_lbf),
                "T": float(T_lbf),
                "spot_at_pricing": float(S0),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
