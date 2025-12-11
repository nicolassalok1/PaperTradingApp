import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_lookback():
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
    st.subheader("Lookback floating – vue Notebook")
    col1, col2 = st.columns(2)
    with col1:
        option_type_lb = st.selectbox("Type", ["call", "put"], key=_k("lb_type"))
        min_lb = st.slider(
            "Min path",
            min_value=0.8 * S0,
            max_value=1.0 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lb_min"),
        )
        max_lb = st.slider(
            "Max path",
            min_value=1.0 * S0,
            max_value=1.2 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lb_max"),
        )
        strike_lb = st.slider(
            "Strike (référence)",
            min_value=0.8 * S0,
            max_value=1.2 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("lb_k"),
        )
    with col2:
        span_lb = st.slider(
            "Span payoff (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=_k("lb_span")
        )
        T_lb = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=float(common_maturity_value),
            step=0.05,
            key=_k("lb_T"),
        )

    view_dyn = view_lookback(
        S0,
        min_lb,
        max_lb,
        option_type=option_type_lb,
        span=span_lb,
        k_ref=float(strike_lb),
        T=float(T_lb),
    )
    premium = float(view_dyn.get("premium", 0.0))
    price_display = abs(premium)
    price_display = abs(premium)
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(S0, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
    ax_ts.axhline(min_lb, color="teal", linestyle=":", label=f"Min = {min_lb:.2f}")
    ax_ts.axhline(max_lb, color="gray", linestyle="--", label=f"Max = {max_lb:.2f}")
    ax_ts.axhline(S0, color="firebrick", linestyle="-.", label=f"S0 = {S0:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {hist_tkr} (Lookback)")
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
    ax_pay.set_title(f"Lookback floating ({option_type_lb})")
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
    expiration_dt = today + datetime.timedelta(days=int((common_maturity_value or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("lb_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("lb_side"))

    if st.button("Ajouter au dashboard", key=_k("lb_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_type_lb,
            "product_type": "Lookback floating",
            "type": "Lookback floating",
            "strike": float(strike_lb),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(S0),
            "maturity_years": float(T_lb),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "min_path": float(min_lb),
                "max_path": float(max_lb),
                "strike_ref": float(strike_lb),
                "span": float(span_lb),
                "T": float(T_lb),
                "spot_at_pricing": float(S0),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
