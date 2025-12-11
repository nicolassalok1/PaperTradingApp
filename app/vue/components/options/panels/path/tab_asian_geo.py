import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_asian_geo():
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
    st.subheader("Asian géométrique – vue Notebook")
    avg_close = float(close_series.mean()) if close_series is not None else S0
    col1, col2 = st.columns(2)
    with col1:
        option_type_ag = st.selectbox("Type", ["call", "put"], key=_k("asian_geo_type"))
        strike_ag = st.slider(
            "Strike",
            min_value=0.6 * S0,
            max_value=1.4 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("asian_geo_k"),
        )
        avg_ag = st.slider(
            "Moyenne (ref)",
            min_value=0.5 * S0,
            max_value=1.5 * S0,
            value=float(floor_n(S0, 0)),
            step=0.5,
            key=_k("asian_geo_avg"),
        )
    with col2:
        r_ag = float(common_rate_value)
        T_ag = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=common_maturity_value,
            step=0.05,
            key=_k("asian_geo_T"),
        )
    iv_ag = _get_cached_iv_for(strike_ag, T_ag, option_type_ag)
    sigma_ag = (
        float(iv_ag)
        if iv_ag is not None and np.isfinite(iv_ag) and iv_ag > 0
        else float(common_sigma_value)
    )
    if iv_ag is not None and np.isfinite(iv_ag) and iv_ag > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_ag:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_asian_geom(
        S0,
        strike_ag,
        avg_ag,
        option_type=option_type_ag,
        r=r_ag,
        q=0.0,
        sigma=sigma_ag,
        T=T_ag,
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(S0, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
    ax_ts.axhline(avg_ag, color="purple", linestyle=":", label=f"Moyenne = {avg_ag:.2f}")
    ax_ts.axhline(strike_ag, color="gray", linestyle="--", label=f"K = {strike_ag:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {hist_tkr} (Asian géo)")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(strike_ag, color="gray", linestyle="--", label=f"K = {strike_ag:.2f}")
    ax_pay.axvline(S0, color="crimson", linestyle="-.", label=f"S0 = {S0:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Asian géométrique ({option_type_ag})")
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
    expiration_dt = today + datetime.timedelta(days=int((T_ag or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("asian_geo_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("asian_geo_side"))

    if st.button("Ajouter au dashboard", key=_k("asian_geo_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_type_ag,
            "product_type": "Asian géométrique",
            "type": "Asian géométrique",
            "strike": float(strike_ag),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(S0),
            "maturity_years": float(T_ag),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "avg_ref": float(avg_ag),
                "sigma_used": float(sigma_ag),
                "r": float(r_ag),
                "spot_at_pricing": float(S0),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("asian_geo_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
