import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import (
    _get_cached_iv_for,
    _k,
    current_spot,
    current_ticker,
    ensure_close_history,
    get_common_div_yield,
    get_common_maturity_value,
    get_common_sigma_value,
    get_option_context,
    get_rate_for_ttm,
    mark_full_width,
    plt,
    render_figures_grid,
    view_asian_geom,
)


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
    hist_tkr = current_ticker(ctx) or ticker
    S0 = float(current_spot(ctx))
    st.subheader("Asian géométrique – vue Notebook")
    avg_close = float(close_series.mean()) if close_series is not None else S0
    col1, col2 = st.columns(2)
    with col1:
        option_type_ag = st.selectbox("Type", ["call", "put"], key=_k("asian_geo_type"))
        st.caption(f"Spot actuel ({hist_tkr}) : {S0:.2f}")
        strike_ag = st.slider(
            "Strike",
            min_value=0.6 * S0,
            max_value=1.4 * S0,
            value=float(S0),
            step=0.5,
            key=_k("asian_geo_k"),
        )
        avg_ag = st.slider(
            "Moyenne (ref)",
            min_value=0.5 * S0,
            max_value=1.5 * S0,
            value=float(S0),
            step=0.5,
            key=_k("asian_geo_avg"),
        )
    with col2:
        T_ag = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=float(get_common_maturity_value()),
            step=0.05,
            key=_k("asian_geo_T"),
        )
        r_ag = float(get_rate_for_ttm(T_ag))
    iv_ag = _get_cached_iv_for(strike_ag, T_ag, option_type_ag)
    sigma_ag = (
        float(iv_ag)
        if iv_ag is not None and np.isfinite(iv_ag) and iv_ag > 0
        else float(get_common_sigma_value())
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
        q=float(get_common_div_yield()),
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
    mark_full_width(fig_ts)

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
    st.metric("Prix de l'option", f"${premium:.6f}")
