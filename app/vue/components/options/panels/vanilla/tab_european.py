import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.vue.components.options.controller_bridge import (
    _choose_option_select,
    _get_cached_iv_for,
    _k,
    build_close_with_strike_fig,
    current_spot,
    current_ticker,
    ensure_close_history,
    get_common_div_yield,
    get_common_maturity_value,
    get_common_sigma_value,
    get_option_context,
    get_rate_for_ttm,
    option_char,
    render_figures_grid,
    view_european,
)


def render_tab_european():
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
    hist_tkr = current_ticker(ctx) or ticker
    # --------------------------------
    spot_base = current_spot(ctx)
    S0 = float(spot_base)

    _, opt_char = _choose_option_select("opt_choice_european", option_char)
    st.caption(f"Spot actuel ({hist_tkr}) : {spot_base:.2f}")
    strike = st.slider(
        "Strike",
        min_value=0.5 * spot_base,
        max_value=1.5 * spot_base,
        value=spot_base,
        step=0.5,
        key=_k("european_k"),
    )
    T_eur = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("european_T"),
    )
    iv_eur = _get_cached_iv_for(
        strike, T_eur, "call" if opt_char == "c" else "put"
    )
    sigma_eur = (
        float(iv_eur)
        if iv_eur is not None and np.isfinite(iv_eur) and iv_eur > 0
        else float(get_common_sigma_value())
    )
    if iv_eur is not None and np.isfinite(iv_eur) and iv_eur > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_eur:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_european(
        float(spot_base),
        strike,
        option_type="call" if opt_char == "c" else "put",
        r=float(get_rate_for_ttm(T_eur)),
        q=float(get_common_div_yield()),
        sigma=float(sigma_eur),
        T=float(T_eur),
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff brut")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(
        float(strike), color="gray", linestyle="--", label=f"K = {float(strike):.2f}"
    )
    ax_pay.axvline(
        float(spot_base),
        color="crimson",
        linestyle="-.",
        label=f"S_0 = {float(spot_base):.2f}",
    )
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title("Vanilla européenne (payoff & P&L)")
    ax_pay.legend(loc="lower right")

    close_fig = build_close_with_strike_fig(close_series, hist_tkr, strike)
    render_figures_grid([close_fig, fig_pay])

    st.session_state[_k("european_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
