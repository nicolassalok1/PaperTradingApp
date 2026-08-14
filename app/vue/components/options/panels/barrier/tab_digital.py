import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import (
    _choose_option_select,
    _get_cached_iv_for,
    _k,
    common_maturity_value,
    common_rate_value,
    common_sigma_value,
    common_spot_value,
    current_spot,
    current_ticker,
    d_common,
    ensure_close_history,
    get_option_context,
    get_rate_for_ttm,
    option_char,
    plt,
    show_and_close,
    view_digital,
)


def render_tab_digital():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    # Legacy defaults to avoid NameError
    option_char = st.session_state.get("option_char", "c")
    common_spot_value = float(current_spot(ctx))
    spot_anchor = float(common_spot_value)
    common_maturity_value = float(st.session_state.get("common_maturity_value", 1.0))
    common_rate_value = float(st.session_state.get("common_rate_value", 0.01))
    common_sigma_value = float(st.session_state.get("common_sigma_value", 0.2))
    d_common = float(st.session_state.get("d_common", 0.0))  # dividend yield

    hist_tkr = current_ticker(ctx) or ticker
    opt_label_dig, opt_char_dig = _choose_option_select("opt_choice_digital", option_char)
    option_label, option_char = opt_label_dig, opt_char_dig
    st.caption(f"Spot actuel ({hist_tkr}) : {spot_anchor:.2f}")
    strike = st.slider(
        "Strike",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("digital_k"),
    )
    T_dig = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("digital_T"),
    )
    opt_type = "call" if opt_char_dig == "c" else "put"
    iv_dig = _get_cached_iv_for(strike, T_dig, opt_type)
    sigma_dig = (
        float(iv_dig)
        if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0
        else float(common_sigma_value)
    )
    if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_dig:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_digital(
        float(spot_anchor),
        strike,
        T=float(T_dig),
        r=float(get_rate_for_ttm(T_dig)),
        q=float(d_common),
        sigma=float(sigma_dig),
        option_type=opt_type,
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, payoff_grid, label="Payoff brut")
    ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax.axvline(float(spot_anchor), color="crimson", linestyle="-.", label=f"S_0 = {float(spot_anchor):.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    ax.set_title("Digital (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("digital_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
