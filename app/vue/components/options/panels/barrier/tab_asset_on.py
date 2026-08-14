import pandas as pd
import streamlit as st
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
    np,
    option_char,
    plt,
    show_and_close,
    view_asset_or_nothing,
)


def render_tab_asset_on():
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
    opt_label_aon, opt_char_aon = _choose_option_select("opt_choice_asset_on", option_char)
    option_label, option_char = opt_label_aon, opt_char_aon
    st.caption(f"Spot actuel ({hist_tkr}) : {spot_anchor:.2f}")
    strike = st.slider(
        "Strike",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("asset_on_k"),
    )
    T_aon = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("asset_on_T"),
    )
    opt_type = "call" if opt_char_aon == "c" else "put"
    iv_aon = _get_cached_iv_for(strike, T_aon, opt_type)
    sigma_aon = (
        float(iv_aon)
        if iv_aon is not None and np.isfinite(iv_aon) and iv_aon > 0
        else float(common_sigma_value)
    )
    if iv_aon is not None and np.isfinite(iv_aon) and iv_aon > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_aon:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_asset_or_nothing(
        float(spot_anchor),
        strike,
        T=float(T_aon),
        r=float(get_rate_for_ttm(T_aon)),
        q=float(d_common),
        sigma=float(sigma_aon),
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
    ax.set_title("Asset-or-nothing (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("asset_on_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
