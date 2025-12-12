import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_chooser():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series([S0], index=pd.Index([pd.Timestamp.today()]))
    hist_tkr = resolve_common_underlying() or ticker

    # --- Contexte exotiques ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))
    spot_anchor = float(S0 if S0 is not None else common_spot_value)
    S0 = float(common_spot_value)
    # -----------------------------
    _, opt_char_chooser = _choose_option_select("opt_choice_chooser", option_char)
    option_char_selected = opt_char_chooser
    strike = st.slider(
        "Strike",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("chooser_k"),
    )
    T_chooser = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("chooser_T"),
    )
    iv_chooser = _get_cached_iv_for(
        strike, T_chooser, "call" if option_char_selected == "c" else "put"
    )
    sigma_chooser = (
        float(iv_chooser)
        if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0
        else float(common_sigma_value)
    )
    if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_chooser:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_chooser(
        float(common_spot_value),
        strike,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(sigma_chooser),
        T=float(T_chooser),
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
        float(common_spot_value),
        color="crimson",
        linestyle="-.",
        label=f"S_0 = {float(common_spot_value):.2f}",
    )
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title("Chooser (payoff & P&L avec prime BS)")
    ax_pay.legend(loc="lower right")

    close_fig = build_close_with_strike_fig(close_series, hist_tkr, strike)
    render_figures_grid([close_fig, fig_pay])

    st.session_state[_k("chooser_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
