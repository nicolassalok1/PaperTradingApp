import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import (
    _get_cached_iv_for,
    _k,
    common_spot_value,
    ensure_close_history,
    get_common_div_yield,
    get_common_maturity_value,
    get_common_sigma_value,
    get_option_context,
    get_rate_for_ttm,
    plt,
    resolve_common_underlying,
    show_and_close,
    view_call_spread,
)


def render_tab_call_spread():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    if not ensure_close_history(ctx):
        return
    hist_tkr = ticker

    # --- Bootstrap du contexte Spreads/Wings ---
    common_spot_value = float(st.session_state.get("common_spot_value", S0 if S0 is not None else 100.0))

    hist_tkr = resolve_common_underlying()
    spot_anchor = float(S0 if S0 is not None else common_spot_value)
    S0 = spot_anchor
    # --- Fin bootstrap ---
    k_long_raw = st.slider(
        "Strike call long",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("call_spread_k_long"),
    )
    k_short_raw = st.slider(
        "Strike call short",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("call_spread_k_short"),
    )
    k_long = min(k_long_raw, k_short_raw)
    k_short = max(k_long_raw, k_short_raw)
    T_call_spread = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("call_spread_T"),
    )
    iv_long = _get_cached_iv_for(k_long, T_call_spread, "call")
    iv_short = _get_cached_iv_for(k_short, T_call_spread, "call")
    sigma_long_cs = (
        float(iv_long)
        if iv_long is not None and np.isfinite(iv_long) and iv_long > 0
        else float(get_common_sigma_value())
    )
    sigma_short_cs = (
        float(iv_short)
        if iv_short is not None and np.isfinite(iv_short) and iv_short > 0
        else float(get_common_sigma_value())
    )
    if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_long, iv_short)):
        iv_long_txt = (
            f"{iv_long:.4f}"
            if iv_long is not None and np.isfinite(iv_long) and iv_long > 0
            else "n/a"
        )
        iv_short_txt = (
            f"{iv_short:.4f}"
            if iv_short is not None and np.isfinite(iv_short) and iv_short > 0
            else "n/a"
        )
        st.caption(f"IV récupérées (cache) ≈ long {iv_long_txt} | short {iv_short_txt}")
        st.caption(f"σ utilisées : long {sigma_long_cs:.4f} | short {sigma_short_cs:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_call_spread(
        float(common_spot_value),
        k_long,
        k_short,
        r=float(get_rate_for_ttm(T_call_spread)),
        q=float(get_common_div_yield()),
        sigma=float(get_common_sigma_value()),
        sigma_long=float(sigma_long_cs),
        sigma_short=float(sigma_short_cs),
        T=float(T_call_spread),
    )
    premium = float(view_dyn.get("premium", 0.0))

    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, payoff_grid, label="Payoff brut")
    ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax.axvline(
        float(common_spot_value),
        color="crimson",
        linestyle="-.",
        label=f"S_0 = {float(common_spot_value):.2f}",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    ax.set_title("Call spread (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("call_spread_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
