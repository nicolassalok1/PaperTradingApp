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
    view_put_spread,
)


def render_tab_put_spread():
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
        "Strike put long",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("put_spread_k_long"),
    )
    k_short_raw = st.slider(
        "Strike put short",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("put_spread_k_short"),
    )
    k_long = max(k_long_raw, k_short_raw)
    k_short = min(k_long_raw, k_short_raw)
    T_put_spread = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("put_spread_T"),
    )
    iv_long_p = _get_cached_iv_for(k_long, T_put_spread, "put")
    iv_short_p = _get_cached_iv_for(k_short, T_put_spread, "put")
    sigma_long_ps = (
        float(iv_long_p)
        if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0
        else float(get_common_sigma_value())
    )
    sigma_short_ps = (
        float(iv_short_p)
        if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0
        else float(get_common_sigma_value())
    )
    if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_long_p, iv_short_p)):
        iv_long_txt = (
            f"{iv_long_p:.4f}"
            if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0
            else "n/a"
        )
        iv_short_txt = (
            f"{iv_short_p:.4f}"
            if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0
            else "n/a"
        )
        st.caption(f"IV récupérées (cache) ≈ long {iv_long_txt} | short {iv_short_txt}")
        st.caption(f"σ utilisées : long {sigma_long_ps:.4f} | short {sigma_short_ps:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_put_spread(
        float(common_spot_value),
        k_long,
        k_short,
        r=float(get_rate_for_ttm(T_put_spread)),
        q=float(get_common_div_yield()),
        sigma=float(get_common_sigma_value()),
        sigma_long=float(sigma_long_ps),
        sigma_short=float(sigma_short_ps),
        T=float(T_put_spread),
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
    ax.set_title("Put spread (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("put_spread_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
