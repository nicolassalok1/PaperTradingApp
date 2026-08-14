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
    view_butterfly,
)


def render_tab_butterfly():
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
    k_center = st.slider(
        "Strike central",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("butterfly_k_center"),
    )
    wing = st.slider(
        "Écart ailes",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * spot_anchor),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("butterfly_wing"),
    )
    k1 = max(0.01, k_center - wing)
    k2 = k_center
    k3 = k_center + wing
    T_bfly = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("butterfly_T"),
    )
    ivs_bfly = [
        _get_cached_iv_for(k1, T_bfly, "call"),
        _get_cached_iv_for(k2, T_bfly, "call"),
        _get_cached_iv_for(k3, T_bfly, "call"),
    ]
    sigma_k1_bfly = (
        float(ivs_bfly[0])
        if ivs_bfly[0] is not None and np.isfinite(ivs_bfly[0]) and ivs_bfly[0] > 0
        else float(get_common_sigma_value())
    )
    sigma_k2_bfly = (
        float(ivs_bfly[1])
        if ivs_bfly[1] is not None and np.isfinite(ivs_bfly[1]) and ivs_bfly[1] > 0
        else float(get_common_sigma_value())
    )
    sigma_k3_bfly = (
        float(ivs_bfly[2])
        if ivs_bfly[2] is not None and np.isfinite(ivs_bfly[2]) and ivs_bfly[2] > 0
        else float(get_common_sigma_value())
    )
    iv_vals_bfly = [v for v in ivs_bfly if v is not None and np.isfinite(v) and v > 0]
    sigma_bfly = float(np.mean(iv_vals_bfly)) if iv_vals_bfly else float(get_common_sigma_value())
    if iv_vals_bfly:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k1, k2, k3], ivs_bfly)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            f"σ utilisées : K1 {sigma_k1_bfly:.4f} | K2 {sigma_k2_bfly:.4f} | K3 {sigma_k3_bfly:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_butterfly(
        float(common_spot_value),
        k1,
        k2,
        k3,
        r=float(get_rate_for_ttm(T_bfly)),
        q=float(get_common_div_yield()),
        sigma=float(get_common_sigma_value()),
        sigma_k1=float(sigma_k1_bfly),
        sigma_k2=float(sigma_k2_bfly),
        sigma_k3=float(sigma_k3_bfly),
        T=float(T_bfly),
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
    ax.set_title("Butterfly (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("butterfly_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
