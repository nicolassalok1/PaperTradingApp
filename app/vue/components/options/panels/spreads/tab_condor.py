import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import (
    _get_cached_iv_for,
    _k,
    current_spot,
    ensure_close_history,
    get_common_div_yield,
    get_common_maturity_value,
    get_common_sigma_value,
    get_option_context,
    get_rate_for_ttm,
    plt,
    resolve_common_underlying,
    show_and_close,
    view_condor,
)


def render_tab_condor():
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
    common_spot_value = float(current_spot(ctx))

    hist_tkr = resolve_common_underlying()
    spot_anchor = float(S0 if S0 is not None else common_spot_value)
    S0 = spot_anchor
    # --- Fin bootstrap ---
    k_center = st.slider(
        "Strike central (condor)",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("condor_center"),
    )
    inner = st.slider(
        "Écart strikes courts",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * spot_anchor),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("condor_inner"),
    )
    outer_raw = st.slider(
        "Écart strikes ailes",
        min_value=max(0.2, 0.03 * float(common_spot_value)),
        max_value=max(1.5, 0.7 * float(common_spot_value)),
        value=max(0.9, 0.1 * float(common_spot_value)),
        step=0.1,
        key=_k("condor_outer"),
    )
    outer = max(outer_raw, inner + max(0.1, 0.01 * float(common_spot_value)))

    k1 = max(0.01, k_center - outer)
    k2 = k_center - inner
    k3 = k_center + inner
    k4 = k_center + outer
    T_condor = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("condor_T"),
    )
    ivs_condor = [
        _get_cached_iv_for(k1, T_condor, "call"),
        _get_cached_iv_for(k2, T_condor, "call"),
        _get_cached_iv_for(k3, T_condor, "call"),
        _get_cached_iv_for(k4, T_condor, "call"),
    ]
    sigma_k1_condor = (
        float(ivs_condor[0])
        if ivs_condor[0] is not None and np.isfinite(ivs_condor[0]) and ivs_condor[0] > 0
        else float(get_common_sigma_value())
    )
    sigma_k2_condor = (
        float(ivs_condor[1])
        if ivs_condor[1] is not None and np.isfinite(ivs_condor[1]) and ivs_condor[1] > 0
        else float(get_common_sigma_value())
    )
    sigma_k3_condor = (
        float(ivs_condor[2])
        if ivs_condor[2] is not None and np.isfinite(ivs_condor[2]) and ivs_condor[2] > 0
        else float(get_common_sigma_value())
    )
    sigma_k4_condor = (
        float(ivs_condor[3])
        if ivs_condor[3] is not None and np.isfinite(ivs_condor[3]) and ivs_condor[3] > 0
        else float(get_common_sigma_value())
    )
    iv_vals_condor = [v for v in ivs_condor if v is not None and np.isfinite(v) and v > 0]
    sigma_condor = float(np.mean(iv_vals_condor)) if iv_vals_condor else float(get_common_sigma_value())
    if iv_vals_condor:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k1, k2, k3, k4], ivs_condor)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            f"σ utilisées : "
            f"K1 {sigma_k1_condor:.4f} | K2 {sigma_k2_condor:.4f} | "
            f"K3 {sigma_k3_condor:.4f} | K4 {sigma_k4_condor:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_condor(
        float(common_spot_value),
        k1,
        k2,
        k3,
        k4,
        r=float(get_rate_for_ttm(T_condor)),
        q=float(get_common_div_yield()),
        sigma=float(get_common_sigma_value()),
        sigma_k1=float(sigma_k1_condor),
        sigma_k2=float(sigma_k2_condor),
        sigma_k3=float(sigma_k3_condor),
        sigma_k4=float(sigma_k4_condor),
        T=float(T_condor),
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
    ax.set_title("Condor (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("condor_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
