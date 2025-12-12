import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_iron_condor():
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
        "Strike central (iron condor)",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("iron_condor_center"),
    )
    inner = st.slider(
        "Écart strikes courts",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * spot_anchor),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("iron_condor_inner"),
    )
    outer_raw = st.slider(
        "Écart strikes ailes",
        min_value=max(0.2, 0.03 * float(common_spot_value)),
        max_value=max(1.5, 0.7 * float(common_spot_value)),
        value=max(0.9, 0.1 * float(common_spot_value)),
        step=0.1,
        key=_k("iron_condor_outer"),
    )
    outer = max(outer_raw, inner + max(0.1, 0.01 * float(common_spot_value)))

    k_put_long = max(0.01, k_center - outer)
    k_put_short = k_center - inner
    k_call_short = k_center + inner
    k_call_long = k_center + outer
    T_iron_condor = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("iron_condor_T"),
    )
    ivs_ic = [
        _get_cached_iv_for(k_put_long, T_iron_condor, "put"),
        _get_cached_iv_for(k_put_short, T_iron_condor, "put"),
        _get_cached_iv_for(k_call_short, T_iron_condor, "call"),
        _get_cached_iv_for(k_call_long, T_iron_condor, "call"),
    ]
    sigma_put_long_ic = (
        float(ivs_ic[0])
        if ivs_ic[0] is not None and np.isfinite(ivs_ic[0]) and ivs_ic[0] > 0
        else float(common_sigma_value)
    )
    sigma_put_short_ic = (
        float(ivs_ic[1])
        if ivs_ic[1] is not None and np.isfinite(ivs_ic[1]) and ivs_ic[1] > 0
        else float(common_sigma_value)
    )
    sigma_call_short_ic = (
        float(ivs_ic[2])
        if ivs_ic[2] is not None and np.isfinite(ivs_ic[2]) and ivs_ic[2] > 0
        else float(common_sigma_value)
    )
    sigma_call_long_ic = (
        float(ivs_ic[3])
        if ivs_ic[3] is not None and np.isfinite(ivs_ic[3]) and ivs_ic[3] > 0
        else float(common_sigma_value)
    )
    iv_vals_ic = [v for v in ivs_ic if v is not None and np.isfinite(v) and v > 0]
    if iv_vals_ic:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k_put_long, k_put_short, k_call_short, k_call_long], ivs_ic)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            "σ utilisées : "
            f"put long {sigma_put_long_ic:.4f} | put short {sigma_put_short_ic:.4f} | "
            f"call short {sigma_call_short_ic:.4f} | call long {sigma_call_long_ic:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_iron_condor(
        float(common_spot_value),
        k_put_long,
        k_put_short,
        k_call_short,
        k_call_long,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_put_long=float(sigma_put_long_ic),
        sigma_put_short=float(sigma_put_short_ic),
        sigma_call_short=float(sigma_call_short_ic),
        sigma_call_long=float(sigma_call_long_ic),
        T=float(T_iron_condor),
    )
    premium = float(view_dyn.get("premium", 0.0))
    price_display = abs(premium)

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
    ax.set_title("Iron Condor (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("iron_condor_pre_price")] = premium
    price = float(price_display)
    # --- Bouton Add-to-Dashboard Clean ---
