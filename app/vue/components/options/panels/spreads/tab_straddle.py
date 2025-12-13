import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_straddle():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    if not ensure_close_history(ctx):
        return
    hist_tkr = current_ticker(ctx) or ticker

    # --- Bootstrap du contexte Spreads/Wings ---
    spot_base = float(current_spot(ctx))
    spot_anchor = float(spot_base)
    S0 = float(spot_base)
    # --- Fin bootstrap ---
    mc_label_to_model = {
        "Black–Scholes (MC)": "bs",
        "Stoch vol (MC)": "rheston",
        "rBergomi (MC)": "rbergomi",
        "SABR (MC)": "sabr",
        "Volterra (MC)": "volterra",
    }
    mc_choice = st.selectbox(
        "Modèle de pricing Monte Carlo",
        options=list(mc_label_to_model.keys()),
        index=0,
        key=f"mc_model_{_k('straddle')}",
        help="Black–Scholes (MC) implémenté, les autres seront ajoutés progressivement.",
    )
    mc_model = mc_label_to_model.get(mc_choice, list(mc_label_to_model.values())[0])
    strike_slider = st.slider(
        "Strike",
        min_value=0.5 * spot_anchor,
        max_value=1.5 * spot_anchor,
        value=spot_anchor,
        step=0.5,
        key=_k("straddle_k"),
    )
    T_straddle = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("straddle_T"),
    )
    iv_straddle = _get_cached_iv_for(strike_slider, T_straddle, "call")
    iv_straddle_put = _get_cached_iv_for(strike_slider, T_straddle, "put")
    sigma_call_straddle = (
        float(iv_straddle)
        if iv_straddle is not None and np.isfinite(iv_straddle) and iv_straddle > 0
        else float(get_common_sigma_value())
    )
    sigma_put_straddle = (
        float(iv_straddle_put)
        if iv_straddle_put is not None and np.isfinite(iv_straddle_put) and iv_straddle_put > 0
        else float(get_common_sigma_value())
    )
    if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_straddle, iv_straddle_put)):
        iv_call_txt = (
            f"{iv_straddle:.4f}"
            if iv_straddle is not None and np.isfinite(iv_straddle) and iv_straddle > 0
            else "n/a"
        )
        iv_put_txt = (
            f"{iv_straddle_put:.4f}"
            if iv_straddle_put is not None and np.isfinite(iv_straddle_put) and iv_straddle_put > 0
            else "n/a"
        )
        st.caption(f"IV récupérées (cache) ≈ call {iv_call_txt} | put {iv_put_txt}")
        st.caption(f"σ utilisées : call {sigma_call_straddle:.4f} | put {sigma_put_straddle:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_straddle(
        float(spot_base),
        strike_slider,
        r=float(get_rate_for_ttm(T_straddle)),
        q=float(get_common_div_yield()),
        sigma=float(get_common_sigma_value()),
        sigma_call=float(sigma_call_straddle),
        sigma_put=float(sigma_put_straddle),
        T=float(T_straddle),
    )
    premium = float(view_dyn.get("premium", 0.0))

    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    pnl_at_s0 = float(payoff_grid[np.searchsorted(s_grid, float(spot_base), side="left") - 1] - premium)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, payoff_grid, label="Payoff brut")
    ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax.axvline(float(spot_base), color="crimson", linestyle="-.", label=f"S_0 = {float(spot_base):.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    ax.set_title("Straddle (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    # Stocke la prime et affiche directement le formulaire d'ajout (équivalent dropdown)
    st.session_state[_k("straddle_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
