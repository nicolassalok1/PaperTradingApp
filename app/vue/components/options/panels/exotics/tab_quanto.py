import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_quanto():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    hist_tkr = current_ticker(ctx) or ticker

    # --- Contexte exotiques ---
    spot_base = float(current_spot(ctx))
    S0 = spot_base
    # -----------------------------
    opt_label_quanto, opt_char_quanto = _choose_option_select("opt_choice_quanto", option_char)
    option_label = opt_label_quanto
    option_char_selected = opt_char_quanto
    st.caption(f"Spot actuel ({hist_tkr}) : {spot_base:.2f}")
    strike = st.slider(
        "Strike",
        min_value=0.5 * spot_base,
        max_value=1.5 * spot_base,
        value=spot_base,
        step=0.5,
        key=_k("quanto_k"),
    )
    fx_rate = st.slider(
        "Taux FX (payout)",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.05,
        key=_k("quanto_fx"),
    )
    opt_type = "call" if option_char_selected.lower() == "c" else "put"
    T_quanto = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(get_common_maturity_value()),
        step=0.05,
        key=_k("quanto_T"),
    )
    iv_quanto = _get_cached_iv_for(strike, T_quanto, opt_type)
    sigma_quanto = (
        float(iv_quanto)
        if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0
        else float(get_common_sigma_value())
    )
    if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_quanto:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    sigma_fx_quanto = st.slider(
        "σ FX (volatilité du taux de change)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        key=_k("quanto_sigma_fx"),
        help="À 0, le quanto se réduit à une vanille convertie : la corrélation n'a aucun effet.",
    )
    rho_quanto = st.slider(
        "ρ (corrélation spot / FX)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key=_k("quanto_rho"),
        help="Ajustement quanto du drift : mu = r_f − q − ρ·σ·σ_FX.",
    )
    view_dyn = view_quanto(
        float(spot_base),
        strike,
        fx_rate=fx_rate,
        r=float(get_rate_for_ttm(T_quanto)),
        q=float(get_common_div_yield()),
        sigma=float(sigma_quanto),
        T=float(T_quanto),
        option_type=opt_type,
        rho=float(rho_quanto),
        sigma_fx=float(sigma_fx_quanto),
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, payoff_grid, label="Payoff brut")
    ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax.axvline(float(spot_base), color="crimson", linestyle="-.", label=f"S_0 = {float(spot_base):.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    ax.set_title("Quanto (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("quanto_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
