import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_forward_start():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series([S0], index=pd.Index([pd.Timestamp.today()]))
    # --------------------------------
    hist_tkr = current_ticker(ctx) or ticker
    S0 = float(current_spot(ctx))
    opt_char_base = option_char if option_char else "c"
    _, fs_char = _choose_option_select("opt_choice_forward_start", opt_char_base)
    option_char = fs_char
    st.caption(f"Spot actuel ({hist_tkr}) : {S0:.2f}")
    spot_start = st.slider(
        "Spot de départ (S_start)",
        min_value=0.5 * float(S0),
        max_value=1.5 * float(S0),
        value=float(S0),
        step=0.5,
        key=_k("forward_start_s_start"),
    )
    strike_fs = st.slider(
        "Strike (K = m × S_start)",
        min_value=0.8 * float(S0),
        max_value=1.2 * float(S0),
        value=float(S0),
        step=0.5,
        key=_k("forward_start_k"),
    )
    m_factor = float(strike_fs / spot_start) if spot_start else 1.0
    T_fs = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("forward_start_T"),
    )
    opt_type = "call" if option_char.lower() == "c" else "put"
    strike_forward = m_factor * spot_start
    figs = []
    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
    ax_ts.axhline(spot_start, color="gray", linestyle="--", label=f"S_start = {spot_start:.2f}")
    ax_ts.axhline(
        strike_forward,
        color="firebrick",
        linestyle=":",
        label=f"K = m*S_start = {strike_forward:.2f}",
    )
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {hist_tkr} (Forward-start)")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()
    figs.append(fig_ts)
    view_dyn = view_forward_start(
        S0,
        spot_start,
        m=m_factor,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        T=float(T_fs),
        option_type=opt_type,
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, payoff_grid, label="Payoff brut")
    ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax.axvline(float(S0), color="crimson", linestyle="-.", label=f"S_0 = {float(S0):.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    ax.set_title("Forward-start (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    figs.append(fig)
    render_figures_grid(figs)

    st.session_state[_k("forward_start_pre_price")] = premium
    price = float(premium)
    st.metric("Prix de l'option", f"${price:.6f}")
