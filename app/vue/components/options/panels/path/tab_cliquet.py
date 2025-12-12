import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_cliquet():
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
    hist_tkr = ticker
    # FIX: runtime crash detected by crawler (option_char unbound)
    option_char = option_char if "option_char" in locals() else "c"
    clq_label, clq_char = _choose_option_select("opt_choice_cliquet_tab", option_char)
    option_label, option_char = clq_label, clq_char
    st.subheader("Cliquet / Ratchet – vue Notebook")
    k_cliquet_anchor = float(S0 if S0 is not None else common_spot_value)
    strike_clq = st.slider(
        "Strike / niveau de référence",
        min_value=0.6 * k_cliquet_anchor,
        max_value=1.4 * k_cliquet_anchor,
        value=float(k_cliquet_anchor),
        step=0.5,
        key=_k("cliquet_k"),
    )
    floor_val = st.slider(
        "Floor", min_value=-0.5, max_value=0.5, value=0.0, step=0.01, key=_k("cliquet_floor")
    )
    cap_val = st.slider(
        "Cap", min_value=0.0, max_value=0.5, value=0.1, step=0.01, key=_k("cliquet_cap")
    )
    T_clq = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("cliquet_T"),
    )

    view_dyn = view_cliquet(
        S0,
        floor=floor_val,
        cap=cap_val,
        T=float(T_clq),
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        n_periods=12,
        n_paths=4000,
        k_ref=float(strike_clq),
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(S0, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
    ax_ts.axhline(S0, color="gray", linestyle="--", label=f"S0 = {S0:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {hist_tkr} (Cliquet)")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff cliquet")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(S0, color="crimson", linestyle="-.", label=f"S0 = {S0:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title("Cliquet / Ratchet (approx)")
    render_figures_grid([fig_ts, fig_pay])
