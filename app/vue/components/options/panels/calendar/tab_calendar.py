import contextlib
import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_calendar():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    hist_tkr = ticker

    # --- Bootstrap du contexte Calendar/Diagonal ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))

    S0 = float(common_spot_value)
    hist_tkr, close_series = load_shared_close_series(S0)
    # --- Fin bootstrap ---
    st.subheader("Calendar spread – vue Notebook")
    s0_ref = float(common_spot_value)

    strike_anchor_cal = float(S0 if S0 is not None else common_spot_value)
    col1, col2 = st.columns(2)
    col1_ctx = col1 if hasattr(col1, "__enter__") else contextlib.nullcontext()
    col2_ctx = col2 if hasattr(col2, "__enter__") else contextlib.nullcontext()
    with col1_ctx:
        option_type_cal = st.selectbox("Type", ["call", "put"], key=_k("calendar_type"))
        strike_cal = st.slider(
            "Strike",
            min_value=0.6 * strike_anchor_cal,
            max_value=1.4 * strike_anchor_cal,
            value=float(strike_anchor_cal),
            step=0.5,
            key=_k("calendar_strike"),
        )
        t_short = st.slider(
            "T court (années)",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key=_k("calendar_t_short"),
        )
        t_long_raw = st.slider(
            "T long (années)",
            min_value=0.1,
            max_value=2.0,
            value=0.75,
            step=0.05,
            key=_k("calendar_t_long"),
        )
        t_long = max(t_long_raw, t_short + 0.01)
        if t_long != t_long_raw:
            st.caption(f"T long ajusté à {t_long:.2f} pour rester après T court.")
    with col2_ctx:
        span_cal = st.slider(
            "Span payoff (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=_k("calendar_span"),
        )

    iv_cal = _get_cached_iv_for(strike_cal, t_long, option_type_cal)
    sigma_cal = (
        float(iv_cal)
        if iv_cal is not None and np.isfinite(iv_cal) and iv_cal > 0
        else float(get_common_sigma_value())
    )
    if iv_cal is not None and np.isfinite(iv_cal) and iv_cal > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_cal:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    r_cal = float(get_rate_for_ttm(t_long))

    view_dyn = view_calendar_spread(
        s0_ref,
        strike_cal,
        T_short=t_short,
        T_long=t_long,
        option_type=option_type_cal,
        r=r_cal,
        q=float(get_common_div_yield()),
        sigma=sigma_cal,
        span=span_cal,
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    forward_start_date = datetime.date.today() + datetime.timedelta(days=int(t_short * 365))
    figs = []
    if close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
        ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
        ax_ts.axhline(strike_cal, color="gray", linestyle="--", label=f"K = {strike_cal:.2f}")
        ax_ts.axvline(
            forward_start_date,
            color="purple",
            linestyle=":",
            label=f"Start ~ {forward_start_date.isoformat()}",
        )
        ax_ts.set_ylabel("Prix")
        ax_ts.set_title(f"Clôtures {hist_tkr} (strike / forward start)")
        ax_ts.legend(loc="best")
        fig_ts.autofmt_xdate()
        mark_full_width(fig_ts)
        figs.append(fig_ts)
    else:
        st.info(
            "Ajoute un ticker commun en haut de l'onglet Options pour tracer l'historique 1 an."
        )

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(strike_cal, color="gray", linestyle="--", label=f"K = {strike_cal:.2f}")
    ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Calendar spread ({option_type_cal})")
    figs.append(fig_pay)
    render_figures_grid(figs)
    st.metric("Prix de l'option", f"${premium:.6f}")
