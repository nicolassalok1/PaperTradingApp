import contextlib
import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_diagonal():
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
    st.subheader("Diagonal spread – vue Notebook")
    s0_ref = float(common_spot_value)

    strike_anchor_diag = float(S0 if S0 is not None else common_spot_value)
    col1, col2 = st.columns(2)
    col1_ctx = col1 if hasattr(col1, "__enter__") else contextlib.nullcontext()
    col2_ctx = col2 if hasattr(col2, "__enter__") else contextlib.nullcontext()
    with col1_ctx:
        option_type_diag = st.selectbox("Type", ["call", "put"], key=_k("diag_type"))
        k_near = st.slider(
            "Strike near",
            min_value=0.6 * strike_anchor_diag,
            max_value=1.4 * strike_anchor_diag,
            value=float(strike_anchor_diag),
            step=0.5,
            key=_k("diag_k_near"),
        )
        k_far = st.slider(
            "Strike far",
            min_value=0.6 * strike_anchor_diag,
            max_value=1.6 * strike_anchor_diag,
            value=float(strike_anchor_diag),
            step=0.5,
            key=_k("diag_k_far"),
        )
        t_near = st.slider(
            "T near (années)",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key=_k("diag_t_near"),
        )
        t_far_raw = st.slider(
            "T far (années)",
            min_value=0.1,
            max_value=2.0,
            value=0.75,
            step=0.05,
            key=_k("diag_t_far"),
        )
        t_far = max(t_far_raw, t_near + 0.01)
        if t_far != t_far_raw:
            st.caption(f"T far ajusté à {t_far:.2f} pour rester après T near.")
    with col2_ctx:
        span_diag = st.slider(
            "Span payoff (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=_k("diag_span"),
        )
    r_diag = float(get_rate_for_ttm(t_far))
    iv_diag = _get_cached_iv_for(k_far, t_far, option_type_diag)
    sigma_diag = (
        float(iv_diag)
        if iv_diag is not None and np.isfinite(iv_diag) and iv_diag > 0
        else float(get_common_sigma_value())
    )
    if iv_diag is not None and np.isfinite(iv_diag) and iv_diag > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_diag:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_diagonal_spread(
        s0_ref,
        k_near,
        k_far,
        T_near=t_near,
        T_far=t_far,
        option_type=option_type_diag,
        r=r_diag,
        q=float(get_common_div_yield()),
        sigma=sigma_diag,
        span=span_diag,
    )
    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]
    payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
    pnl_s0 = payoff_s0 - premium

    forward_start_date = datetime.date.today() + datetime.timedelta(days=int(t_near * 365))
    figs = []
    if close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
        ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
        ax_ts.axhline(k_near, color="gray", linestyle="--", label=f"K near = {k_near:.2f}")
        ax_ts.axhline(k_far, color="firebrick", linestyle=":", label=f"K far = {k_far:.2f}")
        ax_ts.axvline(
            forward_start_date,
            color="purple",
            linestyle=":",
            label=f"Start near ~ {forward_start_date.isoformat()}",
        )
        ax_ts.set_ylabel("Prix")
        ax_ts.set_title(f"Clôtures {hist_tkr} (strikes / start)")
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
    ax_pay.axvline(k_near, color="gray", linestyle="--", label=f"K near = {k_near:.2f}")
    ax_pay.axvline(k_far, color="firebrick", linestyle=":", label=f"K far = {k_far:.2f}")
    ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Diagonal spread ({option_type_diag})")
    figs.append(fig_pay)
    render_figures_grid(figs)
    st.metric("Prix de l'option", f"${premium:.6f}")
