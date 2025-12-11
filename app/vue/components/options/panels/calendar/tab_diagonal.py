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

    strike_anchor_diag = float(common_spot_value)
    col1, col2 = st.columns(2)
    col1_ctx = col1 if hasattr(col1, "__enter__") else contextlib.nullcontext()
    col2_ctx = col2 if hasattr(col2, "__enter__") else contextlib.nullcontext()
    with col1_ctx:
        option_type_diag = st.selectbox("Type", ["call", "put"], key=_k("diag_type"))
        k_near = st.slider(
            "Strike near",
            min_value=0.6 * strike_anchor_diag,
            max_value=1.4 * strike_anchor_diag,
            value=float(floor_n(strike_anchor_diag, 0)),
            step=0.5,
            key=_k("diag_k_near"),
        )
        k_far = st.slider(
            "Strike far",
            min_value=0.6 * strike_anchor_diag,
            max_value=1.6 * strike_anchor_diag,
            value=float(floor_n(strike_anchor_diag * 1.02, 0)),
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
    r_diag = float(common_rate_value)
    iv_diag = _get_cached_iv_for(k_far, t_far, option_type_diag)
    sigma_diag = (
        float(iv_diag)
        if iv_diag is not None and np.isfinite(iv_diag) and iv_diag > 0
        else float(common_sigma_value)
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
        q=0.0,
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

    price = float(premium)
    st.markdown("### Ajouter au dashboard")
    st.metric("Prix calculé", f"${price:.6f}")

    underlying = (
        (
            st.session_state.get("heston_cboe_ticker")
            or st.session_state.get("tkr_common")
            or st.session_state.get("common_underlying")
            or st.session_state.get("ticker_default")
            or ""
        )
        .strip()
        .upper()
    )
    st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
    today = datetime.date.today()
    expiration_dt = today + datetime.timedelta(days=int((t_far or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("diag_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("diag_side"))

    if st.button("Ajouter au dashboard", key=_k("diag_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_type_diag,
            "product_type": "Diagonal spread",
            "type": "Diagonal spread",
            "strike": float(k_near),
            "strike2": float(k_far),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(s0_ref),
            "maturity_years": float(t_far),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "T_near": float(t_near),
                "T_far": float(t_far),
                "sigma_used": float(sigma_diag),
                "r": float(r_diag),
                "span": float(span_diag),
                "spot_at_pricing": float(s0_ref),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    # removed duplicate add_clean block for cleaned UI
    # if "price" in locals() and st.button("Ajouter au dashboard", key=_k("diag_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
