import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_grp_barrier():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    # Fallback defaults for anciennes globals (avoids NameError)
    option_char = st.session_state.get("option_char", "c")
    common_spot_value = float(st.session_state.get("common_spot_value", S0))
    common_maturity_value = float(st.session_state.get("common_maturity_value", 1.0))
    common_rate_value = float(st.session_state.get("common_rate_value", 0.01))
    common_sigma_value = float(st.session_state.get("common_sigma_value", 0.2))

    st.subheader("Barrières (vanilla / binaire) – vue Notebook")
    s0_ref = float(common_spot_value)
    hist_tkr, close_series = load_shared_close_series(s0_ref)

    strike_anchor_bar = float(common_spot_value)
    col1, col2, col3 = st.columns(3)
    with col1:
        strike_b = st.slider(
            "Strike",
            min_value=0.6 * strike_anchor_bar,
            max_value=1.4 * strike_anchor_bar,
            value=float(floor_n(strike_anchor_bar, 0)),
            step=0.5,
            key=_k("barrier_all_strike"),
        )
        barrier_b = st.slider(
            "Barrière",
            min_value=0.5 * strike_anchor_bar,
            max_value=1.8 * strike_anchor_bar,
            value=float(floor_n(strike_anchor_bar, 0)),
            step=0.5,
            key=_k("barrier_all_level"),
        )
        call_put_b = st.selectbox("Type", ["call", "put"], key=_k("barrier_all_type"))
    with col2:
        direction_b = st.selectbox("Direction", ["up", "down"], key=_k("barrier_all_dir"))
        knock_b = st.selectbox("Knock", ["out", "in"], key=_k("barrier_all_knock"))
        binary_b = st.checkbox("Binaire ?", value=False, key=_k("barrier_all_binary"))
        payout_b = st.slider(
            "Payout (si binaire)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key=_k("barrier_all_payout"),
        )
    with col3:
        r_b = float(common_rate_value)
        T_b = st.slider(
            "T (années)",
            min_value=0.05,
            max_value=2.0,
            value=common_maturity_value,
            step=0.05,
            key=_k("barrier_all_T"),
        )
    iv_bar = _get_cached_iv_for(strike_b, T_b, call_put_b)
    sigma_b = (
        float(iv_bar)
        if iv_bar is not None and np.isfinite(iv_bar) and iv_bar > 0
        else float(common_sigma_value)
    )
    if iv_bar is not None and np.isfinite(iv_bar) and iv_bar > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_bar:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    with st.spinner("Calcul..."):
        view_dyn = view_barrier(
            s0_ref,
            strike_b,
            barrier_b,
            direction=direction_b,
            knock=knock_b,
            option_type=call_put_b,
            payout=payout_b,
            binary=binary_b,
            r=r_b,
            q=0.0,
            sigma=sigma_b,
            T=T_b,
        )
        premium = float(view_dyn.get("premium", 0.0))
        s_grid = view_dyn["s_grid"]
        payoff_grid = view_dyn["payoff"]
        pnl_grid = view_dyn["pnl"]
        payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
        pnl_s0 = payoff_s0 - premium

    figs = []
    if close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
        ax_ts.plot(close_series.index, close_series.values, label=f"{hist_tkr} close (1y)")
        ax_ts.axhline(strike_b, color="gray", linestyle="--", label=f"Strike = {strike_b:.2f}")
        ax_ts.axhline(
            barrier_b, color="firebrick", linestyle=":", label=f"Barriere = {barrier_b:.2f}"
        )
        ax_ts.set_ylabel("Prix")
        ax_ts.set_title(f"Clôtures {hist_tkr} (strike/barrière)")
        ax_ts.legend(loc="best")
        fig_ts.autofmt_xdate()
        figs.append(fig_ts)
    else:
        st.info(
            "Ajoute un ticker commun en haut de l'onglet Options pour afficher l'historique 1 an."
        )

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(barrier_b, color="firebrick", linestyle=":", label=f"Barriere = {barrier_b:.2f}")
    ax_pay.axvline(strike_b, color="gray", linestyle="--", label=f"K = {strike_b:.2f}")
    ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Barrier {'binaire' if binary_b else 'vanilla'} ({direction_b} / {knock_b})")
    figs.append(fig_pay)
    render_figures_grid(figs)

    price = float(premium)
    st.markdown("### Ajouter au dashboard")
    st.metric("Prix calculé", f"${abs(price):.6f}")

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
    expiration_dt = today + datetime.timedelta(days=int((T_b or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("barrier_all_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("barrier_all_side"))

    if st.button("Ajouter au dashboard", key=_k("barrier_all_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": call_put_b,
            "product_type": f"Barrier {'binary' if binary_b else 'vanilla'}",
            "type": f"Barrier {'binary' if binary_b else 'vanilla'}",
            "strike": float(strike_b),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(s0_ref),
            "maturity_years": float(T_b),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "barrier": float(barrier_b),
                "barrier_type": direction_b,
                "direction": direction_b,
                "knock": knock_b,
                "binary": bool(binary_b),
                "payout": float(payout_b),
                "spot_at_pricing": float(s0_ref),
                "sigma_used": float(sigma_b),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    # removed duplicate add_clean block for cleaned UI
    # if "price" in locals() and st.button("Ajouter au dashboard", key=_k("barrier_all_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
