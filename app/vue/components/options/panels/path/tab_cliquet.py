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
    if not isinstance(close_series, pd.Series):
        close_series = pd.Series([S0], index=pd.Index([pd.Timestamp.today()]))
    # --------------------------------
    hist_tkr = ticker
    # FIX: runtime crash detected by crawler (option_char unbound)
    option_char = option_char if "option_char" in locals() else "c"
    clq_label, clq_char = _choose_option_select("opt_choice_cliquet_tab", option_char)
    option_label, option_char = clq_label, clq_char
    st.subheader("Cliquet / Ratchet – vue Notebook")
    k_cliquet_anchor = float(common_spot_value)
    strike_clq = st.slider(
        "Strike / niveau de référence",
        min_value=0.6 * k_cliquet_anchor,
        max_value=1.4 * k_cliquet_anchor,
        value=float(floor_n(k_cliquet_anchor, 0)),
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
    st.pyplot(fig_ts, clear_figure=True)

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff cliquet")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(S0, color="crimson", linestyle="-.", label=f"S0 = {S0:.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title("Cliquet / Ratchet (approx)")
    st.pyplot(fig_pay, clear_figure=True)

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
    expiration_dt = today + datetime.timedelta(days=int((T_clq or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("cliquet_qty"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("cliquet_side"))

    if st.button("Ajouter au dashboard", key=_k("cliquet_add"), type="primary"):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_char,
            "product_type": "Cliquet / Ratchet",
            "type": "Cliquet / Ratchet",
            "strike": float(strike_clq),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(S0),
            "maturity_years": float(T_clq),
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "floor": float(floor_val),
                "cap": float(cap_val),
                "strike_ref": float(strike_clq),
                "T": float(T_clq),
                "spot_at_pricing": float(S0),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("cliquet_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
