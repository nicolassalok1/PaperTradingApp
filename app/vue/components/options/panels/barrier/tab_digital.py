import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_digital():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    # Legacy defaults to avoid NameError
    option_char = st.session_state.get("option_char", "c")
    common_spot_value = float(st.session_state.get("common_spot_value", S0))
    common_maturity_value = float(st.session_state.get("common_maturity_value", 1.0))
    common_rate_value = float(st.session_state.get("common_rate_value", 0.01))
    common_sigma_value = float(st.session_state.get("common_sigma_value", 0.2))
    d_common = float(st.session_state.get("d_common", 0.0))  # dividend yield

    hist_tkr = ticker
    opt_label_dig, opt_char_dig = _choose_option_select("opt_choice_digital", option_char)
    option_label, option_char = opt_label_dig, opt_char_dig
    strike = st.slider(
        "Strike",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("digital_k"),
    )
    T_dig = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("digital_T"),
    )
    opt_type = "call" if opt_char_dig == "c" else "put"
    iv_dig = _get_cached_iv_for(strike, T_dig, opt_type)
    sigma_dig = (
        float(iv_dig)
        if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0
        else float(common_sigma_value)
    )
    if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_dig:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_digital(
        float(common_spot_value),
        strike,
        T=float(T_dig),
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(sigma_dig),
        option_type=opt_type,
    )
    premium = float(view_dyn.get("premium", 0.0))
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
    ax.set_title("Digital (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)

    st.session_state[_k("digital_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_dig or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("digital_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("digital_side_inline"))
    st.caption(f"K: {strike:.4f}")
    st.caption(f"T (maturité, années): {float(T_dig):.4f}")

    if st.button("Ajouter au dashboard", key=_k("digital_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": opt_char_dig,
            "product_type": "Digital",
            "type": "Digital",
            "strike": float(strike),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_dig),
            "legs": [
                {
                    "option_type": opt_char_rain,
                    "strike": float(strike),
                    "payout": 1.0,
                    "digital": True,
                },
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Digital",
                "strike": float(strike),
                "spot_at_pricing": float(common_spot_value),
                "sigma_used": float(sigma_dig),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_dig),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("digital_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
