import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_chooser():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    hist_tkr = ticker

    # --- Contexte exotiques ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))
    hist_tkr = resolve_common_underlying()
    S0 = float(common_spot_value)
    # -----------------------------
    opt_label_chooser, opt_char_chooser = _choose_option_select("opt_choice_chooser", option_char)
    option_label = opt_label_chooser
    option_char_selected = opt_char_chooser
    strike = st.slider(
        "Strike",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("chooser_k"),
    )
    T_chooser = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("chooser_T"),
    )
    iv_chooser = _get_cached_iv_for(
        strike, T_chooser, "call" if option_char_selected == "c" else "put"
    )
    sigma_chooser = (
        float(iv_chooser)
        if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0
        else float(common_sigma_value)
    )
    if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_chooser:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_chooser(
        float(common_spot_value),
        strike,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(sigma_chooser),
        T=float(T_chooser),
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
    ax.set_title("Chooser (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("chooser_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_chooser or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("chooser_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("chooser_side_inline"))
    st.caption(f"K: {strike:.4f}")
    st.caption(f"T (maturité, années): {float(T_chooser):.4f}")

    if st.button("Ajouter au dashboard", key=_k("chooser_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": option_char_selected,
            "product_type": "Chooser",
            "type": "Chooser",
            "strike": float(strike),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_chooser),
            "legs": [
                {"option_type": "chooser", "strike": float(strike)},
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Chooser",
                "strike": float(strike),
                "spot_at_pricing": float(common_spot_value),
                "sigma_used": float(sigma_chooser),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_chooser),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("chooser_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
