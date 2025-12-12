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
    hist_tkr = ticker

    # --- Contexte exotiques ---
    common_spot_value = float(st.session_state.get("common_spot_value", S0 if S0 is not None else 100.0))
    hist_tkr = resolve_common_underlying()
    base_spot = float(S0 if S0 is not None else common_spot_value)
    S0 = base_spot
    # -----------------------------
    opt_label_quanto, opt_char_quanto = _choose_option_select("opt_choice_quanto", option_char)
    option_label = opt_label_quanto
    option_char_selected = opt_char_quanto
    strike = st.slider(
        "Strike",
        min_value=0.5 * base_spot,
        max_value=1.5 * base_spot,
        value=base_spot,
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
        value=float(common_maturity_value),
        step=0.05,
        key=_k("quanto_T"),
    )
    iv_quanto = _get_cached_iv_for(strike, T_quanto, opt_type)
    sigma_quanto = (
        float(iv_quanto)
        if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0
        else float(common_sigma_value)
    )
    if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0:
        st.caption(f"IV récupérée (cache) ≈ {iv_quanto:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_quanto(
        float(common_spot_value),
        strike,
        fx_rate=fx_rate,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(sigma_quanto),
        T=float(T_quanto),
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
    ax.set_title("Quanto (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("quanto_pre_price")] = premium
    price = float(premium)

    st.markdown("### Ajouter au dashboard")
    st.metric("Prix calculé", f"${price:.6f}")

    underlying = (
        (
            st.session_state.get("tkr_common")
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
    expiration_dt = today + datetime.timedelta(days=int((T_quanto or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("quanto_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("quanto_side_inline"))
    st.caption(f"K: {strike:.4f} | FX: {fx_rate:.4f}")
    st.caption(f"T (maturité, années): {float(T_quanto):.4f}")

    if st.button("Ajouter au dashboard", key=_k("quanto_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": opt_char_quanto,
            "product_type": "Quanto",
            "type": "Quanto",
            "strike": float(strike),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_quanto),
            "legs": [
                {
                    "option_type": opt_char_rain,
                    "strike": float(strike),
                    "fx_rate": float(fx_rate),
                    "quanto": True,
                },
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Quanto",
                "strike": float(strike),
                "fx_rate": float(fx_rate),
                "spot_at_pricing": float(common_spot_value),
                "sigma_used": float(sigma_quanto),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_quanto),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
