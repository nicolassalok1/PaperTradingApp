import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_butterfly():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    hist_tkr = ticker

    # --- Bootstrap du contexte Spreads/Wings ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))

    hist_tkr = resolve_common_underlying()
    S0 = float(common_spot_value)
    # --- Fin bootstrap ---
    k_center = st.slider(
        "Strike central",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("butterfly_k_center"),
    )
    wing = st.slider(
        "Écart ailes",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * float(common_spot_value)),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("butterfly_wing"),
    )
    k1 = max(0.01, k_center - wing)
    k2 = k_center
    k3 = k_center + wing
    T_bfly = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("butterfly_T"),
    )
    ivs_bfly = [
        _get_cached_iv_for(k1, T_bfly, "call"),
        _get_cached_iv_for(k2, T_bfly, "call"),
        _get_cached_iv_for(k3, T_bfly, "call"),
    ]
    sigma_k1_bfly = (
        float(ivs_bfly[0])
        if ivs_bfly[0] is not None and np.isfinite(ivs_bfly[0]) and ivs_bfly[0] > 0
        else float(common_sigma_value)
    )
    sigma_k2_bfly = (
        float(ivs_bfly[1])
        if ivs_bfly[1] is not None and np.isfinite(ivs_bfly[1]) and ivs_bfly[1] > 0
        else float(common_sigma_value)
    )
    sigma_k3_bfly = (
        float(ivs_bfly[2])
        if ivs_bfly[2] is not None and np.isfinite(ivs_bfly[2]) and ivs_bfly[2] > 0
        else float(common_sigma_value)
    )
    iv_vals_bfly = [v for v in ivs_bfly if v is not None and np.isfinite(v) and v > 0]
    sigma_bfly = float(np.mean(iv_vals_bfly)) if iv_vals_bfly else float(common_sigma_value)
    if iv_vals_bfly:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k1, k2, k3], ivs_bfly)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            f"σ utilisées : K1 {sigma_k1_bfly:.4f} | K2 {sigma_k2_bfly:.4f} | K3 {sigma_k3_bfly:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_butterfly(
        float(common_spot_value),
        k1,
        k2,
        k3,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_k1=float(sigma_k1_bfly),
        sigma_k2=float(sigma_k2_bfly),
        sigma_k3=float(sigma_k3_bfly),
        T=float(T_bfly),
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
    ax.set_title("Butterfly (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)

    st.session_state[_k("butterfly_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_bfly or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("butterfly_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("butterfly_side_inline"))
    st.caption(f"K1: {k1:.4f} | K2: {k2:.4f} | K3: {k3:.4f}")
    st.caption(f"T (maturité, années): {float(T_bfly):.4f}")

    if st.button("Ajouter au dashboard", key=_k("butterfly_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": "call" if option_char.lower() == "c" else "put",
            "product_type": "Butterfly",
            "type": "Butterfly",
            "strike": float(k1),
            "strike2": float(k3),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_bfly),
            "legs": [
                {"option_type": "call", "strike": float(k1)},
                {"option_type": "call", "strike": float(k2)},
                {"option_type": "call", "strike": float(k2)},
                {"option_type": "call", "strike": float(k3)},
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Butterfly",
                "legs": [
                    {"option_type": "call", "strike": float(k1)},
                    {"option_type": "call", "strike": float(k2)},
                    {"option_type": "call", "strike": float(k2)},
                    {"option_type": "call", "strike": float(k3)},
                ],
                "spot_at_pricing": float(common_spot_value),
                "sigma_k1_used": float(sigma_k1_bfly),
                "sigma_k2_used": float(sigma_k2_bfly),
                "sigma_k3_used": float(sigma_k3_bfly),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_bfly),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("butterfly_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
