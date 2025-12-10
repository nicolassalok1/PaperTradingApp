import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_iron_bfly():
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
        "Strike central (iron butterfly)",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("iron_bfly_center"),
    )
    wing = st.slider(
        "Écart ailes",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * float(common_spot_value)),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("iron_bfly_wing"),
    )
    k_put_long = max(0.01, k_center - wing)
    k_call_long = k_center + wing

    # Iron butterfly uses its own pricer (different from iron condor)
    T_iron_bfly = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("iron_bfly_T"),
    )
    ivs_ib = [
        _get_cached_iv_for(k_put_long, T_iron_bfly, "put"),
        _get_cached_iv_for(k_center, T_iron_bfly, "call"),
        _get_cached_iv_for(k_call_long, T_iron_bfly, "call"),
    ]
    sigma_put_long_ib = (
        float(ivs_ib[0])
        if ivs_ib[0] is not None and np.isfinite(ivs_ib[0]) and ivs_ib[0] > 0
        else float(common_sigma_value)
    )
    sigma_call_center_ib = (
        float(ivs_ib[1])
        if ivs_ib[1] is not None and np.isfinite(ivs_ib[1]) and ivs_ib[1] > 0
        else float(common_sigma_value)
    )
    sigma_call_long_ib = (
        float(ivs_ib[2])
        if ivs_ib[2] is not None and np.isfinite(ivs_ib[2]) and ivs_ib[2] > 0
        else float(common_sigma_value)
    )
    sigma_put_center_ib = (
        sigma_call_center_ib  # même strike central, on réutilise la même IV pour le put central
    )
    iv_vals_ib = [v for v in ivs_ib if v is not None and np.isfinite(v) and v > 0]
    sigma_iron_bfly = float(np.mean(iv_vals_ib)) if iv_vals_ib else float(common_sigma_value)
    if iv_vals_ib:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k_put_long, k_center, k_call_long], ivs_ib)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            "σ utilisées : "
            f"put long {sigma_put_long_ib:.4f} | "
            f"put center {sigma_put_center_ib:.4f} | "
            f"call center {sigma_call_center_ib:.4f} | "
            f"call long {sigma_call_long_ib:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
    view_dyn = view_iron_butterfly(
        float(common_spot_value),
        k_put_long,
        k_center,
        k_call_long,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_put_long=float(sigma_put_long_ib),
        sigma_put_center=float(sigma_put_center_ib),
        sigma_call_center=float(sigma_call_center_ib),
        sigma_call_long=float(sigma_call_long_ib),
        T=float(T_iron_bfly),
    )
    premium_raw = price_iron_butterfly_bs(
        float(common_spot_value),
        k_put_long,
        k_center,
        k_call_long,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_put_long=float(sigma_put_long_ib),
        sigma_put_center=float(sigma_put_center_ib),
        sigma_call_center=float(sigma_call_center_ib),
        sigma_call_long=float(sigma_call_long_ib),
        T=float(T_iron_bfly),
    )
    premium = float(premium_raw)
    price_display = abs(premium)

    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["payoff"] - premium

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
    ax.set_title("Iron Butterfly (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)

    st.session_state[_k("iron_bfly_pre_price")] = premium
    price = float(price_display)

    st.markdown("### Ajouter au dashboard")
    st.metric("Prix calculé", f"${price_display:.6f}")

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
    expiration_dt = today + datetime.timedelta(days=int((T_iron_bfly or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("iron_bfly_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("iron_bfly_side_inline"))
    st.caption(
        f"K_put_long: {k_put_long:.4f} | K_centre: {k_center:.4f} | K_call_long: {k_call_long:.4f}"
    )
    st.caption(f"T (maturité, années): {float(T_iron_bfly):.4f}")

    if st.button("Ajouter au dashboard", key=_k("iron_bfly_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": "call" if option_char.lower() == "c" else "put",
            "product_type": "Iron Butterfly",
            "type": "Iron Butterfly",
            "strike": float(k_put_long),
            "strike2": float(k_call_long),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price_display,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_iron_bfly),
            "legs": [
                {"option_type": "put", "strike": float(k_put_long)},
                {"option_type": "put", "strike": float(k_center)},
                {"option_type": "call", "strike": float(k_center)},
                {"option_type": "call", "strike": float(k_call_long)},
            ],
            "T_0": today.isoformat(),
            "price": price_display,
            "misc": {
                "structure": "Iron butterfly",
                "legs": [
                    {"option_type": "put", "strike": float(k_put_long)},
                    {"option_type": "put", "strike": float(k_center)},
                    {"option_type": "call", "strike": float(k_center)},
                    {"option_type": "call", "strike": float(k_call_long)},
                ],
                "premium_raw": float(premium),
                "spot_at_pricing": float(common_spot_value),
                "sigma_put_long_used": float(sigma_put_long_ib),
                "sigma_put_center_used": float(sigma_put_center_ib),
                "sigma_call_center_used": float(sigma_call_center_ib),
                "sigma_call_long_used": float(sigma_call_long_ib),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_iron_bfly),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("iron_bfly_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
