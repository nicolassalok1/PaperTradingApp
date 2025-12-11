import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_iron_condor():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    if not ensure_close_history(ctx):
        return
    hist_tkr = ticker

    # --- Bootstrap du contexte Spreads/Wings ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))

    hist_tkr = resolve_common_underlying()
    S0 = float(common_spot_value)
    # --- Fin bootstrap ---
    k_center = st.slider(
        "Strike central (iron condor)",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("iron_condor_center"),
    )
    inner = st.slider(
        "Écart strikes courts",
        min_value=max(0.1, 0.02 * float(common_spot_value)),
        max_value=max(1.0, 0.5 * float(common_spot_value)),
        value=max(0.5, 0.05 * float(common_spot_value)),
        step=0.1,
        key=_k("iron_condor_inner"),
    )
    outer_raw = st.slider(
        "Écart strikes ailes",
        min_value=max(0.2, 0.03 * float(common_spot_value)),
        max_value=max(1.5, 0.7 * float(common_spot_value)),
        value=max(0.9, 0.1 * float(common_spot_value)),
        step=0.1,
        key=_k("iron_condor_outer"),
    )
    outer = max(outer_raw, inner + max(0.1, 0.01 * float(common_spot_value)))

    k_put_long = max(0.01, k_center - outer)
    k_put_short = k_center - inner
    k_call_short = k_center + inner
    k_call_long = k_center + outer
    T_iron_condor = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("iron_condor_T"),
    )
    ivs_ic = [
        _get_cached_iv_for(k_put_long, T_iron_condor, "put"),
        _get_cached_iv_for(k_put_short, T_iron_condor, "put"),
        _get_cached_iv_for(k_call_short, T_iron_condor, "call"),
        _get_cached_iv_for(k_call_long, T_iron_condor, "call"),
    ]
    sigma_put_long_ic = (
        float(ivs_ic[0])
        if ivs_ic[0] is not None and np.isfinite(ivs_ic[0]) and ivs_ic[0] > 0
        else float(common_sigma_value)
    )
    sigma_put_short_ic = (
        float(ivs_ic[1])
        if ivs_ic[1] is not None and np.isfinite(ivs_ic[1]) and ivs_ic[1] > 0
        else float(common_sigma_value)
    )
    sigma_call_short_ic = (
        float(ivs_ic[2])
        if ivs_ic[2] is not None and np.isfinite(ivs_ic[2]) and ivs_ic[2] > 0
        else float(common_sigma_value)
    )
    sigma_call_long_ic = (
        float(ivs_ic[3])
        if ivs_ic[3] is not None and np.isfinite(ivs_ic[3]) and ivs_ic[3] > 0
        else float(common_sigma_value)
    )
    iv_vals_ic = [v for v in ivs_ic if v is not None and np.isfinite(v) and v > 0]
    if iv_vals_ic:
        iv_txt = " | ".join(
            (
                f"K={k:.2f}: {v:.4f}"
                if v is not None and np.isfinite(v) and v > 0
                else f"K={k:.2f}: n/a"
            )
            for k, v in zip([k_put_long, k_put_short, k_call_short, k_call_long], ivs_ic)
        )
        st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
        st.caption(
            "σ utilisées : "
            f"put long {sigma_put_long_ic:.4f} | put short {sigma_put_short_ic:.4f} | "
            f"call short {sigma_call_short_ic:.4f} | call long {sigma_call_long_ic:.4f}"
        )
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_iron_condor(
        float(common_spot_value),
        k_put_long,
        k_put_short,
        k_call_short,
        k_call_long,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_put_long=float(sigma_put_long_ic),
        sigma_put_short=float(sigma_put_short_ic),
        sigma_call_short=float(sigma_call_short_ic),
        sigma_call_long=float(sigma_call_long_ic),
        T=float(T_iron_condor),
    )
    premium = float(view_dyn.get("premium", 0.0))
    price_display = abs(premium)

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
    ax.set_title("Iron Condor (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("iron_condor_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_iron_condor or 0.0) * 365))
    qty = st.number_input(
        "Quantité", min_value=1, value=1, step=1, key=_k("iron_condor_qty_inline")
    )
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("iron_condor_side_inline"))
    st.caption(
        f"K put long: {k_put_long:.4f} | K put short: {k_put_short:.4f} | "
        f"K call short: {k_call_short:.4f} | K call long: {k_call_long:.4f}"
    )
    st.caption(f"T (maturité, années): {float(T_iron_condor):.4f}")

    if st.button("Ajouter au dashboard", key=_k("iron_condor_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": "call" if option_char.lower() == "c" else "put",
            "product_type": "Iron Condor",
            "type": "Iron Condor",
            "strike": float(k_put_long),
            "strike2": float(k_call_long),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price_display,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_iron_condor),
            "legs": [
                {"option_type": "put", "strike": float(k_put_long)},
                {"option_type": "put", "strike": float(k_put_short)},
                {"option_type": "call", "strike": float(k_call_short)},
                {"option_type": "call", "strike": float(k_call_long)},
            ],
            "T_0": today.isoformat(),
            "price": price_display,
            "misc": {
                "structure": "Iron Condor",
                "legs": [
                    {"option_type": "put", "strike": float(k_put_long)},
                    {"option_type": "put", "strike": float(k_put_short)},
                    {"option_type": "call", "strike": float(k_call_short)},
                    {"option_type": "call", "strike": float(k_call_long)},
                ],
                "premium_raw": float(premium),
                "spot_at_pricing": float(common_spot_value),
                "sigma_put_long_used": float(sigma_put_long_ic),
                "sigma_put_short_used": float(sigma_put_short_ic),
                "sigma_call_short_used": float(sigma_call_short_ic),
                "sigma_call_long_used": float(sigma_call_long_ic),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_iron_condor),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("iron_condor_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
