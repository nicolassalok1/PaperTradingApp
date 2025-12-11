import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_put_spread():
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
    k_long_raw = st.slider(
        "Strike put long",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("put_spread_k_long"),
    )
    k_short_raw = st.slider(
        "Strike put short",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=float(common_spot_value),
        step=0.5,
        key=_k("put_spread_k_short"),
    )
    k_long = max(k_long_raw, k_short_raw)
    k_short = min(k_long_raw, k_short_raw)
    T_put_spread = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("put_spread_T"),
    )
    iv_long_p = _get_cached_iv_for(k_long, T_put_spread, "put")
    iv_short_p = _get_cached_iv_for(k_short, T_put_spread, "put")
    sigma_long_ps = (
        float(iv_long_p)
        if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0
        else float(common_sigma_value)
    )
    sigma_short_ps = (
        float(iv_short_p)
        if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0
        else float(common_sigma_value)
    )
    if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_long_p, iv_short_p)):
        iv_long_txt = (
            f"{iv_long_p:.4f}"
            if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0
            else "n/a"
        )
        iv_short_txt = (
            f"{iv_short_p:.4f}"
            if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0
            else "n/a"
        )
        st.caption(f"IV récupérées (cache) ≈ long {iv_long_txt} | short {iv_short_txt}")
        st.caption(f"σ utilisées : long {sigma_long_ps:.4f} | short {sigma_short_ps:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_put_spread(
        float(common_spot_value),
        k_long,
        k_short,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_long=float(sigma_long_ps),
        sigma_short=float(sigma_short_ps),
        T=float(T_put_spread),
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
    ax.set_title("Put spread (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("put_spread_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_put_spread or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("put_spread_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("put_spread_side_inline"))
    st.caption(f"K long: {k_long:.4f} | K short: {k_short:.4f}")
    st.caption(f"T (maturité, années): {float(T_put_spread):.4f}")

    if st.button("Ajouter au dashboard", key=_k("put_spread_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": "call" if option_char.lower() == "c" else "put",
            "product_type": "Put Spread",
            "type": "Put spread",
            "strike": float(k_long),
            "strike2": float(k_short),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_put_spread),
            "legs": [
                {"option_type": "put", "strike": float(k_long)},
                {"option_type": "put", "strike": float(k_short)},
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Put spread",
                "legs": [
                    {"option_type": "put", "strike": float(k_long)},
                    {"option_type": "put", "strike": float(k_short)},
                ],
                "spot_at_pricing": float(common_spot_value),
                "sigma_put_long_used": float(sigma_long_ps),
                "sigma_put_short_used": float(sigma_short_ps),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_put_spread),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
