import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_strangle():
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
    k_put_raw = st.slider(
        "Strike put",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=0.95 * float(common_spot_value),
        step=0.5,
        key=_k("strangle_k_put"),
    )
    k_call_raw = st.slider(
        "Strike call",
        min_value=0.5 * float(common_spot_value),
        max_value=1.5 * float(common_spot_value),
        value=1.05 * float(common_spot_value),
        step=0.5,
        key=_k("strangle_k_call"),
    )
    k_put = min(k_put_raw, k_call_raw)
    k_call = max(k_put_raw, k_call_raw)
    T_strangle = st.slider(
        "T (années)",
        min_value=0.05,
        max_value=2.0,
        value=float(common_maturity_value),
        step=0.05,
        key=_k("strangle_T"),
    )
    iv_put = _get_cached_iv_for(k_put, T_strangle, "put")
    iv_call = _get_cached_iv_for(k_call, T_strangle, "call")
    sigma_put_strangle = (
        float(iv_put)
        if iv_put is not None and np.isfinite(iv_put) and iv_put > 0
        else float(common_sigma_value)
    )
    sigma_call_strangle = (
        float(iv_call)
        if iv_call is not None and np.isfinite(iv_call) and iv_call > 0
        else float(common_sigma_value)
    )
    if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_put, iv_call)):
        iv_put_txt = (
            f"{iv_put:.4f}" if iv_put is not None and np.isfinite(iv_put) and iv_put > 0 else "n/a"
        )
        iv_call_txt = (
            f"{iv_call:.4f}"
            if iv_call is not None and np.isfinite(iv_call) and iv_call > 0
            else "n/a"
        )
        st.caption(f"IV récupérées (cache) ≈ put {iv_put_txt} | call {iv_call_txt}")
        st.caption(f"σ utilisées : put {sigma_put_strangle:.4f} | call {sigma_call_strangle:.4f}")
    else:
        st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

    view_dyn = view_strangle(
        float(common_spot_value),
        k_put,
        k_call,
        r=float(common_rate_value),
        q=float(d_common),
        sigma=float(common_sigma_value),
        sigma_call=float(sigma_call_strangle),
        sigma_put=float(sigma_put_strangle),
        T=float(T_strangle),
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
    ax.set_title("Strangle (payoff & P&L avec prime BS)")
    ax.legend(loc="lower right")
    show_and_close(fig)

    st.session_state[_k("strangle_pre_price")] = premium
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
    expiration_dt = today + datetime.timedelta(days=int((T_strangle or 0.0) * 365))
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("strangle_qty_inline"))
    side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("strangle_side_inline"))
    st.caption(f"K_put: {k_put:.4f} | K_call: {k_call:.4f}")
    st.caption(f"T (maturité, années): {float(T_strangle):.4f}")

    if st.button("Ajouter au dashboard", key=_k("strangle_add_inline")):
        payload = {
            "underlying": underlying or "N/A",
            "option_type": "call" if option_char.lower() == "c" else "put",
            "product_type": "Strangle",
            "type": "Strangle",
            "strike": float(k_put),
            "strike2": float(k_call),
            "expiration": expiration_dt.isoformat(),
            "quantity": int(qty),
            "avg_price": price,
            "side": side,
            "S0": float(common_spot_value),
            "maturity_years": float(T_strangle),
            "legs": [
                {"option_type": "put", "strike": float(k_put)},
                {"option_type": "call", "strike": float(k_call)},
            ],
            "T_0": today.isoformat(),
            "price": price,
            "misc": {
                "structure": "Strangle",
                "legs": [
                    {"option_type": "put", "strike": float(k_put)},
                    {"option_type": "call", "strike": float(k_call)},
                ],
                "strike_put": float(k_put),
                "strike_call": float(k_call),
                "spot_at_pricing": float(common_spot_value),
                "sigma_put_used": float(sigma_put_strangle),
                "sigma_call_used": float(sigma_call_strangle),
                "r": float(common_rate_value),
                "q": float(d_common),
                "maturity": float(T_strangle),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("strangle_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
