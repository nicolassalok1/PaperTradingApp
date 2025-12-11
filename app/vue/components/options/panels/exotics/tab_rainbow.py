import streamlit as st
import pandas as pd
import numpy as np

from app.vue.components.options.controller_bridge import (
    add_option_to_dashboard_clean,
    common_rate_value,
    common_sigma_value,
    d_common,
    ensure_close_history,
    get_option_context,
    load_close_series_for_ticker,
    log_action,
    render_figures_grid,
    resolve_common_underlying,
    view_rainbow,
)


def render_tab_rainbow():
    """
    Rainbow option (deux sous-jacents) avec pricing MC et push dashboard.
    """
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0_a = ctx["S0"]
    ticker_a = ctx["ticker"]
    close_a = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    if not isinstance(close_a, pd.Series) or close_a.empty:
        close_a = pd.Series([S0_a], index=pd.Index([pd.Timestamp.today()]))
    # --------------------------------

    st.subheader("Rainbow (2 sous-jacents) – vue Notebook")
    col1, col2 = st.columns(2)
    with col1:
        ticker_b = (
            st.text_input(
                "Ticker secondaire (B)",
                value=st.session_state.get("tkr_rainbow_b", ""),
                placeholder="ex: MSFT",
                key=_k("rainbow_tkr_b"),
            )
            .strip()
            .upper()
        )
        st.session_state["tkr_rainbow_b"] = ticker_b
        opt_type = st.selectbox("Type", ["call", "put"], key=_k("rainbow_type"))
        T_val = st.slider(
            "T (années)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, key=_k("rainbow_T")
        )
        span_val = st.slider(
            "Span payoff (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=_k("rainbow_span"),
        )
    with col2:
        rho_val = st.slider(
            "Corrélation ρ", min_value=-0.9, max_value=0.9, value=0.2, step=0.05, key=_k("rainbow_rho")
        )
        sigma_a = st.slider(
            "Sigma A", min_value=0.05, max_value=1.0, value=float(common_sigma_value), step=0.01, key=_k("rainbow_sigma_a")
        )
        sigma_b = st.slider(
            "Sigma B", min_value=0.05, max_value=1.0, value=float(common_sigma_value), step=0.01, key=_k("rainbow_sigma_b")
        )
        n_paths = st.number_input(
            "N paths (MC)", min_value=2000, max_value=50000, value=15000, step=1000, key=_k("rainbow_npaths")
        )

    # Série secondaire (B)
    close_b = None
    if ticker_b:
        close_b = load_close_series_for_ticker(ticker_b, fallback_value=S0_a)
        if close_b is None or getattr(close_b, "empty", True):
            close_b = pd.Series([S0_a], index=pd.Index([pd.Timestamp.today()]))
    S0_b = float(close_b.iloc[-1]) if close_b is not None and not close_b.empty else float(S0_a)

    strike_anchor = np.nanmean([float(S0_a), float(S0_b)])
    strike_val = st.slider(
        "Strike commun",
        min_value=0.5 * strike_anchor,
        max_value=1.5 * strike_anchor,
        value=float(strike_anchor),
        step=0.5,
        key=_k("rainbow_k"),
    )

    try:
        view_dyn = view_rainbow(
            float(S0_a),
            float(S0_b),
            float(strike_val),
            span=span_val,
            option_type=opt_type,
            T=T_val,
            sigma=sigma_a,
            sigma_b=sigma_b,
            r=float(common_rate_value),
            q=float(d_common),
            rho=rho_val,
            n_paths=int(n_paths),
        )
    except Exception as exc:
        st.error(f"Pricing Rainbow indisponible : {exc}")
        return

    premium = float(view_dyn.get("premium", 0.0))
    s_grid = view_dyn["s_grid"]
    payoff_grid = view_dyn["payoff"]
    pnl_grid = view_dyn["pnl"]

    price_cols = st.columns(3)
    price_cols[0].metric("Prix (MC)", f"${premium:.4f}")
    if view_dyn.get("breakevens"):
        bes = ", ".join(f"{b:.2f}" for b in view_dyn["breakevens"])
        price_cols[1].metric("Seuils", bes)
    price_cols[2].metric("ρ utilisé", f"{rho_val:.2f}")

    figs = []
    fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
    ax_ts.plot(close_a.index, close_a.values, label=f"{ticker_a or 'A'} close")
    if close_b is not None and not close_b.empty:
        ax_ts.plot(close_b.index, close_b.values, label=f"{ticker_b or 'B'} close", alpha=0.8)
    ax_ts.axhline(strike_val, color="gray", linestyle="--", label=f"K = {strike_val:.2f}")
    ax_ts.set_ylabel("Prix")
    ax_ts.set_title(f"Clôtures {ticker_a or 'A'} / {ticker_b or 'B'}")
    ax_ts.legend(loc="best")
    fig_ts.autofmt_xdate()
    figs.append(fig_ts)

    fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
    ax_pay.plot(s_grid, payoff_grid, label="Payoff")
    ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
    ax_pay.axvline(strike_val, color="gray", linestyle="--", label=f"K = {strike_val:.2f}")
    ax_pay.axvline(float(S0_a), color="crimson", linestyle="-.", label=f"S0 A = {float(S0_a):.2f}")
    ax_pay.axhline(0, color="black", linewidth=0.8)
    ax_pay.legend(loc="best")
    ax_pay.set_xlabel("Spot sous-jacent A")
    ax_pay.set_ylabel("Payoff / P&L")
    ax_pay.set_title(f"Rainbow ({opt_type})")
    figs.append(fig_pay)
    render_figures_grid(figs)

    st.markdown("### Ajouter au dashboard")
    qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("rainbow_qty"))
    if st.button("Ajouter au dashboard", key=_k("rainbow_add"), type="primary"):
        payload = {
            "underlying": f"{(ticker_a or 'A').upper()} / {(ticker_b or 'B').upper()}",
            "option_type": opt_type,
            "product_type": "Rainbow",
            "type": "Rainbow",
            "strike": float(strike_val),
            "quantity": int(qty),
            "avg_price": float(premium),
            "side": "long",
            "S0": float(S0_a),
            "S0_b": float(S0_b),
            "maturity_years": float(T_val),
            "T_0": datetime.date.today().isoformat(),
            "price": float(premium),
            "misc": {
                "sigma_a": float(sigma_a),
                "sigma_b": float(sigma_b),
                "rho": float(rho_val),
                "span": float(span_val),
                "n_paths": int(n_paths),
                "spot_at_pricing_a": float(S0_a),
                "spot_at_pricing_b": float(S0_b),
            },
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
