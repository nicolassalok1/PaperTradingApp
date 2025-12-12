import datetime
import time

import pandas as pd
import streamlit as st

from app.controller import yieldcurve_controller as yc
from app.vue.components.page_utils import render_closing_history_chart, render_page_header


def render():
    # Try to load existing cache once (no download) to avoid API calls when blocked.
    df_curve_cached, curve_path_cached = yc.load_curve(ensure_cache=False)
    df_curve_fwd = df_curve_cached
    curve_path_fwd = curve_path_cached

    # Reset form fields if flagged by previous submission.
    if st.session_state.pop("_reset_forward_form", False):
        for k in [
            "fwd_symbol",
            "fwd_spot_symbol",
            "fwd_spot_value",
            "fwd_qty",
            "fwd_side",
            "fwd_maturity",
        ]:
            st.session_state.pop(k, None)

    render_page_header(
        "Forwards & Yield Curve",
        "Prix forward, suivi des contrats et visualisation des taux",
        icon="🧭",
        badge="Rates",
    )

    with st.expander("ℹ️ Comprendre Forwards"):
        st.markdown(
            """
            ### 💡 Ce que vous faites dans Forwards

            Cet onglet vous permet d'**acheter (ou vendre) un forward** sur un sous-jacent : vous fixez aujourd'hui un prix d'échange futur.

            - Vous choisissez le sous-jacent, la date d'échéance, le sens (long/short) et le prix forward.
            - Le forward est enregistré dans un fichier JSON (`forwards_portfolio.json`).
            - Il apparaît ensuite dans le Dashboard avec P&L mark-to-market basé sur le prix spot actuel.

            Un **long forward** gagne lorsque le prix spot est au-dessus du prix forward à l'échéance; un **short forward** gagne dans le cas inverse.
            """
        )

    st.markdown("### 🧭 Trade Forward")

    st.markdown("##### Historique des taux (Yield Curve)")

    col_curve, col_refresh, col_upload = st.columns([3, 2, 2])
    with col_curve:
        if st.button("📈 Charger le cache", key="btn_load_curve_cache"):
            with st.spinner("Lecture du cache yield_curve.csv"):
                df_curve_fwd, curve_path_fwd = yc.load_curve(ensure_cache=False)
    with col_refresh:
        if st.button("🔄 Rafraîchir (Stooq)", key="btn_refresh_curve"):
            with st.spinner("Téléchargement des taux..."):
                df_curve_fwd, curve_path_fwd = yc.build_curve()
    with col_upload:
        uploaded = st.file_uploader("Déposer yield_curve.csv", type="csv", key="upload_curve")
        if uploaded is not None:
            import pandas as pd

            try:
                df_up = pd.read_csv(uploaded)
                df_up.to_csv(yc.yield_curve_cache_file(), index=False)
                df_curve_fwd = df_up
                curve_path_fwd = yc.yield_curve_cache_file()
                st.success("Courbe importée et sauvegardée dans le cache.")
            except Exception as exc:
                st.error(f"Import CSV impossible : {exc}")

    if df_curve_fwd is not None and not df_curve_fwd.empty:
        st.caption(f"Source: {curve_path_fwd}")
        df_curve_numeric = df_curve_fwd.apply(pd.to_numeric, errors="coerce")
        if not df_curve_numeric.empty:
            st.line_chart(df_curve_numeric)
    else:
        st.info(
            "Aucune courbe chargée. Utilise les boutons ci-dessus ou dépose `cache/yield_curve.csv` manuellement."
        )

    col_sym, col_fetch = st.columns([3, 1], vertical_alignment="bottom")
    with col_sym:
        fwd_symbol = st.text_input(
            "Underlying symbol",
            placeholder="e.g., AAPL",
            key="fwd_symbol",
        ).upper()
    with col_fetch:
        if st.button("🔎 Récupérer le spot", key="btn_fetch_forward_spot"):
            price_data = yc.get_spot(fwd_symbol) if fwd_symbol else {"price": 0}
            spot_now_fetch = yc.floor_4(price_data.get("price", 0.0))
            if spot_now_fetch > 0:
                st.session_state["fwd_spot_symbol"] = fwd_symbol
                st.session_state["fwd_spot_value"] = spot_now_fetch
                st.success(f"Spot {fwd_symbol} ~ ${spot_now_fetch:.4f}")
            else:
                st.warning("Spot introuvable pour ce ticker.")

    with st.container():
        render_closing_history_chart(fwd_symbol, "forward_history", location_label="Forward")

    today = datetime.date.today()
    default_maturity = today + datetime.timedelta(days=30)
    fwd_maturity = st.date_input(
        "Maturity date",
        value=default_maturity,
        min_value=today,
        key="fwd_maturity",
    )

    fwd_qty = st.number_input(
        "Notional (units)",
        min_value=1,
        value=1,
        step=1,
        key="fwd_qty",
    )

    spot_now = 0.0
    if fwd_symbol and st.session_state.get("fwd_spot_symbol") == fwd_symbol:
        spot_now = yc.floor_4(st.session_state.get("fwd_spot_value", 0.0))
    if spot_now <= 0 and fwd_symbol:
        price_data = yc.get_spot(fwd_symbol)
        spot_now = yc.floor_4(price_data.get("price", 0.0))

    if fwd_symbol:
        if spot_now > 0:
            days_to_mat = max((fwd_maturity - today).days, 0)
            T_years = days_to_mat / 365.0
            r_curve = yc.interpolate_curve_rate(df_curve_fwd, T_years) if T_years > 0 else None
            r_forward = (
                r_curve if r_curve is not None else yc.get_rate()(T_years if T_years > 0 else 0.1)
            )
            r_forward_f = yc.floor_4(r_forward)
            forward_price = yc.compute_forward_price(spot_now, r_forward, T_years)
            forward_price_f = yc.floor_4(forward_price)
            forward_price = forward_price_f
            spot_now_f = yc.floor_4(spot_now)
            T_years_f = yc.floor_3(T_years)
            pill = (
                f"<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.6rem;'>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"Spot {fwd_symbol} = {spot_now_f:.4f}</span>"
                f"<span style='background:#e3f2fd;color:#0d47a1;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"r = {r_forward_f:.4f}</span>"
                f"<span style='background:#fffde7;color:#f57f17;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"T = {T_years_f:.3f}y</span>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:700;'>"
                f"F ~ {forward_price_f:.4f}</span></div>"
            )
            st.markdown(pill, unsafe_allow_html=True)
        else:
            st.warning(
                "Impossible de récupérer le spot, tu peux quand même enregistrer mais le prix forward restera nul."
            )
            forward_price = 0.0
    else:
        forward_price = 0.0

    fwd_side_label = st.radio(
        "Position",
        options=["Long forward", "Short forward"],
        horizontal=True,
        key="fwd_side",
    )
    fwd_side = "long" if str(fwd_side_label).startswith("Long") else "short"

    # Only show the action button once a spot has been fetched/displayed to avoid stale/unknown prices.
    if spot_now > 0:
        if st.button("Enregistrer le forward", type="primary", key="btn_save_forward"):
            if fwd_maturity <= today:
                st.error("La maturité doit être strictement future.")
            elif not fwd_symbol or forward_price <= 0:
                st.error(
                    "Renseigne un symbole et assure-toi que le spot est disponible (>0) pour fixer le prix forward."
                )
            else:
                forwards = yc.load_forwards()
                uid = f"{fwd_symbol}_{fwd_maturity.isoformat()}_{int(time.time())}"
                forward_price_f = yc.floor_4(forward_price)
                forwards[uid] = {
                    "symbol": fwd_symbol,
                    "maturity": fwd_maturity.isoformat(),
                    "forward_price": forward_price_f,
                    "quantity": int(fwd_qty),
                    "side": fwd_side,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                yc.save_forwards(forwards)
                # UI is read-only; balance updates handled by downstream processes if needed.
                st.success(
                    f"Forward {fwd_side.upper()} sur {fwd_symbol} enregistré pour {fwd_maturity} à {forward_price_f:.4f}."
                )
                yc.clear_closing_history_cache()
                # Flag reset so fields are cleared before next render.
                st.session_state["_reset_forward_form"] = True
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("### 📑 Forward Portfolio")
    forwards = yc.load_forwards()
    if forwards:
        rows = yc.prepare_forward_rows(forwards, today=today)
        if rows:
            df_fwd = pd.DataFrame(rows)
            st.dataframe(df_fwd, hide_index=True)
        if st.button("🧹 Vider les forwards", key="clear_forwards"):
            yc.save_forwards({})
            st.success("Forward portfolio cleared.")
            time.sleep(1)
            st.rerun()
    else:
        st.info("Aucun forward pour le moment.")
