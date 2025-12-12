import pandas as pd
import streamlit as st
import altair as alt

from app.controller import yieldcurve_controller as yc
from app.vue.components.page_utils import render_page_header


def render():
    currencies = yc.available_currencies()
    default_currency = "USD" if "USD" in currencies else (currencies[0] if currencies else "USD")
    currency_options = currencies or [default_currency]
    default_index = currency_options.index(default_currency) if default_currency in currency_options else 0

    col_sel_ccy, col_sel_ref = st.columns([1, 1])
    with col_sel_ccy:
        currency = st.selectbox("Currency", currency_options, index=default_index)
    t_ref_options = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    with col_sel_ref:
        risk_free_maturity = st.selectbox(
            "Référence r(T)",
            t_ref_options,
            index=t_ref_options.index(1.0),
            format_func=lambda x: f"{x:.2f}y",
        )

    snapshot = yc.get_curve_snapshot(
        currency=currency, risk_free_maturity=risk_free_maturity, ensure_cache=True
    )

    nodes = snapshot.get("nodes") or []
    grid = snapshot.get("grid") or []
    ns_curve = snapshot.get("ns_curve") or []
    risk_free_rate = snapshot.get("risk_free_rate")
    source_path = snapshot.get("source_path")
    source_kind = snapshot.get("source_kind") or "cache"
    last_updated = snapshot.get("last_updated")
    currency = snapshot.get("currency") or currency

    render_page_header(
        "Yield Curve",
        "Visualisation des taux zc et facteurs d'actualisation (lecture seule)",
        icon="🧭",
        badge="Rates",
    )

    source_label = source_path if source_path else "data/yield_curves/*_nodes.(csv|json)"
    meta = f"Devise: {currency} • Source: {source_kind} -> {source_label}"
    if last_updated:
        meta += f" • Last update: {last_updated}"
    if source_kind == "flat_fallback":
        st.warning("Données indisponibles: courbe plate (DEFAULT_RF_RATE) utilisée en secours.")
    st.caption(meta)

    col_rf, col_ref = st.columns([2, 1])
    with col_rf:
        if risk_free_rate is not None:
            st.metric(
                f"Risk-free rate r(T) ({currency})",
                f"{float(risk_free_rate) * 100:.2f} %",
                f"T = {float(risk_free_maturity):.2f}y",
            )
        else:
            st.warning("Risk-free rate indisponible.")
    with col_ref:
        st.write("")
        st.write("Taux utilisé par les pricers (zc).")

    st.markdown("#### Nœuds de courbe (zc/df)")
    df_nodes = pd.DataFrame(nodes)
    if not df_nodes.empty:
        st.dataframe(
            df_nodes[["tenor", "t_years", "zero_rate", "discount_factor"]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Aucun nœud trouvé, courbe plate par défaut (DEFAULT_RF_RATE).")

    st.markdown("#### Zero rates (observed vs Nelson–Siegel)")
    df_grid = pd.DataFrame(grid)
    df_ns = pd.DataFrame(ns_curve)
    if not df_nodes.empty or not df_ns.empty:
        df_nodes_plot = df_nodes.copy()
        if "t_years" in df_nodes_plot.columns and "zero_rate" in df_nodes_plot.columns:
            df_nodes_plot["rate_pct"] = df_nodes_plot["zero_rate"].astype(float) * 100.0
        else:
            df_nodes_plot = pd.DataFrame()

        df_ns_plot = df_ns.copy()
        if "t_years" in df_ns_plot.columns and "zero_rate" in df_ns_plot.columns:
            df_ns_plot["rate_pct"] = df_ns_plot["zero_rate"].astype(float) * 100.0
        else:
            df_ns_plot = pd.DataFrame()

        x_enc = alt.X("t_years:Q", title="Maturité (années)")
        y_enc = alt.Y("rate_pct:Q", title="Taux zéro-coupon (%)")

        charts = []
        if not df_nodes_plot.empty:
            charts.append(
                alt.Chart(df_nodes_plot)
                .mark_circle(size=70, color="#ffcc66")
                .encode(x=x_enc, y=y_enc, tooltip=["tenor:N", "t_years:Q", "rate_pct:Q"])
            )
        if not df_ns_plot.empty:
            charts.append(
                alt.Chart(df_ns_plot)
                .mark_line(color="#00e5ff")
                .encode(x=x_enc, y=y_enc)
            )
        if charts:
            chart = charts[0]
            for ch in charts[1:]:
                chart = chart + ch
            chart = chart.properties(height=260)
            st.altair_chart(chart, use_container_width=True)

        if not df_grid.empty:
            df_grid = df_grid.sort_values("t_years").set_index("t_years")
            st.dataframe(
                df_grid.reset_index().rename(
                    columns={"t_years": "T (years)", "zero_rate": "Zero rate", "discount_factor": "DF"}
                ),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Aucune courbe à afficher. Ajoute des fichiers *_nodes.csv sous data/yield_curves/.")

    st.markdown("#### Discount factors")
    if not df_grid.empty:
        df_df = df_grid.copy()
        if "t_years" in df_df.columns:
            df_df = df_df.sort_values("t_years").set_index("t_years")
        st.area_chart(df_df[["discount_factor"]], height=220)
    else:
        st.info("Pas de discount factors calculables sans courbe.")
