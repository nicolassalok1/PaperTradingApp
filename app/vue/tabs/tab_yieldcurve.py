import pandas as pd
import streamlit as st
import altair as alt

from app.controller import yieldcurve_controller as yc
from app.vue.components.page_utils import render_page_header


TAB_LABEL = "🧮 Yield Curve"


def _to_pct(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce").astype(float) * 100.0
    except Exception:
        return series


def render_tab() -> None:
    currencies = yc.available_currencies()
    default_currency = "USD" if "USD" in currencies else (currencies[0] if currencies else "USD")
    currency_options = currencies or [default_currency]
    default_index = currency_options.index(default_currency) if default_currency in currency_options else 0

    col_sel_ccy, col_sel_ref = st.columns([1, 1])
    with col_sel_ccy:
        currency = st.selectbox("Currency", currency_options, index=default_index, key="yc_currency")
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
    forwards = snapshot.get("forward_rates") or []
    inst_curve = snapshot.get("inst_curve") or []
    ns_params = snapshot.get("ns_params") or {}
    ns_curve = snapshot.get("ns_curve") or []
    risk_free_rate = snapshot.get("risk_free_rate")
    source_path = snapshot.get("source_path")
    source_kind = snapshot.get("source_kind") or "cache"
    last_updated = snapshot.get("last_updated")
    currency = snapshot.get("currency") or currency

    render_page_header(
        "Yield Curve (courbe des taux)",
        "ZC/DF, forwards et taux instantanés — source: fichiers nodes / cache.",
        icon="🧮",
        badge="Rates",
    )

    source_label = source_path if source_path else "data/yield_curves/*_nodes.(csv|json)"
    meta = f"Devise: {currency} • Source: {source_kind} -> {source_label}"
    if last_updated:
        meta += f" • Last update: {last_updated}"
    if source_kind == "flat_fallback":
        st.warning("Données indisponibles: courbe plate (DEFAULT_RF_RATE) utilisée en secours.")
    st.caption(meta)

    with st.expander("Gestion de la courbe (import / édition)", expanded=False):
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("Invalidate cache", width="stretch"):
                yc.invalidate_curve_cache(currency)
                st.rerun()
        with col_b:
            if st.button("Refresh API (USD/EUR)", width="stretch"):
                ok = yc.refresh_curve_cache_from_api(currency)
                if ok:
                    st.success("Courbe rafraîchie depuis l'API.")
                    st.rerun()
                st.warning("Refresh API indisponible (active `YIELD_CURVE_ENABLE_API=1`) ou échec provider.")
        with col_c:
            uploaded = st.file_uploader(
                "Importer un fichier nodes (CSV/JSON) — sera sauvegardé en `<CCY>_nodes.csv`",
                type=["csv", "json"],
                key="yc_nodes_upload",
            )
            if uploaded is not None:
                raw = uploaded.getvalue()
                parsed = yc.parse_nodes_upload(uploaded.name, raw)
                df_parsed = pd.DataFrame(parsed)
                if df_parsed.empty:
                    st.error("Aucun nœud détecté dans ce fichier.")
                else:
                    st.dataframe(df_parsed, hide_index=True, width="stretch")
                if st.button(
                    f"Sauvegarder pour {currency}",
                    disabled=df_parsed.empty,
                    type="primary",
                    width="stretch",
                ):
                    res = yc.import_curve_nodes_upload(currency, uploaded.name, raw)
                    if res.get("success"):
                        st.success(res.get("message", "OK"))
                        st.rerun()
                    st.error(res.get("message", "Import échoué."))

        st.markdown("##### Éditeur rapide (nœuds)")
        df_nodes_export = pd.DataFrame(nodes)
        editor_cols = ["tenor", "t_years", "zero_rate"]
        df_nodes_editor = df_nodes_export.copy()
        for c in editor_cols:
            if c not in df_nodes_editor.columns:
                df_nodes_editor[c] = None
        df_nodes_editor = df_nodes_editor[editor_cols]

        df_edit = st.data_editor(
            df_nodes_editor,
            num_rows="dynamic",
            width="stretch",
            key=f"yc_nodes_editor_{currency}",
        )
        if st.button("Sauvegarder l'éditeur", width="stretch", disabled=df_edit.empty):
            result = yc.save_curve_nodes(currency, df_edit.to_dict(orient="records"))
            if result.get("success"):
                st.success(f"Sauvegardé: {result.get('path')}")
                st.rerun()
            st.error(result.get("message", "Sauvegarde échouée."))

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
            width="stretch",
        )
    else:
        st.info("Aucun nœud trouvé, courbe plate par défaut (DEFAULT_RF_RATE).")

    with st.expander("Nelson–Siegel (paramètres)", expanded=False):
        if ns_params:
            st.json(ns_params)
        else:
            st.info("Pas assez de points pour calibrer Nelson–Siegel (>= 3 nœuds requis).")

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
            st.altair_chart(chart, width="stretch")

        if not df_grid.empty:
            df_grid = df_grid.sort_values("t_years").set_index("t_years")
            st.dataframe(
                df_grid.reset_index().rename(
                    columns={"t_years": "T (years)", "zero_rate": "Zero rate", "discount_factor": "DF"}
                ),
                hide_index=True,
                width="stretch",
            )
    else:
        st.info("Aucune courbe à afficher. Ajoute des fichiers *_nodes.csv sous data/yield_curves/.")

    st.markdown("#### Discount factors")
    if not df_grid.empty:
        df_df = df_grid.copy()
        if "t_years" not in df_df.columns or "discount_factor" not in df_df.columns:
            st.info("Pas de discount factors calculables sans courbe.")
        else:
            df_df = df_df.sort_values("t_years").copy()
            df_df["discount_factor"] = pd.to_numeric(df_df["discount_factor"], errors="coerce")
            df_df = df_df.dropna(subset=["t_years", "discount_factor"])

            x_axis = alt.Axis(
                tickCount=12,
                labelOverlap="greedy",
                labelExpr=(
                    "datum.value >= 1 ? "
                    "(abs(datum.value - round(datum.value)) < 1e-6 ? round(datum.value) + 'a' : format(datum.value, '.1f') + 'a') : "
                    "datum.value >= (1/12) ? "
                    "(abs(datum.value*12 - round(datum.value*12)) < 1e-6 ? round(datum.value*12) + 'm' : format(datum.value*12, '.1f') + 'm') : "
                    "(abs(datum.value*365 - round(datum.value*365)) < 1e-6 ? round(datum.value*365) + 'j' : format(datum.value*365, '.1f') + 'j')"
                ),
            )

            chart = (
                alt.Chart(df_df)
                .mark_area(color="#88ccff", opacity=0.65, line={"color": "#88ccff"})
                .encode(
                    x=alt.X("t_years:Q", title="Maturité", axis=x_axis),
                    y=alt.Y("discount_factor:Q", title="DF", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[
                        alt.Tooltip("t_years:Q", title="T (années)", format=".4f"),
                        alt.Tooltip("discount_factor:Q", title="Discount factor", format=".6f"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, width="stretch")
    else:
        st.info("Pas de discount factors calculables sans courbe.")

    st.markdown("#### Forwards & instantaneous")
    df_fwds = pd.DataFrame(forwards)
    if not df_fwds.empty and "forward_rate" in df_fwds.columns:
        df_fwds = df_fwds.copy()
        df_fwds["forward_pct"] = _to_pct(df_fwds["forward_rate"])
        st.dataframe(
            df_fwds.rename(
                columns={
                    "start_years": "Start (y)",
                    "end_years": "End (y)",
                    "forward_pct": "Forward rate (%)",
                }
            )[["Start (y)", "End (y)", "Forward rate (%)"]],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Forward rates indisponibles.")

    df_inst = pd.DataFrame(inst_curve)
    if not df_inst.empty and "inst_forward_rate" in df_inst.columns:
        df_inst = df_inst.copy()
        df_inst["fwd_pct"] = _to_pct(df_inst["inst_forward_rate"])
        st.altair_chart(
            alt.Chart(df_inst)
            .mark_line(color="#ff66cc")
            .encode(
                x=alt.X("t_years:Q", title="Maturité (années)"),
                y=alt.Y("fwd_pct:Q", title="Instantaneous forward (%)"),
            )
            .properties(height=220),
            width="stretch",
        )
    else:
        st.info("Instantaneous forward curve indisponible.")


def render() -> None:
    render_tab()
