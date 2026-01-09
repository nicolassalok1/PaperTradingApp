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
    preferred = ["USD", "EUR"]
    target_currencies = [ccy for ccy in preferred if ccy in currencies] or currencies or ["USD"]
    risk_free_maturity = 1.0  # fixed ref to declutter UI

    snapshots: list[dict] = []
    for ccy in target_currencies:
        snapshots.append(
            yc.get_curve_snapshot(
                currency=ccy, risk_free_maturity=risk_free_maturity, ensure_cache=True
            )
        )

    active_snapshot = snapshots[0]
    nodes = active_snapshot.get("nodes") or []
    grid = active_snapshot.get("grid") or []
    forwards = active_snapshot.get("forward_rates") or []
    inst_curve = active_snapshot.get("inst_curve") or []
    ns_params = active_snapshot.get("ns_params") or {}
    ns_curve = active_snapshot.get("ns_curve") or []
    currency = active_snapshot.get("currency") or target_currencies[0]
    source_path = active_snapshot.get("source_path")
    source_kind = active_snapshot.get("source_kind") or "cache"
    last_updated = active_snapshot.get("last_updated")

    render_page_header(
        "Yield Curve (courbe des taux)",
        "Vue rapide multi-devises (USD/EUR) des courbes ZC/DF et gestion des nodes / cache.",
        icon="🧮",
        badge="Rates",
    )

    source_label = source_path if source_path else "data/yield_curves/*_nodes.(csv|json)"
    if source_kind == "flat_fallback":
        st.warning("Données indisponibles: courbe plate (DEFAULT_RF_RATE) utilisée en secours.")

    # Build combined datasets to plot multiple currencies on one chart.
    nodes_frames = []
    ns_frames = []
    for snap in snapshots:
        ccy = snap.get("currency") or "N/A"
        df_n = pd.DataFrame(snap.get("nodes") or [])
        if not df_n.empty and {"t_years", "zero_rate"}.issubset(df_n.columns):
            df_n = df_n.copy()
            df_n["rate_pct"] = pd.to_numeric(df_n["zero_rate"], errors="coerce") * 100.0
            df_n["currency"] = ccy
            nodes_frames.append(df_n[["tenor", "t_years", "rate_pct", "currency"]])

        df_ns_local = pd.DataFrame(snap.get("ns_curve") or [])
        if not df_ns_local.empty and {"t_years", "zero_rate"}.issubset(df_ns_local.columns):
            df_ns_local = df_ns_local.copy()
            df_ns_local["rate_pct"] = pd.to_numeric(df_ns_local["zero_rate"], errors="coerce") * 100.0
            df_ns_local["currency"] = ccy
            ns_frames.append(df_ns_local[["t_years", "rate_pct", "currency"]])

    df_nodes = pd.concat(nodes_frames, ignore_index=True) if nodes_frames else pd.DataFrame()
    df_ns = pd.concat(ns_frames, ignore_index=True) if ns_frames else pd.DataFrame()

    st.markdown("### Courbe de taux (vue rapide)")
    if not df_nodes.empty or not df_ns.empty:
        x_enc = alt.X("t_years:Q", title="Maturité (années)")
        y_enc = alt.Y("rate_pct:Q", title="Taux zéro-coupon (%)")
        color_enc = alt.Color("currency:N", title="Devise")

        charts = []
        if not df_nodes.empty:
            charts.append(
                alt.Chart(df_nodes)
                .mark_circle(size=70)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    color=color_enc,
                    tooltip=["currency:N", "tenor:N", "t_years:Q", "rate_pct:Q"],
                )
            )
        if not df_ns.empty:
            charts.append(
                alt.Chart(df_ns)
                .mark_line()
                .encode(
                    x=x_enc,
                    y=y_enc,
                    color=color_enc,
                    tooltip=["currency:N", "t_years:Q", "rate_pct:Q"],
                )
            )
        if charts:
            chart = charts[0]
            for ch in charts[1:]:
                chart = chart + ch
            chart = chart.properties(height=320)
            st.altair_chart(chart, width="stretch")
    else:
        st.info("Aucune courbe à afficher. Ajoute des fichiers *_nodes.csv sous data/yield_curves/.")

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

        st.markdown("##### Nelson-Siegel (paramètres)")
        if ns_params:
            st.json(ns_params)
        else:
            st.info("Pas assez de points pour calibrer Nelson–Siegel (>= 3 nœuds requis).")


def render() -> None:
    render_tab()
