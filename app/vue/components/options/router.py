import streamlit as st

import pandas as pd
import plotly.graph_objects as go

from app.controller import options_controller as opt_ctrl
from app.vue.components.options.panels.tab_vanilla import render_panel_vanilla
from app.vue.components.options.panels.tab_path import render_panel_path
from app.vue.components.options.panels.tab_barrier import render_panel_barrier
from app.vue.components.options.panels.tab_spreads import render_panel_spreads
from app.vue.components.options.panels.tab_calendar import render_panel_calendar
from app.vue.components.options.panels.tab_exotics import render_panel_exotics
from app.vue.components.options.controller_bridge import *


_IV_SURFACE_DF_KEY = "opt_iv_surface_df"
_IV_SURFACE_TKR_KEY = "opt_iv_surface_ticker"


def _to_float(val, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _ensure_global_defaults() -> None:
    st.session_state.setdefault("common_maturity_value", 1.0)


def _ensure_iv_surface_defaults() -> None:
    # Normalize legacy values safely before widgets are created.
    if "opt_iv_surface_source" in st.session_state:
        try:
            src_raw = str(st.session_state.get("opt_iv_surface_source") or "").strip()
            src_low = src_raw.lower()
            src_norm = "Calibration" if src_low.startswith("calib") else "Yahoo"
            if st.session_state.get("opt_iv_surface_source") != src_norm:
                st.session_state["opt_iv_surface_source"] = src_norm
        except Exception:
            pass

    if "opt_iv_surface_type" in st.session_state:
        try:
            typ_raw = str(st.session_state.get("opt_iv_surface_type") or "").strip()
            typ_low = typ_raw.lower()
            typ_norm = "Put" if typ_low.startswith("p") else "Call"
            if st.session_state.get("opt_iv_surface_type") != typ_norm:
                st.session_state["opt_iv_surface_type"] = typ_norm
        except Exception:
            pass

    if "opt_iv_surface_max_years" in st.session_state:
        try:
            max_years = float(st.session_state.get("opt_iv_surface_max_years"))
        except Exception:
            return
        max_years = min(2.0, max(0.25, max_years))
        if st.session_state.get("opt_iv_surface_max_years") != max_years:
            st.session_state["opt_iv_surface_max_years"] = max_years


def _render_global_params() -> None:
    _ensure_global_defaults()

    with st.expander("Parametres globaux (r / q / sigma)", expanded=True):
        col_r, col_q, col_sig = st.columns([1, 1, 1])
        currency = (st.session_state.get("yc_currency") or "USD").strip().upper()

        with col_r:
            toggle_kwargs = {"value": True} if "opt_use_yield_curve_rate" not in st.session_state else {}
            st.toggle("r depuis Yield Curve", key="opt_use_yield_curve_rate", **toggle_kwargs)
            st.caption(
                f"YC: {currency} (r(T) est resolu dans chaque panneau selon le T selectionne)"
            )

            rate_kwargs = {"value": 0.02} if "common_rate_value" not in st.session_state else {}
            st.number_input(
                "Taux sans risque r",
                min_value=-0.50,
                max_value=1.00,
                step=0.001,
                format="%.6f",
                key="common_rate_value",
                disabled=bool(st.session_state.get("opt_use_yield_curve_rate", True)),
                **rate_kwargs,
            )

        with col_q:
            q_kwargs = {"value": 0.00} if "d_common" not in st.session_state else {}
            st.number_input(
                "Dividend yield q",
                min_value=-0.50,
                max_value=1.00,
                step=0.001,
                format="%.6f",
                key="d_common",
                **q_kwargs,
            )

        with col_sig:
            sig_kwargs = {"value": 0.20} if "common_sigma_value" not in st.session_state else {}
            st.number_input(
                "Volatilite sigma",
                min_value=0.0001,
                max_value=5.0,
                step=0.01,
                format="%.4f",
                key="common_sigma_value",
                **sig_kwargs,
            )


def _pivot_surface(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    k_col = cols.get("k") or cols.get("strike") or "K"
    t_col = cols.get("t") or cols.get("maturity") or "T"
    iv_col = cols.get("iv") or cols.get("iv_market") or cols.get("sigma") or "iv"

    df_clean = df.dropna(subset=[k_col, t_col, iv_col]).copy()
    df_clean["T_days"] = (
        (pd.to_numeric(df_clean[t_col], errors="coerce") * 365.0).round().astype("Int64")
    )
    df_clean[k_col] = pd.to_numeric(df_clean[k_col], errors="coerce")
    df_clean[iv_col] = pd.to_numeric(df_clean[iv_col], errors="coerce")
    df_clean = df_clean.dropna(subset=["T_days", k_col, iv_col])
    if df_clean.empty:
        return pd.DataFrame()

    surf = (
        df_clean.pivot_table(index="T_days", columns=k_col, values=iv_col, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    try:
        surf = surf.interpolate(axis=1, limit_direction="both").interpolate(
            axis=0, limit_direction="both"
        )
    except Exception:
        pass
    return surf


def _render_iv_surface_section(ticker: str) -> None:
    _ensure_iv_surface_defaults()

    tkr = (ticker or "").strip().upper()
    if not tkr:
        return

    st.markdown("### IV marche (surface)")
    st.caption("Source: Yahoo (marche) ou Calibration (Heston).")

    col_src, col_a, col_b = st.columns([1, 1, 1])
    with col_src:
        src_default_idx = 0
        if str(st.session_state.get("opt_iv_surface_source") or "").strip().lower().startswith("calib"):
            src_default_idx = 1
        src_kwargs = {"index": src_default_idx} if "opt_iv_surface_source" not in st.session_state else {}
        st.selectbox("Source", ["Yahoo", "Calibration"], key="opt_iv_surface_source", **src_kwargs)
    with col_a:
        typ_default_idx = 0
        if str(st.session_state.get("opt_iv_surface_type") or "").strip().lower().startswith("p"):
            typ_default_idx = 1
        typ_kwargs = {"index": typ_default_idx} if "opt_iv_surface_type" not in st.session_state else {}
        st.selectbox("Type", ["Call", "Put"], key="opt_iv_surface_type", **typ_kwargs)
    with col_b:
        slider_kwargs = {"value": 2.0} if "opt_iv_surface_max_years" not in st.session_state else {}
        st.slider(
            "Max maturite (annees)",
            min_value=0.25,
            max_value=2.0,
            step=0.25,
            key="opt_iv_surface_max_years",
            **slider_kwargs,
        )

    max_years = float(st.session_state.get("opt_iv_surface_max_years", 2.0))
    source_kind = str(st.session_state.get("opt_iv_surface_source") or "Yahoo").strip()

    def _render_surface(df_in: pd.DataFrame, title_suffix: str) -> None:
        df = df_in.copy()
        if "type" in df.columns:
            cp = (
                "c"
                if str(st.session_state.get("opt_iv_surface_type", "Call"))
                .lower()
                .startswith("c")
                else "p"
            )
            df = df[df["type"].astype(str).str.lower().str.startswith(cp)]

        surf = _pivot_surface(df)
        if surf.empty:
            st.info("Surface IV vide apres filtrage.")
            return

        fig = go.Figure(
            data=go.Surface(
                z=surf.values,
                x=[float(x) for x in surf.columns],
                y=[int(y) for y in surf.index],
                colorscale="Viridis",
                showscale=True,
            )
        )
        fig.update_layout(
            title=(
                f"{tkr} - IV surface ({st.session_state.get('opt_iv_surface_type')}) {title_suffix}"
            ),
            scene=dict(
                xaxis_title="Strike K",
                yaxis_title="TTM (jours)",
                zaxis_title="IV",
            ),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    if source_kind.lower().startswith("calib"):
        df_cal = st.session_state.get("calib_model_surface_df")
        meta = st.session_state.get("calib_model_surface_meta") or {}
        if df_cal is None or getattr(df_cal, "empty", True):
            st.info(
                "Aucune surface Calibration disponible. Lance une calibration puis clique "
                "'Envoyer IV modele vers Options' dans l'onglet Calibration."
            )
            return

        df_iv = df_cal.copy()
        if "T" in df_iv.columns:
            try:
                df_iv["T"] = pd.to_numeric(df_iv["T"], errors="coerce")
                df_iv = df_iv.dropna(subset=["T"])
                df_iv = df_iv[df_iv["T"] <= max_years]
            except Exception:
                pass

        tkr_cal = str(meta.get("ticker") or "").strip().upper()
        if tkr_cal and tkr_cal != tkr:
            st.warning(f"Surface calibration: {tkr_cal} (ticker courant: {tkr}).")

        _render_surface(df_iv, title_suffix="(Calibration)")
        return

    # --- Market (Yahoo) ---
    if st.session_state.get(_IV_SURFACE_TKR_KEY) != tkr:
        st.session_state[_IV_SURFACE_TKR_KEY] = tkr
        st.session_state[_IV_SURFACE_DF_KEY] = None

    refresh = st.button("Refresh IV surface", use_container_width=True, type="secondary")
    df_iv = st.session_state.get(_IV_SURFACE_DF_KEY)

    if refresh:
        try:
            with st.spinner(f"Chargement IV surface (Yahoo) pour {tkr}..."):
                df_iv = opt_ctrl.fetch_iv_surface(
                    tkr,
                    max_maturity_years=float(st.session_state.get("opt_iv_surface_max_years", 2.0)),
                )
            st.session_state[_IV_SURFACE_DF_KEY] = df_iv
        except Exception as exc:
            st.error(f"IV surface indisponible: {exc}")
            st.session_state[_IV_SURFACE_DF_KEY] = None
            return

    if df_iv is None:
        st.info("Clique sur 'Refresh IV surface' pour charger la surface IV.")
        return
    if getattr(df_iv, "empty", True):
        st.info("Surface IV vide.")
        return

    _render_surface(df_iv, title_suffix="(Yahoo)")


def render_options_router():
    st.header("🧮 Options - Interface Professionnelle")

    st.markdown(
        """
        <style>
        [data-testid="stTabs"] button[role="tab"] {
            padding: 0.35rem 0.65rem !important;
            font-size: 0.85rem !important;
            min-height: 2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _render_global_params()

    tkr_common = st.text_input(
        "Ticker commun pour les historiques IV/cl“tures (optionnel)",
        value=st.session_state.get("tkr_common", ""),
        placeholder="ex: AAPL",
    )
    tkr_common_norm = (tkr_common or "").strip().upper()
    st.session_state["tkr_common"] = tkr_common_norm
    st.session_state["common_underlying"] = tkr_common_norm

    ctx = get_option_context()
    if ctx.get("S0") is not None:
        st.session_state["common_spot_value"] = ctx["S0"]

    close_series = ctx.get("close_series")
    close_available = bool(ctx.get("close_available"))
    if close_available and close_series is not None and hasattr(close_series, "empty") and not close_series.empty:
        tkr_label = ctx.get("ticker") or tkr_common_norm or "Ticker"
        render_static_line_chart(
            close_series,
            title=f"{tkr_label} - Clotures (cache)",
            y_label="Prix de cloture",
        )
    else:
        st.info(
            "Aucune cloture disponible pour ce ticker (cache OHLC introuvable). "
            "Verifie le symbole puis reessaie."
        )

    _render_iv_surface_section(ctx.get("ticker") or tkr_common_norm)

    families = [
        "Vanilla / Early Exercise",
        "Path-dependent",
        "BarriSres",
        "Spreads & Wings",
        "Calendriers",
        "Exotiques avanc'es",
    ]

    family_tabs = st.tabs(families)
    for fam_label, fam_tab in zip(families, family_tabs):
        with fam_tab:
            if fam_label == "Vanilla / Early Exercise":
                render_panel_vanilla()
            elif fam_label == "Path-dependent":
                render_panel_path()
            elif fam_label == "BarriSres":
                render_panel_barrier()
            elif fam_label == "Spreads & Wings":
                render_panel_spreads()
            elif fam_label == "Calendriers":
                render_panel_calendar()
            elif fam_label == "Exotiques avanc'es":
                render_panel_exotics()
            else:
                st.error("Famille inconnue.")
