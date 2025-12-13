import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controller.calibration_controller import CalibrationController
from app.controller import yieldcurve_controller as yc
from app.model.options.data.iv_surface import fetch_iv_surface as _fetch_iv_surface
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🧮 Calibration"

_CHAIN_STATE_KEY = "calib_alpaca_chain_df"
_CHAIN_TICKER_KEY = "calib_alpaca_chain_ticker"
_TICKERS_STATE_KEY = "calib_alpaca_underlyings"
_YAHOO_SURFACE_STATE_KEY = "calib_yahoo_surface_df"
_YAHOO_SURFACE_TICKER_KEY = "calib_yahoo_surface_ticker"
_YAHOO_SURFACE_MAX_YEARS_KEY = "calib_yahoo_surface_max_years"
_OPTIONABLE_TICKERS_CSV = Path(
    os.getenv("ALPACA_OPTIONABLE_TICKERS_PATH", "data/alpaca_optionable_tickers.csv")
)

_SURF_K_ALIASES = {"k", "strike", "strike_price", "strikeprice"}
_SURF_T_ALIASES = {"t", "ttm", "tau", "time_to_maturity", "maturity"}
_SURF_S0_ALIASES = {"s0", "spot", "underlying", "underlyingprice"}
_SURF_IV_ALIASES = {"iv", "implied_vol", "impliedvol", "implied_volatility", "impliedvolatility", "sigma"}
_SURF_TYPE_ALIASES = {"type", "option_type", "cp", "right"}


def _plot_heatmap(z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title="IV"),
        )
    )
    fig.update_layout(title=title, xaxis_title="Moneyness", yaxis_title="Time to maturity (y)")
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "scrollZoom": False})


def _grid_to_surface_df(
    *,
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_grid: np.ndarray,
    opt_type: str = "call",
) -> pd.DataFrame:
    m = np.asarray(m_grid, dtype=float).reshape(1, -1)
    t = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    iv = np.asarray(iv_grid, dtype=float)
    if iv.ndim != 2 or iv.shape != (t.shape[0], m.shape[1]):
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    rows = []
    for i_t in range(iv.shape[0]):
        for j_m in range(iv.shape[1]):
            v = float(iv[i_t, j_m])
            if not np.isfinite(v) or v <= 0:
                continue
            T = float(t_grid[i_t])
            K = float(float(S0) * float(m_grid[j_m]))
            rows.append({"K": K, "T": T, "S0": float(S0), "iv": v, "type": str(opt_type)})

    return pd.DataFrame(rows, columns=["K", "T", "S0", "iv", "type"])


def _discover_cached_surfaces(cache_dir: Path) -> list[Path]:
    def _is_surface_csv(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                header = (f.readline() or "").strip().lower()
        except Exception:
            return False
        if not header:
            return False
        cols = {c.strip() for c in header.split(",")}
        return {"k", "t", "s0", "iv"}.issubset(cols)

    if not cache_dir.exists():
        return []
    files = [p for p in cache_dir.glob("*.csv") if p.is_file() and _is_surface_csv(p)]
    return sorted(files, key=lambda p: p.name.lower())


def _median_s0(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "S0" not in df.columns:
        return None
    s0_vals = pd.to_numeric(df["S0"], errors="coerce")
    s0_pos = s0_vals[s0_vals > 0]
    if s0_pos.empty:
        return None
    try:
        return float(s0_pos.median())
    except Exception:
        return None


def _load_optionable_tickers() -> list[str]:
    tickers = st.session_state.get(_TICKERS_STATE_KEY)
    if isinstance(tickers, list) and tickers:
        return tickers

    tickers = []
    try:
        if _OPTIONABLE_TICKERS_CSV.exists():
            df = pd.read_csv(_OPTIONABLE_TICKERS_CSV)
            if not df.empty:
                col = "symbol" if "symbol" in df.columns else df.columns[0]
                syms = df[col].dropna().astype(str)
                tickers = sorted({s.strip().upper() for s in syms if s.strip()})
    except Exception:
        tickers = []

    st.session_state[_TICKERS_STATE_KEY] = tickers
    return tickers


def _get_chain_from_state() -> pd.DataFrame | None:
    df = st.session_state.get(_CHAIN_STATE_KEY)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _parse_constraints(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def _find_surface_col(df: pd.DataFrame, aliases: set[str]):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in cols:
            return cols[alias]
    return None


def _canonicalize_surface_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    k_col = _find_surface_col(df, _SURF_K_ALIASES)
    t_col = _find_surface_col(df, _SURF_T_ALIASES)
    s0_col = _find_surface_col(df, _SURF_S0_ALIASES)
    iv_col = _find_surface_col(df, _SURF_IV_ALIASES)
    typ_col = _find_surface_col(df, _SURF_TYPE_ALIASES)

    if not (k_col and t_col and s0_col and iv_col):
        return pd.DataFrame(columns=["K", "T", "S0", "iv", "type"])

    out = pd.DataFrame(
        {
            "K": pd.to_numeric(df[k_col], errors="coerce"),
            "T": pd.to_numeric(df[t_col], errors="coerce"),
            "S0": pd.to_numeric(df[s0_col], errors="coerce"),
            "iv": pd.to_numeric(df[iv_col], errors="coerce"),
        }
    )
    if typ_col is not None:
        typ = df[typ_col].astype(str).str.lower().str.strip()
        typ = typ.replace({"c": "call", "p": "put", "calll": "call"})
        out["type"] = typ
    else:
        out["type"] = "call"

    out = out.dropna(subset=["K", "T", "S0", "iv"])
    out = out[(out["K"] > 0) & (out["T"] > 0) & (out["S0"] > 0) & (out["iv"] > 0)]
    out = out.reset_index(drop=True)
    return out[["K", "T", "S0", "iv", "type"]]


def _render_surface_filters(df_canon: pd.DataFrame) -> pd.DataFrame:
    if df_canon is None or df_canon.empty:
        st.info("Aucune surface pour filtrage.")
        return df_canon

    dfw = df_canon.copy()
    dfw["moneyness"] = dfw["K"] / dfw["S0"]
    dfw = dfw.dropna(subset=["moneyness"])
    if dfw.empty:
        st.warning("Impossible de calculer la moneyness (K/S0).")
        return df_canon

    m_min = float(dfw["moneyness"].min())
    m_max = float(dfw["moneyness"].max())
    t_min = float(dfw["T"].min())
    t_max = float(dfw["T"].max())
    iv_min = float(dfw["iv"].min())
    iv_max = float(dfw["iv"].max())

    if m_min == m_max:
        m_max = m_min + 1e-6
    if t_min == t_max:
        t_max = t_min + 1e-6
    if iv_min == iv_max:
        iv_max = iv_min + 1e-6

    st.caption(f"Avant filtre: {len(dfw):,} lignes")

    def _clamp_range(key: str, lo: float, hi: float) -> None:
        prev = st.session_state.get(key)
        if isinstance(prev, (list, tuple)) and len(prev) == 2:
            try:
                a = float(prev[0])
                b = float(prev[1])
            except Exception:
                return
            a = max(lo, min(hi, a))
            b = max(lo, min(hi, b))
            if a > b:
                a, b = b, a
            st.session_state[key] = (float(a), float(b))

    _clamp_range("calib_filter_moneyness", m_min, m_max)
    _clamp_range("calib_filter_ttm", t_min, t_max)
    _clamp_range("calib_filter_iv", iv_min, iv_max)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        calls_only = st.checkbox("CALL only", value=True, key="calib_filter_calls_only")
        max_rows = int(st.number_input("Max rows (sample, 0=off)", min_value=0, max_value=200000, value=0, step=1000))
        quantile_on = st.checkbox("Filtrer IV par quantiles", value=False, key="calib_filter_iv_quantile_on")
    with col_b:
        m_rng = st.slider(
            "Moneyness (K/S0)",
            min_value=float(m_min),
            max_value=float(m_max),
            value=st.session_state.get("calib_filter_moneyness", (float(m_min), float(m_max))),
            step=0.01,
            key="calib_filter_moneyness",
        )
        t_rng = st.slider(
            "TTM T (années)",
            min_value=float(t_min),
            max_value=float(t_max),
            value=st.session_state.get("calib_filter_ttm", (float(t_min), float(t_max))),
            step=0.01,
            key="calib_filter_ttm",
        )
        iv_rng = st.slider(
            "IV",
            min_value=float(iv_min),
            max_value=float(iv_max),
            value=st.session_state.get("calib_filter_iv", (float(iv_min), float(iv_max))),
            step=0.01,
            key="calib_filter_iv",
        )

    q_low, q_high = 0.01, 0.99
    if quantile_on:
        q_low, q_high = st.slider(
            "Quantiles IV (low/high)",
            min_value=0.0,
            max_value=1.0,
            value=(0.01, 0.99),
            step=0.01,
            key="calib_filter_iv_quantiles",
        )

    df_f = dfw.copy()
    if calls_only and "type" in df_f.columns:
        df_f = df_f[df_f["type"].astype(str).str.lower().str.startswith("c")]
    df_f = df_f[
        (df_f["moneyness"] >= float(m_rng[0]))
        & (df_f["moneyness"] <= float(m_rng[1]))
        & (df_f["T"] >= float(t_rng[0]))
        & (df_f["T"] <= float(t_rng[1]))
        & (df_f["iv"] >= float(iv_rng[0]))
        & (df_f["iv"] <= float(iv_rng[1]))
    ]

    if quantile_on and not df_f.empty:
        try:
            lo = float(df_f["iv"].quantile(float(q_low)))
            hi = float(df_f["iv"].quantile(float(q_high)))
            df_f = df_f[(df_f["iv"] >= lo) & (df_f["iv"] <= hi)]
        except Exception:
            pass

    if max_rows > 0 and len(df_f) > max_rows:
        df_f = df_f.sample(max_rows, random_state=42).reset_index(drop=True)

    st.caption(f"Après filtre: {len(df_f):,} lignes")
    st.dataframe(df_f.head(30), hide_index=True, use_container_width=True)
    try:
        st.download_button(
            "Télécharger la surface filtrée (CSV)",
            data=df_f[["K", "T", "S0", "iv", "type"]].to_csv(index=False).encode("utf-8"),
            file_name="surface_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception:
        pass

    return df_f[["K", "T", "S0", "iv", "type"]]


def _render_constraints_builder(default_bounds: Dict[str, Any]) -> Dict[str, Any]:
    st.caption("Définissez des bornes ou fixez des paramètres (optionnel).")
    params = ["kappa", "theta", "sigma", "rho", "v0"]
    constraints: Dict[str, Any] = {}

    for name in params:
        bounds = default_bounds.get(name) or [None, None]
        try:
            b_lo = float(bounds[0]) if bounds[0] is not None else None
            b_hi = float(bounds[1]) if bounds[1] is not None else None
        except Exception:
            b_lo, b_hi = None, None

        col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 1.0])
        with col_a:
            mode = st.selectbox(
                name,
                options=["Default", "Bornes", "Fixe"],
                index=0,
                key=f"calib_constr_mode_{name}",
            )
        if mode == "Bornes":
            with col_b:
                mn = st.number_input(
                    f"{name} min",
                    value=(b_lo if b_lo is not None else 0.0),
                    key=f"calib_constr_min_{name}",
                )
            with col_c:
                mx = st.number_input(
                    f"{name} max",
                    value=(b_hi if b_hi is not None else 1.0),
                    key=f"calib_constr_max_{name}",
                )
            constraints[name] = [float(mn), float(mx)]
            with col_d:
                st.caption("bornes")
        elif mode == "Fixe":
            with col_b:
                if b_lo is not None and b_hi is not None:
                    default_v = 0.5 * (b_lo + b_hi)
                elif b_lo is not None:
                    default_v = float(b_lo)
                else:
                    default_v = 0.0
                v = st.number_input(
                    f"{name} value",
                    value=float(default_v),
                    key=f"calib_constr_val_{name}",
                )
            constraints[name] = float(v)
            with col_c:
                st.write("")
            with col_d:
                st.caption("fixe")
        else:
            with col_b:
                st.write("")
            with col_c:
                st.write("")
            with col_d:
                st.caption("—")

    swapped: list[str] = []
    for name, val in list(constraints.items()):
        if isinstance(val, list) and len(val) == 2:
            try:
                mn = float(val[0])
                mx = float(val[1])
            except Exception:
                continue
            if mn > mx:
                swapped.append(name)
                constraints[name] = [mx, mn]
    if swapped:
        st.warning("Bornes inversées corrigées: " + ", ".join(swapped))

    st.markdown("**JSON généré**")
    st.code(json.dumps(constraints, indent=2, ensure_ascii=False), language="json")

    return constraints


def _download_surface_template() -> None:
    df = pd.DataFrame(
        [
            {"K": 100.0, "T": 0.25, "S0": 100.0, "iv": 0.20, "type": "call"},
            {"K": 105.0, "T": 0.25, "S0": 100.0, "iv": 0.22, "type": "call"},
            {"K": 95.0, "T": 1.00, "S0": 100.0, "iv": 0.25, "type": "call"},
        ],
        columns=["K", "T", "S0", "iv", "type"],
    )
    st.download_button(
        "Télécharger un template CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="surface_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_surface_preview_dropdown(df: pd.DataFrame, *, label: str, key: str) -> None:
    if df is None or df.empty:
        st.info("Surface vide.")
        return

    def _render() -> None:
        st.dataframe(df.head(30), hide_index=True, use_container_width=True)
        if len(df) > 30:
            st.caption(f"Aperçu: 30 premières lignes affichées (sur {len(df):,}).")
        st.divider()
        _surface_diagnostics(df)

    title = f"{label} ({len(df):,} lignes)"
    expander = None
    try:
        expander = st.expander(title, expanded=False)
    except Exception:
        expander = None

    if expander is not None:
        with expander:
            _render()
        return

    if hasattr(st, "popover"):
        with st.popover(title):
            _render()
        return

    if st.checkbox(title, value=False, key=key):
        _render()


def _surface_diagnostics(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("Aucune donnée à diagnostiquer.")
        return

    st.caption(f"{len(df):,} lignes | colonnes: {', '.join([str(c) for c in df.columns[:20]])}")

    # Basic sanity checks for expected schema
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    required = ["k", "t", "s0", "iv"]
    missing = [c.upper() for c in required if c not in cols_lower]
    if missing:
        st.warning(f"Colonnes manquantes pour calibration: {', '.join(missing)}")
        return

    k_col = cols_lower["k"]
    t_col = cols_lower["t"]
    s0_col = cols_lower["s0"]
    iv_col = cols_lower["iv"]
    typ_col = cols_lower.get("type")

    k = pd.to_numeric(df[k_col], errors="coerce")
    t = pd.to_numeric(df[t_col], errors="coerce")
    s0 = pd.to_numeric(df[s0_col], errors="coerce")
    iv = pd.to_numeric(df[iv_col], errors="coerce")

    dfw = pd.DataFrame({"K": k, "T": t, "S0": s0, "iv": iv})
    dfw = dfw.dropna(subset=["K", "T", "S0", "iv"])
    dfw = dfw[(dfw["K"] > 0) & (dfw["T"] > 0) & (dfw["S0"] > 0) & (dfw["iv"] > 0)]
    if dfw.empty:
        st.warning("Toutes les lignes ont été filtrées (valeurs non valides).")
        return

    dfw["moneyness"] = dfw["K"] / dfw["S0"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("K", f"[{dfw['K'].min():.2f} ; {dfw['K'].max():.2f}]")
    col_b.metric("T (années)", f"[{dfw['T'].min():.4f} ; {dfw['T'].max():.4f}]")
    col_c.metric("IV", f"[{dfw['iv'].min():.4f} ; {dfw['iv'].max():.4f}]")
    col_d.metric("Moneyness", f"[{dfw['moneyness'].min():.3f} ; {dfw['moneyness'].max():.3f}]")

    if typ_col is not None:
        try:
            counts = (
                df[typ_col]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({"c": "call", "p": "put"})
                .value_counts()
            )
            if not counts.empty:
                st.caption("Répartition `type`: " + ", ".join([f"{k}={int(v)}" for k, v in counts.items()]))
        except Exception:
            pass

    show_scatter = st.checkbox(
        "Afficher le nuage de points (moneyness x T, couleur=IV)",
        value=False,
        key="calib_diag_show_scatter",
    )
    if show_scatter:
        try:
            dfp = dfw.copy()
            if len(dfp) > 4000:
                dfp = dfp.sample(4000, random_state=42)
            fig = go.Figure(
                data=go.Scattergl(
                    x=dfp["moneyness"],
                    y=dfp["T"],
                    mode="markers",
                    marker=dict(
                        color=dfp["iv"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="IV"),
                        size=5,
                        opacity=0.7,
                    ),
                    text=[
                        f"K={k:.2f}, T={tt:.3f}, IV={vv:.3f}"
                        for k, tt, vv in zip(dfp["K"], dfp["T"], dfp["iv"])
                    ],
                    hoverinfo="text",
                )
            )
            fig.update_layout(xaxis_title="Moneyness (K/S0)", yaxis_title="T (years)", height=360)
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "scrollZoom": False})
        except Exception as exc:
            st.warning(f"Plot impossible: {exc}")


def render_tab() -> None:
    ctrl = CalibrationController()

    render_page_header(
        "Calibration",
        "Heston: calibration via optimisation (fallback) ou NN si poids disponibles",
        icon="🧮",
        badge="Models",
    )

    models = ctrl.get_models()
    default_model = "heston" if "heston" in models else (models[0] if models else "heston")
    col_model, col_method = st.columns([1, 1])
    with col_model:
        model_choice = st.selectbox(
            "Modèle",
            options=models or [default_model],
            index=(models.index(default_model) if default_model in models else 0),
            format_func=lambda v: str(v).upper(),
        )

    nn_info = ctrl.get_heston_nn_info()
    weights_exists = bool(nn_info.get("weights_exists"))
    with col_method:
        if weights_exists:
            method = st.selectbox(
                "Méthode",
                options=["least_squares", "neural_net"],
                index=1,
                format_func=lambda x: "Neural Net (rapide)" if x == "neural_net" else "Least Squares (sans poids)",
            )
        else:
            st.caption("Méthode")
            st.info("Poids NN absents → Least Squares")
            method = "least_squares"

    if model_choice != "heston":
        st.info("Sélectionnez HESTON pour lancer la calibration.")
        return

    constraints: Dict[str, Any] | None = None
    with st.expander("Contraintes", expanded=False):
        mode = st.radio(
            "Mode",
            options=["Builder", "JSON"],
            horizontal=True,
            index=0,
            key="calib_constraints_mode",
        )
        if mode == "Builder":
            default_bounds = ctrl.get_heston_default_bounds() or {}
            constraints = _render_constraints_builder(default_bounds)
        else:
            constraints_raw = st.text_area(
                "Contraintes (JSON)",
                value="{}",
                height=120,
                key="calib_constraints",
            )
            constraints = _parse_constraints(constraints_raw)
            if constraints_raw and constraints is None:
                st.warning("JSON invalide.")

    st.markdown("### Surface IV marché")
    _download_surface_template()
    surface_sources = ["Ticker (Yahoo)", "Upload CSV", "Cache (cache/*.csv)", "Alpaca (live)"]
    if st.session_state.get("calib_surface_source") not in surface_sources:
        st.session_state["calib_surface_source"] = surface_sources[0]
    source = st.radio("Source", options=surface_sources, horizontal=True, key="calib_surface_source")

    csv_bytes = None
    surface_path = None
    surface_df = None
    preview_df = None
    surface_ticker = None

    if source == "Ticker (Yahoo)":
        default_ticker = (
            st.session_state.get("calib_yahoo_ticker_input")
            or st.session_state.get("tkr_common")
            or st.session_state.get(_CHAIN_TICKER_KEY)
            or "AAPL"
        )
        col_ticker, col_years, col_load = st.columns([2, 2, 1])
        with col_ticker:
            ticker_raw = st.text_input(
                "Ticker",
                value=str(default_ticker),
                placeholder="ex: AAPL",
                key="calib_yahoo_ticker_input",
            )
        with col_years:
            try:
                max_years_default = float(st.session_state.get("calib_yahoo_max_years", 2.0))
            except Exception:
                max_years_default = 2.0
            max_years_default = min(2.0, max(0.25, max_years_default))
            max_years = st.slider(
                "Max maturité (années)",
                min_value=0.25,
                max_value=2.0,
                step=0.25,
                value=float(max_years_default),
                key="calib_yahoo_max_years",
            )
        with col_load:
            load_clicked = st.button("Load", use_container_width=True, key="calib_yahoo_load_btn")

        ticker = (ticker_raw or "").strip().upper()
        cached_df = st.session_state.get(_YAHOO_SURFACE_STATE_KEY)
        cached_ticker = str(st.session_state.get(_YAHOO_SURFACE_TICKER_KEY) or "").strip().upper()
        cached_max_years = st.session_state.get(_YAHOO_SURFACE_MAX_YEARS_KEY)

        if load_clicked and ticker:
            try:
                with st.spinner(f"Chargement surface (Yahoo) pour {ticker}..."):
                    surface_df = _fetch_iv_surface(ticker, max_maturity_years=float(max_years))
                st.session_state[_YAHOO_SURFACE_STATE_KEY] = surface_df
                st.session_state[_YAHOO_SURFACE_TICKER_KEY] = ticker
                st.session_state[_YAHOO_SURFACE_MAX_YEARS_KEY] = float(max_years)
            except Exception as exc:
                st.error(f"Yahoo indisponible: {exc}")
                st.session_state[_YAHOO_SURFACE_STATE_KEY] = None
                st.session_state[_YAHOO_SURFACE_TICKER_KEY] = None
                st.session_state[_YAHOO_SURFACE_MAX_YEARS_KEY] = None

        cache_matches = (
            isinstance(cached_df, pd.DataFrame)
            and not cached_df.empty
            and bool(cached_ticker)
            and bool(ticker)
            and cached_ticker == ticker
            and (
                cached_max_years is None
                or float(cached_max_years) == float(max_years)
            )
        )
        surface_df = surface_df if isinstance(surface_df, pd.DataFrame) else (cached_df if cache_matches else None)
        if isinstance(surface_df, pd.DataFrame) and not surface_df.empty:
            surface_ticker = ticker or None
            preview_df = surface_df
            _render_surface_preview_dropdown(
                surface_df,
                label=f"Aperçu / diagnostics surface ({surface_ticker or 'Yahoo'})",
                key="calib_surface_preview_yahoo",
            )
        else:
            needs_reload = bool(ticker) and not cache_matches
            if needs_reload:
                st.info("Clique sur `Load` pour récupérer l'option chain Yahoo.")
            else:
                st.info("Aucune surface Yahoo chargée.")

    elif source == "Upload CSV":
        uploaded = st.file_uploader(
            "Charger une surface IV (CSV: K, T, S0, iv, type)", type=["csv"], key="calib_upload"
        )
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()
            try:
                preview_df = pd.read_csv(uploaded)
                _render_surface_preview_dropdown(
                    preview_df,
                    label="Aperçu / diagnostics surface",
                    key="calib_surface_preview_upload",
                )
            except Exception as exc:
                st.warning(f"Impossible de lire le CSV: {exc}")

    elif source == "Cache (cache/*.csv)":
        cache_dir = Path("cache")
        cached = _discover_cached_surfaces(cache_dir)
        if not cached:
            st.info("Aucune surface détectée dans `cache/`. Dépose un CSV ou utilise l'upload.")
        else:
            chosen = st.selectbox(
                "Surface disponible",
                options=cached,
                format_func=lambda p: p.name,
                key="calib_cached_surface",
            )
            surface_path = str(chosen)
            try:
                preview_df = pd.read_csv(chosen)
                _render_surface_preview_dropdown(
                    preview_df,
                    label="Aperçu / diagnostics surface",
                    key="calib_surface_preview_cache",
                )
            except Exception as exc:
                st.warning(f"Impossible de lire {chosen.name}: {exc}")

    elif source == "Alpaca (live)":
        tickers = _load_optionable_tickers()
        default_ticker = st.session_state.get(_CHAIN_TICKER_KEY) or (tickers[0] if tickers else "AAPL")
        col_ticker, col_load = st.columns([3, 1])
        with col_ticker:
            if tickers:
                if default_ticker not in tickers:
                    default_ticker = tickers[0]
                ticker = st.selectbox(
                    "Underlying ticker",
                    options=tickers,
                    index=tickers.index(default_ticker),
                    key="calib_alpaca_ticker",
                )
            else:
                ticker = st.text_input("Underlying ticker", value=str(default_ticker)).upper().strip()
        with col_load:
            load_clicked = st.button("Load chain", use_container_width=True)

        if load_clicked and ticker:
            with st.spinner(f"Loading options for {ticker} from Alpaca..."):
                res = ctrl.download_alpaca_options_chain(ticker)
            if not res.get("success"):
                st.error(res.get("message", "Erreur Alpaca."))
            else:
                surface_ticker = str(res.get("ticker") or ticker).strip().upper()
                st.session_state[_CHAIN_TICKER_KEY] = res.get("ticker") or ticker
                st.session_state[_CHAIN_STATE_KEY] = res.get("df")
                st.success(res.get("message", "OK"))

        surface_df = _get_chain_from_state()
        if surface_df is not None:
            surface_ticker = surface_ticker or str(st.session_state.get(_CHAIN_TICKER_KEY) or "").strip().upper() or None
            preview_df = surface_df
            _render_surface_preview_dropdown(
                surface_df,
                label="Aperçu / diagnostics surface",
                key="calib_surface_preview_alpaca",
            )
        else:
            st.info("Chargez une chaîne d'options Alpaca pour calibrer.")

    calib_df = None
    if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
        canon = _canonicalize_surface_df(preview_df)
        if canon.empty:
            st.warning("Surface non reconnue (colonnes attendues: K, T, S0, iv).")
        else:
            with st.expander("Filtrage surface", expanded=False):
                calib_df = _render_surface_filters(canon)

    with st.expander("Neural Net (entraîner les poids)", expanded=False):
        st.caption(f"Chemin: {nn_info.get('weights_path')}")
        if weights_exists:
            size_bytes = nn_info.get("size_bytes")
            try:
                kb = int(size_bytes) // 1024 if size_bytes is not None else None
            except Exception:
                kb = None
            suffix = f" ({kb} KB)" if kb else ""
            st.success(f"Poids détectés{suffix}.")
            allow_overwrite = st.checkbox(
                "Ré-entraîner et écraser les poids",
                value=False,
                key="calib_nn_train_overwrite",
            )
        else:
            st.warning("Poids absents: entraîne le NN pour activer la méthode Neural Net.")
            allow_overwrite = True

        preset = st.selectbox(
            "Preset",
            options=["Rapide", "Qualité"],
            index=0,
            key="calib_nn_train_preset",
        )
        if preset == "Qualité":
            default_n_samples = 4000
            default_epochs = 30
            default_n_integration = 2000
        else:
            default_n_samples = 1500
            default_epochs = 15
            default_n_integration = 800

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_samples = int(
                st.number_input(
                    "n_samples",
                    min_value=200,
                    max_value=20000,
                    value=int(default_n_samples),
                    step=100,
                    key="calib_nn_train_n_samples",
                )
            )
            epochs = int(
                st.number_input(
                    "epochs",
                    min_value=1,
                    max_value=200,
                    value=int(default_epochs),
                    step=1,
                    key="calib_nn_train_epochs",
                )
            )
        with col_b:
            batch_size = int(
                st.number_input(
                    "batch_size",
                    min_value=8,
                    max_value=512,
                    value=64,
                    step=8,
                    key="calib_nn_train_batch_size",
                )
            )
            lr = float(
                st.number_input(
                    "lr",
                    min_value=1e-5,
                    max_value=1e-2,
                    value=1e-3,
                    step=1e-4,
                    format="%.5f",
                    key="calib_nn_train_lr",
                )
            )
        with col_c:
            n_integration = int(
                st.number_input(
                    "n_integration (pricing)",
                    min_value=200,
                    max_value=4000,
                    value=int(default_n_integration),
                    step=200,
                    key="calib_nn_train_n_integration",
                )
            )
            u_max = float(
                st.number_input(
                    "u_max",
                    min_value=10.0,
                    max_value=150.0,
                    value=50.0,
                    step=5.0,
                    key="calib_nn_train_u_max",
                )
            )
            seed_nn = int(
                st.number_input(
                    "seed",
                    min_value=0,
                    max_value=10_000_000,
                    value=42,
                    step=1,
                    key="calib_nn_train_seed",
                )
            )

        train_label = "Ré-entraîner les poids NN" if weights_exists else "Entraîner les poids NN"
        train_disabled = bool(weights_exists) and not bool(allow_overwrite)
        if st.button(train_label, disabled=train_disabled, use_container_width=True, key="calib_nn_train_btn"):
            with st.spinner("Entraînement du NN (Heston)..."):
                res_train = ctrl.train_heston_nn_weights(
                    {
                        "n_samples": int(n_samples),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "lr": float(lr),
                        "device": "cpu",
                        "seed": int(seed_nn),
                        "u_max": float(u_max),
                        "n_integration": int(n_integration),
                    }
                )
            if res_train.get("success"):
                details = res_train.get("details") or {}
                st.success(
                    f"Poids OK | loss={details.get('final_loss')} | elapsed={details.get('elapsed_s')}s"
                )
                st.rerun()
            else:
                st.error(res_train.get("message", "Entraînement échoué."))

    # Defaults for underlying inputs
    s0_default = _median_s0(calib_df if isinstance(calib_df, pd.DataFrame) else preview_df) or float(
        st.session_state.get("common_spot_value", 100.0)
    )

    st.markdown("### Paramètres du sous-jacent")
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("S0", value=float(s0_default), min_value=0.01, step=1.0)
    with col2:
        r_source = st.selectbox("Source r", options=["Manuel", "Yield Curve"], index=0, key="calib_r_source")
        r_val = 0.02
        if r_source == "Yield Curve":
            try:
                currencies = yc.available_currencies()
            except Exception:
                currencies = []
            default_currency = "USD" if "USD" in currencies else (currencies[0] if currencies else "USD")
            currency_options = currencies or [default_currency]
            default_ccy_index = (
                currency_options.index(default_currency) if default_currency in currency_options else 0
            )
            currency = st.selectbox(
                "Currency",
                options=currency_options,
                index=default_ccy_index,
                key="calib_r_ccy",
            )
            t_ref_options = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            t_ref = st.selectbox("T", options=t_ref_options, index=t_ref_options.index(1.0), key="calib_r_T")
            try:
                r_val = float(yc.get_risk_free_rate(T_ref=float(t_ref), currency=str(currency)))
            except Exception:
                r_val = 0.02
            st.number_input("r(T)", value=float(r_val), step=0.001, format="%.4f", disabled=True)
        else:
            r_val = float(st.number_input("Taux sans risque r", value=0.02, step=0.001, format="%.4f", key="calib_r_val_manual"))
    with col3:
        q = st.number_input("Dividende q", value=0.0, step=0.001, format="%.4f")

    fit_to_observed_only = True
    u_max = 50.0
    n_integration = 2000
    max_nfev = 50
    if method == "least_squares":
        with st.expander("Optimisation (avancé)", expanded=False):
            fit_to_observed_only = st.checkbox(
                "Fit uniquement sur points observés (mask)",
                value=True,
                key="calib_fit_mask_only",
            )
            n_starts = int(st.number_input("Multi-start (n)", min_value=1, max_value=25, value=3, step=1))
            seed = int(st.number_input("Seed (random)", min_value=0, max_value=10_000_000, value=42, step=1))
            max_nfev = int(
                st.number_input("max_nfev", min_value=10, max_value=500, value=50, step=10)
            )
            n_integration = int(
                st.number_input(
                    "N integration (Heston)",
                    min_value=200,
                    max_value=8000,
                    value=2000,
                    step=200,
                )
            )
            u_max = float(st.number_input("u_max", min_value=10.0, max_value=150.0, value=50.0, step=5.0))

    # Build payload + run
    can_run = any([csv_bytes is not None, surface_path is not None, isinstance(calib_df, pd.DataFrame)])
    if st.button("Calibrer", type="primary", disabled=not can_run, use_container_width=True):
        payload: Dict[str, Any] = {
            "model": model_choice,
            "constraints": constraints,
            "S0": float(S0),
            "r": float(r_val),
            "q": float(q),
        }
        if isinstance(calib_df, pd.DataFrame):
            payload["df"] = calib_df
            if source == "Upload CSV":
                payload["source"] = "upload"
            elif source.startswith("Cache"):
                payload["source"] = "cache"
            elif source.startswith("Ticker"):
                payload["source"] = "yahoo"
            else:
                payload["source"] = "alpaca"
            if surface_ticker:
                payload["ticker"] = surface_ticker
        elif surface_path is not None:
            payload["surface_path"] = surface_path
            payload["source"] = "cache"
        else:
            payload["csv_bytes"] = csv_bytes
            payload["source"] = "upload"

        if method == "least_squares":
            payload.update(
                {
                    "fit_to_observed_only": bool(fit_to_observed_only),
                    "u_max": float(u_max),
                    "n_integration": int(n_integration),
                    "max_nfev": int(max_nfev),
                    "n_starts": int(n_starts),
                    "seed": int(seed),
                }
            )
            with st.spinner("Calibration Heston (Least Squares)..."):
                result = ctrl.run_heston_ls_from_surface(payload)
        else:
            with st.spinner("Calibration Heston (NN)..."):
                result = ctrl.run_heston_nn_from_surface(payload)

        st.session_state["last_calibration_result"] = result

    result = st.session_state.get("last_calibration_result")
    if not isinstance(result, dict) or not result:
        return

    if not result.get("success"):
        st.warning(result.get("message", "Calibration échouée."))
        if result.get("details"):
            with st.expander("Détails", expanded=False):
                st.json(result.get("details"))
        return

    st.success(result.get("message", "OK"))

    if str(result.get("method") or "").lower() == "least_squares" and not bool(result.get("converged", True)):
        st.warning("Optimisation non convergée: paramètres retournés = meilleur candidat trouvé.")

    details = result.get("details") or {}
    runs = details.get("runs") if isinstance(details, dict) else None
    if isinstance(runs, list) and runs:
        with st.expander("Détails optimisation (multi-start)", expanded=False):
            best_run = details.get("best_run") if isinstance(details, dict) else None
            if best_run is not None:
                st.caption(f"Best run: {best_run}")
            try:
                rows = []
                for r in runs:
                    if not isinstance(r, dict):
                        continue
                    rows.append(
                        {
                            "idx": r.get("idx"),
                            "ok": r.get("ok"),
                            "converged": r.get("converged"),
                            "cost": r.get("cost"),
                            "nfev": r.get("nfev"),
                            "optimality": r.get("optimality"),
                            "message": r.get("message") or r.get("error"),
                        }
                    )
                df_runs = pd.DataFrame(rows).sort_values("cost", ascending=True, na_position="last")
                st.dataframe(df_runs, hide_index=True, use_container_width=True)
            except Exception as exc:
                st.warning(f"Impossible d'afficher les runs: {exc}")

    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("MAE (IV)", f"{float(metrics.get('mae', 0.0)):.4f}")
        col_b.metric("RMSE (IV)", f"{float(metrics.get('rmse', 0.0)):.4f}")
        col_c.metric("Max |err| (IV)", f"{float(metrics.get('max_abs', 0.0)):.4f}")

    params = result.get("params") or {}
    st.markdown("### Paramètres calibrés")
    st.json(params)

    if st.button("Envoyer IV modèle vers Options", use_container_width=True, type="secondary"):
        try:
            S0_res = float(result.get("S0") or S0)
            m_grid_res = np.array(result.get("m_grid") or [])
            t_grid_res = np.array(result.get("t_grid") or [])
            iv_model_res = np.array(result.get("iv_model") or [])
            df_model_surface = _grid_to_surface_df(
                S0=S0_res, m_grid=m_grid_res, t_grid=t_grid_res, iv_grid=iv_model_res, opt_type="call"
            )
            st.session_state["calib_model_surface_df"] = df_model_surface
            st.session_state["calib_model_surface_meta"] = {
                "ticker": result.get("ticker"),
                "method": result.get("method") or "neural_net",
                "S0": S0_res,
                "r": float(result.get("r") or r_val),
                "q": float(result.get("q") or q),
            }
            st.session_state["opt_iv_surface_source"] = "Calibration"

            tkr = str(result.get("ticker") or "").strip().upper()
            if tkr:
                st.session_state["tkr_common"] = tkr
                st.session_state["common_underlying"] = tkr

            # Best-effort: seed Options global params from calibration
            try:
                st.session_state["common_rate_value"] = float(result.get("r") or r_val)
                st.session_state["d_common"] = float(result.get("q") or q)
                if str(r_source).lower().startswith("man"):
                    st.session_state["opt_use_yield_curve_rate"] = False
            except Exception:
                pass

            # Set sigma from ATM (closest m=1, T=1y)
            try:
                if m_grid_res.size and t_grid_res.size and iv_model_res.size:
                    j_atm = int(np.abs(m_grid_res - 1.0).argmin())
                    i_t = int(np.abs(t_grid_res - 1.0).argmin())
                    atm_iv = float(iv_model_res[i_t, j_atm])
                    if np.isfinite(atm_iv) and atm_iv > 0:
                        st.session_state["common_sigma_value"] = atm_iv
                    try:
                        st.session_state["opt_iv_surface_max_years"] = float(
                            min(2.0, max(0.25, float(np.max(t_grid_res))))
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            st.success("Surface IV modèle envoyée. Ouvrez l’onglet Options → IV surface → Source=Calibration.")
        except Exception as exc:
            st.error(f"Impossible d'envoyer vers Options: {exc}")

    m_grid = np.array(result.get("m_grid") or [])
    t_grid = np.array(result.get("t_grid") or [])
    iv_mkt = np.array(result.get("iv_market") or [])
    iv_model = np.array(result.get("iv_model") or [])
    iv_err = np.array(result.get("iv_error") or [])
    if iv_mkt.size and iv_model.size and iv_err.size:
        st.markdown("### Surfaces IV")
        col_mkt, col_model, col_err = st.columns(3)
        with col_mkt:
            _plot_heatmap(iv_mkt, m_grid, t_grid, "IV marché")
        with col_model:
            _plot_heatmap(iv_model, m_grid, t_grid, "IV modèle")
        with col_err:
            _plot_heatmap(iv_err, m_grid, t_grid, "Erreur (modèle - marché)")
    else:
        st.info("Aucune surface à afficher.")


render = render_tab
