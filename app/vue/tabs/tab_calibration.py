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
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🧮 Calibration"

_CHAIN_STATE_KEY = "calib_alpaca_chain_df"
_CHAIN_TICKER_KEY = "calib_alpaca_chain_ticker"
_TICKERS_STATE_KEY = "calib_alpaca_underlyings"
_OPTIONABLE_TICKERS_CSV = Path(
    os.getenv("ALPACA_OPTIONABLE_TICKERS_PATH", "data/alpaca_optionable_tickers.csv")
)


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

    with st.expander("JSON généré", expanded=False):
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

    with st.expander("Nuage de points (moneyness × T, couleur=IV)", expanded=False):
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
                    text=[f"K={k:.2f}, T={tt:.3f}, IV={vv:.3f}" for k, tt, vv in zip(dfp["K"], dfp["T"], dfp["iv"])],
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
    with col_method:
        if nn_info.get("weights_exists"):
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
    source = st.radio(
        "Source",
        options=["Upload CSV", "Cache (cache/*.csv)", "Alpaca (live)"],
        horizontal=True,
        key="calib_surface_source",
    )

    csv_bytes = None
    surface_path = None
    surface_df = None
    preview_df = None

    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Charger une surface IV (CSV: K, T, S0, iv, type)", type=["csv"], key="calib_upload"
        )
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()
            try:
                preview_df = pd.read_csv(uploaded)
                st.dataframe(preview_df.head(30), hide_index=True, use_container_width=True)
                with st.expander("Diagnostics surface", expanded=False):
                    _surface_diagnostics(preview_df)
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
                st.dataframe(preview_df.head(30), hide_index=True, use_container_width=True)
                with st.expander("Diagnostics surface", expanded=False):
                    _surface_diagnostics(preview_df)
            except Exception as exc:
                st.warning(f"Impossible de lire {chosen.name}: {exc}")

    else:  # Alpaca (live)
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
                st.session_state[_CHAIN_TICKER_KEY] = res.get("ticker") or ticker
                st.session_state[_CHAIN_STATE_KEY] = res.get("df")
                st.success(res.get("message", "OK"))

        surface_df = _get_chain_from_state()
        if surface_df is not None:
            preview_df = surface_df
            st.dataframe(surface_df.head(30), hide_index=True, use_container_width=True)
            with st.expander("Diagnostics surface", expanded=False):
                _surface_diagnostics(surface_df)
        else:
            st.info("Chargez une chaîne d'options Alpaca pour calibrer.")

    # Defaults for underlying inputs
    s0_default = _median_s0(preview_df) or float(st.session_state.get("common_spot_value", 100.0))

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
    can_run = any([csv_bytes is not None, surface_path is not None, isinstance(surface_df, pd.DataFrame)])
    if st.button("Calibrer", type="primary", disabled=not can_run, use_container_width=True):
        payload: Dict[str, Any] = {
            "model": model_choice,
            "constraints": constraints,
            "S0": float(S0),
            "r": float(r_val),
            "q": float(q),
        }
        if isinstance(surface_df, pd.DataFrame):
            payload["df"] = surface_df
            payload["source"] = "alpaca"
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

    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("MAE (IV)", f"{float(metrics.get('mae', 0.0)):.4f}")
        col_b.metric("RMSE (IV)", f"{float(metrics.get('rmse', 0.0)):.4f}")
        col_c.metric("Max |err| (IV)", f"{float(metrics.get('max_abs', 0.0)):.4f}")

    params = result.get("params") or {}
    st.markdown("### Paramètres calibrés")
    st.json(params)

    try:
        export_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Télécharger le résultat (JSON)",
            data=export_bytes,
            file_name="calibration_heston_result.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception:
        pass

    m_grid = np.array(result.get("m_grid") or [])
    t_grid = np.array(result.get("t_grid") or [])
    iv_mkt = np.array(result.get("iv_market") or [])
    iv_model = np.array(result.get("iv_model") or [])
    iv_err = np.array(result.get("iv_error") or [])
    if iv_mkt.size and iv_model.size and iv_err.size:
        st.markdown("### Surfaces IV")
        _plot_heatmap(iv_mkt, m_grid, t_grid, "IV marché")
        _plot_heatmap(iv_model, m_grid, t_grid, "IV modèle")
        _plot_heatmap(iv_err, m_grid, t_grid, "Erreur (modèle - marché)")
    else:
        st.info("Aucune surface à afficher.")


render = render_tab
