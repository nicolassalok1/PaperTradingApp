import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.controller.calibration_controller import CalibrationController
from app.controller import yieldcurve_controller as yc
from app.vue.components.page_utils import render_page_header
from app.vue.components.ui_helpers import render_quickstart
from app.vue.components.surface_ui import (
    canonicalize_surface_df,
    discover_cached_surfaces,
    grid_to_surface_df,
    median_s0,
    plot_iv_heatmap,
    render_surface_filters,
    render_surface_preview_dropdown,
)


TAB_LABEL = "🧪 Calibration avancée"

_CHAIN_STATE_KEY = "adv_calib_alpaca_chain_df"
_CHAIN_TICKER_KEY = "adv_calib_alpaca_chain_ticker"
_TICKERS_STATE_KEY = "adv_calib_alpaca_underlyings"
_YAHOO_SURFACE_STATE_KEY = "adv_calib_yahoo_surface_df"
_YAHOO_SURFACE_TICKER_KEY = "adv_calib_yahoo_surface_ticker"
_YAHOO_SURFACE_MAX_YEARS_KEY = "adv_calib_yahoo_surface_max_years"
_LAST_RESULT_KEY = "last_advanced_calibration_result"

_OPTIONABLE_TICKERS_CSV = Path(
    os.getenv("ALPACA_OPTIONABLE_TICKERS_PATH", "data/alpaca_optionable_tickers.csv")
)
_YAHOO_CHAINS_DIR = Path("cache/YahooOptionChains")


def _parse_json_dict(raw: str) -> Dict[str, Any] | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _as_float_list(raw: str) -> Optional[List[float]]:
    if not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            return None
    return out


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


def _load_chain_meta(ticker: str | None) -> dict[str, Any]:
    if not ticker:
        return {}
    safe = str(ticker).strip().upper()
    if not safe:
        return {}
    try:
        if not _YAHOO_CHAINS_DIR.exists():
            return {}
        matches = sorted(_YAHOO_CHAINS_DIR.glob(f"yahoo_chain_{safe}_*.json"))
        if not matches:
            matches = sorted(_YAHOO_CHAINS_DIR.glob(f"yahoo_chain_{safe}*.json"))
        if not matches:
            return {}
        with open(matches[-1], "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def render_tab() -> None:
    ctrl = CalibrationController()

    render_page_header(
        "Calibration avancée",
        "SABR / Jump Diffusion / Heston / Rough & Volterra (modulaire, MVC-safe)",
        icon="🧪",
        badge="Models",
    )
    render_quickstart(
        "Guide rapide",
        [
            "Charge une surface IV (Yahoo / CSV / Cache / Alpaca), puis filtre-la si besoin.",
            "Règle `S0`, `r`, `q` avant de calibrer (r peut venir de `🧮 Yield Curve`).",
            "Après calibration, clique ‘Envoyer IV modèle vers Options’ pour la visualiser/pricer.",
        ],
        expanded=False,
    )

    specs = ctrl.get_advanced_models()
    if not specs:
        st.error("Aucun modèle avancé disponible.")
        return

    filtered_specs = [
        s for s in specs if isinstance(s, dict) and s.get("key") and s.get("key") != "heston_v1"
    ]
    key_to_spec = {s["key"]: s for s in filtered_specs}

    override_labels = {
        "heston_fft": "Heston",
        "rheston": "rHeston",
        "rbergomi": "rBergomi",
        "volterra": "Volterra SDE",
        "merton_jump_diffusion": "JumpDiffusion",
        "sabr": "SABR",
    }
    desired_order = [
        "heston_fft",
        "rheston",
        "rbergomi",
        "volterra",
        "merton_jump_diffusion",
        "sabr",
    ]
    model_keys = [k for k in desired_order if k in key_to_spec]
    if not model_keys:
        model_keys = list(key_to_spec.keys())
    if not model_keys:
        st.error("Aucun modèle calibrable n'est disponible.")
        return

    constraints: Dict[str, Any] = {}

    st.markdown("### Surface IV marché")
    st.caption("Source: Ticker (Yahoo) uniquement (autres sources désactivées).")
    source = "Ticker (Yahoo)"

    preview_df: pd.DataFrame | None = None
    surface_ticker: str | None = None

    if source == "Ticker (Yahoo)":
        default_ticker = (
            st.session_state.get("adv_calib_yahoo_ticker_input")
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
                key="adv_calib_yahoo_ticker_input",
            )
        with col_years:
            try:
                max_years_default = float(st.session_state.get("adv_calib_yahoo_max_years", 2.0))
            except Exception:
                max_years_default = 2.0
            max_years_default = min(2.0, max(0.25, max_years_default))
            max_years = st.slider(
                "Max maturité (années)",
                min_value=0.25,
                max_value=2.0,
                step=0.25,
                value=float(max_years_default),
                key="adv_calib_yahoo_max_years",
            )
        with col_load:
            load_clicked = st.button("Load", width="stretch", key="adv_calib_yahoo_load_btn")

        ticker = (ticker_raw or "").strip().upper()
        cached_df = st.session_state.get(_YAHOO_SURFACE_STATE_KEY)
        cached_ticker = st.session_state.get(_YAHOO_SURFACE_TICKER_KEY)
        cached_max_years = st.session_state.get(_YAHOO_SURFACE_MAX_YEARS_KEY)

        surface_df = None
        if load_clicked and ticker:
            try:
                with st.spinner(f"Chargement Yahoo IV surface pour {ticker}..."):
                    surface_df = ctrl.fetch_yahoo_iv_surface(ticker, max_maturity_years=float(max_years))
                if surface_df is None or getattr(surface_df, "empty", True):
                    st.warning("Yahoo n'a retourné aucune surface.")
                    surface_df = None
                else:
                    st.session_state[_YAHOO_SURFACE_STATE_KEY] = surface_df
                    st.session_state[_YAHOO_SURFACE_TICKER_KEY] = ticker
                    st.session_state[_YAHOO_SURFACE_MAX_YEARS_KEY] = float(max_years)
                    s0_guess = median_s0(surface_df)
                    if s0_guess:
                        st.session_state["common_spot_value"] = float(s0_guess)
                    st.success(f"Surface Yahoo chargée ({len(surface_df):,} lignes).")
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
            and (cached_max_years is None or float(cached_max_years) == float(max_years))
        )
        surface_df = surface_df if isinstance(surface_df, pd.DataFrame) else (cached_df if cache_matches else None)
        if isinstance(surface_df, pd.DataFrame) and not surface_df.empty:
            surface_ticker = ticker or None
            preview_df = surface_df
            render_surface_preview_dropdown(
                surface_df,
                label=f"Aperçu / diagnostics surface ({surface_ticker or 'Yahoo'})",
                key="adv_calib_surface_preview_yahoo",
                diag_key_prefix="adv_calib",
            )
        else:
            needs_reload = bool(ticker) and not cache_matches
            if needs_reload:
                st.info("Clique sur `Load` pour récupérer l'option chain Yahoo.")
            else:
                st.info("Aucune surface Yahoo chargée.")

    elif source == "Upload CSV":
        uploaded = st.file_uploader(
            "Charger une surface IV (CSV: K, T, S0, iv, type)", type=["csv"], key="adv_calib_upload"
        )
        if uploaded is not None:
            try:
                preview_df = pd.read_csv(uploaded)
                render_surface_preview_dropdown(
                    preview_df,
                    label="Aperçu / diagnostics surface",
                    key="adv_calib_surface_preview_upload",
                    diag_key_prefix="adv_calib",
                )
            except Exception as exc:
                st.warning(f"Impossible de lire le CSV: {exc}")

    elif source == "Cache (cache/*.csv, cache/YahooOptionChains/*.csv)":
        cache_dir = Path("cache")
        cached = discover_cached_surfaces(cache_dir)
        if not cached:
            st.info(
                "Aucune surface détectée dans `cache/` ou `cache/YahooOptionChains/`. Dépose un CSV ou utilise l'upload."
            )
        else:
            chosen = st.selectbox(
                "Surface disponible",
                options=cached,
                format_func=lambda p: p.name,
                key="adv_calib_cached_surface",
            )
            try:
                preview_df = pd.read_csv(chosen)
                render_surface_preview_dropdown(
                    preview_df,
                    label="Aperçu / diagnostics surface",
                    key="adv_calib_surface_preview_cache",
                    diag_key_prefix="adv_calib",
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
                    key="adv_calib_alpaca_ticker",
                )
            else:
                ticker = st.text_input("Underlying ticker", value=str(default_ticker)).upper().strip()
        with col_load:
            load_clicked = st.button("Load chain", width="stretch", key="adv_calib_alpaca_load_btn")

        if load_clicked and ticker:
            with st.spinner(f"Loading options for {ticker} from Alpaca..."):
                res = ctrl.download_alpaca_options_chain(ticker)
            if not res.get("success"):
                st.error(res.get("message", "Erreur Alpaca."))
            else:
                surface_ticker = str(res.get("ticker") or ticker).strip().upper()
                st.session_state[_CHAIN_TICKER_KEY] = surface_ticker
                st.session_state[_CHAIN_STATE_KEY] = res.get("df")
                st.success(res.get("message", "OK"))

        surface_df = _get_chain_from_state()
        if surface_df is not None:
            surface_ticker = surface_ticker or str(st.session_state.get(_CHAIN_TICKER_KEY) or "").strip().upper() or None
            preview_df = surface_df
            s0_guess = median_s0(surface_df)
            if s0_guess:
                st.session_state["common_spot_value"] = float(s0_guess)
            render_surface_preview_dropdown(
                surface_df,
                label="Aperçu / diagnostics surface",
                key="adv_calib_surface_preview_alpaca",
                diag_key_prefix="adv_calib",
            )
        else:
            st.info("Chargez une chaîne d'options Alpaca pour calibrer.")

    calib_df: pd.DataFrame | None = None
    canon_df: pd.DataFrame | None = None
    if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
        canon_df = canonicalize_surface_df(preview_df)
        if canon_df.empty:
            st.warning("Surface non reconnue (colonnes attendues: K, T, S0, iv).")
        else:
            calib_df = canon_df

    chain_meta = _load_chain_meta(surface_ticker or ticker)
    spot_chain = chain_meta.get("spot")
    div_chain = chain_meta.get("div")
    max_chain_years = chain_meta.get("max_maturity_years")

    s0_default = (
        (float(spot_chain) if spot_chain is not None else None)
        or median_s0(calib_df)
        or median_s0(canon_df)
        or float(st.session_state.get("common_spot_value", 100.0))
    )
    S0 = float(s0_default)

    # Risk-free rate from yield curve at max maturity (slider or chain)
    t_ref_for_r = float(max_years)
    try:
        if max_chain_years is not None:
            t_ref_for_r = max(float(t_ref_for_r), float(max_chain_years))
    except Exception:
        t_ref_for_r = float(max_years)

    try:
        currencies = yc.available_currencies()
    except Exception:
        currencies = []
    default_currency = "USD" if "USD" in currencies else (currencies[0] if currencies else "USD")
    currency = default_currency
    try:
        r_val = float(yc.get_risk_free_rate(T_ref=float(t_ref_for_r), currency=str(currency)))
    except Exception:
        r_val = float(st.session_state.get("common_rate_value", 0.02))

    q_val = float(div_chain) if div_chain is not None else float(st.session_state.get("d_common", 0.0))

    st.session_state["common_spot_value"] = float(s0_default)
    st.session_state["common_rate_value"] = float(r_val)
    st.session_state["d_common"] = float(q_val)

    st.markdown("### Paramètres du sous-jacent (auto)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("S0 (spot)", f"{float(s0_default):.4f}")
    with col2:
        st.metric(f"r(T={t_ref_for_r}) via Yield Curve ({currency})", f"{float(r_val):.4f}")
    with col3:
        st.metric("Dividende q", f"{float(q_val):.4f}")

    fit_to_observed_only = True
    max_nfev = 60
    n_starts = 1
    seed = None
    m_grid = None
    t_grid = None
    with st.expander("Optimisation (avancé)", expanded=False):
        fit_to_observed_only = st.checkbox(
            "Fit uniquement sur points observés (mask)",
            value=True,
            key="adv_calib_fit_mask_only",
        )
        col_nfev, col_starts, col_seed = st.columns(3)
        with col_nfev:
            max_nfev = int(
                st.number_input(
                    "max_nfev",
                    min_value=5,
                    max_value=500,
                    value=60,
                    step=5,
                    key="adv_calib_max_nfev",
                )
            )
        with col_starts:
            n_starts = int(
                st.number_input(
                    "Multi-start (n)",
                    min_value=1,
                    max_value=25,
                    value=1,
                    step=1,
                    key="adv_calib_n_starts",
                )
            )
        with col_seed:
            seed_raw = st.text_input(
                "Seed (random, optionnel)",
                value="",
                placeholder="ex: 42",
                key="adv_calib_seed_raw",
            )
        seed = int(seed_raw) if seed_raw.strip().isdigit() else None

        st.caption("Grille fixe optionnelle (valeurs séparées par des virgules). Laisser vide = défauts app.")
        m_raw = st.text_input("m_grid (K/S0)", value="", placeholder="0.8,0.9,1.0,1.1", key="adv_calib_m_grid_raw")
        t_raw = st.text_input(
            "t_grid (années)", value="", placeholder="0.02,0.05,0.1,0.25,0.5,1.0", key="adv_calib_t_grid_raw"
        )
        m_grid = _as_float_list(m_raw)
        t_grid = _as_float_list(t_raw)
        if m_raw.strip() and m_grid is None:
            st.warning("m_grid invalide (liste de floats séparés par des virgules).")
        if t_raw.strip() and t_grid is None:
            st.warning("t_grid invalide (liste de floats séparés par des virgules).")

    st.markdown("### Modèle (sélection juste avant calibration)")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        model_key = st.selectbox(
            "Modèle",
            options=model_keys,
            index=model_keys.index("heston_fft") if "heston_fft" in model_keys else 0,
            format_func=lambda k: override_labels.get(k, key_to_spec.get(k, {}).get("label", k)),
            key="adv_calib_model_select",
        )
    with col_b:
        spec = key_to_spec.get(model_key, {})
        st.caption(f"pricing={spec.get('pricing')} | calibration={spec.get('calibration')}")
        if spec.get("expensive"):
            st.warning("Modèle coûteux: prévoir des runs plus longs.")

    profile = st.radio(
        "Profil de calibration",
        options=["Rapide", "Normal", "Fine"],
        index=1,
        horizontal=True,
        key="adv_calib_profile",
        help="Rapide: itérations limitées. Normal: défaut équilibré. Fine: plus d'itérations et multi-start.",
    )
    profile_map = {
        "Rapide": {"max_nfev": 30, "n_starts": 1},
        "Normal": {"max_nfev": 60, "n_starts": 1},
        "Fine": {"max_nfev": 120, "n_starts": 3},
    }
    cfg = profile_map.get(profile, profile_map["Normal"])
    # Aligne les paramètres d'optimisation sur le profil choisi
    max_nfev = int(cfg["max_nfev"])
    n_starts = int(cfg["n_starts"])

    def _eta_human(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.0f} s"
        mins = seconds / 60.0
        if mins < 60:
            return f"{mins:.1f} min"
        hours = mins / 60.0
        return f"{hours:.1f} h"

    per_eval = 0.05
    if model_key in {"rheston", "rbergomi", "volterra"}:
        per_eval *= 5.0
    eta_seconds = per_eval * float(max_nfev) * float(max(1, n_starts))
    eta_label = _eta_human(eta_seconds)
    st.caption(f"ETA estimée: ~{eta_label} (heuristique; dépend du modèle et des données).")

    can_run = isinstance(calib_df, pd.DataFrame) and not calib_df.empty
    if st.button("Calibrer", type="primary", disabled=not can_run, width="stretch", key="adv_calib_run_btn"):
        progress = st.progress(0, text="Préparation…")
        payload: Dict[str, Any] = {
            "model": model_key,
            "df": calib_df,
            "r": float(r_val),
            "q": float(q_val),
            "S0": float(S0),
            "fit_to_observed_only": bool(fit_to_observed_only),
            "max_nfev": int(max_nfev),
            "n_starts": int(n_starts),
            "seed": seed,
            "constraints": constraints,
        }
        if m_grid is not None:
            payload["m_grid"] = m_grid
        if t_grid is not None:
            payload["t_grid"] = t_grid

        progress.progress(30, text="Calibration en cours…")
        with st.spinner("Calibration..."):
            result = ctrl.run_advanced_surface_calibration(payload)
        progress.progress(100, text="Terminé")
        progress.empty()

        if surface_ticker:
            try:
                result = dict(result or {})
                result["ticker"] = surface_ticker
            except Exception:
                pass
        st.session_state[_LAST_RESULT_KEY] = result

    result = st.session_state.get(_LAST_RESULT_KEY)
    if not isinstance(result, dict) or not result:
        return

    if not result.get("success"):
        st.warning(str(result.get("message") or "Calibration échouée."))
        if result.get("details"):
            with st.expander("Détails", expanded=False):
                st.json(result.get("details"))
        return

    st.success(str(result.get("message") or "OK"))

    details = result.get("details") or {}
    runs = details.get("runs") if isinstance(details, dict) else None
    if isinstance(runs, list) and runs:
        best_run = details.get("best_run") if isinstance(details, dict) else None
        if best_run is not None:
            st.caption(f"Best run: {best_run}")
        # Tableau des runs masqué pour alléger l'UI.

    metrics = result.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("MAE (IV)", f"{float(metrics.get('mae', 0.0)):.4f}")
        col_b.metric("RMSE (IV)", f"{float(metrics.get('rmse', 0.0)):.4f}")
        col_c.metric("Max |err| (IV)", f"{float(metrics.get('max_abs', 0.0)):.4f}")

    st.markdown("### Paramètres calibrés")
    st.json(result.get("params") or {})

    if st.button("Envoyer IV modèle vers Options", width="stretch", type="secondary", key="adv_calib_send_to_opt"):
        try:
            S0_res = float(result.get("S0") or S0)
            m_grid_res = np.array(result.get("m_grid") or [])
            t_grid_res = np.array(result.get("t_grid") or [])
            iv_model_res = np.array(result.get("iv_model") or [])
            df_model_surface = grid_to_surface_df(
                S0=S0_res, m_grid=m_grid_res, t_grid=t_grid_res, iv_grid=iv_model_res, opt_type="call"
            )
            st.session_state["calib_model_surface_df"] = df_model_surface
            st.session_state["calib_model_surface_meta"] = {
                "ticker": result.get("ticker"),
                "method": result.get("method") or "advanced",
                "model": result.get("model") or model_key,
                "S0": S0_res,
                "r": float(result.get("r") or r_val),
                "q": float(result.get("q") or q_val),
            }
            st.session_state["opt_iv_surface_source"] = "Calibration"

            tkr = str(result.get("ticker") or "").strip().upper()
            if tkr:
                st.session_state["tkr_common"] = tkr
                st.session_state["common_underlying"] = tkr

            try:
                st.session_state["common_rate_value"] = float(result.get("r") or r_val)
                st.session_state["d_common"] = float(result.get("q") or q_val)
                st.session_state["opt_use_yield_curve_rate"] = True
            except Exception:
                pass

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

            st.success("Surface IV modèle envoyée. Ouvrez l'onglet Options → IV surface → Source=Calibration.")
        except Exception as exc:
            st.error(f"Impossible d'envoyer vers Options: {exc}")

    m_grid_arr = np.asarray(result.get("m_grid") or [], dtype=float)
    t_grid_arr = np.asarray(result.get("t_grid") or [], dtype=float)
    iv_mkt = np.asarray(result.get("iv_market") or [], dtype=float)
    iv_model = np.asarray(result.get("iv_model") or [], dtype=float)
    iv_err = np.asarray(result.get("iv_error") or [], dtype=float)

    if iv_mkt.size and iv_model.size and iv_err.size:
        st.markdown("### Surfaces IV")
        c_mkt, c_mod, c_err = st.columns(3)
        with c_mkt:
            plot_iv_heatmap(iv_mkt, m_grid_arr, t_grid_arr, "IV marché")
        with c_mod:
            plot_iv_heatmap(iv_model, m_grid_arr, t_grid_arr, "IV modèle")
        with c_err:
            plot_iv_heatmap(iv_err, m_grid_arr, t_grid_arr, "Erreur (modèle - marché)")
    else:
        st.info("Aucune surface à afficher.")

    # Détails bruts supprimés (non affichés)


render = render_tab
