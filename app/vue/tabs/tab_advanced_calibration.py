import json
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from app.controller.calibration_controller import CalibrationController
from app.vue.components.page_utils import render_page_header


TAB_LABEL = "🧪 Advanced Calibration"


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


def render_tab() -> None:
    ctrl = CalibrationController()

    render_page_header(
        "Advanced Calibration",
        "SABR / Jump Diffusion / Heston / Rough & Volterra (modular, MVC-safe)",
        icon="🧪",
        badge="Models",
    )

    specs = ctrl.get_advanced_models()
    if not specs:
        st.error("Aucun modèle avancé disponible.")
        return

    key_to_spec = {s.get("key"): s for s in specs if isinstance(s, dict) and s.get("key")}
    model_keys = list(key_to_spec.keys())

    col_a, col_b = st.columns([2, 1])
    with col_a:
        model_key = st.selectbox(
            "Model",
            options=model_keys,
            index=model_keys.index("heston_fft") if "heston_fft" in model_keys else 0,
            format_func=lambda k: key_to_spec.get(k, {}).get("label", k),
        )
    with col_b:
        spec = key_to_spec.get(model_key, {})
        st.caption(f"pricing={spec.get('pricing')} | calibration={spec.get('calibration')}")
        if spec.get("expensive"):
            st.warning("Expensive model: expect slower runs.")

    st.markdown("### Market IV surface")
    uploaded = st.file_uploader("Upload CSV (columns: K, T, S0, iv, type)", type=["csv"], key="adv_calib_upload")
    if uploaded is None:
        st.info("Upload a CSV surface to run an advanced calibration.")
        return

    csv_bytes = uploaded.getvalue()
    try:
        preview_df = pd.read_csv(uploaded)
        st.dataframe(preview_df.head(20), hide_index=True, use_container_width=True)
    except Exception:
        preview_df = None

    col_r, col_q, col_s0 = st.columns(3)
    with col_r:
        r_val = float(st.number_input("r (risk-free)", value=float(st.session_state.get("common_rate_value", 0.02)), step=0.001))
    with col_q:
        q_val = float(st.number_input("q (dividend)", value=float(st.session_state.get("d_common", 0.0)), step=0.001))
    with col_s0:
        auto_s0 = st.checkbox("Auto S0 from CSV (if present)", value=True, key="adv_calib_auto_s0")
        S0_val = None
        if not auto_s0:
            S0_val = float(st.number_input("S0", value=100.0, step=1.0))

    with st.expander("Grid / settings", expanded=False):
        fit_to_observed_only = st.checkbox("Fit to observed only", value=True, key="adv_fit_observed_only")
        col_nfev, col_starts, col_seed = st.columns(3)
        with col_nfev:
            max_nfev = int(st.number_input("max_nfev", value=60, min_value=5, step=5))
        with col_starts:
            n_starts = int(st.number_input("n_starts", value=1, min_value=1, step=1))
        with col_seed:
            seed_raw = st.text_input("seed (optional)", value="", placeholder="e.g. 42")
        seed = int(seed_raw) if seed_raw.strip().isdigit() else None

        st.caption("Optional custom fixed grid (comma-separated). Leave empty to use app defaults.")
        m_raw = st.text_input("m_grid (K/S0)", value="", placeholder="0.8,0.9,1.0,1.1")
        t_raw = st.text_input("t_grid (years)", value="", placeholder="0.02,0.05,0.1,0.25,0.5,1.0")
        m_grid = _as_float_list(m_raw)
        t_grid = _as_float_list(t_raw)

    with st.expander("Model constraints (JSON)", expanded=False):
        st.caption(
            "Examples:\n"
            "- SABR: {\"beta\": 0.5}\n"
            "- FFT: {\"fft_cfg\": {\"alpha\": 1.5, \"n\": 2048, \"eta\": 0.25}}\n"
            "- rHeston: {\"fft_cfg\": {...}, \"markovian_cfg\": {\"n_factors\": 12, \"steps_per_year\": 120}}\n"
            "- rBergomi: {\"mc_cfg\": {\"n_design\": 24, \"n_paths\": 4000, \"n_steps\": 60}}\n"
            "- Volterra: {\"mc_cfg\": {\"kernel_type\": \"fractional\", \"H\": 0.1, \"n_design\": 24}}"
        )
        constraints_raw = st.text_area("constraints", value="{}", height=140, key="adv_calib_constraints_json")
        constraints = _parse_json_dict(constraints_raw) or {}

    if st.button("Run calibration", type="primary", use_container_width=True):
        payload: Dict[str, Any] = {
            "model": model_key,
            "csv_bytes": csv_bytes,
            "r": r_val,
            "q": q_val,
            "S0": S0_val,
            "fit_to_observed_only": fit_to_observed_only,
            "max_nfev": max_nfev,
            "n_starts": n_starts,
            "seed": seed,
            "constraints": constraints,
        }
        if m_grid is not None:
            payload["m_grid"] = m_grid
        if t_grid is not None:
            payload["t_grid"] = t_grid

        with st.spinner("Calibrating..."):
            result = ctrl.run_advanced_surface_calibration(payload)

        if not result.get("success"):
            st.error(str(result.get("message") or "Calibration failed."))
            details = result.get("details")
            if details:
                st.json(details)
            return

        st.success(str(result.get("message") or "OK"))
        metrics = result.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (IV)", f"{float(metrics.get('mae', 0.0)):.4f}")
            col2.metric("RMSE (IV)", f"{float(metrics.get('rmse', 0.0)):.4f}")
            col3.metric("Max |err| (IV)", f"{float(metrics.get('max_abs', 0.0)):.4f}")

        st.markdown("### Calibrated parameters")
        st.json(result.get("params") or {})

        m_grid_arr = np.asarray(result.get("m_grid") or [], dtype=float)
        t_grid_arr = np.asarray(result.get("t_grid") or [], dtype=float)
        iv_mkt = np.asarray(result.get("iv_market") or [], dtype=float)
        iv_model = np.asarray(result.get("iv_model") or [], dtype=float)
        iv_err = np.asarray(result.get("iv_error") or [], dtype=float)

        if iv_mkt.size and iv_model.size and iv_err.size:
            st.markdown("### IV surfaces")
            c_mkt, c_mod, c_err = st.columns(3)
            with c_mkt:
                _plot_heatmap(iv_mkt, m_grid_arr, t_grid_arr, "IV market")
            with c_mod:
                _plot_heatmap(iv_model, m_grid_arr, t_grid_arr, "IV model")
            with c_err:
                _plot_heatmap(iv_err, m_grid_arr, t_grid_arr, "Error (model - market)")

        with st.expander("Details / runs", expanded=False):
            st.json(result.get("details") or {})


render = render_tab

