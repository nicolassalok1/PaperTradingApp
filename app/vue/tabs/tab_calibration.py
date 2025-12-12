import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controller.calibration_controller import CalibrationController

TAB_LABEL = "🧮 Calibration"


def _plot_heatmap(z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title="IV"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Moneyness",
        yaxis_title="Time to maturity (y)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "scrollZoom": False})


def render_tab():
    ctrl = CalibrationController()
    st.title("Calibration")
    st.caption("Calibration Heston via réseau de neurones (inférence uniquement).")

    # Model selector
    models = ctrl.get_models()
    model_choice = st.selectbox(
        "Modèle",
        options=models,
        index=0 if models else 0,
        format_func=lambda v: str(v).upper(),
    )

    # Common inputs
    with st.expander("Contraintes (JSON)", expanded=False):
        constraints_raw = st.text_area("Contraintes", value="{}", height=120, key="calib_constraints")
        constraints = None
        if constraints_raw:
            try:
                constraints = json.loads(constraints_raw)
            except Exception as exc:
                st.warning(f"JSON invalide: {exc}")

    if model_choice != "heston":
        st.info("Sélectionnez HESTON pour lancer la calibration NN.")
        return

    st.subheader("Surface IV marché")
    uploaded_file = st.file_uploader("Charger une surface IV (CSV)", type=["csv"])
    if uploaded_file:
        try:
            preview_df = pd.read_csv(uploaded_file).head()
            st.dataframe(preview_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"Impossible de lire le CSV: {exc}")

    st.subheader("Paramètres du sous-jacent")
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("S0", value=100.0, min_value=0.01, step=1.0)
    with col2:
        r = st.number_input("Taux sans risque r", value=0.02, step=0.001, format="%.4f")
    with col3:
        q = st.number_input("Dividende q", value=0.0, step=0.001, format="%.4f")

    if st.button("Calibrate Heston (NN)", type="primary"):
        csv_bytes = uploaded_file.getvalue() if uploaded_file else None
        payload = {
            "model": model_choice,
            "source": "file_upload" if csv_bytes is not None else "api_placeholder",
            "csv_bytes": csv_bytes,
            "constraints": constraints if isinstance(constraints, dict) else None,
            "S0": S0,
            "r": r,
            "q": q,
        }
        result = ctrl.run_heston_nn_calibration(payload)
        st.session_state["last_calibration_result"] = result

    result = st.session_state.get("last_calibration_result")
    if result:
        if not result.get("success"):
            st.warning(result.get("message", "Calibration échouée."))
            return
        st.success(result.get("message", "OK"))
        params = result.get("params") or {}
        st.json(params)

        m_grid = np.array(result.get("m_grid") or [])
        t_grid = np.array(result.get("t_grid") or [])
        iv_mkt = np.array(result.get("iv_market") or [])
        iv_model = np.array(result.get("iv_model") or [])
        iv_err = np.array(result.get("iv_error") or [])
        if iv_mkt.size and iv_model.size and iv_err.size:
            st.markdown("#### Surfaces IV")
            _plot_heatmap(iv_mkt, m_grid, t_grid, "IV marché")
            _plot_heatmap(iv_model, m_grid, t_grid, "IV modèle")
            _plot_heatmap(iv_err, m_grid, t_grid, "Erreur (modèle - marché)")
        else:
            st.info("Aucune surface à afficher.")


# Backward-compatible alias
render = render_tab
