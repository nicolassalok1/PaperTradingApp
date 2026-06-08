"""
Streamlit component for side-by-side model comparison after calibration.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False

try:
    import pandas as pd
    _PANDAS = True
except Exception:
    _PANDAS = False


_MODEL_LABELS: Dict[str, str] = {
    "heston_v1": "Heston",
    "sabr": "SABR",
    "merton_jump_diffusion": "Jump Diffusion",
    "rheston": "rHeston",
    "rbergomi": "rBergomi",
    "volterra": "Volterra SDE",
}


def _label(key: str) -> str:
    return _MODEL_LABELS.get(str(key), str(key))


def render_model_comparison(history: Dict[str, Any]) -> None:
    """
    Render a side-by-side comparison of all calibrated models stored in history.
    history: dict mapping model_key -> result_dict (as returned by the controller)
    """
    if not history:
        st.info("Aucun modèle calibré. Lance une calibration pour alimenter la comparaison.")
        return

    valid = {k: v for k, v in history.items() if isinstance(v, dict) and v.get("success")}
    if not valid:
        st.warning("Aucun résultat de calibration valide dans l'historique.")
        return

    # --- Metrics table ---
    st.markdown("#### Métriques comparatives")
    if _PANDAS:
        rows = []
        for key, res in valid.items():
            m = res.get("metrics") or {}
            m_vw = res.get("metrics_vw") or {}
            rows.append({
                "Modèle": _label(key),
                "MAE (IV)": m.get("mae", float("nan")),
                "RMSE (IV)": m.get("rmse", float("nan")),
                "Max |err|": m.get("max_abs", float("nan")),
                "MAE (vega-wt)": m_vw.get("mae_vw", float("nan")),
                "RMSE (vega-wt)": m_vw.get("rmse_vw", float("nan")),
                "N pts": int(m.get("n", 0)),
            })
        df = pd.DataFrame(rows).set_index("Modèle")
        st.dataframe(
            df.style.format("{:.4f}", subset=["MAE (IV)", "RMSE (IV)", "Max |err|", "MAE (vega-wt)", "RMSE (vega-wt)"]),
            use_container_width=True,
        )

    if not _PLOTLY:
        st.info("Plotly non disponible — visualisation des surfaces désactivée.")
        return

    # --- IV surface heatmaps ---
    st.markdown("#### Surfaces IV calibrées")

    n_models = len(valid)
    # Cap columns to avoid too narrow layout
    n_cols = min(n_models, 3)
    model_keys = list(valid.keys())

    # Market surface reference (take from first result — they share the same market data)
    first_res = next(iter(valid.values()))
    m_grid_ref = np.asarray(first_res.get("m_grid") or [], dtype=float)
    t_grid_ref = np.asarray(first_res.get("t_grid") or [], dtype=float)
    iv_mkt_ref = np.asarray(first_res.get("iv_market") or [], dtype=float)

    def _heatmap_fig(z: np.ndarray, title: str, colorscale: str = "RdYlGn") -> "go.Figure":
        fig = go.Figure(go.Heatmap(
            z=z,
            x=[f"{m:.2f}" for m in m_grid_ref],
            y=[f"{t:.2f}y" for t in t_grid_ref],
            colorscale=colorscale,
            colorbar=dict(thickness=10),
            hovertemplate="m=%{x}<br>T=%{y}<br>IV=%{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Moneyness",
            yaxis_title="Maturité",
            height=280,
            margin=dict(l=40, r=10, t=40, b=40),
        )
        return fig

    # Show market reference once
    if iv_mkt_ref.ndim == 2 and m_grid_ref.size and t_grid_ref.size:
        with st.expander("Surface marché (référence)", expanded=False):
            st.plotly_chart(_heatmap_fig(iv_mkt_ref, "Surface IV marché"), use_container_width=True)

    # IV model surfaces
    cols_iv = st.columns(n_cols)
    for i, key in enumerate(model_keys):
        res = valid[key]
        iv_model = np.asarray(res.get("iv_model") or [], dtype=float)
        if iv_model.ndim != 2:
            continue
        with cols_iv[i % n_cols]:
            st.plotly_chart(
                _heatmap_fig(iv_model, f"IV modèle — {_label(key)}"),
                use_container_width=True,
            )

    # IV error heatmaps
    st.markdown("#### Erreurs IV (modèle − marché)")
    cols_err = st.columns(n_cols)
    for i, key in enumerate(model_keys):
        res = valid[key]
        iv_err = np.asarray(res.get("iv_error") or [], dtype=float)
        if iv_err.ndim != 2:
            continue
        with cols_err[i % n_cols]:
            fig_err = _heatmap_fig(iv_err, f"Erreur — {_label(key)}", colorscale="RdBu")
            # Centre the colorscale on 0
            abs_max = float(np.nanmax(np.abs(iv_err))) if np.any(np.isfinite(iv_err)) else 0.05
            fig_err.update_traces(zmin=-abs_max, zmax=abs_max)
            st.plotly_chart(fig_err, use_container_width=True)

    # --- Bar chart: RMSE per model ---
    st.markdown("#### Comparaison RMSE")
    model_names = [_label(k) for k in valid]
    rmse_vals = [(valid[k].get("metrics") or {}).get("rmse", float("nan")) for k in valid]
    rmse_vw_vals = [(valid[k].get("metrics_vw") or {}).get("rmse_vw", float("nan")) for k in valid]

    fig_bar = go.Figure()
    fig_bar.add_bar(name="RMSE uniforme", x=model_names, y=rmse_vals, marker_color="#4C8BF5")
    fig_bar.add_bar(name="RMSE vega-pondéré", x=model_names, y=rmse_vw_vals, marker_color="#F5A623", opacity=0.85)
    fig_bar.update_layout(
        barmode="group",
        xaxis_title="Modèle",
        yaxis_title="RMSE (IV)",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


__all__ = ["render_model_comparison"]
