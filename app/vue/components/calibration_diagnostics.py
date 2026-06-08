"""
Streamlit component for rendering calibration diagnostics.
Receives pre-computed diagnostics dict (from CalibrationController.compute_diagnostics).
No model-layer imports allowed here.
"""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False


def render_calibration_diagnostics(diag: Dict[str, Any]) -> None:
    """
    Render three diagnostic panels:
    1. Résidus par maturité
    2. Résidus par moneyness
    3. Décomposition du smile

    diag: output of CalibrationController.compute_diagnostics(result_dict)
    """
    if not diag:
        st.info("Diagnostics non disponibles (données insuffisantes).")
        return

    tab_mat, tab_mon, tab_smile = st.tabs(["Résidus / maturité", "Résidus / moneyness", "Décomposition smile"])

    with tab_mat:
        by_mat = diag.get("by_maturity") or []
        if not by_mat:
            st.info("Pas de données par maturité.")
        elif _PLOTLY:
            T_labels = [f"T={d['T']:.2f}" for d in by_mat]
            mae_vals = [d["mae"] if d["n"] > 0 else None for d in by_mat]
            rmse_vals = [d["rmse"] if d["n"] > 0 else None for d in by_mat]

            fig = go.Figure()
            fig.add_bar(name="MAE", x=T_labels, y=mae_vals, marker_color="#4C8BF5")
            fig.add_bar(name="RMSE", x=T_labels, y=rmse_vals, marker_color="#F5A623", opacity=0.8)
            fig.update_layout(
                barmode="group",
                title="Erreur IV par maturité",
                xaxis_title="Maturité (années)",
                yaxis_title="Erreur IV",
                height=350,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            try:
                import pandas as pd
                df_mat = pd.DataFrame(by_mat).rename(
                    columns={"T": "Maturité", "mae": "MAE", "rmse": "RMSE", "n": "N pts"}
                )
                st.dataframe(
                    df_mat.set_index("Maturité").style.format("{:.4f}", subset=["MAE", "RMSE"]),
                    use_container_width=True,
                )
            except Exception:
                pass
        else:
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(by_mat), use_container_width=True)
            except Exception:
                st.json(by_mat)

    with tab_mon:
        by_mon = diag.get("by_moneyness") or []
        if not by_mon:
            st.info("Pas de données par moneyness.")
        elif _PLOTLY:
            color_map = {"OTM Put": "#E74C3C", "ATM": "#2ECC71", "OTM Call": "#3498DB"}
            colors = [color_map.get(d["label"], "#95A5A6") for d in by_mon]
            mae_vals = [d["mae"] if d["n"] > 0 else None for d in by_mon]

            fig = go.Figure(go.Bar(
                x=[f"m={d['m']:.2f}" for d in by_mon],
                y=mae_vals,
                marker_color=colors,
                text=[d["label"] for d in by_mon],
                textposition="outside",
            ))
            fig.update_layout(
                title="MAE IV par moneyness",
                xaxis_title="Moneyness (K/S0)",
                yaxis_title="MAE (IV)",
                height=350,
                margin=dict(l=40, r=20, t=40, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(by_mon), use_container_width=True)
            except Exception:
                st.json(by_mon)

    with tab_smile:
        smile = diag.get("smile") or {}
        maturities = smile.get("maturities") or []
        atm_iv = smile.get("atm_iv_by_t") or []
        skew = smile.get("skew_by_t") or []
        curvature = smile.get("curvature_by_t") or []

        if not maturities:
            st.info("Pas de données de décomposition smile.")
        elif _PLOTLY:
            def _safe(vals):
                return [v if v is not None else float("nan") for v in vals]

            t_labels = [f"{t:.2f}y" for t in maturities]

            fig = go.Figure()
            fig.add_scatter(x=t_labels, y=_safe(atm_iv), mode="lines+markers", name="ATM IV", line=dict(color="#2ECC71"))
            fig.add_scatter(x=t_labels, y=_safe(skew), mode="lines+markers", name="Skew (RR)", line=dict(color="#E74C3C", dash="dash"))
            fig.add_scatter(x=t_labels, y=_safe(curvature), mode="lines+markers", name="Courbure (BF)", line=dict(color="#3498DB", dash="dot"))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.update_layout(
                title="Décomposition smile par maturité",
                xaxis_title="Maturité",
                yaxis_title="IV",
                height=380,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Skew = IV(m=0.95) − IV(m=1.05) · Courbure = IV(0.90) + IV(1.10) − 2·IV(ATM)")
        else:
            try:
                import pandas as pd
                df_smile = pd.DataFrame({
                    "Maturité": maturities,
                    "ATM IV": atm_iv,
                    "Skew": skew,
                    "Courbure": curvature,
                })
                st.dataframe(df_smile, use_container_width=True)
            except Exception:
                pass


__all__ = ["render_calibration_diagnostics"]
