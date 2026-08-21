"""p4 skeptic probe (G7_viewB) — does the title/legend overlap in
`_base_layout` persist when the legend fits on ONE line (wide screen)?

Builds self-contained HTML figures reproducing the view's layout
(`margin.t=48`, legend h / y=1.02 / yanchor=bottom) with the geometry the
Streamlit frontend forces (title xanchor=left, x=0, bold, 16 px — see
`applyStreamlitTheme` in the Streamlit JS bundle) at 900 px and 1400 px.
The bboxes of `.gtitle` and `.legend` are then measured in a browser.

Usage: python p4_g7_viewB_title_legend_overlap.py <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _base_layout(fig: go.Figure, *, height: int = 380, ytitle: str = "", xtitle: str = "") -> None:
    # verbatim copy of app/vue/tabs/tab_iv_dashboard.py::_base_layout
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=xtitle or None,
        yaxis_title=ytitle or None,
        hovermode="closest",
    )


def _streamlit_like(fig: go.Figure, width: int) -> None:
    # what Streamlit's frontend applies on top for theme="streamlit"
    fig.update_layout(
        width=width,
        template="plotly_dark",
        title=dict(text=f"<b>{fig.layout.title.text}</b>", xanchor="left", x=0, font=dict(size=16)),
        legend=dict(font=dict(size=12)),
        font=dict(size=12),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )


def series_fig(n_entries: int = 7) -> go.Figure:
    idx = pd.bdate_range("2024-01-01", periods=500)
    vol = pd.Series(0.15 + 0.03 * np.sin(np.arange(500) / 20), index=idx)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=vol, mode="lines", name="Vol réalisée (20 j)"))
    for level, name in ((0.18, "75e percentile"), (0.12, "25e percentile"), (0.15, "Moyenne")):
        fig.add_trace(go.Scatter(x=[idx[0]], y=[level], mode="lines", name=name, line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=[idx[-1]], y=[0.16], mode="markers", name="RV courante"))
    if n_entries >= 6:
        fig.add_trace(go.Scatter(x=idx[-30:], y=vol.iloc[-30:] * 1.1, mode="lines+markers", name="IV ATM (historique local)"))
    if n_entries >= 7:
        fig.add_trace(go.Scatter(x=[idx[-1]], y=[0.17], mode="markers", name="IV ATM courante", marker=dict(symbol="star", size=13)))
    fig.update_layout(title="Série de volatilité et bandes de régime")
    _base_layout(fig, height=400, ytitle="Vol annualisée")
    fig.update_yaxes(tickformat=".0%")
    return fig


def forward_fig() -> go.Figure:
    x = np.linspace(0.08, 0.25, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x * 0.6 + 0.07, mode="markers", name="Observations"))
    fig.add_trace(go.Scatter(x=x, y=x * 0.6 + 0.07, mode="lines", name="Régression (R² = 0.412)"))
    fig.add_trace(go.Scatter(x=[0.08, 0.25], y=[0.08, 0.25], mode="lines", name="y = x (aucun changement)"))
    fig.update_layout(title="Vol forward 30 j vs vol courante — y = 0.602x + 0.066")
    _base_layout(fig, height=380, xtitle="Vol courante", ytitle="Vol forward moyenne 30 j")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "series_900_7entries": (series_fig(7), 900),
        "series_1400_7entries": (series_fig(7), 1400),
        "series_1400_5entries": (series_fig(5), 1400),  # no IV layer (Alpaca options off)
        "forward_560": (forward_fig(), 560),
        "forward_700": (forward_fig(), 700),
    }
    for name, (fig, width) in cases.items():
        _streamlit_like(fig, width)
        path = out_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs=True, full_html=True, div_id="fig")
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
