"""
Pure helper functions for CRR heatmap generation (UI-agnostic).
These wrap the existing numerical utilities under `app.model.options.shared`.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from app.model.options.core import shared as opt_shared


def heatmap_axis(center: float, span: float, n_points: int = 31) -> np.ndarray:
    """
    Build a symmetric axis around a reference level.

    Args:
        center: reference level (spot or strike).
        span: half-range around the center.
        n_points: number of grid points.
    """
    return opt_shared.heatmap_axis(center, span, n_points=n_points)


def compute_crr_heatmaps(
    spot_axis: Iterable[float],
    strike_axis: Iterable[float],
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CRR call/put matrices on the provided axes.

    Returns:
        (call_matrix, put_matrix) as numpy arrays.
    """
    return opt_shared.compute_american_crr_heatmaps(spot_axis, strike_axis, T, r, sigma, n_steps)


def compute_american_crr_heatmaps(
    spot_axis: Iterable[float],
    strike_axis: Iterable[float],
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible alias for CRR heatmap computation.
    """
    return compute_crr_heatmaps(spot_axis, strike_axis, T, r, sigma, n_steps)


def render_heatmap(
    matrix: np.ndarray,
    x_axis: Iterable[float],
    y_axis: Iterable[float],
    title: str = "Heatmap",
    x_label: str = "Strike",
    y_label: str = "Spot",
) -> go.Figure:
    """
    Build a Plotly heatmap figure (pure, UI-agnostic).
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(x_axis),
            y=list(y_axis),
            colorscale="Viridis",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig
