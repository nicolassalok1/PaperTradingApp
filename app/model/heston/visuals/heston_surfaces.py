import numpy as np
import plotly.graph_objects as go


def plot_surface_3d(maturities, strikes, iv_grid, title="IV Surface"):
    fig = go.Figure(data=[go.Surface(z=iv_grid, x=strikes, y=maturities)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Maturity",
            zaxis_title="IV",
        ),
        height=600,
    )
    return fig


def plot_heatmap(maturities, strikes, iv_grid, title="IV Heatmap"):
    fig = go.Figure(
        data=go.Heatmap(
            z=iv_grid,
            x=strikes,
            y=maturities,
            colorscale="Viridis",
        )
    )
    fig.update_layout(title=title, height=500)
    return fig


__all__ = ["plot_surface_3d", "plot_heatmap"]
