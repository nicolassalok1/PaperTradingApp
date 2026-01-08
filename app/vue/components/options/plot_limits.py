"""
Small helpers to constrain chart/figure widths inside the Options tab.
"""

from __future__ import annotations

import matplotlib.figure as mpl_fig


def limit_figure_width(fig: mpl_fig.Figure, max_width_px: float = 500.0) -> mpl_fig.Figure:
    """
    Scale down a matplotlib figure so its rendered width does not exceed max_width_px.
    Preserves aspect ratio by scaling height proportionally.
    """
    try:
        dpi = float(fig.get_dpi() or 100.0)
        max_width_in = float(max_width_px) / dpi
        width_in, height_in = fig.get_size_inches()
        if width_in > max_width_in and width_in > 0:
            scale = max_width_in / width_in
            fig.set_size_inches(max_width_in, height_in * scale, forward=True)
    except Exception:
        # Best-effort resize; ignore if figure metadata is unavailable.
        return fig
    return fig
