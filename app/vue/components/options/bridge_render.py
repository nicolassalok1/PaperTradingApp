"""
Rendering helpers extracted from controller_bridge (Step-6 refactor).

These are the matplotlib/altair UI helpers the options bridge defines locally.
controller_bridge re-exports them unchanged, so the public façade (__all__ +
signatures, guarded by tests/characterization) and the ~57 `import *` consumers
are unaffected. View layer: may import streamlit + controllers, never model/utils.
"""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render_static_line_chart(series, title: str | None = None, y_label: str | None = None) -> bool:
    """
    Render a non-interactive Altair line chart for a pandas Series.
    Returns True if rendered, False otherwise.
    """
    if series is None or getattr(series, "empty", True):
        return False
    try:
        df = series.reset_index()
    except Exception:
        return False

    if df.shape[1] < 2:
        return False

    x_col, y_col = df.columns[:2]
    y_title = y_label or str(y_col)

    x_enc = alt.X(f"{x_col}:T", title="Date")
    try:
        # If conversion to temporal fails, fall back to nominal axis
        _ = pd.to_datetime(df[x_col])
    except Exception:
        x_enc = alt.X(f"{x_col}:O", title=str(x_col))

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=x_enc,
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[alt.Tooltip(f"{x_col}:T", title="Date"), alt.Tooltip(f"{y_col}:Q", title=y_title)],
        )
        .properties(title=title or "", height=260)
        .interactive(False)
        .configure_view(continuousHeight=260, strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)
    return True


def render_figures_grid(figs):
    """
    Render a list of matplotlib figures in responsive 2-column rows.
    On narrow viewports Streamlit stacks columns automatically.
    """
    if not figs:
        return
    for i in range(0, len(figs), 2):
        pair = [f for f in figs[i : i + 2] if f is not None]
        if not pair:
            continue
        cols = st.columns(len(pair))
        for col, fig in zip(cols, pair):
            col.pyplot(fig, clear_figure=True)
            plt.close(fig)


def build_close_with_strike_fig(close_series, ticker: str, strike: float | None):
    """Build a closing-price figure with an optional horizontal strike overlay."""
    if close_series is None or getattr(close_series, "empty", True):
        return None
    tkr = (ticker or "Ticker").strip().upper() or "Ticker"
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(close_series.index, close_series.values, label=f"{tkr} close")
        if strike is not None:
            ax.axhline(float(strike), color="gray", linestyle="--", label=f"Strike = {float(strike):.2f}")
        ax.set_ylabel("Prix")
        ax.set_title(f"Clôtures {tkr} (strike)")
        ax.legend(loc="best")
        fig.autofmt_xdate()
        return fig
    except Exception:
        return None


def show_and_close(fig):
    """Render a matplotlib figure in Streamlit and close it to avoid figure leaks."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        # When running outside `streamlit run` (e.g., tests), skip rendering to avoid warnings.
        if get_script_run_ctx() is None:
            plt.close(fig)
            return
    except Exception:
        plt.close(fig)
        return

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
