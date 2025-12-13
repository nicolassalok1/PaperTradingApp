"""
Controller bridge exposing options helpers and pricing views to the UI.
All logic is routed through app.controller.options_controller (no direct model calls).
"""

from __future__ import annotations

import datetime
import streamlit as st
import altair as alt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from types import SimpleNamespace

from app.controller import options_controller as oc
from app.vue.components.options import ui_helpers as opt_ui
from app.vue.state.options_context import get_option_context
from app.controller.options_controller import floor_n

# Helpers
_choose_option_select = opt_ui._choose_option_select
_render_option_text = opt_ui._render_option_text
render_method_explainer = opt_ui.render_method_explainer
_get_cached_iv_for = oc._get_cached_iv_for
_render_heatmaps_for_current_option = opt_ui._render_heatmaps_for_current_option
# Legacy globals (kept for backward compatibility with older UI modules).
common_spot_value = opt_ui.common_spot_value
common_maturity_value = opt_ui.common_maturity_value
common_rate_value = opt_ui.common_rate_value
common_sigma_value = opt_ui.common_sigma_value
d_common = opt_ui.d_common
option_char = opt_ui.option_char
_k = opt_ui._k

# Close series guard
def ensure_close_history(ctx: dict) -> bool:
    """
    Ensure closing prices are available for the current global ticker.
    When unavailable, show a notice and skip rendering.
    """
    available = bool(ctx.get("close_available"))
    if available:
        return True
    ticker = ctx.get("ticker") or resolve_common_underlying()
    tkr_display = (ticker or "").strip().upper() or "N/A"
    st.info(
        f"Clotures introuvables pour le ticker global ({tkr_display}). "
        "Renseigne un ticker valide puis recharge les clÙtures pour afficher ce panneau."
    )
    return False


def current_ticker(ctx: dict) -> str:
    """Return the active ticker used for charts/labels."""
    return (ctx.get("ticker") or resolve_common_underlying() or "").strip().upper()


def current_spot(ctx: dict) -> float:
    """
    Derive a spot anchor from the latest close when available,
    then fallback to S0/context defaults.
    """
    close_series = ctx.get("close_series")
    spot = None
    try:
        if hasattr(close_series, "dropna") and len(close_series.dropna()) > 0:
            spot = float(close_series.dropna().iloc[-1])
    except Exception:
        spot = None
    if spot is None and ctx.get("S0") is not None:
        try:
            spot = float(ctx.get("S0"))
        except Exception:
            spot = None
    if spot is None:
        spot = float(st.session_state.get("common_spot_value", 100.0))
    return float(spot)


def get_common_maturity_value(default: float = 1.0) -> float:
    try:
        return float(st.session_state.get("common_maturity_value", default))
    except Exception:
        return float(default)


def get_common_rate_value(default: float = 0.01) -> float:
    try:
        return float(st.session_state.get("common_rate_value", default))
    except Exception:
        return float(default)


def get_common_sigma_value(default: float = 0.2) -> float:
    try:
        return float(st.session_state.get("common_sigma_value", default))
    except Exception:
        return float(default)


def get_common_div_yield(default: float = 0.0) -> float:
    try:
        return float(st.session_state.get("d_common", default))
    except Exception:
        return float(default)


def get_rate_for_ttm(T: float, default: float = 0.01) -> float:
    """
    Resolve the risk-free rate for a given maturity:
    - If Options is configured to use Yield Curve, query yc.get_risk_free_rate(T).
    - Otherwise, fall back to the global manual rate (common_rate_value).
    """
    try:
        use_yc = bool(st.session_state.get("opt_use_yield_curve_rate", True))
    except Exception:
        use_yc = False

    if use_yc:
        currency = (st.session_state.get("yc_currency") or "USD").strip().upper()
        try:
            from app.controller import yieldcurve_controller as yc  # local import (UI helper)

            return float(
                yc.get_risk_free_rate(T_ref=float(T), currency=currency, ensure_cache=True)
            )
        except Exception:
            pass

    return float(get_common_rate_value(default))

# View/payoff builders
view_asset_or_nothing = oc.view_asset_or_nothing
view_barrier = oc.view_barrier
view_butterfly = oc.view_butterfly
view_call_spread = oc.view_call_spread
view_calendar_spread = oc.view_calendar_spread
view_chooser = oc.view_chooser
view_cliquet = oc.view_cliquet
view_condor = oc.view_condor
view_diagonal_spread = oc.view_diagonal_spread
view_digital = oc.view_digital
view_european = oc.view_european
view_american = oc.view_american
view_bermudan = oc.view_bermudan
view_forward_start = oc.view_forward_start
view_iron_butterfly = oc.view_iron_butterfly
view_iron_condor = oc.view_iron_condor
view_lookback = oc.view_lookback
view_lookback_fixed = oc.view_lookback_fixed
view_put_spread = oc.view_put_spread
view_quanto = oc.view_quanto
view_rainbow = oc.view_rainbow
view_straddle = oc.view_straddle
view_strangle = oc.view_strangle
view_asian_arith = oc.view_asian_arith
view_asian_geom = oc.view_asian_geom
build_crr_tree = oc.build_crr_tree
plot_crr_tree = oc.plot_crr_tree

# Market data
get_data = oc.get_data
load_or_fetch_closing_history = oc.load_or_fetch_closing_history
load_close_series_for_ticker = oc.load_close_series_for_ticker
clear_closing_history_cache = oc.clear_closing_history_cache

# Pricing extras
price_iron_butterfly_bs = oc.price_iron_butterfly_bs
compute_price_mc = oc.compute_price_mc
price_european_from_market = oc.price_european_from_market
try:
    price_option_mc_unified = oc.price_option_mc_unified
except Exception:  # pragma: no cover - safety fallback
    def price_option_mc_unified(*args, **kwargs):
        raise ImportError("price_option_mc_unified indisponible")


def _bootstrap_fake_streamlit():
    """
    When running modules in bare Python (tests, scripts) Streamlit emits warnings because
    there is no ScriptRunContext. In that situation we stub the minimal API surface we use
    so imports and helpers work without noisy warnings.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    if get_script_run_ctx() is not None or getattr(st, "_codex_fake_streamlit", False):
        return

    class FakeState(dict):
        def __getattr__(self, x):
            return self.get(x, None)

    st.session_state = FakeState()

    # UI primitives used across option panels
    st.write = lambda *args, **kwargs: None
    st.markdown = lambda *args, **kwargs: None
    st.caption = lambda *args, **kwargs: None
    st.text = lambda *args, **kwargs: None
    st.metric = lambda *args, **kwargs: None
    st.line_chart = lambda *args, **kwargs: None
    st.bar_chart = lambda *args, **kwargs: None
    st.dataframe = lambda *args, **kwargs: None
    st.pyplot = lambda *args, **kwargs: None

    st.slider = lambda *args, **kwargs: kwargs.get("value", 0)
    st.selectbox = lambda *args, **kwargs: kwargs.get("options", [None])[kwargs.get("index", 0)] if kwargs.get("options") else None
    st.number_input = lambda *args, **kwargs: kwargs.get("value", 0)
    st.button = lambda *args, **kwargs: False
    st.columns = lambda n: [SimpleNamespace() for _ in range(n)]

    st._codex_fake_streamlit = True


_bootstrap_fake_streamlit()


def resolve_common_underlying() -> str:
    """Return the shared ticker set by the user (empty string if unset)."""
    ticker = (
        st.session_state.get("tkr_common")
        or st.session_state.get("common_underlying")
        or st.session_state.get("ticker_default")
        or ""
    )
    return str(ticker or "").strip().upper()


def load_shared_close_series(fallback_value: float):
    """
    Load close series for the shared ticker only if the user provided one.
    Returns (ticker, series|None).
    """
    ticker = resolve_common_underlying()
    if not ticker or load_close_series_for_ticker is None:
        return ticker, None
    try:
        return ticker, load_close_series_for_ticker(ticker, fallback_value=fallback_value)
    except Exception:
        return ticker, None


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

__all__ = [
    "_choose_option_select",
    "_render_option_text",
    "render_method_explainer",
    "_get_cached_iv_for",
    "_render_heatmaps_for_current_option",
    "common_spot_value",
    "common_maturity_value",
    "common_rate_value",
    "common_sigma_value",
    "d_common",
    "option_char",
    "_k",
    "ensure_close_history",
    "current_ticker",
    "current_spot",
    "get_common_maturity_value",
    "get_common_rate_value",
    "get_common_sigma_value",
    "get_common_div_yield",
    "get_rate_for_ttm",
    "view_asset_or_nothing",
    "view_barrier",
    "view_butterfly",
    "view_call_spread",
    "view_calendar_spread",
    "view_chooser",
    "view_cliquet",
    "view_condor",
    "view_diagonal_spread",
    "view_digital",
    "view_european",
    "view_american",
    "view_bermudan",
    "view_forward_start",
    "view_iron_butterfly",
    "view_iron_condor",
    "view_lookback",
    "view_lookback_fixed",
    "view_put_spread",
    "view_quanto",
    "view_rainbow",
    "view_straddle",
    "view_strangle",
    "view_asian_arith",
    "view_asian_geom",
    "build_crr_tree",
    "plot_crr_tree",
    "get_option_context",
    "compute_price_mc",
    "price_european_from_market",
    "get_data",
    "load_or_fetch_closing_history",
    "load_close_series_for_ticker",
    "clear_closing_history_cache",
    "price_iron_butterfly_bs",
    "resolve_common_underlying",
    "load_shared_close_series",
    "render_static_line_chart",
    "render_figures_grid",
    "build_close_with_strike_fig",
    "show_and_close",
    "floor_n",
    "datetime",
    "np",
    "pd",
    "plt",
]
