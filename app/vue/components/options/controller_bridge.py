"""
Controller bridge exposing options helpers and pricing views to the UI.
All logic is routed through app.controller.options_controller (no direct model calls).
"""

from __future__ import annotations

import datetime
import math
import streamlit as st

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
render_add_to_dashboard_button = opt_ui.render_add_to_dashboard_button
common_spot_value = opt_ui.common_spot_value
common_maturity_value = opt_ui.common_maturity_value
common_rate_value = opt_ui.common_rate_value
common_sigma_value = opt_ui.common_sigma_value
d_common = opt_ui.d_common
option_char = opt_ui.option_char
_k = opt_ui._k

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

# Book / PnL / logging
add_option_to_dashboard_clean = oc.add_option_to_dashboard_clean
log_action = oc.log_action
load_options_book = oc.load_options_book
save_options_book = oc.save_options_book
compute_option_pnl = oc.compute_option_pnl
load_expired = oc.load_expired
save_expired = oc.save_expired

# Market data
get_data = oc.get_data
load_or_fetch_closing_history = oc.load_or_fetch_closing_history
load_close_series_for_ticker = oc.load_close_series_for_ticker
clear_closing_history_cache = oc.clear_closing_history_cache

# Pricing extras
price_iron_butterfly_bs = oc.price_iron_butterfly_bs
compute_price_mc = oc.compute_price_mc
price_european_from_cboe = oc.price_european_from_cboe


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
        or st.session_state.get("heston_cboe_ticker")
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
    "render_add_to_dashboard_button",
    "common_spot_value",
    "common_maturity_value",
    "common_rate_value",
    "common_sigma_value",
    "d_common",
    "option_char",
    "_k",
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
    "add_option_to_dashboard_clean",
    "log_action",
    "get_option_context",
    "load_options_book",
    "save_options_book",
    "compute_option_pnl",
    "load_expired",
    "save_expired",
    "compute_price_mc",
    "price_european_from_cboe",
    "get_data",
    "load_or_fetch_closing_history",
    "load_close_series_for_ticker",
    "clear_closing_history_cache",
    "price_iron_butterfly_bs",
    "resolve_common_underlying",
    "load_shared_close_series",
    "show_and_close",
    "floor_n",
    "math",
    "datetime",
    "np",
    "pd",
    "plt",
]
