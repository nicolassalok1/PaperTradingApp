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
from app.vue.components.options.plot_limits import (
    MAX_CHART_WIDTH_PX,
    limit_figure_width,
    mark_full_width,
)
from app.vue.components.options import ui_helpers as opt_ui
from app.vue.state.options_context import get_option_context
from app.controller.options_controller import floor_n

# Helpers
_choose_option_select = opt_ui._choose_option_select
_render_option_text = opt_ui._render_option_text
render_method_explainer = opt_ui.render_method_explainer
# The ui_helpers wrapper injects the loaded IV surface (opt_iv_surface_df) and the
# shared ticker before delegating to the controller; the bare controller lookup has
# neither and always resolves to None.
_get_cached_iv_for = opt_ui._get_cached_iv_for
_render_heatmaps_for_current_option = opt_ui._render_heatmaps_for_current_option
sigma_from_cache_or_default = opt_ui.sigma_from_cache_or_default
# Legacy globals (kept for backward compatibility with older UI modules).
common_spot_value = opt_ui.common_spot_value
common_maturity_value = opt_ui.common_maturity_value
common_rate_value = opt_ui.common_rate_value
common_sigma_value = opt_ui.common_sigma_value
d_common = opt_ui.d_common
option_char = opt_ui.option_char
_k = opt_ui._k

# Close series guard + context/rate helpers — extracted to bridge_context (Step-6).
# Re-exported here so the public façade and `import *` consumers are unchanged.
from app.vue.components.options.bridge_context import (  # noqa: E402
    ensure_close_history,
    current_ticker,
    current_spot,
    get_common_maturity_value,
    get_common_rate_value,
    get_common_sigma_value,
    get_common_div_yield,
    get_rate_for_ttm,
    resolve_common_underlying,
    load_shared_close_series,
)

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


# Rendering helpers — extracted to bridge_render (Step-6). Re-exported unchanged.
# bridge_render carries main's plot_limits width-limiting (ported into the submodule).
from app.vue.components.options.bridge_render import (  # noqa: E402
    render_static_line_chart,
    render_figures_grid,
    build_close_with_strike_fig,
    show_and_close,
)

__all__ = [
    "_choose_option_select",
    "_render_option_text",
    "render_method_explainer",
    "_get_cached_iv_for",
    "_render_heatmaps_for_current_option",
    "sigma_from_cache_or_default",
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
    "mark_full_width",
    "limit_figure_width",
    "build_close_with_strike_fig",
    "show_and_close",
    "floor_n",
    "datetime",
    "np",
    "pd",
    "plt",
]
