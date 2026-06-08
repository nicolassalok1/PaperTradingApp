"""Behavior tests for the controller_bridge helpers extracted in the Step-6 refactor.

These functions had ZERO tests before; testing them through the bridge façade both
guards the extraction (transcription, not just signature) and locks behavior. No network.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Importing the bridge triggers _bootstrap_fake_streamlit() -> st is stubbed for bare runs.
import app.vue.components.options.controller_bridge as cb

pytestmark = pytest.mark.unit


def test_current_ticker_uppercases_ctx():
    assert cb.current_ticker({"ticker": "aapl"}) == "AAPL"
    assert cb.current_ticker({}) == ""


def test_current_spot_prefers_close_then_s0_then_default():
    s = pd.Series([1.0, 2.0, 3.5])
    assert cb.current_spot({"close_series": s}) == 3.5
    assert cb.current_spot({"S0": 42.0}) == 42.0
    assert cb.current_spot({}) == 100.0  # session-state default


def test_common_value_defaults():
    assert cb.get_common_rate_value() == 0.01
    assert cb.get_common_sigma_value() == 0.2
    assert cb.get_common_maturity_value() == 1.0
    assert cb.get_common_div_yield() == 0.0


def test_get_rate_for_ttm_manual_fallback():
    cb.st.session_state["opt_use_yield_curve_rate"] = False
    assert cb.get_rate_for_ttm(1.0, default=0.03) == 0.03


def test_ensure_close_history_paths(monkeypatch):
    monkeypatch.setattr(cb.st, "info", lambda *a, **k: None, raising=False)
    assert cb.ensure_close_history({"close_available": True}) is True
    assert cb.ensure_close_history({"close_available": False}) is False


def test_render_static_line_chart_guards():
    assert cb.render_static_line_chart(None) is False
    assert cb.render_static_line_chart(pd.Series([], dtype=float)) is False


def test_build_close_with_strike_fig_none_on_empty():
    assert cb.build_close_with_strike_fig(None, "AAPL", 100.0) is None


def test_resolve_common_underlying_empty_by_default():
    # FakeState returns "" for unset keys -> empty resolved ticker.
    for k in ("tkr_common", "common_underlying", "ticker_default"):
        cb.st.session_state.pop(k, None)
    assert cb.resolve_common_underlying() == ""
