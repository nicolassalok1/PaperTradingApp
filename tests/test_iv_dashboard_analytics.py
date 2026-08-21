"""Unit tests for the 🌡️ Vol Implicite analytics (pure math, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model.iv_dashboard import analytics as ivx

pytestmark = pytest.mark.unit


def _mean_reverting_vol(n: int = 700, seed: int = 7) -> pd.Series:
    """Synthetic mean-reverting (OU-like) vol series in [0.05, 0.9]."""
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    vol[0] = 0.20
    for i in range(1, n):
        vol[i] = vol[i - 1] + 0.15 * (0.20 - vol[i - 1]) + rng.normal(0.0, 0.01)
    vol = np.clip(vol, 0.05, 0.9)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(vol, index=idx)


# --------------------------------------------------------------------------- #
# Realized vol
# --------------------------------------------------------------------------- #
def test_realized_vol_constant_prices_is_zero():
    closes = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=80))
    rv = ivx.compute_realized_vol(closes, window=20)
    assert not rv.dropna().empty
    assert float(rv.dropna().abs().max()) < 1e-12


def test_realized_vol_known_magnitude():
    # Alternating +1% / -1% log returns -> daily std ~= 1%, annualized ~= 15.9%
    n = 260
    log_rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    closes = pd.Series(
        100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])),
        index=pd.bdate_range("2024-01-01", periods=n + 1),
    )
    rv = ivx.compute_realized_vol(closes, window=20).dropna()
    expected = 0.01 * np.sqrt(252)
    assert rv.iloc[-1] == pytest.approx(expected, rel=0.05)


def test_realized_vol_warmup_is_nan():
    closes = pd.Series(
        np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30)
    )
    rv = ivx.compute_realized_vol(closes, window=20)
    # first (window - 1) return rows have no full window yet
    assert rv.iloc[: 20 - 2].isna().all()


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #
def test_percentile_series_monotonic_hits_one():
    vol = pd.Series(np.linspace(0.1, 0.5, 300))
    pct = ivx.compute_percentile_series(vol, window=252, min_periods=60)
    assert pct.iloc[:59].isna().all()
    assert pct.iloc[-1] == pytest.approx(1.0)


def test_percentile_within_bounds_and_nan():
    hist = pd.Series(np.linspace(0.1, 0.3, 100))
    assert ivx.percentile_within(hist, 0.05) == pytest.approx(0.0)
    assert ivx.percentile_within(hist, 0.5) == pytest.approx(1.0)
    mid = ivx.percentile_within(hist, 0.2)
    assert 0.4 < mid < 0.6
    assert np.isnan(ivx.percentile_within(hist, float("nan")))
    assert np.isnan(ivx.percentile_within(pd.Series(dtype=float), 0.2))


# --------------------------------------------------------------------------- #
# Regime classification (legacy thresholds)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p, key, signal",
    [
        (0.95, "high", "down"),
        (0.81, "high", "down"),
        (0.7, "above", "neutral"),
        (0.5, "normal", "neutral"),
        (0.3, "below", "neutral"),
        (0.19, "low", "up"),
        (0.05, "low", "up"),
    ],
)
def test_classify_regime_buckets(p, key, signal):
    out = ivx.classify_regime(p)
    assert out["key"] == key
    assert out["signal_key"] == signal
    assert out["label"] and out["signal_label"]


def test_classify_regime_nan():
    out = ivx.classify_regime(float("nan"))
    assert out["key"] == "unknown"
    assert out["signal_key"] == "unknown"


# --------------------------------------------------------------------------- #
# Forward-vol analysis
# --------------------------------------------------------------------------- #
def test_analyze_forward_vol_mean_reverting_series():
    vol = _mean_reverting_vol()
    res = ivx.analyze_forward_vol(vol, forward_window=30)

    df = res["df"]
    assert not df.empty
    assert len(df) == len(vol) - 30  # exactly the last forward_window rows dropped

    reg1, reg2 = res["reg_forward"], res["reg_diff"]
    assert np.isfinite(reg1["slope"]) and np.isfinite(reg2["slope"])
    # A mean-reverting series must show slope1 < 1 and slope2 < 0
    assert reg1["slope"] < 1.0
    assert reg2["slope"] < 0.0
    assert 0.0 <= reg1["r2"] <= 1.0
    assert np.isfinite(res["intersection"])
    assert res["n_high"] + res["n_low"] == len(df)
    assert res["insights"], "insights should not be empty"
    assert any("mean" in i.lower() for i in res["insights"])


def test_analyze_forward_vol_regime_regressions_present_on_long_series():
    vol = _mean_reverting_vol(n=900, seed=11)
    res = ivx.analyze_forward_vol(vol, forward_window=30)
    # With a long OU series both regimes should have > 10 points
    assert res["reg_high"] is not None
    assert res["reg_low"] is not None
    assert res["reg_high"]["n"] == res["n_high"]
    assert res["reg_low"]["n"] == res["n_low"]


def test_analyze_forward_vol_insufficient_data_raises():
    vol = pd.Series(np.full(40, 0.2))  # 40 - 30 = 10 < MIN_ANALYSIS_POINTS
    with pytest.raises(ValueError):
        ivx.analyze_forward_vol(vol, forward_window=30)


def test_analyze_forward_vol_handles_percentile_gaps():
    vol = _mean_reverting_vol(n=400, seed=3)
    pct = ivx.compute_percentile_series(vol, window=252, min_periods=60)
    res = ivx.analyze_forward_vol(vol, forward_window=30, percentile=pct)
    # Percentile warm-up NaNs must NOT shrink the regression sample
    assert len(res["df"]) == len(vol) - 30


# --------------------------------------------------------------------------- #
# Controller sanitization (no network: service call monkeypatched)
# --------------------------------------------------------------------------- #
def test_controller_rejects_empty_symbol():
    from app.controller import iv_dashboard_controller as ctrl

    with pytest.raises(ValueError):
        ctrl.get_iv_analysis("  ")


def test_controller_clamps_and_normalizes(monkeypatch):
    from app.controller import iv_dashboard_controller as ctrl

    captured: dict = {}

    def _fake(symbol, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(ctrl._svc, "get_iv_dashboard_data", _fake)
    out = ctrl.get_iv_analysis(
        " spy ", years=99, rv_window=1, forward_window=1000, percentile_window=10
    )
    assert out == {"ok": True}
    assert captured["symbol"] == "SPY"
    assert captured["years"] == 10.0
    assert captured["rv_window"] == 5
    assert captured["forward_window"] == 90
    assert captured["percentile_window"] == 60
