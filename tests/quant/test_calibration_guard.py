"""Unit tests for the generic anti-false-success calibration guard.

A calibrator must never report success=True while returning a degenerate
(all-NaN) model surface or non-finite metrics. See the rHeston overflow bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import (
    SurfaceCalibrationResult,
    apply_degeneracy_guard,
)

pytestmark = pytest.mark.unit


def _result(*, success, iv_model, metrics):
    return SurfaceCalibrationResult(
        success=success,
        message="OK",
        model="test",
        method="test",
        params={},
        metrics=metrics,
        iv_model=iv_model,
    )


def test_guard_flips_all_nan_surface_to_failure():
    res = _result(
        success=True,
        iv_model=np.full((3, 5), np.nan),
        metrics={"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "n": 0.0},
    )
    out = apply_degeneracy_guard(res)
    assert out is res
    assert res.success is False
    assert "dégénér" in res.message.lower()


def test_guard_flips_non_finite_metrics_to_failure():
    iv = np.full((3, 5), np.nan)
    iv[0, 0] = 0.2  # one finite cell, but metrics are non-finite
    res = _result(
        success=True,
        iv_model=iv,
        metrics={"mae": float("inf"), "rmse": 0.1, "max_abs": 0.2, "n": 1.0},
    )
    apply_degeneracy_guard(res)
    assert res.success is False
    assert "métriques non finies" in res.message  # accurate reason, not "surface NaN"


def test_guard_keeps_partial_nan_success():
    # Legitimate SABR-style result: some maturities NaN, but finite metrics.
    iv = np.full((3, 5), np.nan)
    iv[1:, :] = 0.2
    res = _result(
        success=True,
        iv_model=iv,
        metrics={"mae": 0.001, "rmse": 0.002, "max_abs": 0.004, "n": 10.0},
    )
    apply_degeneracy_guard(res)
    assert res.success is True
    assert res.message == "OK"


def test_guard_is_noop_on_already_failed_result():
    res = _result(success=False, iv_model=None, metrics=None)
    apply_degeneracy_guard(res)
    assert res.success is False
