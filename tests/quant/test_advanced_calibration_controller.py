"""End-to-end test of the advanced-calibration controller path, including the
anti-false-success guard wiring. Uses SABR (fast, analytic) through the full
DataFrame -> dispatch -> result-dict pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.controller.calibration_controller import CalibrationController

pytestmark = pytest.mark.slow


def _call_surface_df():
    rows = []
    for T in (0.25, 0.5, 1.0):
        for mny in (0.9, 0.95, 1.0, 1.05, 1.1):
            iv = 0.2 + 0.1 * (mny - 1.0) ** 2 + 0.03 * (T - 0.25)
            rows.append({"K": 100.0 * mny, "T": T, "S0": 100.0, "iv": iv, "type": "call"})
    return pd.DataFrame(rows)


def test_controller_sabr_end_to_end():
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": "sabr",
            "df": _call_surface_df(),
            "r": 0.02,
            "q": 0.0,
            "S0": 100.0,
            "fit_to_observed_only": True,
            "max_nfev": 80,
            "n_starts": 1,
            "seed": 0,
            "constraints": {},
        }
    )
    assert res["success"] is True
    assert res["model"] == "sabr"
    iv_model = np.asarray(res["iv_model"], dtype=float)
    assert iv_model.ndim == 2
    assert np.isfinite(iv_model).any()
    assert all(np.isfinite(float(v)) for v in res["metrics"].values())


def test_controller_heston_falls_back_to_least_squares_when_the_nn_is_unavailable(monkeypatch):
    """The Heston tab advertises `calibration=least_squares` and a torch-free least-squares
    calibrator exists, yet the dispatcher returned "PyTorch non installé" whenever the NN
    warm start could not run — the default model of the tab was dead without runtime-ml.
    """
    import app.controller.calibration_controller as cc

    monkeypatch.setattr(
        cc, "predict_params", lambda *a, **k: {"success": False, "message": "PyTorch non installé"}
    )
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": "heston_v1",
            "df": _call_surface_df(),
            "r": 0.02,
            "q": 0.0,
            "S0": 100.0,
            "fit_to_observed_only": True,
            "max_nfev": 40,
            "n_starts": 1,
            "seed": 0,
            "constraints": {},
        }
    )
    assert res["success"] is True, res.get("message")
    assert res["model"] == "heston_v1"
    assert res["method"] == "least_squares_cf_v1"
    params = res["params"]
    assert set(params) >= {"kappa", "theta", "sigma", "rho", "v0"}
    assert all(np.isfinite(float(v)) for v in params.values())
    # Feller-agnostic sanity: a positive-variance model, correlation inside (-1, 1).
    assert params["theta"] > 0 and params["v0"] > 0 and -1.0 < params["rho"] < 1.0
    iv_model = np.asarray(res["iv_model"], dtype=float)
    assert iv_model.shape == (len(res["t_grid"]), len(res["m_grid"]))
    assert res["metrics"]["rmse"] < 0.05
