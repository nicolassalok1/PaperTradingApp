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
