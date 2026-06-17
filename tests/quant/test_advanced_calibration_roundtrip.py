"""Round-trip / robustness tests for the advanced surface calibrators.

For analytic models (SABR, Merton): generate an IV surface from the model with
known params, calibrate, and assert the fit recovers it within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import SurfaceGrid, CalibratorSettings

pytestmark = pytest.mark.slow

_M = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
_T = np.array([0.25, 0.5, 1.0])


def _placeholder_surface(S0=100.0, r=0.02, q=0.0):
    return SurfaceGrid(
        S0=S0, r=r, q=q, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )


def _market_from_model(model, params, S0=100.0, r=0.02, q=0.0):
    iv = np.asarray(model.implied_vol_surface(_placeholder_surface(S0, r, q), params), dtype=float)
    mask = np.isfinite(iv) & (iv > 0)
    market = SurfaceGrid(S0=S0, r=r, q=q, m_grid=_M, t_grid=_T, iv_market=iv, mask=mask)
    return market, mask


def test_sabr_roundtrip_recovers_surface():
    from app.model.volatility_models.sabr.model import SABRAnalyticModel
    from app.model.volatility_models.sabr.calibrator import SABRAnalyticCalibrator

    model = SABRAnalyticModel(beta=0.5)
    market, mask = _market_from_model(model, {"alpha": 0.3, "rho": -0.4, "nu": 0.6})
    assert mask.sum() > 0
    res = SABRAnalyticCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=200, n_starts=2, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 1e-2  # analytic per-maturity fit -> near-exact


def test_merton_roundtrip_recovers_surface():
    from app.model.volatility_models.jump_diffusion.model import MertonJumpDiffusionModel
    from app.model.volatility_models.jump_diffusion.calibrator import MertonJumpDiffusionCalibrator

    model = MertonJumpDiffusionModel()
    market, mask = _market_from_model(model, {"sigma": 0.2, "lam": 0.5, "muj": -0.1, "sigj": 0.3})
    assert mask.sum() > 0
    res = MertonJumpDiffusionCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=200, n_starts=3, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 3e-2  # global 4-param fit + FFT/IV discretisation
