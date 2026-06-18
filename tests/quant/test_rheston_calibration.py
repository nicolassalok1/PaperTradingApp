"""rHeston FFT calibration: regression for the Riccati-overflow all-NaN bug,
plus a round-trip on a model-generated surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import SurfaceGrid, CalibratorSettings
from app.model.volatility_models.rheston.calibrator_fft import RHestonFFTMarkovianCalibrator
from app.model.volatility_models.rheston.model_fft import RHestonFFTMarkovianModel

pytestmark = pytest.mark.slow

_M = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
_T = np.array([0.25, 0.5, 1.0])


def _smooth_smile_surface():
    M, T = np.meshgrid(_M, _T)
    iv = 0.2 + 0.1 * (M - 1.0) ** 2 + 0.03 * (T - 0.25)
    return SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=iv, mask=np.isfinite(iv),
    )


def test_rheston_calibrator_does_not_return_all_nan_surface():
    # Mirrors the verified bug: a smooth smile produced an all-NaN model surface.
    surface = _smooth_smile_surface()
    res = RHestonFFTMarkovianCalibrator().calibrate(
        surface, settings=CalibratorSettings(max_nfev=40, n_starts=1, seed=0)
    )
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model).any(), "rHeston returned an all-NaN surface (overflow bug)"


def test_rheston_model_surface_is_finite():
    # The model's own IV surface must be finite for normal params.
    model = RHestonFFTMarkovianModel()
    src = SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )
    iv = np.asarray(
        model.implied_vol_surface(
            src, {"H": 0.1, "kappa": 2.0, "theta": 0.04, "xi": 0.6, "rho": -0.5, "v0": 0.04}
        ),
        dtype=float,
    )
    assert iv.shape == (len(_T), len(_M))
    assert np.isfinite(iv).sum() > 0, "rHeston model surface is entirely NaN (overflow bug)"


def test_rheston_roundtrip_finite_and_reasonable():
    model = RHestonFFTMarkovianModel()
    src = SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )
    iv = np.asarray(
        model.implied_vol_surface(
            src, {"H": 0.1, "kappa": 2.0, "theta": 0.04, "xi": 0.6, "rho": -0.5, "v0": 0.04}
        ),
        dtype=float,
    )
    mask = np.isfinite(iv) & (iv > 0)
    assert mask.sum() > 0  # generation must not collapse to all-NaN
    market = SurfaceGrid(S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T, iv_market=iv, mask=mask)
    res = RHestonFFTMarkovianCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=60, n_starts=1, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 5e-2  # loose: Markovian approx
