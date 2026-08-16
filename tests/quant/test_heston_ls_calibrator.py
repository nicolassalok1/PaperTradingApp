"""`calibrate_heston_least_squares` — the torch-free Heston fit used by the LS refine step
of the NN pipeline and by the Calibration avancée fallback.

Its residual vector appended a soft Feller penalty *only when violated*, so its length
changed between evaluations (15 -> 16 -> 15) and scipy's `least_squares` died with
"could not broadcast input array from shape (15,) into shape (16,)" as soon as the trust
region crossed the Feller boundary — i.e. on ordinary starting points. Every start then
failed and the calibration reported "Optimisation échouée".

Oracle: a market surface that is *exactly attainable* (priced by Heston with known
parameters, then inverted to IV) must be refit to a small IV error from a Feller-violating
start; and no start may abort.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.heston_calibrator import calibrate_heston_least_squares
from app.model.calibration.heston_pricer import price_grid_from_params
from app.model.calibration.implied_vol import implied_vol_grid

pytestmark = pytest.mark.slow

S0, R, Q = 100.0, 0.02, 0.0
M_GRID = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
T_GRID = np.array([0.25, 0.5, 1.0])
TRUE = (1.5, 0.05, 0.4, -0.6, 0.04)  # kappa, theta, sigma, rho, v0 — Feller holds (0.16 < 0.15? no: 2*1.5*0.05=0.15 < 0.16 -> mildly violated on purpose)


def _market_iv():
    px = price_grid_from_params(S0, M_GRID, T_GRID, R, Q, TRUE)
    return implied_vol_grid(px, S0, M_GRID, T_GRID, R, Q)


def test_feller_crossing_start_does_not_abort_any_run():
    iv_mkt = _market_iv()
    out = calibrate_heston_least_squares(
        S0=S0,
        r=R,
        q=Q,
        m_grid=M_GRID,
        t_grid=T_GRID,
        iv_market=iv_mkt,
        mask=np.ones_like(iv_mkt, dtype=bool),
        max_nfev=40,
        n_starts=1,
        seed=0,
        x0=(0.5, 0.04, 1.0, -0.5, 0.04),  # sigma^2 = 1 >> 2*kappa*theta = 0.04
        u_max=40.0,
        n_integration=600,
    )
    assert out["success"], out.get("message")
    errors = [run.get("error") for run in out["runs"] if not run.get("ok")]
    assert errors == [], errors


def test_attainable_surface_is_refit_closely():
    iv_mkt = _market_iv()
    out = calibrate_heston_least_squares(
        S0=S0,
        r=R,
        q=Q,
        m_grid=M_GRID,
        t_grid=T_GRID,
        iv_market=iv_mkt,
        mask=np.ones_like(iv_mkt, dtype=bool),
        max_nfev=120,
        n_starts=2,
        seed=0,
        u_max=40.0,
        n_integration=600,
    )
    assert out["success"], out.get("message")
    p = out["params"]
    px = price_grid_from_params(S0, M_GRID, T_GRID, R, Q, tuple(p[k] for k in ("kappa", "theta", "sigma", "rho", "v0")))
    iv_fit = implied_vol_grid(px, S0, M_GRID, T_GRID, R, Q)
    rmse = float(np.sqrt(np.nanmean((iv_fit - iv_mkt) ** 2)))
    assert rmse < 2e-3, rmse
