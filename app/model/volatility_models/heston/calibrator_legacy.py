from __future__ import annotations

from typing import Any, Dict

import numpy as np

from app.model.calibration.base_calibrator import (
    BaseSurfaceCalibrator,
    CalibratorSettings,
    SurfaceCalibrationResult,
    SurfaceGrid,
)
from app.model.calibration.heston_calibrator import calibrate_heston_least_squares
from app.model.calibration.heston_pricer import price_grid_from_params
from app.model.calibration.implied_vol import implied_vol_grid
from app.model.calibration.loss_surface import (
    compute_bs_vega_grid,
    effective_mask,
    iv_error_metrics,
    iv_error_metrics_weighted,
)


class HestonLegacyLeastSquaresCalibrator(BaseSurfaceCalibrator):
    """
    Adapter around the existing V1 Heston calibrator/pricer.
    Keeps the numerical core untouched while exposing the unified API.
    """

    model = "heston_v1"
    method = "least_squares_cf_v1"

    def calibrate(
        self,
        surface: SurfaceGrid,
        *,
        constraints: Dict[str, Any] | None = None,
        settings: CalibratorSettings | None = None,
    ) -> SurfaceCalibrationResult:
        settings = settings or CalibratorSettings()
        S0 = float(surface.S0)
        r = float(surface.r)
        q = float(surface.q)
        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)
        iv_mkt = np.asarray(surface.iv_market, dtype=float)
        mask_eff = effective_mask(iv_mkt, surface.mask, fit_to_observed_only=settings.fit_to_observed_only)

        out = calibrate_heston_least_squares(
            S0=S0,
            r=r,
            q=q,
            m_grid=m_grid,
            t_grid=t_grid,
            iv_market=iv_mkt,
            mask=mask_eff,
            constraints=constraints,
            fit_to_observed_only=settings.fit_to_observed_only,
            max_nfev=int(settings.max_nfev),
            n_starts=int(settings.n_starts),
            seed=settings.seed,
        )
        if not out.get("success"):
            return SurfaceCalibrationResult(
                success=False,
                message=str(out.get("message") or "Calibration échouée."),
                model=self.model,
                method=self.method,
                params={},
                details=out,
            )

        params = out.get("params") or {}
        params_tuple = (
            params.get("kappa"),
            params.get("theta"),
            params.get("sigma"),
            params.get("rho"),
            params.get("v0"),
        )
        if any(p is None for p in params_tuple):
            return SurfaceCalibrationResult(
                success=False,
                message="Paramètres incomplets après calibration.",
                model=self.model,
                method=self.method,
                params=params,
                details=out,
            )

        price_grid = price_grid_from_params(S0, m_grid, t_grid, r, q, params_tuple)
        iv_model = implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q)
        iv_error = np.where(mask_eff, iv_model - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)
        vega_weights = compute_bs_vega_grid(S0, m_grid, t_grid, r, q, iv_mkt)
        metrics_vw = iv_error_metrics_weighted(iv_error, mask_eff, vega_weights)

        return SurfaceCalibrationResult(
            success=True,
            message=str(out.get("message") or "OK"),
            model=self.model,
            method=self.method,
            params={k: float(v) for k, v in params.items()},
            metrics=metrics,
            metrics_vw=metrics_vw,
            iv_model=iv_model,
            iv_error=iv_error,
            vega_weights=vega_weights,
            details=out,
        )


__all__ = ["HestonLegacyLeastSquaresCalibrator"]

