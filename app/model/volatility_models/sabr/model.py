from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from app.model.calibration.base_calibrator import SurfaceGrid
from app.model.volatility_models.base import VolSurfaceModel
from app.model.volatility_models.sabr.analytic import hagan_black_iv


class SABRAnalyticModel(VolSurfaceModel):
    model_key = "sabr"
    label = "SABR (Hagan analytic)"

    def __init__(self, *, beta: float = 0.5):
        self._beta = float(beta)

    @property
    def beta(self) -> float:
        return float(self._beta)

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        # per-maturity calibration; beta typically fixed
        return {
            "alpha": (1e-6, 5.0),
            "rho": (-0.999, 0.999),
            "nu": (1e-6, 5.0),
        }

    def implied_vol_surface(self, surface: SurfaceGrid, params: Dict[str, Any], **kwargs: Any) -> np.ndarray:
        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)
        S0 = float(surface.S0)
        r = float(surface.r)
        q = float(surface.q)

        alpha_curve = params.get("alpha_by_t") or params.get("alpha")
        rho_curve = params.get("rho_by_t") or params.get("rho")
        nu_curve = params.get("nu_by_t") or params.get("nu")

        def _curve_at(curve, i: int, default: float) -> float:
            if isinstance(curve, (list, tuple, np.ndarray)) and len(curve) == len(t_grid):
                try:
                    return float(curve[i])
                except Exception:
                    return float(default)
            if isinstance(curve, dict):
                # allow {str(T): val} style
                return float(curve.get(str(float(t_grid[i])), default))
            try:
                return float(curve)
            except Exception:
                return float(default)

        iv = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
        for i_t, T in enumerate(t_grid):
            F = S0 * np.exp((r - q) * float(T))
            a = _curve_at(alpha_curve, i_t, 0.2)
            rh = _curve_at(rho_curve, i_t, -0.3)
            n = _curve_at(nu_curve, i_t, 0.5)
            for j_m, m in enumerate(m_grid):
                K = float(S0 * float(m))
                iv[i_t, j_m] = hagan_black_iv(F=F, K=K, T=float(T), alpha=a, beta=float(self._beta), rho=rh, nu=n)
        return iv


__all__ = ["SABRAnalyticModel"]

