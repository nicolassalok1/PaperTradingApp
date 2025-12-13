from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional dependency
    least_squares = None

from app.model.calibration.base_calibrator import (
    BaseSurfaceCalibrator,
    CalibratorSettings,
    SurfaceCalibrationResult,
    SurfaceGrid,
)
from app.model.calibration.loss_surface import effective_mask, iv_error_metrics
from app.model.volatility_models.sabr.analytic import hagan_black_iv_vectorized
from app.model.volatility_models.sabr.model import SABRAnalyticModel


class SABRAnalyticCalibrator(BaseSurfaceCalibrator):
    model = "sabr"
    method = "least_squares_iv"

    def __init__(self, *, beta: float = 0.5):
        self._model = SABRAnalyticModel(beta=beta)

    @property
    def beta(self) -> float:
        return float(self._model.beta)

    @staticmethod
    def _bounds() -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([1e-6, -0.999, 1e-6], dtype=float)  # alpha, rho, nu
        ub = np.array([5.0, 0.999, 5.0], dtype=float)
        return lb, ub

    def calibrate(
        self,
        surface: SurfaceGrid,
        *,
        constraints: Dict[str, Any] | None = None,
        settings: CalibratorSettings | None = None,
    ) -> SurfaceCalibrationResult:
        if least_squares is None:
            return SurfaceCalibrationResult(
                success=False,
                message="SciPy indisponible: least_squares requis pour SABR.",
                model=self.model,
                method=self.method,
                params={},
            )

        settings = settings or CalibratorSettings()
        S0 = float(surface.S0)
        r = float(surface.r)
        q = float(surface.q)
        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)
        iv_mkt = np.asarray(surface.iv_market, dtype=float)
        mask_eff = effective_mask(iv_mkt, surface.mask, fit_to_observed_only=settings.fit_to_observed_only)

        lb, ub = self._bounds()
        # Allow beta override in constraints (fixed beta only)
        beta = float(self.beta)
        if isinstance(constraints, dict) and "beta" in constraints:
            try:
                beta = float(constraints.get("beta"))
            except Exception:
                beta = beta
            self._model = SABRAnalyticModel(beta=beta)

        alpha_by_t: list[float] = []
        rho_by_t: list[float] = []
        nu_by_t: list[float] = []
        runs: list[Dict[str, Any]] = []

        rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()

        for i_t, T in enumerate(t_grid):
            m_row = mask_eff[i_t, :]
            if not np.any(m_row):
                alpha_by_t.append(float("nan"))
                rho_by_t.append(float("nan"))
                nu_by_t.append(float("nan"))
                continue

            F = S0 * np.exp((r - q) * float(T))
            K = (S0 * m_grid[m_row]).astype(float)
            iv_slice = iv_mkt[i_t, m_row].astype(float)

            # heuristic initial guess
            atm_idx = int(np.abs(m_grid[m_row] - 1.0).argmin())
            iv_atm = float(iv_slice[atm_idx]) if iv_slice.size else 0.2
            alpha0 = max(1e-4, iv_atm * (F ** (1.0 - beta)))
            x0 = np.array([alpha0, -0.3, 0.5], dtype=float)
            x0 = np.minimum(np.maximum(x0, lb), ub)

            def residuals(x: np.ndarray) -> np.ndarray:
                a, rh, n = [float(v) for v in x]
                iv_model = hagan_black_iv_vectorized(F=F, K=K, T=float(T), alpha=a, beta=beta, rho=rh, nu=n)
                return (iv_model - iv_slice).astype(float)

            # multi-start: x0 + random box samples
            candidates = [x0]
            while len(candidates) < max(1, int(settings.n_starts or 1)):
                cand = np.array(
                    [
                        float(np.exp(rng.uniform(np.log(lb[0]), np.log(ub[0])))),
                        float(rng.uniform(lb[1], ub[1])),
                        float(np.exp(rng.uniform(np.log(lb[2]), np.log(ub[2])))),
                    ],
                    dtype=float,
                )
                candidates.append(cand)

            best = None
            best_cost = float("inf")
            best_out: Dict[str, Any] | None = None
            for idx, c in enumerate(candidates):
                try:
                    opt = least_squares(residuals, x0=c, bounds=(lb, ub), max_nfev=int(settings.max_nfev))
                    cost = float(getattr(opt, "cost", np.nan))
                    out = {
                        "T": float(T),
                        "idx": int(idx),
                        "ok": True,
                        "converged": bool(getattr(opt, "success", False)),
                        "cost": cost,
                        "nfev": int(getattr(opt, "nfev", 0) or 0),
                        "message": str(getattr(opt, "message", "")),
                        "x0": [float(v) for v in np.asarray(c, dtype=float)],
                        "x": [float(v) for v in np.asarray(opt.x, dtype=float)],
                    }
                except Exception as exc:
                    out = {"T": float(T), "idx": int(idx), "ok": False, "error": str(exc)}
                    cost = float("inf")
                runs.append(out)
                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best = np.asarray(out.get("x") or c, dtype=float)
                    best_out = out

            if best is None:
                alpha_by_t.append(float("nan"))
                rho_by_t.append(float("nan"))
                nu_by_t.append(float("nan"))
                continue

            a, rh, n = [float(v) for v in best]
            alpha_by_t.append(a)
            rho_by_t.append(rh)
            nu_by_t.append(n)
            if best_out:
                best_out["best"] = True

        params = {"beta": float(beta), "alpha_by_t": alpha_by_t, "rho_by_t": rho_by_t, "nu_by_t": nu_by_t}
        iv_model_full = self._model.implied_vol_surface(surface, params)
        iv_error = np.where(mask_eff, iv_model_full - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)

        return SurfaceCalibrationResult(
            success=True,
            message="OK",
            model=self.model,
            method=self.method,
            params=params,
            metrics=metrics,
            iv_model=iv_model_full,
            iv_error=iv_error,
            details={"runs": runs, "beta": float(beta)},
        )


__all__ = ["SABRAnalyticCalibrator"]

