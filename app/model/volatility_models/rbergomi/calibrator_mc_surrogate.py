from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator
except Exception:  # pragma: no cover - optional dependency
    RBFInterpolator = None

from app.model.calibration.base_calibrator import (
    BaseSurfaceCalibrator,
    CalibratorSettings,
    SurfaceCalibrationResult,
    SurfaceGrid,
)
from app.model.calibration.implied_vol import bs_call_price, implied_vol_grid
from app.model.calibration.loss_surface import (
    compute_bs_vega_grid,
    effective_mask,
    iv_error_metrics,
    iv_error_metrics_weighted,
)
from app.model.calibration.optimizers import latin_hypercube_samples
from app.model.volatility_models.rbergomi.pricing_mc import price_call_grid_mc
from app.model.volatility_models.rbergomi.simulator import RBergomiSimConfig, simulate_rbergomi_paths


@dataclass(frozen=True)
class RBergomiCalibrationConfig:
    # surrogate design
    n_design: int = 24
    n_surrogate_candidates: int = 200
    # MC pricing
    n_paths: int = 4000
    n_steps: int = 60


def _coerce_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_param_constraint(val: Any) -> tuple[float | None, float | None, float | None]:
    if isinstance(val, (int, float)):
        v = float(val)
        return v, v, v
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return _coerce_float(val[0]), _coerce_float(val[1]), None
    if isinstance(val, dict):
        if "value" in val:
            v = _coerce_float(val.get("value"))
            if v is None:
                return None, None, None
            return v, v, v
        return _coerce_float(val.get("min")), _coerce_float(val.get("max")), None
    return None, None, None


class RBergomiMCSurrogateCalibrator(BaseSurfaceCalibrator):
    model = "rbergomi"
    method = "mc_surrogate"

    DEFAULT_BOUNDS: Dict[str, tuple[float, float]] = {
        "H": (0.02, 0.49),
        "eta": (0.05, 5.0),
        "rho": (-0.999, 0.999),
        "xi0": (1e-4, 1.5),
    }
    PARAM_ORDER = ("H", "eta", "rho", "xi0")

    def _build_bounds(self, constraints: Dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, str | None]:
        lower = {k: float(v[0]) for k, v in self.DEFAULT_BOUNDS.items()}
        upper = {k: float(v[1]) for k, v in self.DEFAULT_BOUNDS.items()}
        if isinstance(constraints, dict):
            for name in self.PARAM_ORDER:
                if name not in constraints:
                    continue
                mn, mx, fixed = _parse_param_constraint(constraints.get(name))
                if fixed is not None:
                    lower[name] = fixed
                    upper[name] = fixed
                    continue
                if mn is not None:
                    lower[name] = max(lower[name], mn)
                if mx is not None:
                    upper[name] = min(upper[name], mx)
        lb = np.array([lower[p] for p in self.PARAM_ORDER], dtype=float)
        ub = np.array([upper[p] for p in self.PARAM_ORDER], dtype=float)
        if not np.isfinite(lb).all() or not np.isfinite(ub).all():
            return lb, ub, "Bornes invalides (non finies)."
        if np.any(lb > ub):
            return lb, ub, "Bornes invalides: min > max."
        return lb, ub, None

    def calibrate(
        self,
        surface: SurfaceGrid,
        *,
        constraints: Dict[str, Any] | None = None,
        settings: CalibratorSettings | None = None,
    ) -> SurfaceCalibrationResult:
        if RBFInterpolator is None:
            return SurfaceCalibrationResult(
                success=False,
                message="SciPy indisponible: RBFInterpolator requis pour surrogate.",
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

        idx = np.argwhere(mask_eff)
        if idx.size == 0:
            return SurfaceCalibrationResult(
                success=False,
                message="Aucun point valide pour calibration.",
                model=self.model,
                method=self.method,
                params={},
            )

        # Market prices on the grid (constant across evaluations)
        strikes = (S0 * m_grid).astype(float)
        px_mkt_grid = np.zeros((len(t_grid), len(m_grid)), dtype=float)
        scale_grid = np.ones_like(px_mkt_grid, dtype=float)
        for i_t, T in enumerate(t_grid):
            for j_m, K in enumerate(strikes):
                iv = float(iv_mkt[i_t, j_m])
                if not (np.isfinite(iv) and iv > 0 and float(T) > 0):
                    px_mkt_grid[i_t, j_m] = float("nan")
                    scale_grid[i_t, j_m] = float("nan")
                    continue
                px = float(bs_call_price(S0, float(K), float(T), r, q, iv))
                px_mkt_grid[i_t, j_m] = px
                scale_grid[i_t, j_m] = max(1e-4, px)

        lb, ub, err = self._build_bounds(constraints)
        if err:
            return SurfaceCalibrationResult(
                success=False,
                message=str(err),
                model=self.model,
                method=self.method,
                params={},
            )

        calib_cfg = RBergomiCalibrationConfig()
        if isinstance(constraints, dict) and isinstance(constraints.get("mc_cfg"), dict):
            d = constraints.get("mc_cfg") or {}
            try:
                calib_cfg = RBergomiCalibrationConfig(
                    n_design=int(d.get("n_design", calib_cfg.n_design)),
                    n_surrogate_candidates=int(d.get("n_surrogate_candidates", calib_cfg.n_surrogate_candidates)),
                    n_paths=int(d.get("n_paths", calib_cfg.n_paths)),
                    n_steps=int(d.get("n_steps", calib_cfg.n_steps)),
                )
            except Exception:
                calib_cfg = calib_cfg

        rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()
        bounds_list = [(float(lo), float(hi)) for lo, hi in zip(lb, ub)]
        design = latin_hypercube_samples(n=int(calib_cfg.n_design), bounds=bounds_list, rng=rng)
        # Always include a sensible anchor point
        anchor = np.array([[0.1, 1.0, -0.5, 0.04]], dtype=float)
        anchor = np.minimum(np.maximum(anchor, lb[None, :]), ub[None, :])
        X = np.vstack([anchor, design]).astype(float)

        def mc_objective(x: np.ndarray, *, seed: int | None) -> tuple[float, np.ndarray]:
            H, eta, rho, xi0 = [float(v) for v in x]
            sim_cfg = RBergomiSimConfig(n_paths=int(calib_cfg.n_paths), n_steps=int(calib_cfg.n_steps), seed=seed)
            ST_by_T = simulate_rbergomi_paths(S0=S0, r=r, q=q, t_grid=t_grid, H=H, eta=eta, rho=rho, xi0=xi0, cfg=sim_cfg)
            price_grid = price_call_grid_mc(ST_by_T=ST_by_T, strikes=strikes, t_grid=t_grid, r=r)
            err_grid = (price_grid - px_mkt_grid) / scale_grid
            err_grid = np.where(mask_eff, err_grid, np.nan)
            loss = float(np.nanmean(err_grid * err_grid)) if np.isfinite(err_grid).any() else float("inf")
            return float(loss), price_grid

        losses: list[float] = []
        design_details: list[Dict[str, Any]] = []
        best_idx = None
        best_loss = float("inf")

        # Evaluate design points (noisy objective; use deterministic seeds)
        for i, x in enumerate(X):
            seed_i = int(rng.integers(0, 2**31 - 1))
            loss, _ = mc_objective(x, seed=seed_i)
            losses.append(float(loss))
            meta = {"idx": int(i), "x": [float(v) for v in x.tolist()], "loss": float(loss), "seed": int(seed_i)}
            design_details.append(meta)
            if np.isfinite(loss) and loss < best_loss:
                best_loss = float(loss)
                best_idx = int(i)

        if best_idx is None:
            return SurfaceCalibrationResult(
                success=False,
                message="Design MC échoué (loss non finie).",
                model=self.model,
                method=self.method,
                params={},
                details={"design": design_details},
            )

        y = np.asarray(losses, dtype=float).reshape(-1, 1)
        try:
            surrogate = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=1e-6)
        except Exception as exc:
            return SurfaceCalibrationResult(
                success=False,
                message=f"Surrogate RBF indisponible: {exc}",
                model=self.model,
                method=self.method,
                params={},
                details={"design": design_details},
            )

        # Explore surrogate cheaply
        cand = rng.uniform(size=(int(calib_cfg.n_surrogate_candidates), len(lb)))
        cand = lb[None, :] + cand * (ub - lb)[None, :]
        try:
            yhat = surrogate(cand).reshape(-1)
        except Exception:
            yhat = np.full((cand.shape[0],), np.inf, dtype=float)

        best_surr_idx = int(np.nanargmin(yhat)) if np.isfinite(yhat).any() else best_idx
        x_surr = cand[best_surr_idx, :] if best_surr_idx < cand.shape[0] else X[best_idx, :]

        # Final MC evaluation at surrogate best (fresh seed)
        final_seed = int(rng.integers(0, 2**31 - 1))
        best_loss_mc, price_grid_best = mc_objective(x_surr, seed=final_seed)
        params = {k: float(v) for k, v in zip(self.PARAM_ORDER, x_surr)}

        iv_model = implied_vol_grid(price_grid_best, S0, m_grid, t_grid, r, q)
        iv_error = np.where(mask_eff, iv_model - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)
        vega_weights = compute_bs_vega_grid(S0, m_grid, t_grid, r, q, iv_mkt)
        metrics_vw = iv_error_metrics_weighted(iv_error, mask_eff, vega_weights)

        details = {
            "design": design_details,
            "best_design_idx": int(best_idx),
            "best_design_loss": float(best_loss),
            "final_seed": int(final_seed),
            "final_loss": float(best_loss_mc),
            "mc_cfg": {
                "n_design": calib_cfg.n_design,
                "n_surrogate_candidates": calib_cfg.n_surrogate_candidates,
                "n_paths": calib_cfg.n_paths,
                "n_steps": calib_cfg.n_steps,
            },
        }

        return SurfaceCalibrationResult(
            success=True,
            message="OK (MC surrogate, expensive)",
            model=self.model,
            method=self.method,
            params=params,
            metrics=metrics,
            metrics_vw=metrics_vw,
            iv_model=iv_model,
            iv_error=iv_error,
            vega_weights=vega_weights,
            details=details,
        )


__all__ = ["RBergomiMCSurrogateCalibrator", "RBergomiCalibrationConfig"]

