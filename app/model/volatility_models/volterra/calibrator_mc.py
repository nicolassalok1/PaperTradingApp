from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from app.model.calibration.base_calibrator import (
    BaseSurfaceCalibrator,
    CalibratorSettings,
    SurfaceCalibrationResult,
    SurfaceGrid,
)
from app.model.calibration.implied_vol import bs_call_price, implied_vol_grid
from app.model.calibration.loss_surface import effective_mask, iv_error_metrics
from app.model.calibration.optimizers import latin_hypercube_samples
from app.model.volatility_models.rbergomi.pricing_mc import price_call_grid_mc
from app.model.volatility_models.volterra.kernels import exponential_kernel, fractional_kernel
from app.model.volatility_models.volterra.simulator import VolterraSimConfig, simulate_volterra_sv_paths


@dataclass(frozen=True)
class VolterraCalibrationConfig:
    n_design: int = 24
    n_paths: int = 4000
    n_steps: int = 60
    kernel_type: str = "fractional"  # "fractional" | "exponential"
    H: float = 0.1
    lam: float = 1.0


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


class VolterraSDECalibrator(BaseSurfaceCalibrator):
    model = "volterra"
    method = "mc_proxy_search"

    DEFAULT_BOUNDS: Dict[str, tuple[float, float]] = {
        "kappa": (0.0, 10.0),
        "theta": (1e-4, 1.5),
        "xi": (1e-3, 5.0),
        "rho": (-0.999, 0.999),
        "v0": (1e-4, 1.5),
    }
    PARAM_ORDER = ("kappa", "theta", "xi", "rho", "v0")

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

        cfg = VolterraCalibrationConfig()
        if isinstance(constraints, dict) and isinstance(constraints.get("mc_cfg"), dict):
            d = constraints.get("mc_cfg") or {}
            try:
                cfg = VolterraCalibrationConfig(
                    n_design=int(d.get("n_design", cfg.n_design)),
                    n_paths=int(d.get("n_paths", cfg.n_paths)),
                    n_steps=int(d.get("n_steps", cfg.n_steps)),
                    kernel_type=str(d.get("kernel_type", cfg.kernel_type)),
                    H=float(d.get("H", cfg.H)),
                    lam=float(d.get("lam", cfg.lam)),
                )
            except Exception:
                cfg = cfg

        if cfg.kernel_type.lower().startswith("exp"):
            kernel = exponential_kernel(cfg.lam)
            kernel_meta = {"type": "exponential", "lam": float(cfg.lam)}
        else:
            kernel = fractional_kernel(cfg.H)
            kernel_meta = {"type": "fractional", "H": float(cfg.H)}

        rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()
        bounds_list = [(float(lo), float(hi)) for lo, hi in zip(lb, ub)]
        X = latin_hypercube_samples(n=int(cfg.n_design), bounds=bounds_list, rng=rng)

        # include anchor
        anchor = np.array([[2.0, 0.04, 0.6, -0.5, 0.04]], dtype=float)
        anchor = np.minimum(np.maximum(anchor, lb[None, :]), ub[None, :])
        X = np.vstack([anchor, X]).astype(float)

        best_idx = None
        best_loss = float("inf")
        best_price_grid = None
        design_details: list[Dict[str, Any]] = []

        for i, x in enumerate(X):
            kappa, theta, xi, rho, v0 = [float(v) for v in x]
            seed_i = int(rng.integers(0, 2**31 - 1))
            sim_cfg = VolterraSimConfig(n_paths=int(cfg.n_paths), n_steps=int(cfg.n_steps), seed=seed_i)
            ST_by_T = simulate_volterra_sv_paths(
                S0=S0,
                r=r,
                q=q,
                t_grid=t_grid,
                kernel=kernel,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                cfg=sim_cfg,
            )
            price_grid = price_call_grid_mc(ST_by_T=ST_by_T, strikes=strikes, t_grid=t_grid, r=r)
            err_grid = (price_grid - px_mkt_grid) / scale_grid
            err_grid = np.where(mask_eff, err_grid, np.nan)
            loss = float(np.nanmean(err_grid * err_grid)) if np.isfinite(err_grid).any() else float("inf")

            meta = {"idx": int(i), "x": [float(v) for v in x.tolist()], "loss": float(loss), "seed": int(seed_i)}
            design_details.append(meta)
            if np.isfinite(loss) and loss < best_loss:
                best_loss = float(loss)
                best_idx = int(i)
                best_price_grid = price_grid

        if best_idx is None or best_price_grid is None:
            return SurfaceCalibrationResult(
                success=False,
                message="Recherche MC échouée (loss non finie).",
                model=self.model,
                method=self.method,
                params={},
                details={"design": design_details, "kernel": kernel_meta, "mc_cfg": cfg.__dict__},
            )

        x_best = X[best_idx, :]
        params = {k: float(v) for k, v in zip(self.PARAM_ORDER, x_best)}
        params["kernel"] = kernel_meta

        iv_model = implied_vol_grid(best_price_grid, S0, m_grid, t_grid, r, q)
        iv_error = np.where(mask_eff, iv_model - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)

        return SurfaceCalibrationResult(
            success=True,
            message="OK (Volterra MC proxy, research/expensive)",
            model=self.model,
            method=self.method,
            params=params,
            metrics=metrics,
            iv_model=iv_model,
            iv_error=iv_error,
            details={"design": design_details, "best_design_idx": int(best_idx), "best_loss": float(best_loss)},
        )


__all__ = ["VolterraSDECalibrator", "VolterraCalibrationConfig"]

