from __future__ import annotations

from typing import Any, Dict

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
from app.model.calibration.implied_vol import bs_call_price, implied_vol_grid
from app.model.calibration.loss_surface import effective_mask, iv_error_metrics
from app.model.volatility_models.common.fft import FFTConfig, carr_madan_fft_call_prices, interp_prices
from app.model.volatility_models.jump_diffusion.cf import bates_log_return_cf


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


class BatesCalibrator(BaseSurfaceCalibrator):
    model = "bates"
    method = "least_squares_fft_prices"

    DEFAULT_BOUNDS: Dict[str, tuple[float, float]] = {
        "kappa": (0.1, 10.0),
        "theta": (1e-4, 1.5),
        "sigma": (1e-3, 5.0),
        "rho": (-0.999, 0.999),
        "v0": (1e-4, 1.5),
        "lam": (0.0, 5.0),
        "muj": (-1.0, 1.0),
        "sigj": (1e-4, 2.0),
    }
    PARAM_ORDER = ("kappa", "theta", "sigma", "rho", "v0", "lam", "muj", "sigj")

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
        if least_squares is None:
            return SurfaceCalibrationResult(
                success=False,
                message="SciPy indisponible: least_squares requis.",
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

        K = np.empty(len(idx), dtype=float)
        T = np.empty(len(idx), dtype=float)
        px_mkt = np.empty(len(idx), dtype=float)
        scale = np.empty(len(idx), dtype=float)

        for k, (i_t, j_m) in enumerate(idx):
            tt = float(t_grid[int(i_t)])
            kk = float(S0 * float(m_grid[int(j_m)]))
            iv = float(iv_mkt[int(i_t), int(j_m)])
            px = float(bs_call_price(S0, kk, tt, r, q, iv))
            K[k] = kk
            T[k] = tt
            px_mkt[k] = px
            scale[k] = max(1e-4, px)

        uniq_T = sorted({float(t) for t in T.tolist() if float(t) > 0})

        lb, ub, err = self._build_bounds(constraints)
        if err:
            return SurfaceCalibrationResult(
                success=False,
                message=str(err),
                model=self.model,
                method=self.method,
                params={},
            )

        fft_cfg = None
        if isinstance(constraints, dict) and isinstance(constraints.get("fft_cfg"), dict):
            d = constraints.get("fft_cfg") or {}
            try:
                fft_cfg = FFTConfig(
                    alpha=float(d.get("alpha", 1.5)),
                    n=int(d.get("n", 2048)),
                    eta=float(d.get("eta", 0.25)),
                )
            except Exception:
                fft_cfg = None
        cfg = fft_cfg or FFTConfig()

        # Simple, conservative initial guess (keeps stability)
        try:
            atm_j = int(np.abs(m_grid - 1.0).argmin())
            iv_atm = float(iv_mkt[0, atm_j])
            theta0 = float(np.clip(iv_atm * iv_atm, lb[1], ub[1]))
            v00 = theta0
        except Exception:
            theta0 = 0.04
            v00 = 0.04

        x0 = np.array([2.0, theta0, 0.5, -0.5, v00, 0.2, -0.05, 0.2], dtype=float)
        x0 = np.minimum(np.maximum(x0, lb), ub)

        rng = np.random.default_rng(settings.seed) if settings.seed is not None else np.random.default_rng()
        candidates: list[np.ndarray] = [x0, np.minimum(np.maximum(0.5 * (lb + ub), lb), ub)]
        while len(candidates) < max(1, int(settings.n_starts or 1)):
            cand = np.empty_like(x0)
            for i, name in enumerate(self.PARAM_ORDER):
                lo = float(lb[i])
                hi = float(ub[i])
                if abs(hi - lo) < 1e-12:
                    cand[i] = lo
                    continue
                if name == "rho":
                    cand[i] = float(rng.uniform(lo, hi))
                    continue
                if name in ("kappa", "theta", "sigma", "v0", "lam", "sigj"):
                    lo_pos = max(lo, 1e-12)
                    hi_pos = max(hi, lo_pos * 1.000001)
                    cand[i] = float(np.exp(rng.uniform(np.log(lo_pos), np.log(hi_pos))))
                    continue
                cand[i] = float(rng.uniform(lo, hi))
            candidates.append(np.minimum(np.maximum(cand, lb), ub))

        def _model_prices_for_params(x: np.ndarray) -> np.ndarray:
            kappa, theta, sigma, rho, v0, lam, muj, sigj = [float(v) for v in x]
            out = np.full_like(px_mkt, np.nan, dtype=float)
            for tt in uniq_T:
                where = np.where(np.abs(T - tt) < 1e-12)[0]
                if where.size == 0:
                    continue
                strikes = K[where]

                def _cf(u: np.ndarray, T_in: float) -> np.ndarray:
                    return bates_log_return_cf(
                        u,
                        T_in,
                        r=r,
                        q=q,
                        kappa=kappa,
                        theta=theta,
                        sigma=sigma,
                        rho=rho,
                        v0=v0,
                        lam=lam,
                        muj=muj,
                        sigj=sigj,
                    )

                K_grid, C_grid = carr_madan_fft_call_prices(
                    S0=S0, r=r, q=q, T=float(tt), cf_log_return=_cf, cfg=cfg
                )
                prices = np.asarray(interp_prices(K_grid=K_grid, C_grid=C_grid, K=strikes), dtype=float).reshape(-1)
                if prices.size < where.size:
                    padded = np.full(where.size, np.nan, dtype=float)
                    padded[: prices.size] = prices
                    prices = padded
                if prices.size > where.size:
                    prices = prices[: where.size]
                out[where] = prices
            return out

        def residuals(x: np.ndarray) -> np.ndarray:
            x = np.minimum(np.maximum(np.asarray(x, dtype=float), lb), ub)
            px_model = _model_prices_for_params(x)
            res = (px_model - px_mkt) / scale
            res = np.where(np.isfinite(res), res, 0.0)

            # soft Feller penalty
            kappa, theta, sigma, _, _, _, _, _ = [float(v) for v in x]
            feller_gap = float(max(0.0, sigma * sigma - 2.0 * kappa * theta))
            if feller_gap > 0:
                res = np.concatenate([res, np.array([feller_gap], dtype=float)])
            return res.astype(float)

        runs: list[Dict[str, Any]] = []
        best_x: np.ndarray | None = None
        best_cost = float("inf")
        best_meta: Dict[str, Any] | None = None

        for i, cand in enumerate(candidates):
            try:
                opt = least_squares(residuals, x0=cand, bounds=(lb, ub), max_nfev=int(settings.max_nfev))
                cost = float(getattr(opt, "cost", np.nan))
                meta = {
                    "idx": int(i),
                    "ok": True,
                    "converged": bool(getattr(opt, "success", False)),
                    "cost": cost,
                    "nfev": int(getattr(opt, "nfev", 0) or 0),
                    "message": str(getattr(opt, "message", "")),
                    "x0": [float(v) for v in np.asarray(cand, dtype=float)],
                    "x": [float(v) for v in np.asarray(opt.x, dtype=float)],
                }
            except Exception as exc:
                meta = {"idx": int(i), "ok": False, "error": str(exc)}
                cost = float("inf")
            runs.append(meta)
            if np.isfinite(cost) and cost < best_cost:
                best_cost = float(cost)
                best_x = np.asarray(meta.get("x") or cand, dtype=float)
                best_meta = meta

        if best_x is None:
            first_err = next((r.get("error") for r in runs if r.get("error")), "Optimisation échouée.")
            return SurfaceCalibrationResult(
                success=False,
                message=f"Optimisation échouée: {first_err}",
                model=self.model,
                method=self.method,
                params={},
                details={"runs": runs},
            )

        params = {k: float(v) for k, v in zip(self.PARAM_ORDER, best_x)}

        # price grid -> implied vol grid for UI
        price_grid = np.zeros((len(t_grid), len(m_grid)), dtype=float)
        for i_t, tt in enumerate(t_grid):
            if float(tt) <= 0:
                continue

            def _cf(u: np.ndarray, T_in: float) -> np.ndarray:
                return bates_log_return_cf(
                    u,
                    T_in,
                    r=r,
                    q=q,
                    kappa=float(params["kappa"]),
                    theta=float(params["theta"]),
                    sigma=float(params["sigma"]),
                    rho=float(params["rho"]),
                    v0=float(params["v0"]),
                    lam=float(params["lam"]),
                    muj=float(params["muj"]),
                    sigj=float(params["sigj"]),
                )

            K_grid, C_grid = carr_madan_fft_call_prices(S0=S0, r=r, q=q, T=float(tt), cf_log_return=_cf, cfg=cfg)
            strikes = (S0 * m_grid).astype(float)
            prices = np.asarray(interp_prices(K_grid=K_grid, C_grid=C_grid, K=strikes), dtype=float).reshape(-1)
            row = np.full_like(price_grid[i_t, :], np.nan, dtype=float)
            n = min(row.size, prices.size)
            row[:n] = prices[:n]
            price_grid[i_t, :] = row

        iv_model = implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q)
        iv_error = np.where(mask_eff, iv_model - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)

        if best_meta is not None:
            best_meta["best"] = True

        return SurfaceCalibrationResult(
            success=True,
            message="OK",
            model=self.model,
            method=self.method,
            params=params,
            metrics=metrics,
            iv_model=iv_model,
            iv_error=iv_error,
            details={"runs": runs, "fft_cfg": {"alpha": cfg.alpha, "n": cfg.n, "eta": cfg.eta}},
        )


__all__ = ["BatesCalibrator"]

