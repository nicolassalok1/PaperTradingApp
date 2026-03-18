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
from app.model.calibration.loss_surface import (
    compute_bs_vega_grid,
    effective_mask,
    iv_error_metrics,
    iv_error_metrics_weighted,
)
from app.model.volatility_models.common.fft import FFTConfig, carr_madan_fft_call_prices, interp_prices
from app.model.volatility_models.rheston.cf_markovian import RHestonMarkovianConfig, rheston_log_return_cf_markovian


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


class RHestonFFTMarkovianCalibrator(BaseSurfaceCalibrator):
    model = "rheston"
    method = "least_squares_fft_prices"

    DEFAULT_BOUNDS: Dict[str, tuple[float, float]] = {
        "H": (0.02, 0.49),
        "kappa": (0.1, 10.0),
        "theta": (1e-4, 1.5),
        "xi": (1e-3, 5.0),
        "rho": (-0.999, 0.999),
        "v0": (1e-4, 1.5),
    }
    PARAM_ORDER = ("H", "kappa", "theta", "xi", "rho", "v0")

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

        fft_cfg = FFTConfig()
        if isinstance(constraints, dict) and isinstance(constraints.get("fft_cfg"), dict):
            d = constraints.get("fft_cfg") or {}
            try:
                fft_cfg = FFTConfig(alpha=float(d.get("alpha", 1.5)), n=int(d.get("n", 1024)), eta=float(d.get("eta", 0.25)))
            except Exception:
                pass

        mk_cfg = RHestonMarkovianConfig()
        if isinstance(constraints, dict) and isinstance(constraints.get("markovian_cfg"), dict):
            d = constraints.get("markovian_cfg") or {}
            try:
                mk_cfg = RHestonMarkovianConfig(
                    n_factors=int(d.get("n_factors", mk_cfg.n_factors)),
                    steps_per_year=int(d.get("steps_per_year", mk_cfg.steps_per_year)),
                    x_min_mult=float(d.get("x_min_mult", mk_cfg.x_min_mult)),
                    x_max_mult=float(d.get("x_max_mult", mk_cfg.x_max_mult)),
                )
            except Exception:
                mk_cfg = mk_cfg

        # Heuristic x0 (roughness fixed near 0.1)
        try:
            atm_j = int(np.abs(m_grid - 1.0).argmin())
            iv_atm = float(iv_mkt[0, atm_j])
            theta0 = float(np.clip(iv_atm * iv_atm, lb[2], ub[2]))
            v00 = theta0
        except Exception:
            theta0 = 0.04
            v00 = 0.04
        x0 = np.array([0.1, 2.0, theta0, 0.6, -0.5, v00], dtype=float)
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
                if name in ("kappa", "theta", "xi", "v0"):
                    lo_pos = max(lo, 1e-12)
                    hi_pos = max(hi, lo_pos * 1.000001)
                    cand[i] = float(np.exp(rng.uniform(np.log(lo_pos), np.log(hi_pos))))
                    continue
                cand[i] = float(rng.uniform(lo, hi))
            candidates.append(np.minimum(np.maximum(cand, lb), ub))

        def _model_prices_for_params(x: np.ndarray) -> np.ndarray:
            H, kappa, theta, xi, rho, v0 = [float(v) for v in x]
            out = np.full_like(px_mkt, np.nan, dtype=float)

            # Precompute CF on required maturities for current params (vectorized over u in FFT)
            for tt in uniq_T:
                where = np.where(np.abs(T - tt) < 1e-12)[0]
                if where.size == 0:
                    continue
                strikes = K[where]

                def _cf(u: np.ndarray, T_in: float) -> np.ndarray:
                    d = rheston_log_return_cf_markovian(
                        u,
                        [float(T_in)],
                        r=r,
                        q=q,
                        kappa=kappa,
                        theta=theta,
                        xi=xi,
                        rho=rho,
                        v0=v0,
                        H=H,
                        cfg=mk_cfg,
                    )
                    return d[float(T_in)]

                K_grid, C_grid = carr_madan_fft_call_prices(S0=S0, r=r, q=q, T=float(tt), cf_log_return=_cf, cfg=fft_cfg)
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

        # Build surface for UI (prices -> implied vols)
        price_grid = np.zeros((len(t_grid), len(m_grid)), dtype=float)
        for i_t, tt in enumerate(t_grid):
            if float(tt) <= 0:
                continue

            def _cf(u: np.ndarray, T_in: float) -> np.ndarray:
                d = rheston_log_return_cf_markovian(
                    u,
                    [float(T_in)],
                    r=r,
                    q=q,
                    kappa=float(params["kappa"]),
                    theta=float(params["theta"]),
                    xi=float(params["xi"]),
                    rho=float(params["rho"]),
                    v0=float(params["v0"]),
                    H=float(params["H"]),
                    cfg=mk_cfg,
                )
                return d[float(T_in)]

            K_grid, C_grid = carr_madan_fft_call_prices(S0=S0, r=r, q=q, T=float(tt), cf_log_return=_cf, cfg=fft_cfg)
            strikes = (S0 * m_grid).astype(float)
            prices = np.asarray(interp_prices(K_grid=K_grid, C_grid=C_grid, K=strikes), dtype=float).reshape(-1)
            row = np.full_like(price_grid[i_t, :], np.nan, dtype=float)
            n = min(row.size, prices.size)
            row[:n] = prices[:n]
            price_grid[i_t, :] = row

        iv_model = implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q)
        iv_error = np.where(mask_eff, iv_model - iv_mkt, np.nan)
        metrics = iv_error_metrics(iv_error, mask_eff)
        vega_weights = compute_bs_vega_grid(S0, m_grid, t_grid, r, q, iv_mkt)
        metrics_vw = iv_error_metrics_weighted(iv_error, mask_eff, vega_weights)

        if best_meta is not None:
            best_meta["best"] = True

        return SurfaceCalibrationResult(
            success=True,
            message="OK (rHeston approx, expensive)",
            model=self.model,
            method=self.method,
            params=params,
            metrics=metrics,
            metrics_vw=metrics_vw,
            iv_model=iv_model,
            iv_error=iv_error,
            vega_weights=vega_weights,
            details={
                "runs": runs,
                "fft_cfg": {"alpha": fft_cfg.alpha, "n": fft_cfg.n, "eta": fft_cfg.eta},
                "markovian_cfg": {
                    "n_factors": mk_cfg.n_factors,
                    "steps_per_year": mk_cfg.steps_per_year,
                    "x_min_mult": mk_cfg.x_min_mult,
                    "x_max_mult": mk_cfg.x_max_mult,
                },
            },
        )


__all__ = ["RHestonFFTMarkovianCalibrator"]

