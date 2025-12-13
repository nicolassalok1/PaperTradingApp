from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional dependency
    least_squares = None

from app.model.calibration.heston_pricer import call_price_cf
from app.model.calibration.implied_vol import bs_call_price


PARAM_ORDER = ("kappa", "theta", "sigma", "rho", "v0")

DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "kappa": (0.1, 10.0),
    "theta": (1e-4, 1.5),
    "sigma": (1e-3, 5.0),
    "rho": (-0.999, 0.999),
    "v0": (1e-4, 1.5),
}


def _coerce_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_param_constraint(val: Any) -> tuple[float | None, float | None, float | None]:
    """
    Returns (min, max, fixed) where fixed sets min=max=fixed.
    Supported formats:
      - number: fixed value
      - [min, max] / (min, max)
      - {"min": ..., "max": ...} or {"value": ...}
    """
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


def build_bounds(constraints: Dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Build (lb, ub) arrays for PARAM_ORDER.
    Applies optional per-parameter bounds from constraints.
    """
    lower = {k: float(v[0]) for k, v in DEFAULT_BOUNDS.items()}
    upper = {k: float(v[1]) for k, v in DEFAULT_BOUNDS.items()}

    if isinstance(constraints, dict):
        for name in PARAM_ORDER:
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

    lb = np.array([lower[p] for p in PARAM_ORDER], dtype=float)
    ub = np.array([upper[p] for p in PARAM_ORDER], dtype=float)

    if not np.isfinite(lb).all() or not np.isfinite(ub).all():
        return lb, ub, "Bornes invalides (non finies)."
    if np.any(lb > ub):
        return lb, ub, "Bornes invalides: min > max."
    return lb, ub, None


def initial_guess_from_surface(
    iv_market: np.ndarray, m_grid: np.ndarray, t_grid: np.ndarray, *, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """
    Heuristic initial guess:
      - v0 ~ ATM(short)^2
      - theta ~ ATM(long)^2
      - kappa ~ 2, sigma ~ 0.5, rho ~ -0.5
    Clipped to bounds.
    """
    atm_idx = int(np.abs(np.asarray(m_grid, dtype=float) - 1.0).argmin()) if len(m_grid) else 0
    iv_atm = None
    try:
        iv_atm = np.asarray(iv_market, dtype=float)[:, atm_idx]
    except Exception:
        iv_atm = None

    def _pick_at(idx: int) -> float | None:
        if iv_atm is None or iv_atm.size == 0:
            return None
        try:
            v = float(iv_atm[idx])
        except Exception:
            return None
        return v if np.isfinite(v) and v > 0 else None

    iv_short = _pick_at(0) or _pick_at(min(1, len(t_grid) - 1)) or 0.2
    iv_long = _pick_at(len(t_grid) - 1) or 0.2

    x0 = np.array([2.0, iv_long * iv_long, 0.5, -0.5, iv_short * iv_short], dtype=float)
    return np.minimum(np.maximum(x0, lb), ub)


def _build_fit_points(
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_market: np.ndarray,
    mask: np.ndarray | None,
    *,
    fit_to_observed_only: bool,
    r: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns vectors (K, T, px_mkt, scale) for least-squares.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    iv_market = np.asarray(iv_market, dtype=float)

    if mask is None or not fit_to_observed_only:
        mask_eff = np.isfinite(iv_market) & (iv_market > 0)
    else:
        mask_eff = np.asarray(mask, dtype=bool) & np.isfinite(iv_market) & (iv_market > 0)

    idx = np.argwhere(mask_eff)
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    K = np.empty(len(idx), dtype=float)
    T = np.empty(len(idx), dtype=float)
    px_mkt = np.empty(len(idx), dtype=float)
    scale = np.empty(len(idx), dtype=float)

    for n, (i_t, j_m) in enumerate(idx):
        t = float(t_grid[int(i_t)])
        m = float(m_grid[int(j_m)])
        vol = float(iv_market[int(i_t), int(j_m)])
        k = float(m * S0)
        px = float(bs_call_price(S0, k, t, r, q, vol))
        K[n] = k
        T[n] = t
        px_mkt[n] = px
        scale[n] = max(1.0, abs(px))

    return K, T, px_mkt, scale


def calibrate_heston_least_squares(
    *,
    S0: float,
    r: float,
    q: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_market: np.ndarray,
    mask: np.ndarray | None = None,
    constraints: Dict[str, Any] | None = None,
    fit_to_observed_only: bool = True,
    u_max: float = 50.0,
    n_integration: int = 2000,
    max_nfev: int = 50,
    n_starts: int = 1,
    seed: int | None = 42,
) -> Dict[str, Any]:
    if least_squares is None:
        return {"success": False, "message": "SciPy indisponible: least_squares manquant.", "params": {}}

    if S0 <= 0:
        return {"success": False, "message": "S0 invalide.", "params": {}}

    lb, ub, err = build_bounds(constraints)
    if err:
        return {"success": False, "message": err, "params": {}}

    K, T, px_mkt, scale = _build_fit_points(
        S0, m_grid, t_grid, iv_market, mask, fit_to_observed_only=fit_to_observed_only, r=r, q=q
    )
    if K.size == 0:
        return {"success": False, "message": "Aucun point valide pour la calibration.", "params": {}}

    def residuals(x: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = [float(v) for v in x]
        params = (kappa, theta, sigma, rho, v0)
        res = np.empty_like(px_mkt, dtype=float)
        for i in range(len(px_mkt)):
            model_px = call_price_cf(
                S0,
                float(K[i]),
                float(T[i]),
                float(r),
                float(q),
                params,
                u_max=float(u_max),
                N=int(n_integration),
            )
            res[i] = (float(model_px) - float(px_mkt[i])) / float(scale[i])

        # Feller condition penalty (soft)
        feller_gap = float(max(0.0, sigma * sigma - 2.0 * kappa * theta))
        if feller_gap > 0:
            res = np.concatenate([res, np.array([feller_gap], dtype=float)])
        return res

    def _run_one(x0: np.ndarray) -> Dict[str, Any]:
        try:
            opt = least_squares(
                residuals,
                x0=np.asarray(x0, dtype=float),
                bounds=(lb, ub),
                max_nfev=int(max_nfev),
                method="trf",
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        x = np.asarray(opt.x, dtype=float)
        params = {k: float(v) for k, v in zip(PARAM_ORDER, x)}
        return {
            "ok": True,
            "converged": bool(getattr(opt, "success", False)),
            "message": str(getattr(opt, "message", "")),
            "params": params,
            "x": x.tolist(),
            "nfev": int(getattr(opt, "nfev", 0) or 0),
            "cost": float(getattr(opt, "cost", np.nan)),
            "optimality": float(getattr(opt, "optimality", np.nan)),
        }

    def _sample_x0(rng: np.random.Generator) -> np.ndarray:
        x = np.empty(len(PARAM_ORDER), dtype=float)
        for i, name in enumerate(PARAM_ORDER):
            lo = float(lb[i])
            hi = float(ub[i])
            if not np.isfinite(lo) or not np.isfinite(hi):
                x[i] = 0.0
                continue
            if abs(hi - lo) < 1e-12:
                x[i] = lo
                continue
            if name == "rho":
                x[i] = float(rng.uniform(lo, hi))
                continue
            lo_pos = max(lo, 1e-12)
            hi_pos = max(hi, lo_pos * 1.000001)
            # log-uniform for wide ranges, else uniform
            if hi_pos / lo_pos > 50.0:
                x[i] = float(np.exp(rng.uniform(np.log(lo_pos), np.log(hi_pos))))
            else:
                x[i] = float(rng.uniform(lo, hi))
        return np.minimum(np.maximum(x, lb), ub)

    n_starts_eff = int(n_starts or 1)
    if n_starts_eff < 1:
        n_starts_eff = 1

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    candidates: list[np.ndarray] = []
    # 1) heuristic guess from surface
    candidates.append(initial_guess_from_surface(iv_market, m_grid, t_grid, lb=lb, ub=ub))
    # 2) mid-point of bounds
    candidates.append(np.minimum(np.maximum(0.5 * (lb + ub), lb), ub))
    # 3+) random guesses
    while len(candidates) < n_starts_eff:
        candidates.append(_sample_x0(rng))

    runs: list[Dict[str, Any]] = []
    best_idx: int | None = None
    best_cost: float = float("inf")

    for i, x0 in enumerate(candidates):
        out = _run_one(x0)
        out["idx"] = int(i)
        out["x0"] = [float(v) for v in np.asarray(x0, dtype=float)]
        runs.append(out)
        if not out.get("ok"):
            continue
        cost = out.get("cost")
        try:
            cost_f = float(cost)
        except Exception:
            continue
        if not np.isfinite(cost_f):
            continue
        if cost_f < best_cost:
            best_cost = cost_f
            best_idx = int(i)

    if best_idx is None:
        first_err = next((r.get("error") for r in runs if r.get("error")), "Optimisation échouée.")
        return {"success": False, "message": f"Optimisation échouée: {first_err}", "params": {}, "runs": runs}

    best = runs[best_idx]
    return {
        "success": True,
        "converged": bool(best.get("converged", False)),
        "message": str(best.get("message", "")),
        "params": best.get("params") or {},
        "nfev": int(best.get("nfev", 0) or 0),
        "cost": float(best.get("cost", np.nan)),
        "optimality": float(best.get("optimality", np.nan)),
        "n_starts": int(n_starts_eff),
        "seed": int(seed) if seed is not None else None,
        "best_run": int(best_idx),
        "runs": runs,
    }


__all__ = [
    "PARAM_ORDER",
    "DEFAULT_BOUNDS",
    "build_bounds",
    "initial_guess_from_surface",
    "calibrate_heston_least_squares",
]
