from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_bs_vega_grid(
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    r: float,
    q: float,
    iv_grid: np.ndarray,
) -> np.ndarray:
    """
    BS vega for each (T, moneyness) grid point.
    vega = S0 * sqrt(T) * N'(d1) * exp(-q*T)
    Returns array of shape (len(t_grid), len(m_grid)), NaN where iv <= 0 or T <= 0.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    iv_grid = np.asarray(iv_grid, dtype=float)
    S0 = float(S0)
    r = float(r)
    q = float(q)

    vega = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
    for i, T in enumerate(t_grid):
        if T <= 0:
            continue
        sqrtT = float(np.sqrt(T))
        for j, m in enumerate(m_grid):
            K = float(m * S0)
            if K <= 0:
                continue
            iv = float(iv_grid[i, j])
            if not np.isfinite(iv) or iv <= 0:
                continue
            d1 = (float(np.log(S0 / K)) + (r - q + 0.5 * iv * iv) * T) / (iv * sqrtT)
            n_d1 = float(np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi))
            vega[i, j] = S0 * sqrtT * n_d1 * float(np.exp(-q * T))
    return vega


def iv_error_metrics_weighted(
    iv_error: np.ndarray,
    mask: np.ndarray | None,
    vega_weights: np.ndarray | None,
) -> Dict[str, float]:
    """
    Vega-weighted IV error metrics.
    Weights are normalised so they sum to 1 over the active mask.
    Returns mae_vw, rmse_vw, max_abs_vw.
    """
    err = np.asarray(iv_error, dtype=float)
    if mask is None:
        m = np.isfinite(err)
    else:
        m = np.asarray(mask, dtype=bool) & np.isfinite(err)

    if not np.any(m):
        return {"mae_vw": float("nan"), "rmse_vw": float("nan"), "max_abs_vw": float("nan")}

    e = err[m]

    if vega_weights is not None:
        w = np.asarray(vega_weights, dtype=float)[m]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        w_sum = float(np.sum(w))
        if w_sum > 0:
            w = w / w_sum
            mae_vw = float(np.sum(w * np.abs(e)))
            rmse_vw = float(np.sqrt(np.sum(w * e * e)))
            max_abs_vw = float(np.max(np.abs(e)))
            return {"mae_vw": mae_vw, "rmse_vw": rmse_vw, "max_abs_vw": max_abs_vw}

    # Fallback to unweighted if no valid weights
    return {
        "mae_vw": float(np.mean(np.abs(e))),
        "rmse_vw": float(np.sqrt(np.mean(e * e))),
        "max_abs_vw": float(np.max(np.abs(e))),
    }


def effective_mask(iv_market: np.ndarray, mask: np.ndarray | None, *, fit_to_observed_only: bool) -> np.ndarray:
    iv_market = np.asarray(iv_market, dtype=float)
    base = np.isfinite(iv_market) & (iv_market > 0)
    if mask is None or not fit_to_observed_only:
        return base
    return base & np.asarray(mask, dtype=bool)


def iv_error_metrics(iv_error: np.ndarray, mask: np.ndarray | None) -> Dict[str, float]:
    err = np.asarray(iv_error, dtype=float)
    if mask is None:
        m = np.isfinite(err)
    else:
        m = np.asarray(mask, dtype=bool) & np.isfinite(err)
    if not np.any(m):
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "n": 0.0}
    e = err[m]
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "max_abs": float(np.max(np.abs(e))),
        "n": float(e.size),
    }


def grid_points_from_mask(
    *,
    S0: float,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    iv_market: np.ndarray,
    mask: np.ndarray | None,
    fit_to_observed_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns vectors (K, T, iv) for points selected by mask.
    """
    m_grid = np.asarray(m_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    iv_market = np.asarray(iv_market, dtype=float)
    m = effective_mask(iv_market, mask, fit_to_observed_only=fit_to_observed_only)
    idx = np.argwhere(m)
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([])
    K = np.empty(len(idx), dtype=float)
    T = np.empty(len(idx), dtype=float)
    iv = np.empty(len(idx), dtype=float)
    for k, (i_t, j_m) in enumerate(idx):
        T[k] = float(t_grid[int(i_t)])
        K[k] = float(S0) * float(m_grid[int(j_m)])
        iv[k] = float(iv_market[int(i_t), int(j_m)])
    return K, T, iv


__all__ = [
    "compute_bs_vega_grid",
    "effective_mask",
    "iv_error_metrics",
    "iv_error_metrics_weighted",
    "grid_points_from_mask",
]

