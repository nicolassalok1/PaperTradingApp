from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


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


__all__ = ["effective_mask", "iv_error_metrics", "grid_points_from_mask"]

