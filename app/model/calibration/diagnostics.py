"""
Pure-model diagnostic functions for calibration results.
No Streamlit imports allowed here (model layer).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def residuals_by_maturity(
    iv_error: np.ndarray,
    mask: np.ndarray | None,
    t_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Per-maturity slice: MAE and RMSE of IV error.
    Returns list of dicts sorted by maturity: [{T, mae, rmse, n}, ...]
    """
    iv_error = np.asarray(iv_error, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    results = []
    for i, T in enumerate(t_grid):
        row = iv_error[i, :]
        if mask is not None:
            m = np.asarray(mask, dtype=bool)[i, :] & np.isfinite(row)
        else:
            m = np.isfinite(row)
        e = row[m]
        if e.size == 0:
            results.append({"T": float(T), "mae": float("nan"), "rmse": float("nan"), "n": 0})
            continue
        results.append(
            {
                "T": float(T),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e * e))),
                "n": int(e.size),
            }
        )
    return results


def residuals_by_moneyness(
    iv_error: np.ndarray,
    mask: np.ndarray | None,
    m_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Per-moneyness column: MAE of IV error.
    Returns list of dicts: [{m, label, mae, rmse, n}, ...]
    """
    iv_error = np.asarray(iv_error, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    results = []
    for j, m_val in enumerate(m_grid):
        col = iv_error[:, j]
        if mask is not None:
            msk = np.asarray(mask, dtype=bool)[:, j] & np.isfinite(col)
        else:
            msk = np.isfinite(col)
        e = col[msk]

        if m_val < 0.97:
            label = "OTM Put"
        elif m_val > 1.03:
            label = "OTM Call"
        else:
            label = "ATM"

        if e.size == 0:
            results.append({"m": float(m_val), "label": label, "mae": float("nan"), "rmse": float("nan"), "n": 0})
            continue
        results.append(
            {
                "m": float(m_val),
                "label": label,
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e * e))),
                "n": int(e.size),
            }
        )
    return results


def smile_decomposition(
    iv_market: np.ndarray,
    mask: np.ndarray | None,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
) -> Dict[str, Any]:
    """
    Per-maturity smile decomposition:
    - atm_iv_by_t: IV at moneyness closest to 1.0
    - skew_by_t:   iv(m~0.95) - iv(m~1.05)  (risk-reversal proxy)
    - curvature_by_t: iv(m~0.90) + iv(m~1.10) - 2*iv(m~1.0)  (butterfly proxy)

    Returns dict with lists indexed by maturity.
    """
    iv_market = np.asarray(iv_market, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    def _nearest_iv(i_t: int, target_m: float) -> float | None:
        j = int(np.abs(m_grid - target_m).argmin())
        val = float(iv_market[i_t, j])
        if mask is not None and not bool(np.asarray(mask, dtype=bool)[i_t, j]):
            return None
        return val if np.isfinite(val) and val > 0 else None

    maturities: List[float] = []
    atm_iv: List[float | None] = []
    skew: List[float | None] = []
    curvature: List[float | None] = []

    for i, T in enumerate(t_grid):
        maturities.append(float(T))
        iv_atm = _nearest_iv(i, 1.00)
        iv_put = _nearest_iv(i, 0.95)
        iv_call = _nearest_iv(i, 1.05)
        iv_wing_put = _nearest_iv(i, 0.90)
        iv_wing_call = _nearest_iv(i, 1.10)

        atm_iv.append(iv_atm)

        if iv_put is not None and iv_call is not None:
            skew.append(iv_put - iv_call)
        else:
            skew.append(None)

        if iv_wing_put is not None and iv_wing_call is not None and iv_atm is not None:
            curvature.append(iv_wing_put + iv_wing_call - 2.0 * iv_atm)
        else:
            curvature.append(None)

    return {
        "maturities": maturities,
        "atm_iv_by_t": atm_iv,
        "skew_by_t": skew,
        "curvature_by_t": curvature,
    }


def compute_all_diagnostics(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper: compute all diagnostics from a calibration result dict
    (as returned by CalibrationController.run_advanced_surface_calibration).
    """
    iv_error = np.asarray(result_dict.get("iv_error") or [], dtype=float)
    iv_market = np.asarray(result_dict.get("iv_market") or [], dtype=float)
    m_grid = np.asarray(result_dict.get("m_grid") or [], dtype=float)
    t_grid = np.asarray(result_dict.get("t_grid") or [], dtype=float)
    mask_raw = result_dict.get("mask")
    mask = np.asarray(mask_raw, dtype=bool) if mask_raw is not None else None

    if iv_error.ndim != 2 or m_grid.size == 0 or t_grid.size == 0:
        return {}

    return {
        "by_maturity": residuals_by_maturity(iv_error, mask, t_grid),
        "by_moneyness": residuals_by_moneyness(iv_error, mask, m_grid),
        "smile": smile_decomposition(iv_market, mask, m_grid, t_grid),
    }


__all__ = [
    "compute_all_diagnostics",
    "residuals_by_maturity",
    "residuals_by_moneyness",
    "smile_decomposition",
]
