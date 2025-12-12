from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import griddata
except Exception:  # pragma: no cover - optional dependency
    griddata = None


_MONEYNESS_ALIASES = {"moneyness", "k", "k_over_s", "k/s", "k_over_s0", "k/s0"}
_TTM_ALIASES = {"ttm", "tau", "maturity", "t", "time_to_maturity"}
_IV_ALIASES = {"iv", "implied_vol", "implied_volatility", "vol", "sigma"}


def load_market_surface_csv(file_bytes_or_path) -> pd.DataFrame:
    """
    Load a market IV surface CSV (bytes or path) and normalize column names:
      - moneyness in _MONEYNESS_ALIASES
      - ttm in _TTM_ALIASES
      - iv in _IV_ALIASES
    Returns a DataFrame with float columns: moneyness, ttm, iv.
    """
    df = None
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(file_bytes_or_path))
    elif isinstance(file_bytes_or_path, (str, Path)):
        df = pd.read_csv(file_bytes_or_path)
    elif isinstance(file_bytes_or_path, pd.DataFrame):
        df = file_bytes_or_path.copy()
    else:
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])

    cols = {str(c).strip().lower(): c for c in df.columns}

    def _find_col(candidates: Iterable[str]):
        for alias in candidates:
            if alias in cols:
                return cols[alias]
        return None

    m_col = _find_col(_MONEYNESS_ALIASES)
    t_col = _find_col(_TTM_ALIASES)
    iv_col = _find_col(_IV_ALIASES)
    if not (m_col and t_col and iv_col):
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])

    out = pd.DataFrame(
        {
            "moneyness": pd.to_numeric(df[m_col], errors="coerce"),
            "ttm": pd.to_numeric(df[t_col], errors="coerce"),
            "iv": pd.to_numeric(df[iv_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["moneyness", "ttm", "iv"])
    return out


def default_grid() -> Tuple[np.ndarray, np.ndarray]:
    m_grid = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], dtype=float)
    t_grid = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], dtype=float)
    return m_grid, t_grid


def make_fixed_grid(
    df: pd.DataFrame, m_grid: np.ndarray, t_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project market IV points onto a fixed (t_grid, m_grid) mesh.
    Returns (iv_grid, mask) where mask=True for observed points.
    """
    if df is None:
        df = pd.DataFrame(columns=["moneyness", "ttm", "iv"])
    iv_grid = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
    mask = np.zeros_like(iv_grid, dtype=bool)

    for _, row in df.iterrows():
        try:
            m = float(row["moneyness"])
            t = float(row["ttm"])
            iv = float(row["iv"])
        except Exception:
            continue
        i_t = (np.abs(t_grid - t)).argmin()
        i_m = (np.abs(m_grid - m)).argmin()
        iv_grid[i_t, i_m] = iv
        mask[i_t, i_m] = True

    if np.isfinite(iv_grid).any():
        # interpolate missing values
        known_mask = np.isfinite(iv_grid)
        pts = np.argwhere(known_mask)
        vals = iv_grid[known_mask]
        tgt_T, tgt_M = np.meshgrid(np.arange(len(t_grid)), np.arange(len(m_grid)), indexing="ij")
        tgt_points = np.vstack([tgt_T.ravel(), tgt_M.ravel()]).T
        if griddata is not None and len(vals) >= 3:
            try:
                interp = griddata(pts, vals, tgt_points, method="linear")
                iv_interp = interp.reshape(iv_grid.shape)
            except Exception:
                iv_interp = np.full_like(iv_grid, np.nan, dtype=float)
        else:
            iv_interp = np.full_like(iv_grid, np.nan, dtype=float)
        iv_fill = iv_interp
        if not np.isfinite(iv_fill).any():
            median_iv = float(np.nanmedian(vals))
            iv_fill = np.full_like(iv_grid, median_iv)
        # fill NaN with nearest if available
        if griddata is not None and len(vals) >= 1:
            try:
                nearest = griddata(pts, vals, tgt_points, method="nearest").reshape(iv_grid.shape)
                iv_fill = np.where(np.isfinite(iv_fill), iv_fill, nearest)
            except Exception:
                pass
        iv_grid = np.where(np.isfinite(iv_grid), iv_grid, iv_fill)
    else:
        iv_grid[:] = 0.2  # default flat surface

    return iv_grid.astype(float), mask


__all__ = ["load_market_surface_csv", "make_fixed_grid", "default_grid"]
