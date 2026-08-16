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

_K_ALIAS = {"k", "strike", "strike_price", "strikeprice", "k_rel"}
_T_ALIAS = {"t", "tau", "ttm", "time_to_maturity", "maturity"}
_S0_ALIAS = {"s0", "spot", "underlying", "underlyingprice"}
_IV_ALIAS = {"iv", "impliedvol", "implied_vol", "impliedvolatility"}
_TYPE_ALIAS = {"type", "option_type", "cp"}

M_GRID_DEFAULT = np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2], dtype=float)
T_GRID_DEFAULT = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 1.0], dtype=float)


def _read_csv(file_bytes_or_path):
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_bytes_or_path))
    if isinstance(file_bytes_or_path, (str, Path)):
        return pd.read_csv(file_bytes_or_path)
    if isinstance(file_bytes_or_path, pd.DataFrame):
        return file_bytes_or_path.copy()
    return None


def _find_col(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in cols:
            return cols[alias]
    return None


def load_market_surface_csv(file_bytes_or_path) -> pd.DataFrame:
    """
    Backward-compat loader supporting moneyness/ttm/iv columns (existing behavior).
    """
    df = _read_csv(file_bytes_or_path)
    if df is None:
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


def load_csv(file_bytes_or_path) -> pd.DataFrame:
    """
    New loader for CSV with columns: K, T, S0, iv, type (CALL-only).
    Returns canonical df: moneyness, ttm, iv.
    """
    df = _read_csv(file_bytes_or_path)
    if df is None:
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])

    k_col = _find_col(df, _K_ALIAS)
    t_col = _find_col(df, _T_ALIAS)
    s0_col = _find_col(df, _S0_ALIAS)
    iv_col = _find_col(df, _IV_ALIAS)
    typ_col = _find_col(df, _TYPE_ALIAS)

    if not (k_col and t_col and s0_col and iv_col):
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])

    df_work = df.copy()
    if typ_col:
        try:
            df_work = df_work[df_work[typ_col].str.lower() == "call"]
        except Exception:
            df_work = df_work

    k = pd.to_numeric(df_work[k_col], errors="coerce")
    t = pd.to_numeric(df_work[t_col], errors="coerce")
    s0 = pd.to_numeric(df_work[s0_col], errors="coerce")
    iv = pd.to_numeric(df_work[iv_col], errors="coerce")

    out = pd.DataFrame({"K": k, "T": t, "S0": s0, "iv": iv})
    out = out[(out["K"] > 0) & (out["S0"] > 0) & (out["T"] > 0) & (out["iv"] > 0)]
    if out.empty:
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])
    out["moneyness"] = out["K"] / out["S0"]
    out["ttm"] = out["T"]
    out = out.dropna(subset=["moneyness", "ttm", "iv"])
    return out[["moneyness", "ttm", "iv", "S0"]]


def default_grid() -> Tuple[np.ndarray, np.ndarray]:
    return M_GRID_DEFAULT.copy(), T_GRID_DEFAULT.copy()


def _interpolate_grid(points: np.ndarray, values: np.ndarray, m_grid: np.ndarray, t_grid: np.ndarray):
    # Nearest-node snapping decides which nodes count as *observed* (Yahoo expiries and
    # strikes never sit exactly on the grid, so a tolerance is needed). It must NOT decide
    # the node's *value*: an observation can be snapped from far away — a 1.9-year put onto
    # the 1-year slot, a 0.4-moneyness wing onto the 0.8 edge — and last-write-wins would
    # hand exactly those corrupted nodes to the calibrators. Values come from interpolating
    # the scattered observations at the node; the snapped value is only a last-resort seed.
    seeded = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
    mask = np.zeros_like(seeded, dtype=bool)
    for (m, t), iv in zip(points, values):
        i_t = int(np.abs(t_grid - t).argmin())
        i_m = int(np.abs(m_grid - m).argmin())
        seeded[i_t, i_m] = iv
        mask[i_t, i_m] = True

    if len(values) == 0:
        return np.full_like(seeded, 0.2), mask
    iv_grid = np.full_like(seeded, np.nan)

    if griddata is not None and len(values) >= 3:
        try:
            T_mesh, M_mesh = np.meshgrid(t_grid, m_grid, indexing="ij")
            interp = griddata(points[:, [1, 0]], values, (T_mesh, M_mesh), method="linear")
        except Exception:
            interp = np.full_like(iv_grid, np.nan)
    else:
        interp = np.full_like(iv_grid, np.nan)

    if griddata is not None and not np.isfinite(interp).all():
        try:
            T_mesh, M_mesh = np.meshgrid(t_grid, m_grid, indexing="ij")
            nearest = griddata(points[:, [1, 0]], values, (T_mesh, M_mesh), method="nearest")
            interp = np.where(np.isfinite(interp), interp, nearest)
        except Exception:
            pass

    # Priority: linear interpolation > nearest observation > snapped seed > median.
    iv_grid = np.where(np.isfinite(interp), interp, seeded)
    if not np.isfinite(iv_grid).all():
        median_iv = float(np.nanmedian(values))
        iv_grid = np.where(np.isfinite(iv_grid), iv_grid, median_iv)
    return iv_grid.astype(float), mask


def build_fixed_grid(df: pd.DataFrame):
    """
    Build fixed grid (m_grid, t_grid) and interpolate IVs onto it.
    Returns iv_grid, mask, m_grid, t_grid.
    """
    m_grid, t_grid = default_grid()
    if df is None:
        df = pd.DataFrame(columns=["moneyness", "ttm", "iv"])
    df = df.dropna(subset=["moneyness", "ttm", "iv"])
    pts = df[["moneyness", "ttm", "iv"]].to_numpy(dtype=float)
    if pts.size == 0:
        iv_grid = np.full((len(t_grid), len(m_grid)), 0.2, dtype=float)
        mask = np.zeros_like(iv_grid, dtype=bool)
        return iv_grid, mask, m_grid, t_grid

    points = pts[:, :2]
    values = pts[:, 2]
    iv_grid, mask = _interpolate_grid(points, values, m_grid, t_grid)
    return iv_grid, mask, m_grid, t_grid


def make_fixed_grid(
    df: pd.DataFrame, m_grid: np.ndarray, t_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy signature retained; delegates to build_fixed_grid using provided grids.
    """
    df = df if df is not None else pd.DataFrame(columns=["moneyness", "ttm", "iv"])
    pts = df[["moneyness", "ttm", "iv"]].dropna().to_numpy(dtype=float)
    if pts.size == 0:
        iv_grid = np.full((len(t_grid), len(m_grid)), 0.2, dtype=float)
        mask = np.zeros_like(iv_grid, dtype=bool)
        return iv_grid, mask
    iv_grid, mask = _interpolate_grid(pts[:, :2], pts[:, 2], m_grid, t_grid)
    return iv_grid, mask


__all__ = ["load_market_surface_csv", "load_csv", "build_fixed_grid", "make_fixed_grid", "default_grid"]
