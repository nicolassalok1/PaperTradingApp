"""
Yield curve services (no UI).
Handles curve loading, caching, and forward rate computations.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from app.model.market_data.realtime import get_data
from app.model.market_data.rates import get_r
from app.model.yieldcurve.loader import (
    YIELD_CURVE_CACHE_FILE,
    download_yield_curve_to_cache,
    load_yield_curve_csv,
)
from app.utils.io import load_json_file, save_json_file
from app.utils.math_utils import floor_4
from app.utils.paths import JSON_DIR

FORWARDS_FILE = JSON_DIR / "forwards_portfolio.json"


def load_curve(ensure_cache: bool = False) -> Tuple[pd.DataFrame | None, Path | None]:
    return load_yield_curve_csv(ensure_cache=ensure_cache)


def build_curve(period: str = "1y") -> Tuple[pd.DataFrame | None, Path | None]:
    return download_yield_curve_to_cache(period=period)


def refresh_curve(period: str = "1y") -> Tuple[pd.DataFrame | None, Path | None]:
    """
    Convenience wrapper for rebuilding the curve cache.
    """
    return download_yield_curve_to_cache(period=period)


def yield_curve_cache_file() -> Path:
    return YIELD_CURVE_CACHE_FILE


def get_spot(sym: str):
    return get_data(sym)


def get_rate():
    return get_r


def load_forwards() -> dict:
    return load_json_file(FORWARDS_FILE, {})


def save_forwards(data: dict) -> None:
    save_json_file(FORWARDS_FILE, data)


def compute_forward_price(spot: float, r: float, T: float) -> float:
    try:
        return float(spot) * math.exp(float(r) * float(T))
    except Exception:
        return 0.0


def _latest_curve_points(df_curve: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Extract the latest curve row and return sorted (tenors, rates) in years/decimals.
    """
    if df_curve is None or df_curve.empty:
        return [], []
    df_latest = df_curve.dropna(how="all").tail(1)
    if df_latest.empty:
        return [], []
    tenor_years = {"3M": 0.25, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}
    tenors: List[float] = []
    rates: List[float] = []
    for col, years in tenor_years.items():
        if col in df_latest.columns:
            val = pd.to_numeric(df_latest[col], errors="coerce").iloc[-1]
            if pd.notna(val):
                tenors.append(float(years))
                rates.append(float(val) / 100.0 if val > 1 else float(val))
    if len(tenors) < 2:
        return [], []
    tenors_sorted, rates_sorted = zip(*sorted(zip(tenors, rates)))
    return list(tenors_sorted), list(rates_sorted)


def _interpolate_from_points(
    tenors: List[float], rates: List[float], T_years: float
) -> float | None:
    if not tenors or not rates:
        return None
    try:
        series = pd.Series(rates, index=tenors).sort_index()
        return float(series.reindex(series.index.union([T_years])).interpolate().loc[T_years])
    except Exception:
        return None


def interpolate_curve_rate(df_curve: pd.DataFrame, T_years: float) -> float | None:
    """
    Approximate a rate using the latest curve row and simple interpolation across tenors.
    """
    tenors, rates = _latest_curve_points(df_curve)
    return _interpolate_from_points(tenors, rates, T_years)


def _get_curve_rate(df_curve: pd.DataFrame, T_years: float) -> float | None:
    """
    Interpolate curve rate then fall back to model rate if unavailable.
    """
    rate = interpolate_curve_rate(df_curve, T_years)
    if rate is not None:
        return rate
    try:
        return float(get_r(T_years))
    except Exception:
        return None


def compute_forward_rate(
    df_curve: pd.DataFrame, start_years: float, end_years: float
) -> float | None:
    """
    Compute forward rate between start and end using interpolated zero rates.
    """
    if end_years <= start_years or end_years <= 0:
        return None
    r1 = _get_curve_rate(df_curve, max(start_years, 0.0))
    r2 = _get_curve_rate(df_curve, end_years)
    if r1 is None or r2 is None:
        return None
    try:
        df1 = math.exp(-r1 * max(start_years, 0.0))
        df2 = math.exp(-r2 * end_years)
        return -(math.log(df2) - math.log(df1)) / (end_years - max(start_years, 0.0))
    except Exception:
        return None


def compute_forward_rates(
    df_curve: pd.DataFrame,
    horizons: Iterable[Tuple[float, float]] | None = None,
) -> Dict[Tuple[float, float], float]:
    """
    Compute a set of forward rates for given (t1, t2) horizons.
    """
    if horizons is None:
        horizons = [(0.0, 0.25), (0.25, 1.0), (1.0, 2.0), (2.0, 5.0)]
    rates: Dict[Tuple[float, float], float] = {}
    for t1, t2 in horizons:
        val = compute_forward_rate(df_curve, float(t1), float(t2))
        if val is not None and math.isfinite(val):
            rates[(float(t1), float(t2))] = float(val)
    return rates


def compute_instantaneous_rate(
    df_curve: pd.DataFrame, t_years: float, h: float = 1e-4
) -> float | None:
    """
    Estimate instantaneous forward rate f(t) using central difference on discount factors.
    """
    if t_years < 0:
        return None
    h = max(float(h), 1e-6)
    t_plus = float(t_years) + h
    t_minus = max(float(t_years) - h, 0.0)
    r_plus = _get_curve_rate(df_curve, t_plus)
    r_minus = _get_curve_rate(df_curve, t_minus)
    if r_plus is None or r_minus is None:
        return None
    try:
        df_plus = math.exp(-r_plus * t_plus)
        df_minus = math.exp(-r_minus * t_minus)
        return -(math.log(df_plus) - math.log(df_minus)) / (2 * h)
    except Exception:
        return None


def compute_instantaneous_rates(
    df_curve: pd.DataFrame, maturities: Iterable[float]
) -> Dict[float, float]:
    """
    Compute instantaneous forward rates for provided maturities.
    """
    results: Dict[float, float] = {}
    for t in maturities:
        val = compute_instantaneous_rate(df_curve, float(t))
        if val is not None and math.isfinite(val):
            results[float(t)] = float(val)
    return results


def prepare_forward_rows(forwards: dict, today: datetime.date | None = None) -> List[dict]:
    """
    Enrich forward portfolio entries with spot and maturity metadata for UI consumption.
    """
    today = today or datetime.date.today()
    rows: List[dict] = []
    for _, fwd in (forwards or {}).items():
        sym = fwd.get("symbol", "")
        qty = int(fwd.get("quantity", 0) or 0)
        price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
        side = fwd.get("side", "long")
        maturity_str = fwd.get("maturity")
        try:
            maturity_dt = datetime.date.fromisoformat(maturity_str) if maturity_str else None
        except Exception:
            maturity_dt = None
        days_to_mat = (maturity_dt - today).days if maturity_dt else None
        spot_now_val = floor_4(get_spot(sym).get("price", 0.0)) if sym else 0.0
        rows.append(
            {
                "Symbol": sym,
                "Side": side.capitalize(),
                "Quantity": qty,
                "Forward Price": price_fwd,
                "Spot Now": spot_now_val,
                "Maturity": maturity_str,
                "Days to mat": days_to_mat,
            }
        )
    return rows


__all__ = [
    "load_curve",
    "build_curve",
    "refresh_curve",
    "yield_curve_cache_file",
    "get_spot",
    "get_rate",
    "load_forwards",
    "save_forwards",
    "compute_forward_price",
    "compute_forward_rate",
    "compute_forward_rates",
    "compute_instantaneous_rate",
    "compute_instantaneous_rates",
    "interpolate_curve_rate",
    "prepare_forward_rows",
]
