"""
Yield curve services (no UI).
Handles curve loading, caching, and forward rate computations.
"""

from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests
import numpy as np
import pandas as pd

from app.model.market_data.realtime import get_data
from app.model.yieldcurve.base import YieldCurve
from app.model.yieldcurve.curves import FlatYieldCurve, NodeYieldCurve
from app.model.yieldcurve.engine import (
    fit_nelson_siegel_grid_search,
    nelson_siegel_yield,
)
from app.model.yieldcurve.loader import (
    YIELD_CURVE_CACHE_FILE,
    download_yield_curve_to_cache,
    load_yield_curve_csv,
)
from app.model.yieldcurve.parsing import load_nodes_from_file
from app.utils.io import load_json_file, save_json_file
from app.utils.math_utils import floor_4
from app.utils.paths import JSON_DIR

FORWARDS_FILE = JSON_DIR / "forwards_portfolio.json"
YIELD_CURVE_DATA_DIR = JSON_DIR / "yield_curves"
YIELD_CURVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))
DEFAULT_CURVE_CCY = os.getenv("DEFAULT_CURVE_CCY", "USD").upper()
MAX_CACHE_AGE_HOURS = float(os.getenv("YIELD_CURVE_CACHE_MAX_AGE_HOURS", "24"))
ENABLE_YC_API = os.getenv("YIELD_CURVE_ENABLE_API", "0").lower() in {"1", "true", "yes"}


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


def _cache_path_for_currency(currency: str) -> Path:
    return YIELD_CURVE_DATA_DIR / f"{currency.upper()}_nodes.csv"


def _file_mtime(path: Path | None) -> datetime.datetime | None:
    if path and path.exists():
        try:
            return datetime.datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            return None
    return None


def _is_cache_stale(path: Path | None) -> bool:
    if not path or not path.exists():
        return True
    if MAX_CACHE_AGE_HOURS <= 0:
        return False
    mtime = _file_mtime(path)
    if not mtime:
        return True
    age = datetime.datetime.now() - mtime
    return age.total_seconds() > MAX_CACHE_AGE_HOURS * 3600


def get_spot(sym: str):
    return get_data(sym)


def get_rate(currency: str | None = None):
    return lambda T: get_risk_free_rate(T, currency=currency)


def load_forwards() -> dict:
    return load_json_file(FORWARDS_FILE, {})


def save_forwards(data: dict) -> None:
    save_json_file(FORWARDS_FILE, data)


def _save_nodes_to_cache(currency: str, nodes: list[dict]) -> Path | None:
    if not nodes:
        return None
    path = _cache_path_for_currency(currency)
    try:
        df = pd.DataFrame(nodes)
        df.to_csv(path, index=False)
        return path
    except Exception:
        return None


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


def _find_nodes_file(currency: str) -> Path | None:
    if not currency:
        return None
    code = currency.upper()
    for ext in ("csv", "json"):
        candidate = YIELD_CURVE_DATA_DIR / f"{code}_nodes.{ext}"
        if candidate.exists():
            return candidate
    for path in YIELD_CURVE_DATA_DIR.glob("*_nodes.*"):
        stem = path.stem.split("_")[0].upper()
        if stem == code:
            return path
    return None


def available_currencies() -> List[str]:
    codes = {path.stem.split("_")[0].upper() for path in YIELD_CURVE_DATA_DIR.glob("*_nodes.*")}
    return sorted(codes) if codes else [DEFAULT_CURVE_CCY]


def _nodes_from_cache_csv(ensure_cache: bool) -> tuple[list[dict], Path | None]:
    df_curve, source_path = load_yield_curve_csv(ensure_cache=ensure_cache)
    if df_curve is None or df_curve.empty:
        return [], source_path
    tenors, rates = _latest_curve_points(df_curve)
    nodes: list[dict] = []
    for t, r in zip(tenors, rates):
        if not math.isfinite(t) or not math.isfinite(r):
            continue
        tenor_label = f"{int(round(t * 12))}M" if t < 1 else f"{int(round(t))}Y"
        nodes.append(
            {
                "tenor": tenor_label,
                "t_years": float(t),
                "zero_rate": float(r),
                "discount_factor": math.exp(-float(r) * float(t)),
            }
        )
    return nodes, source_path


def _load_nodes_for_currency(currency: str, ensure_cache: bool = True) -> tuple[list[dict], Path | None]:
    path = _find_nodes_file(currency)
    if path:
        nodes = load_nodes_from_file(path)
        return nodes, path
    nodes, source = _nodes_from_cache_csv(ensure_cache)
    return nodes, source


def _safe_get_json(url: str, params: Dict[str, Any] | None = None, timeout: float = 5.0) -> Dict[str, Any] | None:
    """
    Local, minimal clone of a safe HTTP GET returning JSON or None.
    Never laisse remonter les exceptions jusqu'à la Vue.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _fetch_usd_nodes_from_fred() -> list[dict]:
    """
    Fetch USD curve nodes from FRED (optional). Returns list of node dicts.
    Kept strictement côté modèle pour respecter le MVC.
    """
    fred_series: Dict[str, str] = {
        "1M": "DTB1M",
        "3M": "DTB3",
        "6M": "DTB6",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "5Y": "DGS5",
        "10Y": "DGS10",
    }
    tenor_years: Dict[str, float] = {
        "1M": 1.0 / 12.0,
        "3M": 0.25,
        "6M": 0.5,
        "1Y": 1.0,
        "2Y": 2.0,
        "5Y": 5.0,
        "10Y": 10.0,
    }

    api_key = os.getenv("FRED_API_KEY")

    def _fetch_series(series_id: str) -> float | None:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key or "",
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        data = _safe_get_json(url, params=params, timeout=5)
        if data is None:
            return None
        try:
            obs = data.get("observations", [])
            if not obs:
                return None
            val_str = obs[0].get("value")
            val = float(val_str)
            return val / 100.0
        except Exception:
            return None

    nodes: list[dict] = []
    for tenor, series in fred_series.items():
        t_years = tenor_years.get(tenor)
        if t_years is None:
            continue
        val = _fetch_series(series)
        if val is None:
            continue
        nodes.append(
            {
                "tenor": tenor,
                "t_years": float(t_years),
                "zero_rate": float(val),
                "discount_factor": None,
            }
        )
    return nodes


def _fetch_eur_nodes_from_ecb() -> list[dict]:
    """
    Fetch EUR curve nodes from a lightweight ECB SDW endpoint (optional).
    """
    ecb_base = "https://data-api.ecb.europa.eu/service/data"
    series_map: Dict[str, tuple[str, float]] = {
        "0.5Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_6M", 0.5),
        "1Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_1Y", 1.0),
        "2Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_2Y", 2.0),
        "5Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_5Y", 5.0),
        "10Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_10Y", 10.0),
    }

    def _fetch_single_series(series_code: str) -> float | None:
        url = f"{ecb_base}/{series_code}"
        params = {"lastNObservations": 1, "format": "jsondata"}
        data = _safe_get_json(url, params=params, timeout=5)
        if data is None:
            return None
        try:
            obs = data.get("data", {}).get("dataSets", [{}])[0].get("series", {})
            first_series = next(iter(obs.values()))
            values = first_series.get("observations", {})
            first_obs = next(iter(values.values()))
            val = float(first_obs[0])
            return val / 100.0
        except Exception:
            return None

    nodes: list[dict] = []
    for tenor, (series_code, t_years) in series_map.items():
        val = _fetch_single_series(series_code)
        if val is None:
            continue
        nodes.append(
            {
                "tenor": tenor,
                "t_years": float(t_years),
                "zero_rate": float(val),
                "discount_factor": None,
            }
        )
    return nodes


def _fetch_nodes_from_provider(currency: str) -> list[dict]:
    ccy = currency.upper()
    try:
        if ccy == "USD":
            return _fetch_usd_nodes_from_fred()
        if ccy == "EUR":
            return _fetch_eur_nodes_from_ecb()
    except Exception:
        return []
    return []


def _build_curve_from_nodes(nodes: list[dict], fallback_rate: float = DEFAULT_RF) -> NodeYieldCurve | FlatYieldCurve:
    if nodes:
        return NodeYieldCurve(nodes, default_rate=fallback_rate)
    return FlatYieldCurve(rate=fallback_rate)


def _pick_currency(currency: str | None = None) -> str:
    if currency:
        return currency.upper()
    currencies = available_currencies()
    if DEFAULT_CURVE_CCY in currencies:
        return DEFAULT_CURVE_CCY
    if currencies:
        return currencies[0]
    return DEFAULT_CURVE_CCY


_CURVE_CACHE: Dict[str, tuple[YieldCurve, Path | None, list[dict], str, datetime.datetime | None]] = {}


def _compute_nelson_siegel_from_curve(
    curve: YieldCurve, maturities_grid: List[float]
) -> tuple[Dict[str, float] | None, List[dict]]:
    """
    Calibrate a Nelson–Siegel curve from the underlying nodes (if any).
    Returns (params, curve_points) where params is a dict with beta0/beta1/beta2/tau,
    and curve_points is a list of dict(t_years, zero_rate) evaluated on maturities_grid.
    Fails silently and returns (None, []) on any error.
    """
    raw_nodes = getattr(curve, "nodes", None)
    if not raw_nodes:
        return None, []

    ts: List[float] = []
    rs: List[float] = []
    for n in raw_nodes:
        try:
            t = float(getattr(n, "t_years", 0.0))
            r = float(getattr(n, "zero_rate", DEFAULT_RF))
        except Exception:
            continue
        if not math.isfinite(t) or not math.isfinite(r) or t <= 0:
            continue
        ts.append(t)
        rs.append(r)

    if len(ts) < 3:
        # Too few points to robustly calibrate NS; skip silently.
        return None, []

    try:
        beta0, beta1, beta2, tau = fit_nelson_siegel_grid_search(ts, rs)
        if not all(math.isfinite(x) for x in (beta0, beta1, beta2, tau)):
            return None, []
        params = {
            "beta0": float(beta0),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "tau": float(tau),
        }

        # Evaluate NS curve on provided grid (or on a fallback dense grid).
        if maturities_grid:
            grid = sorted({float(t) for t in maturities_grid if t > 0})
        else:
            t_min = min(ts)
            t_max = max(ts)
            if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= 0:
                return params, []
            grid = np.linspace(t_min, t_max, 32).tolist()

        t_arr = np.asarray(grid, dtype=float)
        y_arr = nelson_siegel_yield(t_arr, beta0, beta1, beta2, tau)

        ns_curve: List[dict] = []
        for t, r in zip(t_arr, y_arr):
            try:
                t_f = float(t)
                r_f = float(r)
            except Exception:
                continue
            if not math.isfinite(t_f) or not math.isfinite(r_f) or t_f <= 0:
                continue
            ns_curve.append({"t_years": t_f, "zero_rate": r_f})
        return params, ns_curve
    except Exception:
        return None, []


def get_active_curve(
    currency: str | None = None,
    ensure_cache: bool = True,
    allow_api: bool = False,
) -> tuple[YieldCurve, Path | None, list[dict], str, datetime.datetime | None]:
    ccy = _pick_currency(currency)
    if ccy in _CURVE_CACHE:
        return _CURVE_CACHE[ccy]

    nodes, source_path = _load_nodes_for_currency(ccy, ensure_cache=ensure_cache)
    source_kind = "cache" if nodes else "flat_fallback"
    last_updated = _file_mtime(source_path)

    if allow_api and ENABLE_YC_API and (_is_cache_stale(source_path) or not nodes):
        fetched = _fetch_nodes_from_provider(ccy)
        if fetched:
            path = _save_nodes_to_cache(ccy, fetched)
            if path:
                source_path = path
                nodes = fetched
                source_kind = "api_refresh"
                last_updated = _file_mtime(path)

    curve = _build_curve_from_nodes(nodes, fallback_rate=DEFAULT_RF)
    if not nodes:
        source_kind = "flat_fallback"
    _CURVE_CACHE[ccy] = (curve, source_path, nodes, source_kind, last_updated)
    return curve, source_path, nodes, source_kind, last_updated


def refresh_curve_cache_from_api(currency: str | None = None) -> bool:
    """
    Optional manual refresh using API provider; never called during pricing/UI render.
    """
    if not ENABLE_YC_API:
        return False
    ccy = _pick_currency(currency)
    fetched = _fetch_nodes_from_provider(ccy)
    if not fetched:
        return False
    path = _save_nodes_to_cache(ccy, fetched)
    if not path:
        return False
    # invalidate cache entry
    _CURVE_CACHE.pop(ccy, None)
    return True


def get_curve_snapshot(
    currency: str | None = None,
    risk_free_maturity: float = 1.0,
    ensure_cache: bool = True,
    grid_size: int = 16,
) -> dict[str, Any]:
    ccy = _pick_currency(currency)
    curve, source_path, nodes, source_kind, last_updated = get_active_curve(
        currency=currency, ensure_cache=ensure_cache, allow_api=False
    )
    maturities_nodes = [n.get("t_years") for n in nodes if n.get("t_years") is not None]
    maturities_nodes = [float(t) for t in maturities_nodes if math.isfinite(float(t))]
    base_grid = sorted(set(maturities_nodes))
    max_mat = max(base_grid) if base_grid else max(float(risk_free_maturity), 5.0)
    dense_grid = [round(max_mat * i / max(grid_size, 2), 6) for i in range(1, grid_size + 1)]
    maturities_grid = sorted({t for t in base_grid + dense_grid if t > 0})
    grid = [
        {
            "t_years": t,
            "zero_rate": curve.zero_rate(t),
            "discount_factor": curve.discount_factor(t),
        }
        for t in maturities_grid
    ]
    risk_free_rate = curve.zero_rate(risk_free_maturity)

    # Nelson–Siegel smoothing (optional, fully contained in the model layer).
    ns_params, ns_curve = _compute_nelson_siegel_from_curve(curve, maturities_grid)
    nodes_table = []
    for n in getattr(curve, "nodes", []):
        nodes_table.append(
            {
                "tenor": getattr(n, "tenor", None) or f"{getattr(n, 't_years', 0.0)}y",
                "t_years": float(getattr(n, "t_years", 0.0)),
                "zero_rate": float(getattr(n, "zero_rate", 0.0)),
                "discount_factor": float(getattr(n, "discount_factor", 0.0)),
            }
        )
    return {
        "currency": ccy,
        "nodes": nodes_table,
        "grid": grid,
        "risk_free_rate": risk_free_rate,
        "risk_free_maturity": float(risk_free_maturity),
        "source_path": source_path,
        "source_kind": source_kind,
        "last_updated": last_updated.isoformat() if last_updated else None,
        "ns_params": ns_params,
        "ns_curve": ns_curve,
    }


def get_risk_free_rate(
    T_ref: float = 1.0, currency: str | None = None, ensure_cache: bool = True
) -> float:
    curve, _, _, _, _ = get_active_curve(currency=currency, ensure_cache=ensure_cache, allow_api=False)
    return float(curve.zero_rate(T_ref))


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
        return float(get_risk_free_rate(T_years))
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
    "available_currencies",
    "yield_curve_cache_file",
    "get_spot",
    "get_rate",
    "get_active_curve",
    "refresh_curve_cache_from_api",
    "get_curve_snapshot",
    "get_risk_free_rate",
    "load_forwards",
    "save_forwards",
    "compute_forward_price",
    "compute_forward_rate",
    "compute_forward_rates",
    "compute_instantaneous_rate",
    "compute_instantaneous_rates",
    "interpolate_curve_rate",
    "prepare_forward_rows",
    "YieldCurve",
]
