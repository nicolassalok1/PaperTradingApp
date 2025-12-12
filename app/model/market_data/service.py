from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import requests

from app.utils.paths import CACHE_CSV_DIR
from app.utils.symbol_mapper import map_to_stooq

STOOQ_BASE = "https://stooq.pl/q/d/l/"


def _stooq_params(symbol: str, start: Optional[str], end: Optional[str], freq: str) -> dict:
    params = {"s": map_to_stooq(symbol), "i": freq.lower()}
    if start:
        params["d1"] = start.replace("-", "")
    if end:
        params["d2"] = end.replace("-", "")
    return params


def _download_stooq_csv(params: dict) -> pd.DataFrame:
    resp = requests.get(STOOQ_BASE, params=params, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def fetch_historical_prices(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "D",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical prices from Stooq. Returns columns:
    date, open, high, low, close, volume
    """
    mapped = map_to_stooq(symbol)
    if not mapped:
        return pd.DataFrame()

    cache_key = (
        f"{mapped}_{start or 'start'}_{end or 'end'}_{freq}"
        .replace(" ", "_")
        .replace("/", "-")
    )
    cache_path = CACHE_CSV_DIR / f"stooq_{cache_key}.csv"
    if cache and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            if not cached.empty:
                return cached
        except Exception:
            pass

    params = _stooq_params(symbol, start, end, freq)
    try:
        df = _download_stooq_csv(params)
    except Exception:
        df = pd.DataFrame()

    if cache and df is not None and not df.empty:
        try:
            CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception:
            pass
    return df


def fetch_spot_price(symbol: str):
    """Spot price derived from the latest close via Stooq."""
    df = fetch_historical_prices(symbol, freq="D", cache=True)
    if df is None or df.empty or "close" not in df.columns:
        return None
    return float(df["close"].iloc[-1])


__all__ = ["fetch_historical_prices", "fetch_spot_price"]
