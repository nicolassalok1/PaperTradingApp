"""
Minimal market data API.
- Spots: Alpaca -> fallback Yahoo chart JSON (no yfinance)
- OHLC: Alpaca -> fallback Yahoo chart JSON -> cached under cache/ohlc_*.csv
- Options: CBOE calls + puts merged metadata
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

from app.utils.paths import CACHE_CSV_DIR

try:  # optional dependency
    from alpaca_trade_api import REST as AlpacaREST
    from alpaca_trade_api.rest import TimeFrame
except Exception:  # pragma: no cover - optional dependency
    AlpacaREST = None  # type: ignore
    TimeFrame = None  # type: ignore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _http_get_json(
    url: str, *, params: dict | None = None, timeout: int = 12, retries: int = 3
) -> dict:
    """GET JSON with headers, timeout, and simple retries."""
    last_exc = None
    for k in range(retries):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": _UA, "Accept": "application/json,text/plain,*/*"},
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.4 * (k + 1))
    raise last_exc  # type: ignore[misc]


def _yahoo_chart(sym: str, *, range_: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Direct Yahoo Finance chart endpoint (no yfinance dependency)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    data = _http_get_json(url, params={"range": range_, "interval": interval})
    res = (data or {}).get("chart", {}).get("result") or []
    if not res:
        return pd.DataFrame()
    r0 = res[0]
    ts = r0.get("timestamp") or []
    q = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
    closes = q.get("close") or []
    if not ts or not closes:
        return pd.DataFrame()
    idx = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True).dt.tz_convert(None)
    df = pd.DataFrame({"Close": pd.Series(closes, dtype="float64").values}, index=idx)
    df = df.dropna()
    return df


def _load_env_fallback() -> None:
    """Tiny .env loader so Alpaca keys are available without dotenv."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _alpaca_credentials() -> Tuple[str | None, str | None, str]:
    _load_env_fallback()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    return key, secret, base


def _period_to_days(period: str) -> int:
    """Approximate period -> day count."""
    if not period:
        return 365
    p = period.strip().lower()
    try:
        if p.endswith("y"):
            return int(float(p[:-1]) * 365)
        if p.endswith("mo"):
            return int(float(p[:-2]) * 30)
        if p.endswith("w"):
            return int(float(p[:-1]) * 7)
        if p.endswith("d"):
            return int(float(p[:-1]))
        return int(float(p))
    except Exception:
        return 365


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _cache_path(sym: str, period: str, interval: str) -> Path:
    safe = f"{sym}_{period}_{interval}".replace("/", "-").replace(" ", "_")
    return CACHE_CSV_DIR / f"ohlc_{safe}.csv"


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Date + standard OHLC columns exist."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Normalize column names
    rename_map = {
        "timestamp": "Date",
        "time": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df.columns = [rename_map.get(str(c).lower(), c) for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    cols = ["Date"] + [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.reset_index(drop=True)


def _fetch_ohlc_alpaca(sym: str, period: str, interval: str) -> pd.DataFrame:
    """Daily bars from Alpaca if credentials + SDK available."""
    if AlpacaREST is None or TimeFrame is None:
        return pd.DataFrame()
    key, secret, base = _alpaca_credentials()
    if not key or not secret:
        return pd.DataFrame()
    tf = TimeFrame.Day if interval.lower() == "1d" else TimeFrame.Hour
    start = dt.datetime.utcnow() - dt.timedelta(days=_period_to_days(period))
    try:
        client = AlpacaREST(key, secret, base, api_version="v2")
        bars = client.get_bars(sym, tf, start=start.isoformat())
        df = getattr(bars, "df", None)
        if df is None:
            df = pd.DataFrame(bars)
        return _normalize_ohlc(df)
    except Exception as exc:
        logging.warning(f"[OHLC] Alpaca failed for {sym}: {exc}")
        return pd.DataFrame()


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def make_alpaca_client():
    """Instantiate an Alpaca REST client if credentials are configured."""
    if AlpacaREST is None:
        return None
    key, secret, base = _alpaca_credentials()
    if not key or not secret:
        return None
    try:
        return AlpacaREST(key, secret, base, api_version="v2")
    except Exception as exc:
        logging.warning(f"[alpaca] client init failed: {exc}")
        return None


def fetch_spot_price(symbol: str):
    """Spot price: Alpaca latest trade -> Yahoo chart fallback (no yfinance)."""
    sym = _normalize_symbol(symbol)
    if not sym:
        return None

    client = make_alpaca_client()
    if client is not None:
        try:
            trade = client.get_latest_trade(sym)
            px = getattr(trade, "price", None)
            if px is None and hasattr(client, "get_latest_quote"):
                try:
                    quote = client.get_latest_quote(sym)
                    px = getattr(quote, "bidprice", None) or getattr(quote, "askprice", None)
                except Exception:
                    px = None
            if px is not None:
                return float(px)
        except Exception as exc:
            logging.warning(f"[spot] Alpaca failed for {sym}: {exc}")

    try:
        df = _yahoo_chart(sym, range_="5d", interval="1d")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as exc:
        logging.warning(f"[spot] yahoo chart failed for {sym}: {exc}")

    return float("nan")


def fetch_closing_prices(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Closing prices (single symbol).
    Tries Alpaca then Yahoo chart API (no yfinance), caches to cache/ohlc_<symbol>_<period>_<interval>.csv.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return pd.DataFrame()

    cache_path = _cache_path(sym, period, interval)
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            if cached is not None and not cached.empty:
                return cached
        except Exception:
            pass

    df = _fetch_ohlc_alpaca(sym, period, interval)
    if df is None or df.empty:
        try:
            df_chart = _yahoo_chart(sym, range_=period, interval=interval)
            if df_chart is not None and not df_chart.empty:
                df_chart = df_chart.reset_index().rename(columns={"index": "Date"})
                df_chart["Date"] = pd.to_datetime(df_chart["Date"])
                df = _normalize_ohlc(df_chart)
                if "Close" in df.columns and "close" not in df.columns:
                    df["close"] = df["Close"]
        except Exception as exc:
            logging.warning(f"[OHLC] yahoo chart failed for {sym}: {exc}")
            df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    _save_cache(df, cache_path)
    return df


def load_or_fetch_closing_history(
    ticker: str, *, period: str = "2y", interval: str = "1d"
) -> Tuple[pd.DataFrame | None, Path | None, bool]:
    """
    Backward-compatible wrapper returning (df, cache_path, from_cache).
    """
    sym = _normalize_symbol(ticker)
    if not sym:
        return None, None, False
    cache_path = _cache_path(sym, period, interval)
    from_cache = cache_path.exists()
    df = fetch_closing_prices(sym, period=period, interval=interval)
    if df is None or df.empty:
        return None, cache_path, from_cache
    return df, cache_path, from_cache


def clear_closing_history_cache(ticker: str, *, period: str = "2y", interval: str = "1d") -> None:
    """Remove the cached OHLC CSV for a ticker/period/interval."""
    sym = _normalize_symbol(ticker)
    if not sym:
        return
    path = _cache_path(sym, period, interval)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def fetch_options_details(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    CBOE options download (calls + puts) with shared metadata.
    Returns (calls_df, puts_df, spot, rf, div).
    """
    try:
        from app.model.options.logic import download_options_cboe
    except Exception as exc:  # pragma: no cover - soft dependency
        logging.warning(f"[options] download unavailable: {exc}")
        return pd.DataFrame(), pd.DataFrame(), float("nan"), float("nan"), float("nan")

    sym = _normalize_symbol(symbol)
    if not sym:
        return pd.DataFrame(), pd.DataFrame(), float("nan"), float("nan"), float("nan")

    calls_df, spot_c, rf_c, div_c = download_options_cboe(sym, "call")
    puts_df, spot_p, rf_p, div_p = download_options_cboe(sym, "put")

    def _pick(v_primary, v_fallback):
        try:
            return v_primary if pd.notna(v_primary) else v_fallback
        except Exception:
            return v_primary or v_fallback

    spot = _pick(spot_c, spot_p)
    rf = _pick(rf_c, rf_p)
    div = _pick(div_c, div_p)
    if (spot is None or (isinstance(spot, float) and pd.isna(spot))):
        spot = fetch_spot_price(sym)

    return (
        calls_df,
        puts_df,
        float(spot) if spot is not None else float("nan"),
        float(rf) if rf is not None else float("nan"),
        float(div) if div is not None else float("nan"),
    )


def fetch_options_chain_yahoo(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Download Yahoo options chain for all available expirations and return:
    (calls_df, puts_df, spot)
    Columns expected downstream: strike, iv, expiry (date), type ('call'/'put')
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return pd.DataFrame(), pd.DataFrame(), float("nan")

    base = f"https://query2.finance.yahoo.com/v7/finance/options/{sym}"

    root = _http_get_json(base)
    chain = ((root.get("optionChain") or {}).get("result") or [])
    if not chain:
        return pd.DataFrame(), pd.DataFrame(), float("nan")
    c0 = chain[0]
    quote = c0.get("quote") or {}
    spot = quote.get("regularMarketPrice")
    expirations = c0.get("expirationDates") or []
    if not expirations:
        return pd.DataFrame(), pd.DataFrame(), float(spot) if spot is not None else float("nan")

    all_calls: list[pd.DataFrame] = []
    all_puts: list[pd.DataFrame] = []

    for exp in expirations:
        try:
            data = _http_get_json(base, params={"date": int(exp)})
            r = ((data.get("optionChain") or {}).get("result") or [])
            if not r:
                continue
            opt = (((r[0].get("options") or [])[:1]) or [{}])[0]
            calls = opt.get("calls") or []
            puts = opt.get("puts") or []

            if calls:
                dfc = pd.DataFrame(calls)
                all_calls.append(dfc)
            if puts:
                dfp = pd.DataFrame(puts)
                all_puts.append(dfp)
        except Exception:
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    def _normalize(df: pd.DataFrame, typ: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "impliedVolatility" in out.columns and "iv" not in out.columns:
            out["iv"] = out["impliedVolatility"].astype(float)
        if "strike" in out.columns:
            out["strike"] = out["strike"].astype(float)
        if "expiration" in out.columns:
            out["expiry"] = (
                pd.to_datetime(out["expiration"], unit="s", utc=True).dt.tz_convert(None).dt.date
            )
        elif "expiry" in out.columns:
            out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
        out["type"] = typ
        keep = ["contractSymbol", "strike", "iv", "expiry", "type", "bid", "ask", "lastPrice"]
        cols = [c for c in keep if c in out.columns]
        return out[cols].dropna(subset=["strike", "iv", "expiry"])

    calls_df = _normalize(calls_df, "call")
    puts_df = _normalize(puts_df, "put")

    if spot is None or (isinstance(spot, float) and pd.isna(spot)):
        spot = fetch_spot_price(sym)

    return calls_df, puts_df, float(spot) if spot is not None else float("nan")


__all__ = [
    "make_alpaca_client",
    "fetch_spot_price",
    "fetch_closing_prices",
    "fetch_options_details",
    "fetch_options_chain_yahoo",
    "load_or_fetch_closing_history",
    "clear_closing_history_cache",
]
