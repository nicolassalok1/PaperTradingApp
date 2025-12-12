"""
Market data API backed by Stooq (free) with optional Alpaca spot.
Exposes the legacy functions used across the app without Yahoo dependencies.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

from app.model.market_data.service import fetch_historical_prices as _fetch_stooq_history
from app.model.market_data.service import fetch_spot_price as _fetch_stooq_spot
from app.utils.paths import CACHE_CSV_DIR
from app.utils.symbol_mapper import map_to_stooq

try:  # optional dependency
    from alpaca_trade_api import REST as AlpacaREST
    from alpaca_trade_api.rest import TimeFrame
except Exception:  # pragma: no cover
    AlpacaREST = None  # type: ignore
    TimeFrame = None  # type: ignore


def _load_env_fallback() -> None:
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


def make_alpaca_client():
    """Instantiate an Alpaca REST client if credentials are configured."""
    if AlpacaREST is None:
        return None
    key, secret, base = _alpaca_credentials()
    if not key or not secret:
        return None
    try:
        return AlpacaREST(key, secret, base, api_version="v2")
    except Exception:
        return None


def fetch_spot_price(symbol: str):
    """Spot price: Alpaca latest trade -> Stooq last close."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    client = make_alpaca_client()
    if client is not None:
        try:
            trade = client.get_latest_trade(sym)
            px = getattr(trade, "price", None)
            if px is not None:
                return float(px)
        except Exception:
            pass

    return _fetch_stooq_spot(sym)


def fetch_closing_prices(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Historical prices via Stooq. period/interval kept for compatibility.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()
    df = _fetch_stooq_history(sym, freq="d")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"date": "Date", "close": sym})
    return df[["Date", sym]]


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe = f"{ticker}_{period}_{interval}".replace("/", "_").replace(" ", "_")
    return CACHE_CSV_DIR / f"closing_{safe}.csv"


def load_or_fetch_closing_history(
    ticker: str, *, period: str = "2y", interval: str = "1d"
) -> Tuple[pd.DataFrame | None, Path | None, bool]:
    tk = (ticker or "").strip().upper()
    if not tk:
        return None, None, False
    path = _cache_path(tk, period, interval)
    from_cache = False
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if df is not None and not df.empty:
                return df, path, True
        except Exception:
            pass
    df = fetch_closing_prices(tk, period=period, interval=interval)
    if df is not None and not df.empty:
        try:
            CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        except Exception:
            pass
        return df, path, from_cache
    return None, path, from_cache


def clear_closing_history_cache(ticker: str, *, period: str = "2y", interval: str = "1d") -> None:
    tk = (ticker or "").strip().upper()
    path = _cache_path(tk, period, interval)
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

    sym = (symbol or "").strip().upper()
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
    if spot is None or (isinstance(spot, float) and pd.isna(spot)):
        spot = fetch_spot_price(sym)

    return (
        calls_df,
        puts_df,
        float(spot) if spot is not None else float("nan"),
        float(rf) if rf is not None else float("nan"),
        float(div) if div is not None else float("nan"),
    )


__all__ = [
    "make_alpaca_client",
    "fetch_spot_price",
    "fetch_closing_prices",
    "fetch_options_details",
    "load_or_fetch_closing_history",
    "clear_closing_history_cache",
]
