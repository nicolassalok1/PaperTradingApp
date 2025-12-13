"""
Market data API backed by Stooq (free) with optional Alpaca spot.
Exposes the legacy functions used across the app, including Yahoo option chains.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

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
    return CACHE_CSV_DIR / f"ohlc_{safe}.csv"


def load_or_fetch_closing_history(
    ticker: str, *, period: str = "2y", interval: str = "1d"
) -> Tuple[pd.DataFrame | None, Path | None, bool]:
    tk = (ticker or "").strip().upper()
    if not tk:
        return None, None, False
    path = _cache_path(tk, period, interval)
    # Support legacy filename if present
    legacy_path = CACHE_CSV_DIR / f"closing_{tk}_{period}_{interval}.csv"
    from_cache = False
    for p in (path, legacy_path):
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, parse_dates=["Date"])
            if df is not None and not df.empty:
                return df, p, True
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


def fetch_options_details(
    symbol: str,
    *,
    max_maturity_years: float | None = 2.0,
    max_expiries: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Yahoo option chain download (calls + puts) with shared metadata.
    Returns (calls_df, puts_df, spot, rf, div).
    """
    return fetch_options_details_yahoo(
        symbol, max_maturity_years=max_maturity_years, max_expiries=max_expiries
    )


def _yahoo_options_url(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    return f"https://query2.finance.yahoo.com/v7/finance/options/{sym}"


def _fetch_yahoo_options_json(symbol: str, *, expiry_ts: int | None = None) -> dict:
    """
    Fetch raw JSON payload from Yahoo Finance options endpoint.
    `expiry_ts` is the UNIX timestamp returned by Yahoo in `expirationDates`.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {}

    url = _yahoo_options_url(sym)
    params: dict[str, int] = {}
    if expiry_ts is not None:
        params["date"] = int(expiry_ts)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logging.warning(f"[yahoo-options] fetch failed for {sym} (date={expiry_ts}): {exc}")
        return {}


def _parse_yahoo_option_chain(payload: dict) -> tuple[float | None, list[int], list[dict]]:
    """
    Return (spot, expiry_ts_list, options_blocks) from a Yahoo options payload.
    `options_blocks` is usually a list with one element containing 'calls' and 'puts'.
    """
    if not isinstance(payload, dict):
        return None, [], []

    chain = payload.get("optionChain") or {}
    results = chain.get("result") or []
    if not results or not isinstance(results, list):
        return None, [], []

    root = results[0] if isinstance(results[0], dict) else {}
    expiries = root.get("expirationDates") or []
    expiries = [int(x) for x in expiries if isinstance(x, (int, float, str)) and str(x).isdigit()]

    quote = root.get("quote") or {}
    spot = (
        quote.get("regularMarketPrice")
        or quote.get("regularMarketPreviousClose")
        or quote.get("previousClose")
        or quote.get("postMarketPrice")
        or quote.get("preMarketPrice")
    )
    try:
        spot_val = float(spot) if spot is not None else None
    except Exception:
        spot_val = None

    options_blocks = root.get("options") or []
    if not isinstance(options_blocks, list):
        options_blocks = []
    options_blocks = [b for b in options_blocks if isinstance(b, dict)]
    return spot_val, expiries, options_blocks


def _sample_evenly(items: list[int], max_n: int) -> list[int]:
    if max_n <= 0:
        return []
    if len(items) <= max_n:
        return items
    if max_n == 1:
        return [items[0]]
    n = len(items)
    idxs = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    out: list[int] = []
    seen: set[int] = set()
    for idx in idxs:
        j = int(idx)
        if j in seen:
            continue
        seen.add(j)
        out.append(items[j])
    return out


def fetch_options_details_yahoo(
    symbol: str,
    *,
    max_maturity_years: float | None = 2.0,
    max_expiries: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Download Yahoo option chain across multiple expiries and return:
    (calls_df, puts_df, spot, rf, div).

    Notes:
    - Yahoo provides IV per contract; rates/dividend yield are not reliably exposed here
      so rf/div are returned as 0.0 (pricing uses Yield Curve / UI inputs elsewhere).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame(), pd.DataFrame(), float("nan"), 0.0, 0.0

    today = dt.date.today()
    base_payload = _fetch_yahoo_options_json(sym)
    spot, expiry_ts_list, options_blocks = _parse_yahoo_option_chain(base_payload)

    if spot is None:
        spot = fetch_spot_price(sym)
    try:
        spot_val = float(spot) if spot is not None else float("nan")
    except Exception:
        spot_val = float("nan")

    expiry_candidates: list[tuple[float, int]] = []
    for ts in expiry_ts_list:
        try:
            exp_date = dt.datetime.utcfromtimestamp(int(ts)).date()
            T = (exp_date - today).days / 365.0
            if T <= 0:
                continue
            if max_maturity_years is not None and T > float(max_maturity_years):
                continue
            expiry_candidates.append((float(T), int(ts)))
        except Exception:
            continue

    expiry_candidates.sort(key=lambda x: x[0])
    selected_expiries = [ts for _, ts in expiry_candidates]
    selected_expiries = _sample_evenly(selected_expiries, int(max_expiries))

    def _rows_from_block(block: dict, expiry_ts: int) -> tuple[list[dict], list[dict]]:
        try:
            exp_date = dt.datetime.utcfromtimestamp(int(expiry_ts)).date()
        except Exception:
            return [], []
        T = (exp_date - today).days / 365.0
        if T <= 0:
            return [], []

        def _norm_contract(contract: dict, opt_type: str) -> dict | None:
            if not isinstance(contract, dict):
                return None
            strike = contract.get("strike")
            iv = contract.get("impliedVolatility")
            try:
                strike_f = float(strike)
                iv_f = float(iv)
            except Exception:
                return None
            if not (strike_f > 0.0 and iv_f > 0.0):
                return None
            return {
                "underlying": sym,
                "contractSymbol": contract.get("contractSymbol"),
                "expiry": exp_date,
                "expiry_ts": int(expiry_ts),
                "T": float(T),
                "strike": float(strike_f),
                "iv": float(iv_f),
                "bid": contract.get("bid"),
                "ask": contract.get("ask"),
                "lastPrice": contract.get("lastPrice"),
                "openInterest": contract.get("openInterest"),
                "volume": contract.get("volume"),
                "inTheMoney": contract.get("inTheMoney"),
                "type": opt_type,
                "S0": float(spot_val) if pd.notna(spot_val) else float("nan"),
            }

        calls = block.get("calls") or []
        puts = block.get("puts") or []
        call_rows = [
            r for r in (_norm_contract(c, "call") for c in calls if isinstance(c, dict)) if r
        ]
        put_rows = [
            r for r in (_norm_contract(p, "put") for p in puts if isinstance(p, dict)) if r
        ]
        return call_rows, put_rows

    calls_rows: list[dict] = []
    puts_rows: list[dict] = []

    base_block = options_blocks[0] if options_blocks else None
    base_expiry_ts = None
    try:
        if isinstance(base_block, dict):
            base_expiry_ts = int(base_block.get("expirationDate"))
    except Exception:
        base_expiry_ts = None

    remaining = list(selected_expiries)
    if base_expiry_ts is not None and base_expiry_ts in remaining and isinstance(base_block, dict):
        cr, pr = _rows_from_block(base_block, base_expiry_ts)
        calls_rows.extend(cr)
        puts_rows.extend(pr)
        remaining = [ts for ts in remaining if ts != base_expiry_ts]

    for expiry_ts in remaining:
        payload = _fetch_yahoo_options_json(sym, expiry_ts=expiry_ts)
        _, _, blocks = _parse_yahoo_option_chain(payload)
        if not blocks:
            continue
        block = blocks[0]
        cr, pr = _rows_from_block(block, expiry_ts)
        calls_rows.extend(cr)
        puts_rows.extend(pr)

    calls_df = pd.DataFrame(calls_rows)
    puts_df = pd.DataFrame(puts_rows)

    return calls_df, puts_df, float(spot_val), 0.0, 0.0


__all__ = [
    "make_alpaca_client",
    "fetch_spot_price",
    "fetch_closing_prices",
    "fetch_options_details",
    "fetch_options_details_yahoo",
    "load_or_fetch_closing_history",
    "clear_closing_history_cache",
]
