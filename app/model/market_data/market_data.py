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
from urllib.parse import quote

import pandas as pd
import requests

from app.model.market_data.service import fetch_historical_prices as _fetch_stooq_history
from app.model.market_data.service import fetch_spot_price as _fetch_stooq_spot
from app.utils.paths import CACHE_CSV_DIR
from app.utils.secrets import get_secret
from app.utils.symbol_mapper import map_to_stooq

try:  # optional dependency
    from alpaca_trade_api import REST as AlpacaREST
    from alpaca_trade_api.rest import TimeFrame
except Exception:  # pragma: no cover
    AlpacaREST = None  # type: ignore
    TimeFrame = None  # type: ignore


ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"

_YAHOO_HEADERS_BASE: dict[str, str] = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}
_YAHOO_OPTIONS_SESSION: requests.Session | None = None
_YAHOO_OPTIONS_CRUMB: str | None = None


def _alpaca_credentials() -> Tuple[str | None, str | None, str]:
    key = get_secret("APCA_API_KEY_ID")
    secret = get_secret("APCA_API_SECRET_KEY")
    base = get_secret("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
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


def _alpaca_data_headers() -> dict[str, str] | None:
    key, secret, _ = _alpaca_credentials()
    if not key or not secret:
        return None
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _alpaca_latest_trade_price(symbol: str, *, feed: str | None = "iex") -> float | None:
    """
    Latest trade via Alpaca Stock Market Data API.

    Uses IEX by default (works on free plans); can be overridden via feed.
    """
    headers = _alpaca_data_headers()
    if not headers:
        return None

    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
    params: dict[str, str] = {}
    if feed:
        params["feed"] = str(feed)

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        trade = payload.get("trade") if isinstance(payload, dict) else None
        if not isinstance(trade, dict):
            trade = {}
        px = trade.get("p") or trade.get("price") or payload.get("price")
        return float(px) if px is not None else None
    except Exception:
        return None


def _alpaca_latest_quote_mid(symbol: str, *, feed: str | None = "iex") -> float | None:
    """Fallback mid price from the latest quote via Alpaca Stock Market Data API."""
    headers = _alpaca_data_headers()
    if not headers:
        return None

    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
    params: dict[str, str] = {}
    if feed:
        params["feed"] = str(feed)

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        quote = payload.get("quote") if isinstance(payload, dict) else None
        if not isinstance(quote, dict):
            quote = {}
        bid = quote.get("bp") or quote.get("bid_price") or quote.get("bidPrice")
        ask = quote.get("ap") or quote.get("ask_price") or quote.get("askPrice")

        bid_f = float(bid) if bid is not None else None
        ask_f = float(ask) if ask is not None else None
        if bid_f is not None and ask_f is not None:
            return 0.5 * (bid_f + ask_f)
        return ask_f if ask_f is not None else bid_f
    except Exception:
        return None


def fetch_spot_price(symbol: str):
    """
    Spot price: Alpaca latest trade (data API) -> quote mid -> legacy SDK -> Stooq last close.

    Notes:
    - We prefer the data API because `alpaca_trade_api` is an optional dependency.
    - Stooq can be rate-limited; we treat it as a best-effort fallback.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    feed = os.getenv("ALPACA_STOCK_DATA_FEED") or "iex"
    px = _alpaca_latest_trade_price(sym, feed=feed)
    if px is None and feed:
        px = _alpaca_latest_trade_price(sym, feed=None)
    if px is None:
        px = _alpaca_latest_quote_mid(sym, feed=feed)
        if px is None and feed:
            px = _alpaca_latest_quote_mid(sym, feed=None)
    if px is not None:
        return float(px)

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
    Historical closing prices.
    Primary source: Stooq (free).
    Fallback: Yahoo chart endpoint (when Stooq is rate-limited/unavailable).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()
    df = _fetch_stooq_history(sym, freq="d")
    if df is not None and not df.empty:
        df = df.rename(columns={"date": "Date", "close": sym})
        return df[["Date", sym]]

    df_y = _fetch_yahoo_ohlc(sym, period=period, interval=interval)
    if df_y is None or df_y.empty:
        return pd.DataFrame()
    df_y = df_y.rename(columns={"Close": sym})
    return df_y[["Date", sym]]


def _period_to_days(period: str) -> int | None:
    p = (period or "").strip().lower()
    if not p:
        return None
    try:
        if p.endswith("y"):
            return int(float(p[:-1]) * 365)
        if p.endswith("mo"):
            return int(float(p[:-2]) * 30)
        if p.endswith("w"):
            return int(float(p[:-1]) * 7)
        if p.endswith("d"):
            return int(float(p[:-1]))
    except Exception:
        return None
    return None


def _filter_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    days = _period_to_days(period)
    if not days or days <= 0:
        return df
    try:
        cutoff = pd.Timestamp.now(tz=None).normalize() - pd.Timedelta(days=int(days))
        return df[df["Date"] >= cutoff]
    except Exception:
        return df


def _standardize_ohlc(df: pd.DataFrame, *, ticker: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cols = {str(c).strip().lower(): c for c in df.columns}

    date_col = (
        cols.get("date")
        or cols.get("datetime")
        or cols.get("timestamp")
        or cols.get("time")
        or next(iter(df.columns), None)
    )
    if date_col is None:
        return pd.DataFrame()

    def _pick(*names: str) -> str | None:
        for n in names:
            c = cols.get(n)
            if c is not None:
                return c
        return None

    open_col = _pick("open")
    high_col = _pick("high")
    low_col = _pick("low")
    close_col = _pick("close")
    vol_col = _pick("volume", "vol")

    if close_col is None and ticker:
        tkr = (ticker or "").strip().upper()
        close_col = next(
            (c for c in df.columns if str(c).strip().upper() == tkr),
            None,
        )

    if close_col is None and len(df.columns) >= 2:
        close_col = df.columns[1]

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    if open_col is not None:
        out["Open"] = pd.to_numeric(df[open_col], errors="coerce")
    if high_col is not None:
        out["High"] = pd.to_numeric(df[high_col], errors="coerce")
    if low_col is not None:
        out["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    if vol_col is not None:
        out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce")

    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return out


def _find_stooq_cache_path(symbol: str) -> Path | None:
    mapped = map_to_stooq(symbol)
    if not mapped:
        return None
    candidates = [
        CACHE_CSV_DIR / f"stooq_{mapped}_start_end_D.csv",
        CACHE_CSV_DIR / f"stooq_{mapped}_start_end_d.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def _fetch_yahoo_ohlc(symbol: str, *, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    rng = (period or "2y").strip().lower()
    itv = (interval or "1d").strip().lower()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        resp = requests.get(url, params={"range": rng, "interval": itv}, headers=headers, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logging.info(f"[yahoo-chart] fetch failed for {sym} ({rng}/{itv}): {exc}")
        return pd.DataFrame()

    try:
        result = (payload.get("chart") or {}).get("result") or []
        root = result[0] if result and isinstance(result[0], dict) else {}
        timestamps = root.get("timestamp") or []
        indicators = (root.get("indicators") or {}).get("quote") or []
        quote0 = indicators[0] if indicators and isinstance(indicators[0], dict) else {}
        if not timestamps or not quote0:
            return pd.DataFrame()

        dt_index = pd.to_datetime(pd.Series(timestamps, dtype="int64"), unit="s", utc=True)
        dates = dt_index.dt.tz_convert(None)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": quote0.get("open", []),
                "High": quote0.get("high", []),
                "Low": quote0.get("low", []),
                "Close": quote0.get("close", []),
                "Volume": quote0.get("volume", []),
            }
        )
        return _standardize_ohlc(df, ticker=sym)
    except Exception:
        return pd.DataFrame()


def fetch_ohlc_history(symbol: str, *, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLC history for a symbol.
    Primary source: Stooq (cached when available).
    Fallback: Yahoo chart endpoint.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    try:
        df_stooq = _fetch_stooq_history(sym, freq="d")
        df_std = _standardize_ohlc(df_stooq, ticker=sym)
        df_std = _filter_to_period(df_std, period)
        if df_std is not None and not df_std.empty:
            return df_std
    except Exception:
        pass

    df_y = _fetch_yahoo_ohlc(sym, period=period, interval=interval)
    return _filter_to_period(df_y, period) if df_y is not None else pd.DataFrame()


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
            df_raw = pd.read_csv(p)
            df = _standardize_ohlc(df_raw, ticker=tk)
            df = _filter_to_period(df, period)
            if df is not None and not df.empty:
                # If we loaded the legacy close-only cache, materialize the new OHLC filename.
                if p == legacy_path:
                    try:
                        CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
                        df.to_csv(path, index=False)
                    except Exception:
                        pass
                    return df, path, True

                # If we have a close-only cache, attempt an in-place OHLC upgrade.
                needs_upgrade = not {"Open", "High", "Low"}.issubset(set(df.columns))
                if needs_upgrade:
                    df_full = pd.DataFrame()
                    try:
                        stooq_path = _find_stooq_cache_path(tk)
                        if stooq_path is not None:
                            df_full = _standardize_ohlc(pd.read_csv(stooq_path), ticker=tk)
                        if df_full is None or df_full.empty:
                            df_full = fetch_ohlc_history(tk, period=period, interval=interval)
                        df_full = _filter_to_period(df_full, period)
                    except Exception:
                        df_full = pd.DataFrame()

                    if df_full is not None and not df_full.empty:
                        try:
                            CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
                            df_full.to_csv(path, index=False)
                        except Exception:
                            pass
                        return df_full, path, False

                return df, p, True
        except Exception:
            pass

    # Best-effort: rebuild `ohlc_*` from an existing stooq_* cache (no network).
    stooq_path = _find_stooq_cache_path(tk)
    if stooq_path is not None:
        try:
            df_raw = pd.read_csv(stooq_path)
            df = _standardize_ohlc(df_raw, ticker=tk)
            df = _filter_to_period(df, period)
            if df is not None and not df.empty:
                try:
                    CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path, index=False)
                except Exception:
                    pass
                return df, path, False
        except Exception:
            pass

    df = fetch_ohlc_history(tk, period=period, interval=interval)
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


def _get_yahoo_options_session() -> requests.Session:
    global _YAHOO_OPTIONS_SESSION
    if _YAHOO_OPTIONS_SESSION is None:
        _YAHOO_OPTIONS_SESSION = requests.Session()
    return _YAHOO_OPTIONS_SESSION


def _refresh_yahoo_crumb(session: requests.Session) -> str | None:
    """
    Yahoo Finance options endpoint requires a `crumb` tied to cookies.
    Best-effort crumb retrieval using the public test endpoint.
    """
    try:
        session.get("https://fc.yahoo.com", headers=_YAHOO_HEADERS_BASE, timeout=12)
    except Exception:
        pass

    try:
        resp = session.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers={**_YAHOO_HEADERS_BASE, "Accept": "*/*"},
            timeout=12,
        )
        resp.raise_for_status()
        crumb = (resp.text or "").strip()
        if not crumb or "<html" in crumb.lower():
            return None
        return crumb
    except Exception as exc:
        logging.warning(f"[yahoo-crumb] fetch failed: {exc}")
        return None


def _get_yahoo_crumb(session: requests.Session, *, force_refresh: bool = False) -> str | None:
    global _YAHOO_OPTIONS_CRUMB
    if force_refresh or not _YAHOO_OPTIONS_CRUMB:
        _YAHOO_OPTIONS_CRUMB = _refresh_yahoo_crumb(session)
    return _YAHOO_OPTIONS_CRUMB


def _yahoo_options_url(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    return f"https://query1.finance.yahoo.com/v7/finance/options/{sym}"


def _fetch_yahoo_options_json(symbol: str, *, expiry_ts: int | None = None) -> dict:
    """
    Fetch raw JSON payload from Yahoo Finance options endpoint.
    `expiry_ts` is the UNIX timestamp returned by Yahoo in `expirationDates`.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {}

    url = _yahoo_options_url(sym)
    session = _get_yahoo_options_session()

    params: dict[str, int | str] = {}
    if expiry_ts is not None:
        params["date"] = int(expiry_ts)

    headers = {**_YAHOO_HEADERS_BASE, "Accept": "application/json"}
    crumb = _get_yahoo_crumb(session)
    if crumb:
        params["crumb"] = crumb
    try:
        resp = session.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 401 and "Invalid Crumb" in (resp.text or ""):
            crumb2 = _get_yahoo_crumb(session, force_refresh=True)
            if crumb2:
                params["crumb"] = crumb2
                resp = session.get(url, params=params, headers=headers, timeout=12)
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
    "fetch_ohlc_history",
    "fetch_options_details",
    "fetch_options_details_yahoo",
    "load_or_fetch_closing_history",
    "clear_closing_history_cache",
]
