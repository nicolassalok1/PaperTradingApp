"""
Server-side Yahoo Finance loader for the SPX/VIX exercise.

Python port of `reference/engine/yahoo.ts`: fetches ^GSPC and ^VIX daily closes
from the Yahoo chart API and inner-joins them on date. Runs server-side (the
Streamlit process), so there is no browser CORS constraint and no need for
`yfinance`. Returns a DataFrame shaped exactly like `engine.load_csv`
(DatetimeIndex, float columns SPX/VIX) — ready to feed `engine.backtest`.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ^GSPC, ^VIX (URL-encoded caret)
_SYMBOLS = {"SPX": "%5EGSPC", "VIX": "%5EVIX"}
_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_TIMEOUT = 20


def _fetch_series(symbol: str, period1: int, period2: int) -> dict[str, float]:
    """Daily closes for one Yahoo symbol -> {YYYY-MM-DD: close}. Tries both hosts."""
    last_err: Exception | None = None
    for host in _HOSTS:
        url = (
            f"https://{host}/v8/finance/chart/{symbol}"
            f"?period1={period1}&period2={period2}&interval=1d"
        )
        try:
            res = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if res.status_code != 200:
                raise RuntimeError(f"Yahoo {symbol} HTTP {res.status_code}")
            payload = res.json()
            result = (payload.get("chart") or {}).get("result") or []
            if not result:
                raise RuntimeError(f"Yahoo {symbol}: payload vide")
            r0 = result[0]
            ts = r0.get("timestamp") or []
            quote = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
            close = quote.get("close") or []
            if not ts or not close:
                raise RuntimeError(f"Yahoo {symbol}: payload vide")
            out: dict[str, float] = {}
            for t, c in zip(ts, close):
                if c is None:
                    continue
                try:
                    cf = float(c)
                except (TypeError, ValueError):
                    continue
                if cf != cf:  # NaN
                    continue
                day = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
                out[day] = cf  # last value for a date wins (daily bars are unique)
            return out
        except Exception as exc:  # noqa: BLE001 - fall through to the next host
            last_err = exc
    raise last_err or RuntimeError(f"Yahoo fetch failed for {symbol}")


def fetch_yahoo_prices(start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch & inner-join ^GSPC/^VIX daily closes into an engine-ready DataFrame."""
    period1 = int(
        datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )
    period2 = (
        int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        if end
        else int(time.time())
    )

    spx = _fetch_series(_SYMBOLS["SPX"], period1, period2)
    vix = _fetch_series(_SYMBOLS["VIX"], period1, period2)

    rows = [(d, s, vix[d]) for d, s in spx.items() if d in vix]
    if len(rows) < 500:
        raise RuntimeError(f"Seulement {len(rows)} lignes SPX/VIX alignées depuis Yahoo.")

    df = pd.DataFrame(rows, columns=["date", "SPX", "VIX"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").astype(float).sort_index().dropna()
    return df[["SPX", "VIX"]]
