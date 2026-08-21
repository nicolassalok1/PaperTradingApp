"""p4 repro — yahoo-period-string (offline, deterministic).

Measures, without any network call, the exact `range` query parameter that the
last-resort Yahoo leg of fetch_daily_closes() sends for the tab's `years`
values, when both Alpaca feeds and Stooq fail. Also checks what the code does
when Yahoo answers with its "invalid range" error envelope (result=null) or a
400, i.e. whether the failure is silent.

NOTE: whether Yahoo's v8 chart endpoint actually rejects '3y' cannot be
measured offline; the Yahoo-documented set used by yfinance is
{1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import app.model.iv_dashboard.service as svc  # noqa: E402
import app.model.market_data.market_data as md  # noqa: E402
import app.model.market_data.service as stooq  # noqa: E402

VALID_YF = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

# Alpaca legs fail
svc._fetch_closes_alpaca = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("alpaca down"))
# Stooq leg fails (download error, no cache) — patch the downloader used by fetch_historical_prices
stooq._download_stooq_csv = lambda params: (_ for _ in ()).throw(requests.ConnectionError("stooq down"))
stooq.CACHE_OHLC_DIR = Path(__file__).parent / "_p4_nocache"  # non-existent -> no cache hit
stooq.CACHE_CSV_DIR = stooq.CACHE_OHLC_DIR

captured = []


class YahooErr:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"chart": {"result": None, "error": {"code": "Bad Request", "description": "Invalid input - range ..."}}}


def fake_get(url, params=None, headers=None, timeout=None):
    captured.append((url, dict(params or {})))
    return YahooErr()


md.requests.get = fake_get

print("=== range param sent to Yahoo by fetch_daily_closes(years=...) ===")
for years in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0):
    captured.clear()
    df, tag, log = svc.fetch_daily_closes("SPY", years=years)
    url, params = captured[-1] if captured else ("<no yahoo call>", {})
    rng = params.get("range")
    print(f"  years={years:>4}: yahoo called={bool(captured)} range={rng!r} valid_yf_range={rng in VALID_YF} "
          f"-> tag={tag!r} rows={len(df)} last_log={log[-1]!r}")

print("\n=== default tab value (service.get_iv_dashboard_data years=2.0) ===")
captured.clear()
try:
    svc.get_iv_dashboard_data("SPY", years=2.0, include_current_iv=False)
except RuntimeError as exc:
    print("  RuntimeError:", exc)
print("  yahoo params:", captured[-1][1] if captured else None)
