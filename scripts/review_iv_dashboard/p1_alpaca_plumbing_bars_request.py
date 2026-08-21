"""Probe: _fetch_closes_alpaca request construction against alpaca-py 0.12.0 (offline, client.get_stock_bars mocked)."""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import alpaca  # noqa: E402
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
import alpaca.data.historical as hist  # noqa: E402
import app.model.iv_dashboard.service as svc  # noqa: E402

print("alpaca-py version:", getattr(alpaca, "__version__", "?"), "| has alpaca.data.historical.option?",
      (Path(alpaca.__file__).parent / "data" / "historical" / "option.py").exists())
print("DataFeed('iex') ->", repr(DataFeed("iex")), "| Adjustment.SPLIT ->", repr(Adjustment.SPLIT))
try:
    DataFeed("sip_delayed")
except Exception as exc:
    print("DataFeed('sip_delayed') raises ->", type(exc).__name__, "(code falls back to raw string, pydantic then rejects?)")
try:
    StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day, feed="delayed_sip")
    print("StockBarsRequest(feed='delayed_sip') accepted")
except Exception as exc:
    print("StockBarsRequest(feed='delayed_sip') rejected ->", type(exc).__name__)

end = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=16)
req = StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day, start=end - dt.timedelta(days=800),
                       end=end, feed=DataFeed("iex"), adjustment=Adjustment.SPLIT)
print("request fields sent:", {k: v for k, v in req.to_request_fields().items() if k != "symbol_or_symbols"})

# Mock get_stock_bars to return a BarSet-like object with a MultiIndex df incl. today's partial bar
captured = {}
class FakeBars:
    def __init__(self, df): self.df = df
def fake_get_stock_bars(self, request):
    captured["req"] = request
    ts = pd.to_datetime(["2026-08-18 04:00", "2026-08-19 04:00", "2026-08-20 04:00", "2026-08-21 04:00"], utc=True)
    df = pd.DataFrame({"symbol": ["SPY"] * 4, "timestamp": ts, "open": 1, "high": 1, "low": 1,
                       "close": [640.0, 641.0, 642.0, 650.0], "volume": 1}).set_index(["symbol", "timestamp"])
    return FakeBars(df)
hist.StockHistoricalDataClient.get_stock_bars = fake_get_stock_bars
svc._alpaca_keys = lambda: ("k", "s")
out = svc._fetch_closes_alpaca("SPY", end - dt.timedelta(days=800), feed="iex")
print("parsed closes:\n", out.to_string(index=False))
print("today's (partial) bar kept if the server returns it? ->", (out["Date"].max().date() == dt.date(2026, 8, 21)))
print("request end sent to server (naive UTC):", captured["req"].end, "| adjustment:", captured["req"].adjustment, "| feed:", captured["req"].feed)
