"""p4 skeptic probe (G4_fallback, code-reading lens). NO network: requests.get is patched module-wide.

1. Yahoo leg: which `range` value does fetch_daily_closes(years=...) actually send?
2. download_options_alpaca(max_pages=3): page-size param actually sent?
3. html-exception: can the message carrying str(exc) (service L368) ever be iv_log[-1] -> iv_error?
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRATCH = Path(tempfile.mkdtemp(prefix="p4_fb_"))

import app.model.iv_dashboard.service as svc  # noqa: E402
import app.model.options.logic as logic  # noqa: E402
import app.model.market_data.service as mds  # noqa: E402

calls = []


class Resp:
    status_code = 503
    text = "down"
    headers = {}

    def raise_for_status(self):
        raise requests.HTTPError("503 Server Error")

    def json(self):
        return {}


def fake_get(url, headers=None, params=None, timeout=None, **kw):
    calls.append((url.split("/")[2], dict(params or {})))
    return Resp()


requests.get = fake_get
# isolate Stooq / chain caches so nothing is served from disk
mds.CACHE_OHLC_DIR = SCRATCH / "ohlc"
mds.CACHE_CSV_DIR = SCRATCH / "csv"
logic.CACHE_ALPACA_OPTION_CHAINS_DIR = SCRATCH / "chains"
logic.CACHE_CSV_DIR = SCRATCH / "csv"
logic._load_alpaca_credentials = lambda: ("k", "s", None)
logic.time.sleep = lambda *_: None


def _boom(*a, **k):
    raise RuntimeError("alpaca down")


svc._fetch_closes_alpaca = _boom

# 1. Yahoo range actually sent
for years in (1.0, 2.0, 3.0, 5.0):
    calls.clear()
    df, tag, log = svc.fetch_daily_closes("SPY", years=years)
    yahoo = [p for h, p in calls if "yahoo" in h]
    print(f"years={years}: yahoo params sent = {yahoo} ; result tag={tag!r}")

# 2. page size sent by download_options_alpaca(max_pages=3)
calls.clear()
logic.download_options_alpaca("SPY", feed="indicative", max_pages=3, cache_to_csv=False, include_spot=False)
print("download_options_alpaca(max_pages=3) alpaca params:", [p for h, p in calls if "alpaca" in h])

# 3. reachability of L368 text as iv_log[-1]
svc.fetch_spot_price = lambda s: 640.0
svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
info, log = svc.fetch_current_atm_iv("SPY")
print("fetch_current_atm_iv on 503: info=", info)
for m in log:
    print("   log:", m[:140])
print("iv_error (= iv_log[-1]) contains the HTTP exc text?", "503" in log[-1])
