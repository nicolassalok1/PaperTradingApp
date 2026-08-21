"""p4 repro — stale-chain-cache-as-current-iv (offline, deterministic).

Independent of the finder's script: own fake chain cache, own failure modes.

Scenario: a chain cache options_alpaca_SPY.csv was written N days ago by
download_options_alpaca (T = (expiry - cache_day)/365, opra carries the real
expiry). Today the Alpaca snapshot endpoint fails (ConnectionError / 403 / 429).
What does fetch_current_atm_iv return, and what does record_iv_observation
persist ?  Also: what happens when the cached chain has NaN IVs (indicative
feed without greeks) ?
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import app.model.iv_dashboard.service as svc  # noqa: E402
import app.model.options.logic as logic  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="p4_stale_"))
for sub in ("chains", "csv", "iv"):
    (SCRATCH / sub).mkdir(parents=True, exist_ok=True)
logic.CACHE_ALPACA_OPTION_CHAINS_DIR = SCRATCH / "chains"
logic.CACHE_CSV_DIR = SCRATCH / "csv"
svc.CACHE_IV_HISTORY_DIR = SCRATCH / "iv"
svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
logic._load_alpaca_credentials = lambda: ("k", "s", None)
logic.time.sleep = lambda *_: None  # no real backoff
SPOT = 500.0
svc.fetch_spot_price = lambda s: SPOT
logic.fetch_spot_price = lambda s: SPOT

TODAY = dt.date.today()


def write_stale_cache(age_days: int, iv_value, real_dte: int = 30) -> Path:
    cache_day = TODAY - dt.timedelta(days=age_days)
    expiry = TODAY + dt.timedelta(days=real_dte)
    rows = []
    for K in range(470, 531, 5):
        for t, letter in (("call", "C"), ("put", "P")):
            rows.append(
                {
                    "symbol": "SPY",
                    "opra": f"SPY{expiry:%y%m%d}{letter}{int(K * 1000):08d}",
                    "K": float(K),
                    "T": (expiry - cache_day).days / 365.0,  # as download_options_alpaca computed it on cache_day
                    "S0": 490.0,
                    "iv": iv_value,
                    "type": t,
                }
            )
    p = logic.CACHE_ALPACA_OPTION_CHAINS_DIR / "options_alpaca_SPY.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    old = (dt.datetime.now() - dt.timedelta(days=age_days)).timestamp()
    os.utime(p, (old, old))
    return p


class Resp:
    def __init__(self, status):
        self.status_code = status
        self.headers = {}
        self.url = "https://data.alpaca.markets/v1beta1/options/snapshots/SPY"

    def json(self):
        return {"message": "forbidden."}

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}", response=self)


def make_get(mode):
    calls = []

    def _get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params or {}))
        if mode == "conn":
            raise requests.ConnectionError("Failed to establish a new connection")
        return Resp(mode)

    _get.calls = calls
    return _get


def run(label, mode, age_days, iv_value, real_dte=30):
    print(f"\n=== {label}")
    write_stale_cache(age_days, iv_value, real_dte)
    hist = svc.CACHE_IV_HISTORY_DIR / "iv_daily_SPY.csv"
    if hist.exists():
        hist.unlink()
    requests.get = make_get(mode)
    info, log = svc.fetch_current_atm_iv("SPY")
    print("  HTTP attempts:", len(requests.get.calls))
    for m in log:
        print("  log:", m)
    if info is None:
        print("  info: None")
        return
    real_expiry = TODAY + dt.timedelta(days=real_dte)
    print(
        f"  info: iv={info['iv']} expiry={info['expiry']} dte={info['dte']} "
        f"method={info['method']!r} feed={info['feed']!r} n={info['n_contracts']}"
    )
    print(f"  real expiry (from opra in the cache) = {real_expiry} -> shift = {(info['expiry'] - real_expiry).days:+d} d (cache age {age_days} d)")
    # what get_iv_dashboard_data does next (service.py L567-568)
    svc.record_iv_observation("SPY", info)
    print("  persisted iv_daily_SPY.csv:", pd.read_csv(hist).to_dict("records"))


run("A. network down, 7-day-old cache with IVs (0.22), real expiry today+30", "conn", 7, 0.22)
run("B. 403 on snapshots AND on fallback chain, 3-day-old cache with IVs", 403, 3, 0.22)
run("C. 429 persistent, 10-day-old cache with IVs (real dte 30 -> 40 after shift)", 429, 10, 0.22)
run("D. network down, 7-day-old cache WITHOUT IVs (indicative feed, NaN greeks)", "conn", 7, float("nan"))
run("E. network down, 20-day-old cache, real expiry today+30 (shift pushes dte to 50, still in 15-60)", "conn", 20, 0.22)
run("F. network down, 40-day-old cache, real expiry today+30 -> reported dte 70 (out of band)", "conn", 40, 0.22)

# does the cached CSV already carry the real expiry ?
p = logic.CACHE_ALPACA_OPTION_CHAINS_DIR / "options_alpaca_SPY.csv"
row = pd.read_csv(p).iloc[0]
print("\ncache row opra:", row["opra"], "-> _decode_opra expiry:", svc._decode_opra(row["opra"])[1],
      "| T-derived expiry:", TODAY + dt.timedelta(days=int(round(float(row["T"]) * 365.0))))
