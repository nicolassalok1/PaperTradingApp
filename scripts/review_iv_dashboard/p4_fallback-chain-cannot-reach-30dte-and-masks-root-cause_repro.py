"""p4 repro — fallback-chain-cannot-reach-30dte-and-masks-root-cause (offline).

Checks, with an own fake Alpaca server (symbol-sorted pages, server default
page size when no `limit` param is sent):
  1. which query params download_options_alpaca(max_pages=3) actually sends
     (is `limit` ever present ?)
  2. how many contracts / which DTE range the 3-page fallback yields for a
     SPY-like chain (daily expiries, many strikes) vs a thin monthly chain,
     for server default page = 100 (Alpaca doc default) and 1000 (sensitivity)
  3. on a 403 for the filtered snapshot call, which message ends up as
     iv_error (= iv_log[-1]) and where the 403 text sits.
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

import app.model.iv_dashboard.service as svc  # noqa: E402
import app.model.options.logic as logic  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="p4_fb_"))
(SCRATCH / "chains").mkdir()
(SCRATCH / "csv").mkdir()
(SCRATCH / "iv").mkdir()
logic.CACHE_ALPACA_OPTION_CHAINS_DIR = SCRATCH / "chains"
logic.CACHE_CSV_DIR = SCRATCH / "csv"
svc.CACHE_IV_HISTORY_DIR = SCRATCH / "iv"
svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
logic._load_alpaca_credentials = lambda: ("k", "s", None)
logic.time.sleep = lambda *_: None
SPOT = 500.0
svc.fetch_spot_price = lambda s: SPOT
logic.fetch_spot_price = lambda s: SPOT
TODAY = dt.date.today()


def chain_spy_like(days=70, n_strikes=300):
    """Daily (Mon-Fri) expiries, n_strikes $1-spaced strikes, calls + puts."""
    out = {}
    for d in range(0, days + 1):
        e = TODAY + dt.timedelta(days=d)
        if e.weekday() >= 5:
            continue
        for K in range(int(SPOT) - n_strikes // 2, int(SPOT) + n_strikes // 2):
            for t in "CP":
                out[f"SPY{e:%y%m%d}{t}{K * 1000:08d}"] = {"impliedVolatility": 0.2, "latestQuote": {"bp": 4.0, "ap": 4.2}}
    return dict(sorted(out.items()))


def chain_thin_monthly(n_strikes=40):
    """Third-Friday monthly expiries only, few strikes."""
    out = {}
    for m in range(0, 4):
        y, mo = TODAY.year, TODAY.month + m
        while mo > 12:
            mo -= 12
            y += 1
        first = dt.date(y, mo, 1)
        third_fri = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)
        if third_fri <= TODAY:
            continue
        for K in range(int(SPOT) - n_strikes // 2 * 5, int(SPOT) + n_strikes // 2 * 5, 5):
            for t in "CP":
                out[f"XYZ{third_fri:%y%m%d}{t}{K * 1000:08d}"] = {"impliedVolatility": 0.3}
    return dict(sorted(out.items()))


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
        self.headers = {}
        self.url = "https://data.alpaca.markets/v1beta1/options/snapshots/X"

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: Forbidden for url: {self.url}", response=self)


class Server:
    def __init__(self, chain, *, default_page=100, status_filtered=200, status_unfiltered=200):
        self.items = list(chain.items())
        self.default_page = default_page
        self.status_filtered = status_filtered
        self.status_unfiltered = status_unfiltered
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        filtered = "expiration_date_gte" in params
        status = self.status_filtered if filtered else self.status_unfiltered
        if status != 200:
            return Resp(status, {"message": "forbidden."})
        items = self.items
        if filtered:
            lo = dt.date.fromisoformat(params["expiration_date_gte"])
            hi = dt.date.fromisoformat(params["expiration_date_lte"])
            items = [(s, v) for s, v in items if lo <= svc._decode_opra(s)[1] <= hi]
        page = int(params["limit"]) if params.get("limit") else self.default_page
        start = int(params.get("page_token") or 0)
        chunk = items[start:start + page]
        nxt = str(start + page) if start + page < len(items) else None
        return Resp(200, {"snapshots": dict(chunk), "next_page_token": nxt})


def fallback_only(label, chain, default_page):
    srv = Server(chain, default_page=default_page)
    requests.get = srv.get
    df = logic.download_options_alpaca("SPY", feed="indicative", max_pages=svc._SNAPSHOT_MAX_PAGES, cache_to_csv=False)
    dtes = sorted({int(round(t * 365)) for t in df["T"]}) if len(df) else []
    print(f"  {label}: server_default_page={default_page} calls={len(srv.calls)} params[0]={srv.calls[0]} "
          f"'limit' sent? {any('limit' in c for c in srv.calls)} -> {len(df)} contracts, DTEs={dtes[:8]}{'...' if len(dtes) > 8 else ''}")
    usable = [d for d in dtes if 15 <= d <= 60]
    print(f"     contracts reaching the 15-60 DTE band: {sum(1 for t in df['T'] if 15 <= round(t*365) <= 60)} (expiries {usable})")


print("=== 1+2. fallback download_options_alpaca(max_pages=3): params sent and reach ===")
spy = chain_spy_like()
print(f"SPY-like chain: {len(spy)} contracts, {len({s[3:9] for s in spy})} expiries")
fallback_only("SPY-like", spy, 100)
fallback_only("SPY-like", spy, 1000)
thin = chain_thin_monthly()
print(f"thin monthly chain: {len(thin)} contracts, {len({s[3:9] for s in thin})} expiries")
fallback_only("thin   ", thin, 100)

print("\n=== 3. 403 on filtered snapshots: what reaches the user ===")
for label, st_unf in (("403 on both calls (OPRA agreement missing)", 403), ("403 filtered only, unfiltered OK", 200)):
    srv = Server(spy, default_page=100, status_filtered=403, status_unfiltered=st_unf)
    requests.get = srv.get
    info, log = svc.fetch_current_atm_iv("SPY")
    print(f"  {label}: calls={len(srv.calls)} info={info if info is None else {k: info[k] for k in ('iv','dte','method')}}")
    for i, m in enumerate(log):
        print(f"     log[{i}]: {m[:120]}")
    print(f"     UI iv_error = {log[-1]!r} | 403 text in iv_error? {'403' in log[-1]}")

print("\n=== 3b. ConnectionError on filtered snapshots, fallback also down, no cache ===")
def down(url, headers=None, params=None, timeout=None):
    raise requests.ConnectionError("Failed to establish a new connection")
requests.get = down
info, log = svc.fetch_current_atm_iv("SPY")
print("  iv_error =", repr(log[-1]))
