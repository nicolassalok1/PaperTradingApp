"""Probe: _fetch_atm_snapshots / fetch_current_atm_iv plumbing with a mocked Alpaca server (NO network).

Scenarios:
  A. server honours filters, SPY-like chain  -> does the 3x1000 page cap truncate? is it logged?
  B. server IGNORES expiry/strike filters    -> what does the user see?
  C. 403 (OPRA not signed) on snapshots      -> which message surfaces as iv_error?
  D. network down + stale chain cache        -> stale IVs / shifted expiries reported as current?
  E. fallback download_options_alpaca(max_pages=3): page size actually requested
"""
from __future__ import annotations
import datetime as dt
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRATCH = Path(os.environ.get("PROBE_SCRATCH", tempfile.mkdtemp(prefix="iv_probe_")))

import app.model.iv_dashboard.service as svc  # noqa: E402
import app.model.options.logic as logic  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(message)s")

# ---- isolate caches & creds ------------------------------------------------
(SCRATCH / "chains").mkdir(parents=True, exist_ok=True)
(SCRATCH / "csv").mkdir(parents=True, exist_ok=True)
(SCRATCH / "iv").mkdir(parents=True, exist_ok=True)
logic.CACHE_ALPACA_OPTION_CHAINS_DIR = SCRATCH / "chains"
logic.CACHE_CSV_DIR = SCRATCH / "csv"
svc.CACHE_IV_HISTORY_DIR = SCRATCH / "iv"
svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
logic._load_alpaca_credentials = lambda: ("k", "s", None)
logic.time.sleep = lambda *_: None
SPOT = 640.0
svc.fetch_spot_price = lambda s: SPOT

TODAY = dt.date.today()


def build_chain(spot=SPOT, days=90, strike_lo=300, strike_hi=900):
    """SPY-like: Mon/Wed/Fri expiries, $1 strikes, calls+puts, keys sorted like the API (by symbol)."""
    out = {}
    for d in range(0, days + 1):
        e = TODAY + dt.timedelta(days=d)
        if e.weekday() not in (0, 2, 4):
            continue
        for K in range(strike_lo, strike_hi + 1):
            for t in ("C", "P"):
                sym = f"SPY{e:%y%m%d}{t}{int(K*1000):08d}"
                out[sym] = {
                    "latestQuote": {"bp": 5.0, "ap": 5.1},
                    "latestTrade": {"p": 5.05},
                    "impliedVolatility": 0.16 + 0.0001 * d,
                    "greeks": {"delta": 0.5},
                }
    return dict(sorted(out.items()))


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.headers = {}
        self.url = "https://data.alpaca.markets/v1beta1/options/snapshots/SPY?feed=indicative&limit=1000"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}", response=self)


class FakeServer:
    def __init__(self, chain, *, honour_filters=True, status=200, default_page=100):
        self.chain = chain
        self.honour = honour_filters
        self.status = status
        self.default_page = default_page
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        if self.status != 200:
            return FakeResp(self.status, {"message": "forbidden."})
        items = list(self.chain.items())
        if self.honour:
            def keep(sym):
                strike, expiry, _ = svc._decode_opra(sym)
                ok = True
                if "expiration_date_gte" in params:
                    ok &= expiry >= dt.date.fromisoformat(params["expiration_date_gte"])
                if "expiration_date_lte" in params:
                    ok &= expiry <= dt.date.fromisoformat(params["expiration_date_lte"])
                if "strike_price_gte" in params:
                    ok &= strike >= float(params["strike_price_gte"])
                if "strike_price_lte" in params:
                    ok &= strike <= float(params["strike_price_lte"])
                return ok
            items = [(s, v) for s, v in items if keep(s)]
        page = int(params.get("limit") or self.default_page)
        start = int(params.get("page_token") or 0)
        chunk = items[start:start + page]
        nxt = str(start + page) if start + page < len(items) else None
        return FakeResp(200, {"snapshots": dict(chunk), "next_page_token": nxt})


def run(label, server):
    requests.get = server.get
    info, log = svc.fetch_current_atm_iv("SPY")
    print(f"\n=== {label}")
    print(f"  HTTP calls: {len(server.calls)}; limits requested: {[c.get('limit') for c in server.calls]}")
    for m in log:
        print("  log:", m)
    print("  info:", None if info is None else {k: info[k] for k in ("iv", "expiry", "dte", "n_contracts", "method")})
    return info, log


chain = build_chain()
n_filtered = sum(1 for s in chain if 15 <= (svc._decode_opra(s)[1] - TODAY).days <= 60 and abs(svc._decode_opra(s)[0] / SPOT - 1) <= 0.10)
print(f"Synthetic SPY chain: {len(chain)} contracts total; {n_filtered} inside the server filters (15-60 DTE, +-10% strikes)")

# A
srv = FakeServer(chain, honour_filters=True)
info, log = run("A. filters honoured, 3 pages x 1000 cap", srv)
print(f"  -> cap hit? {len(srv.calls) == svc._SNAPSHOT_MAX_PAGES and n_filtered > 3000}; any log mentioning truncation/pages? "
      f"{any('page' in m.lower() or 'tronq' in m.lower() for m in log)}")

# B
srv = FakeServer(chain, honour_filters=False)
run("B. filters IGNORED by server (unfiltered chain sorted by symbol)", srv)

# C
srv = FakeServer(chain, status=403)
info, log = run("C. 403 on snapshots (e.g. feed=opra without OPRA agreement)", srv)
print("  -> iv_error shown in UI (= iv_log[-1]):", repr(log[-1]))

# E (direct)
requests.get = FakeServer(chain, honour_filters=False).get
srv_e = FakeServer(chain, honour_filters=False)
requests.get = srv_e.get
df = logic.download_options_alpaca("SPY", feed="indicative", max_pages=svc._SNAPSHOT_MAX_PAGES, cache_to_csv=False)
print(f"\n=== E. fallback download_options_alpaca(max_pages=3): calls={len(srv_e.calls)} limit params={[c.get('limit') for c in srv_e.calls]} "
      f"-> {len(df)} contracts, DTE range = [{(df['T']*365).min():.0f}, {(df['T']*365).max():.0f}] days")

# D: stale cache (written 7 days ago with T relative to that date) + network down
stale_date = TODAY - dt.timedelta(days=7)
exp = TODAY + dt.timedelta(days=30)   # real expiry
rows = []
for K in range(600, 681):
    for t in ("call", "put"):
        rows.append({"symbol": "SPY", "opra": f"SPY{exp:%y%m%d}{'C' if t=='call' else 'P'}{K*1000:08d}",
                     "K": float(K), "T": (exp - stale_date).days / 365.0, "S0": 630.0, "iv": 0.25, "type": t})
cache_path = logic.CACHE_ALPACA_OPTION_CHAINS_DIR / "options_alpaca_SPY.csv"
pd.DataFrame(rows).to_csv(cache_path, index=False)
os.utime(cache_path, (dt.datetime.now().timestamp() - 7 * 86400,) * 2)


class DownServer(FakeServer):
    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        raise requests.ConnectionError("Failed to establish a new connection")


info, log = run("D. network down + 7-day-old chain cache (iv=0.25 stored, real expiry today+30)", DownServer({}))
if info:
    svc.record_iv_observation("SPY", info)   # what get_iv_dashboard_data does right after (L567-568)
    hist = pd.read_csv(svc.CACHE_IV_HISTORY_DIR / "iv_daily_SPY.csv")
    print(f"  -> reported expiry {info['expiry']} vs real {exp} (shift {(info['expiry'] - exp).days:+d} d); iv {info['iv']} comes from a 7-day-old file; "
          f"persisted as today's observation: {hist.to_dict('records')}")
