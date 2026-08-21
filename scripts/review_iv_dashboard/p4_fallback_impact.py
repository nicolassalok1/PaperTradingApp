"""
Phase-4 skeptic probe — G4_fallback (impact & severity lens). NO network: requests.get is mocked.

Scenarios (all through the real service/logic code, caches redirected to a temp dir):
  S1  snapshots fail (ConnectionError) + 10-day-old chain cache WITH iv  -> what is displayed / persisted?
  S2  same, but the cache's `opra` column already carries the real expiry -> is the fix simpler than claimed?
  S3  snapshots fail + 10-day-old cache WITHOUT iv (indicative w/o greeks) -> any number displayed?
  S4  snapshots 403 / 401-HTML / ConnectionError, NO cache -> iv_error text shown to the user + fallback latency
  S5  401 nginx HTML as the bars exception -> how many lines land in the journal?
Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_fallback_impact.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from app.model.iv_dashboard import service as svc  # noqa: E402
from app.model.options import logic  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="p4_fallback_"))
CHAIN_DIR = TMP / "AlpacaOptionChains"
IVH_DIR = TMP / "IVHistory"
CHAIN_DIR.mkdir()
IVH_DIR.mkdir()

# --- redirect caches & credentials, never touch the real cache/ -------------
logic.CACHE_ALPACA_OPTION_CHAINS_DIR = CHAIN_DIR
logic.CACHE_CSV_DIR = TMP
svc.CACHE_IV_HISTORY_DIR = IVH_DIR
logic._load_alpaca_credentials = lambda: ("k", "s", "https://paper-api.alpaca.markets")
svc._alpaca_keys = lambda: ("k", "s")
svc.fetch_spot_price = lambda sym: 500.0
logic.fetch_spot_price = lambda sym: 500.0

TODAY = dt.date.today()
CACHE_AGE_DAYS = 10
CACHE_DATE = TODAY - dt.timedelta(days=CACHE_AGE_DAYS)


def opra(sym: str, expiry: dt.date, typ: str, k: float) -> str:
    return f"{sym}{expiry.strftime('%y%m%d')}{typ}{int(round(k * 1000)):08d}"


def write_cache(sym: str, *, with_iv: bool, mtime_days_ago: int) -> Path:
    """Chain as download_options_alpaca would have written it `mtime_days_ago` days ago."""
    rows = []
    for dte_at_cache in (2, 9, 30, 44):  # expiries as seen from the cache date
        expiry = CACHE_DATE + dt.timedelta(days=dte_at_cache)
        for k in range(470, 531, 5):
            for typ in ("C", "P"):
                rows.append(
                    {
                        "symbol": sym,
                        "opra": opra(sym, expiry, typ, k),
                        "K": float(k),
                        "T": dte_at_cache / 365.0,
                        "S0": 500.0,
                        "iv": 0.25 if with_iv else float("nan"),
                        "type": "call" if typ == "C" else "put",
                    }
                )
    p = CHAIN_DIR / f"options_alpaca_{sym}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    ts = time.time() - mtime_days_ago * 86400
    os.utime(p, (ts, ts))
    return p


class FakeResp:
    def __init__(self, status: int, text: str = "", payload=None):
        self.status_code = status
        self.text = text
        self.headers = {}
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error: {self.text[:40]}", response=self)


NGINX_401 = (
    "<html>\n<head><title>401 Authorization Required</title></head>\n<body>\n"
    "<center><h1>401 Authorization Required</h1></center>\n<hr><center>nginx</center>\n</body>\n</html>\n"
)


def patch_get(mode: str):
    calls: list[dict] = []

    def _get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "params": dict(params or {})})
        if mode == "conn":
            raise requests.ConnectionError("HTTPSConnectionPool(host='data.alpaca.markets'): Max retries exceeded")
        if mode == "403":
            return FakeResp(403, "forbidden", {"message": "subscription does not permit querying opra"})
        if mode == "401html":
            return FakeResp(401, NGINX_401, None)
        raise AssertionError(mode)

    requests.get = _get
    return calls


def run_iv(sym="SPY"):
    t0 = time.perf_counter()
    info, log = svc.fetch_current_atm_iv(sym)
    return info, log, time.perf_counter() - t0


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- S1
section("S1  ConnectionError + 10-day-old cache WITH iv=0.25 (calibration-tab leftover)")
p = write_cache("SPY", with_iv=True, mtime_days_ago=CACHE_AGE_DAYS)
calls = patch_get("conn")
info, log, elapsed = run_iv()
for ln in log:
    print("  log:", ln[:140])
print(f"  HTTP attempts: {len(calls)} | elapsed {elapsed:.2f}s (retry/backoff inside download_options_alpaca)")
print("  -> current_iv returned to the view:", info)
if info:
    real_exp_30 = CACHE_DATE + dt.timedelta(days=30)
    print(f"  displayed expiry {info['expiry']} dte={info['dte']} | real expiry of that contract {real_exp_30} "
          f"(true dte today = {(real_exp_30 - TODAY).days}) | shift = {(info['expiry'] - real_exp_30).days} d")
    print(f"  caption the user sees: 'méthode : {info['method']} · flux : {info['feed']}'  (no hint of cache/age)")
    svc.record_iv_observation("SPY", info)
    hist = pd.read_csv(IVH_DIR / "iv_daily_SPY.csv")
    print("  persisted iv_daily_SPY.csv row:", hist.iloc[-1].to_dict())

# --------------------------------------------------------------------------- S2
section("S2  Does the cache already carry the real expiry? (fix simplicity check)")
df_cache = pd.read_csv(p)
dec = [svc._decode_opra(o) for o in df_cache["opra"].head(3)]
print("  first opra symbols :", list(df_cache["opra"].head(3)))
print("  _decode_opra -> (K, expiry, type):", dec)
print("  => real expiry is recoverable from the existing `opra` column with svc._decode_opra; "
      "no new column needed.")

# --------------------------------------------------------------------------- S3
section("S3  ConnectionError + 10-day-old cache WITHOUT iv (indicative feed w/o greeks)")
write_cache("SPY", with_iv=False, mtime_days_ago=CACHE_AGE_DAYS)
patch_get("conn")
info, log, elapsed = run_iv()
for ln in log:
    print("  log:", ln[:140])
print("  -> current_iv:", info, "| iv_error shown:", repr(log[-1]))

# --------------------------------------------------------------------------- S4
section("S4  NO cache: iv_error text shown in st.warning + latency of the useless fallback")
(CHAIN_DIR / "options_alpaca_SPY.csv").unlink()
for mode in ("403", "401html", "conn"):
    calls = patch_get(mode)
    info, log, elapsed = run_iv()
    limits = [c["params"].get("limit") for c in calls]
    pages_unfiltered = sum(1 for c in calls if "expiration_date_gte" not in c["params"])
    print(f"  [{mode:7}] elapsed {elapsed:.2f}s | HTTP calls {len(calls)} (unfiltered fallback calls: {pages_unfiltered}, "
          f"limit params: {limits})")
    print(f"            iv_error (st.warning) = {log[-1]!r}")
    root = next((l for l in log if "Snapshots filtrés indisponibles" in l), "")
    print(f"            root cause only in expander: {root[:110]!r}")

# --------------------------------------------------------------------------- S5
section("S5  401 nginx HTML as bars exception -> journal lines")


def _boom(sym, start, feed=None):
    raise RuntimeError(NGINX_401)


svc._fetch_closes_alpaca = _boom
svc.fetch_ohlc_history = lambda sym, period="2y", interval="1d": pd.DataFrame(
    {"Date": pd.date_range(end=pd.Timestamp.today(), periods=600, freq="B"), "Close": 500.0}
)
df, src, log = svc.fetch_daily_closes("SPY", years=2.0, extra_days=47)
joined = "\n".join(log)
print(f"  source={src} bars={len(df)} | log entries={len(log)} | rendered journal lines={joined.count(chr(10)) + 1}")
for ln in log:
    print("  entry lines:", ln.count("\n") + 1, "| first 70 chars:", ln[:70].replace("\n", "\\n"))
short = [" ".join(l.split())[:160] for l in log]
print("  with _short_exc(160):", [s[:90] + ("..." if len(s) > 90 else "") for s in short])

print("\ntemp dir:", TMP)
