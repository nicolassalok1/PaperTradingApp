"""p4 skeptic probe (code-reading lens) for G5_cache: corrupt CSV, local date, unsanitized path.
Offline only. Monkeypatches CACHE_IV_HISTORY_DIR to a temp dir. Never touches app/."""
import datetime as dt
import logging
import sys
import tempfile
import unittest.mock as um
from math import erf, exp, log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

print("pandas", pd.__version__, "python", sys.version.split()[0])

from app.model.calibration.implied_vol import implied_vol_call  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="p4_ivhist_"))
svc.CACHE_IV_HISTORY_DIR = tmp
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(message)s")

info = {"iv": 0.2, "dte": 30, "n_contracts": 4, "method": "greeks Alpaca", "spot": 100.0}

print("\n[1] corrupt-file scenarios (record_iv_observation then load_iv_history)")
cases = {
    "EMPTY0": b"",
    "WS": b"\n\n",
    "HDRONLY": b"date,iv,dte,n_contracts,method,spot\n",
    "NODATECOL": b"foo,iv\n2026-01-01,0.1\n",
    "TRUNC_ROW": b"date,iv,dte,n_contracts,method,spot\n2026-08-19,0.18,30,4,greeks Alpaca,1",
    "EXCELDATE": b"date,iv,dte,n_contracts,method,spot\n19/08/2026,0.18,30,4,greeks Alpaca,100\n",
}
for name, content in cases.items():
    p = svc._iv_history_path(name)
    p.write_bytes(content)
    before = p.stat().st_size
    svc.record_iv_observation(name, info)
    after = p.stat().st_size
    hist = svc.load_iv_history(name)
    print(f"  {name:10s} size {before}->{after}  load rows={len(hist)}  written={after != before}")

print("\n[2] pathlib behaviour of _iv_history_path")
for s in ["A/B", "A\\B", "../evil", "..", "BRK.B", "BRK-B", "^VIX", "ES=F", "A:B", "A?B"]:
    p = svc._iv_history_path(s)
    try:
        inside = p.resolve().relative_to(tmp.resolve()) is not None
    except ValueError:
        inside = False
    try:
        ex = p.exists()
    except OSError as e:
        ex = f"OSError {e}"
    print(f"  {s!r:12} -> {str(p.relative_to(tmp))!r:26} parent={p.parent.name!r:14} inside={inside} exists()={ex}")
svc.record_iv_observation("A/B", info)
svc.record_iv_observation("A:B", info)
svc.record_iv_observation("A?B", info)
print("  files under tmp:", sorted(str(x.relative_to(tmp)) for x in tmp.rglob("*")))
print("  load_iv_history('A/B') rows:", len(svc.load_iv_history("A/B")), " ('A:B'):", len(svc.load_iv_history("A:B")))

print("\n[3] timezone: does ZoneInfo('America/New_York') resolve in this venv (Windows needs tzdata)?")
try:
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    print("  ZoneInfo OK:", dt.datetime.now(ny).tzinfo)
except Exception as e:  # noqa: BLE001
    print("  ZoneInfo FAILED:", type(e).__name__, e)
try:
    import tzdata

    print("  tzdata package present", getattr(tzdata, "__version__", "?"))
except ImportError:
    print("  tzdata package NOT installed")
try:
    from dateutil import tz as dtz

    print("  dateutil.tz.gettz('America/New_York') ->", dtz.gettz("America/New_York"))
except Exception as e:  # noqa: BLE001
    print("  dateutil failed", e)


print("\n[4] DTE / T off-by-one arithmetic (pure)")


def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_call(S, K, T, r, sig):
    d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * N(d1) - K * exp(-r * T) * N(d2)


S = K = 100.0
r = 0.0
sig_true = 0.16
expiry = dt.date(2026, 9, 19)
us_now_et = dt.datetime(2026, 8, 20, 15, 30)  # 15:30 ET Aug 20 (US session live)
sgt_local = us_now_et + dt.timedelta(hours=12)  # 03:30 Aug 21 SGT
dte_ny = (expiry - us_now_et.date()).days
dte_sgt = (expiry - sgt_local.date()).days
T_true = (dte_ny + 0.5 / 24) / 365.0  # ~30 days + 30 min to 16:00 ET close
price = bs_call(S, K, T_true, r, sig_true)
for label, d in (("NY date", dte_ny), ("SGT date", dte_sgt)):
    T = max(d, 1) / 365.0
    iv = implied_vol_call(price, S, K, T, r, 0.0)
    print(f"  {label}: dte={d} T={T:.5f} inverted IV={iv:.5f} (true {sig_true}) err={(iv - sig_true) * 1e4:+.1f} bp")
sess_open = dt.datetime(2026, 8, 20, 9, 30)
sess_close = dt.datetime(2026, 8, 20, 16, 0)
for tzoff in (+8, +7, +2, -4, -7):
    flip = any(((t + dt.timedelta(hours=tzoff + 4)).date() != sess_open.date()) for t in (sess_open, sess_close))
    print(f"  UTC{tzoff:+d}: local date differs from session date during 09:30-16:00 ET -> {flip}")

print("\n[5] cache key collapse for a UTC+8 user running at 03:30 and 22:00 local on Aug 21")
sym = "COLLAPSE"


class FakeDate(dt.date):
    _today = dt.date(2026, 8, 21)

    @classmethod
    def today(cls):
        return cls._today


with um.patch.object(svc.dt, "date", FakeDate):
    svc.record_iv_observation(sym, {**info, "iv": 0.20})  # 03:30 SGT = US Aug-20 session
    svc.record_iv_observation(sym, {**info, "iv": 0.25})  # 22:00 SGT = US Aug-21 session
print(svc.load_iv_history(sym)[["date", "iv"]].to_string(index=False))

print("\n[6] record_iv_observation surfaces nothing to the caller (returns None, no log param)")
import inspect  # noqa: E402

print("  signature:", inspect.signature(svc.record_iv_observation))
