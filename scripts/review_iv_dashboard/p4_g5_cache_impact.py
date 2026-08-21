"""Phase 4 skeptic probe - G5_cache impact/severity (review-only, no network, no tracked file touched)."""
import datetime as dt
import logging
import sys
import tempfile
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app.model.iv_dashboard.service as svc  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(message)s")
tmp = Path(tempfile.mkdtemp(prefix="p4_g5_"))
svc.CACHE_IV_HISTORY_DIR = tmp
INFO = {"iv": 0.18, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 500.0}

print("=== [1] corrupt cache file ===")
p = svc._iv_history_path("SPY")
p.write_bytes(b"")
svc.record_iv_observation("SPY", INFO)
print("0-byte file -> size after record:", p.stat().st_size, "| load rows:", len(svc.load_iv_history("SPY")))
p.write_text("date,iv,dte,n_contracts,method,spot\n")  # header only
svc.record_iv_observation("SPY", INFO)
print("header-only file -> size after record:", p.stat().st_size, "| load rows:", len(svc.load_iv_history("SPY")))
p.write_text("garbage line without commas\n")
svc.record_iv_observation("SPY", INFO)
print("garbage 1-col file -> load rows:", len(svc.load_iv_history("SPY")))
df = pd.DataFrame({"date": pd.date_range("2024-08-21", periods=500).strftime("%Y-%m-%d"),
                   "iv": np.random.rand(500), "dte": 30, "n_contracts": 6,
                   "method": "greeks Alpaca", "spot": 500.0})
ts = []
for _ in range(20):
    t0 = time.perf_counter(); df.to_csv(p, index=False); ts.append(time.perf_counter() - t0)
print(f"to_csv window for 500-row history: median {np.median(ts)*1e3:.2f} ms, max {max(ts)*1e3:.2f} ms")

print("\n=== [2] local date vs exchange date ===")
NY = ZoneInfo("America/New_York")
for tzname in ("Asia/Makassar", "Asia/Jakarta", "Asia/Singapore", "Europe/Paris", "America/Los_Angeles", "Pacific/Honolulu"):
    tz = ZoneInfo(tzname)
    mism = 0; total = 0
    t = dt.datetime(2026, 8, 20, 9, 30, tzinfo=NY)
    end = dt.datetime(2026, 8, 20, 16, 0, tzinfo=NY)
    first = None
    while t <= end:
        total += 1
        if t.astimezone(tz).date() != t.date():
            mism += 1
            if first is None:
                first = f"{t.astimezone(tz).strftime('%H:%M')} local = {t.strftime('%H:%M')} ET"
        t += dt.timedelta(minutes=1)
    print(f"{tzname:20s} RTH minutes with local date != NY date: {mism}/{total} ({100*mism/total:.0f}%)"
          + (f"  from {first}" if first else ""))

sigma = 0.16
T_int = 30 / 365
T_true = (30 + 3 / 24) / 365
T_off1 = 29 / 365
print(f"IV bias from integer-DTE convention itself (13:00 ET, 30 d): {sigma*np.sqrt(T_int/T_true)*100-16:+.2f} pp")
print(f"IV bias from off-by-one local date (29 vs 30 d):           {sigma*np.sqrt(T_int/T_off1)*100-16:+.2f} pp")
print("NOTE: bias applies only to the BS-inversion path; with 'greeks Alpaca' method T is unused.")
print("ZoneInfo('America/New_York') loads on this Windows venv:", dt.datetime.now(NY).tzname())

svc.CACHE_IV_HISTORY_DIR = tmp / "ov"
p2 = svc._iv_history_path("QQQ")
svc.record_iv_observation("QQQ", {**INFO, "iv": 0.20, "spot": 1.0})
svc.record_iv_observation("QQQ", {**INFO, "iv": 0.25, "spot": 2.0})
print("two sessions under same local date -> rows kept:", len(pd.read_csv(p2)), "(1 = second overwrote first)")

print("\n=== [3] unsanitized symbol ===")
svc.CACHE_IV_HISTORY_DIR = tmp / "sym"
for s in ("A/B", "A\\B", "A:B", "A?B", "A*B", "A|B", "A<B>", '"A"', "../X", "SPY,QQQ", "NUL", "CON", "BRK.B"):
    try:
        h = svc.load_iv_history(s)
        print(f"load_iv_history({s!r:10}) -> rows={len(h)} path={svc._iv_history_path(s).name!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"load_iv_history({s!r:10}) RAISED {type(exc).__name__}: {exc}")
for s in ("A:B", "A?B", "A/B"):
    svc.record_iv_observation(s, INFO)
print("files after writes:", sorted(str(q.relative_to(tmp / 'sym')) for q in (tmp / 'sym').rglob('*') if q.is_file()))
print("tmp:", tmp)
