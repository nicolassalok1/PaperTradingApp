"""Probe: record_iv_observation / load_iv_history cache semantics + local-date drift (offline)."""
from __future__ import annotations
import datetime as dt
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app.model.iv_dashboard.service as svc  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="iv_cache_probe_"))
svc.CACHE_IV_HISTORY_DIR = SCRATCH
logging.basicConfig(level=logging.WARNING, format="LOG %(levelname)s %(message)s")
INFO = {"iv": 0.1812, "dte": 30, "n_contracts": 40, "method": "greeks Alpaca", "spot": 640.0}
path = svc._iv_history_path("SPY")

print("[1] normal upsert twice same day -> rows:")
svc.record_iv_observation("SPY", INFO)
svc.record_iv_observation("SPY", {**INFO, "iv": 0.2})
print(pd.read_csv(path).to_dict("records"))

print("\n[2] empty (0-byte) CSV, e.g. after a crash mid-write -> is today's observation kept?")
path.write_text("")
svc.record_iv_observation("SPY", INFO)
print("  file size after record:", path.stat().st_size, "bytes; load_iv_history ->", len(svc.load_iv_history("SPY")), "rows")

print("\n[3] CSV with header only / missing 'date' column -> DataFrame.get default path")
path.write_text("foo,iv\n1,0.1\n")
svc.record_iv_observation("SPY", INFO)
print("  content:", repr(path.read_text()))

print("\n[4] dtype of 'date' on reload")
path.unlink()
svc.record_iv_observation("SPY", INFO)
raw = pd.read_csv(path)
print("  raw read_csv date dtype:", raw["date"].dtype, "| load_iv_history date dtype:", svc.load_iv_history("SPY")["date"].dtype)

print("\n[5] concurrent read-modify-write (8 threads, distinct dates) -> rows kept?")
path.unlink()
def worker(i):
    orig_today = svc.dt.date.today
    # emulate distinct days by monkeypatching today() per thread is unsafe; instead write rows directly via same function with distinct info
    svc.record_iv_observation("SPY", {**INFO, "iv": 0.1 + i / 100})
ths = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
[t.start() for t in ths]; [t.join() for t in ths]
print("  rows after 8 same-date concurrent upserts:", len(pd.read_csv(path)), "(expected 1; >1 means duplicate rows, exception logged means lost write)")

print("\n[6] local-date drift: dt.date.today() (server-local) vs US exchange date")
for tzname in ("Asia/Singapore", "Europe/Paris", "America/New_York"):
    tz = ZoneInfo(tzname)
    # US session 2026-08-20 15:30 ET (market open)
    et = dt.datetime(2026, 8, 20, 15, 30, tzinfo=ZoneInfo("America/New_York"))
    local = et.astimezone(tz)
    print(f"  {tzname:18} local clock {local:%Y-%m-%d %H:%M} -> date.today()={local.date()} while US session date = {et.date()}"
          f" -> key mismatch: {local.date() != et.date()}")
print("  consequence for Asia/Singapore: 02:00 SGT Aug-21 (US Aug-20 session live) writes row 'date=2026-08-21';"
      " 22:00 SGT Aug-21 (US Aug-21 session live) overwrites the SAME key -> the Aug-20 session observation is lost;"
      " dte/T are also computed from the SGT date (expiry - today) -> 1 day short.")
