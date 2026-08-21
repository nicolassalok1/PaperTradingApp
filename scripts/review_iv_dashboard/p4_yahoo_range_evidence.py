"""
Phase-4 skeptic probe — `yahoo-period-string` (no network).

Uses the artefacts of the earlier LIVE run (orch_live_alpaca.out.txt, keys revoked -> Alpaca 401)
to decide whether Yahoo accepted range='3y':
  * fetch_daily_closes(years=2) -> period '3y' -> fetch_ohlc_history -> Stooq first, then Yahoo(range='3y')
  * Stooq history is cached forever in cache/OHLC/stooq_<sym>.us_start_end_d.csv on success
  * in the same run fetch_spot_price fell through to the Stooq spot and returned None (Stooq down/limited)
If the live run served bars while no Stooq SPY cache file was created after the cache dir was populated,
the bars can only have come from Yahoo with range='3y'.
Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_yahoo_range_evidence.py
"""
from __future__ import annotations

import datetime as dt
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

out = ROOT / "scripts" / "review_iv_dashboard" / "orch_live_alpaca.out.txt"
txt = out.read_text(encoding="utf-8", errors="ignore")
run_ts = dt.datetime.fromtimestamp(out.stat().st_mtime)
print("live run output written at :", run_ts)
print("bars line                  :", next((l.strip() for l in txt.splitlines() if "barres daily via fallback" in l), None))
print("spot line                  :", next((l.strip() for l in txt.splitlines() if "Spot indisponible" in l), None))

ohlc = ROOT / "cache" / "OHLC"
print("cache/OHLC dir mtime       :", dt.datetime.fromtimestamp(ohlc.stat().st_mtime))
files = sorted(ohlc.glob("*"))
for f in files:
    print("   ", f.name, dt.datetime.fromtimestamp(f.stat().st_mtime))
spy_stooq = [f for f in files if re.match(r"stooq_spy\.us", f.name, re.I)]
print("Stooq SPY cache present    :", bool(spy_stooq))

# period actually sent for the default years=2.0
years = 2.0
period = f"{max(1, int(math.ceil(float(years))) + 1)}y"
print("period sent to Yahoo       :", period)

# expected bar count if a >=777-day series is cut at start = now - (2*365.25 + 47) days
lookback = int(2.0 * 365.25) + int(20 * 1.6) + 15
print("lookback days              :", lookback, "-> ~", round(lookback / 365.25 * 252), "trading days (run reported 534)")

verdict = (not spy_stooq) and ("barres daily via fallback" in txt) and ("Spot indisponible" in txt)
print("\n=> Yahoo served range='3y' during the live run:", verdict)
