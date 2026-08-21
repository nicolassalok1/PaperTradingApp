"""Probe: does the fetch lookback (years + extra_days) cover the percentile window for the HEADLINE percentile? (offline)"""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app.model.iv_dashboard.service as svc  # noqa: E402

rng = np.random.default_rng(0)

def fake_fetch(symbol, *, years, extra_days):
    lookback_days = int(float(years) * 365.25) + max(0, int(extra_days))
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)
    dates = pd.bdate_range(start.date(), dt.date.today())   # ~ trading days (holidays ignored -> optimistic)
    px = 600 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": px}), "probe", [f"{len(dates)} bars (probe)"]

svc.fetch_daily_closes = fake_fetch
print(f"{'years':>5} {'rv':>4} {'pct_win':>7} {'bars':>5} {'rv_pts':>6} {'pct obs at last point':>22} {'displayed label'}")
for years in (1.0, 2.0, 3.0, 5.0):
    for rv_w in (20, 120):
        for pw in (252, 504, 756):
            res = svc.get_iv_dashboard_data("SPY", years=years, rv_window=rv_w, percentile_window=pw, include_current_iv=False)
            closes = res["series"]["close"]
            # recompute: how many RV observations feed the LAST percentile value
            df, _, _ = fake_fetch("SPY", years=years, extra_days=int(rv_w * 1.6) + 15)
            n_bars = len(df)
            n_rv = n_bars - rv_w   # log returns: n_bars-1, rolling(rv_w, min_periods=rv_w): n_bars-1-rv_w+1
            obs = min(n_rv, pw)
            flag = "" if obs >= pw else f"  <-- only {obs}/{pw} obs ({obs/pw:.0%})"
            print(f"{years:5.1f} {rv_w:4d} {pw:7d} {n_bars:5d} {n_rv:6d} {obs:22d} 'Percentile ({pw} j)'{flag}")
