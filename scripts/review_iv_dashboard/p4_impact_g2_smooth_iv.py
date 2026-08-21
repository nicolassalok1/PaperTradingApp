"""Follow-up: smoother IV proxies (IV anchored to longer-run RV, not RV20_t) on real AAPL closes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model.iv_dashboard import analytics as ivx  # noqa: E402

raw = pd.read_csv(ROOT / "cache" / "OHLC" / "stooq_aapl.us_start_end_d.csv", parse_dates=["date"]).sort_values("date")
closes = raw.set_index("date")["close"].astype(float)
closes = closes[closes.index >= "2015-01-01"]
rv = ivx.compute_realized_vol(closes, 20).dropna()
m63 = rv.rolling(63).mean()
m126 = rv.rolling(126).mean()
m252 = rv.rolling(252).mean()
PWIN = 252
idx = rv.index[PWIN + 126:]
trail = {t: rv.loc[:t].tail(PWIN) for t in idx}
calm = [t for t in idx if float(rv.loc[t]) <= float(trail[t].median())]


def rates(iv_series):
    sig = {t: ivx.classify_regime(ivx.percentile_within(trail[t], float(iv_series.loc[t])))["signal_key"] for t in idx}
    d_all = np.mean([sig[t] == "down" for t in idx])
    d_calm = np.mean([sig[t] == "down" for t in calm])
    u_all = np.mean([sig[t] == "up" for t in idx])
    return d_all, d_calm, u_all


print(f"{'IV proxy':<40} {'MR↓ all':>8} {'MR↓ calm':>9} {'MR↑ all':>8}")
for lbl, s in [
    ("RV20 (baseline)", rv),
    ("mean63(RV20) x1.10", m63 * 1.10),
    ("mean126(RV20) x1.15", m126 * 1.15),
    ("mean252(RV20) x1.15", m252 * 1.15),
    ("max(RV20, mean63) + 3pts", np.maximum(rv, m63) + 0.03),
    ("0.5*RV20 + 0.5*mean126 + 3pts", 0.5 * rv + 0.5 * m126 + 0.03),
    ("0.3*RV20 + 0.7*mean252 + 4pts", 0.3 * rv + 0.7 * m252 + 0.04),
]:
    d, dc, u = rates(s)
    print(f"{lbl:<40} {d:>8.1%} {dc:>9.1%} {u:>8.1%}")
print(f"n days={len(idx)}, calm days={len(calm)}")
