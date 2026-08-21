"""p4 probe: regime-split-out-of-range frequency on REAL AAPL history for the
user-reachable parameter corners (duration 1y/2y, rv 5..120, fwd 5..90).
Reports how often slope>=1, intersection out of [min,max], empty regime."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, WORKTREE)
from app.model.iv_dashboard import analytics as ivx  # noqa: E402

cands = glob.glob(os.path.join(WORKTREE, "cache", "**", "stooq_aapl*.csv"), recursive=True)
df = pd.read_csv(cands[0])
dcol = next(c for c in df.columns if c.lower() == "date")
ccol = next(c for c in df.columns if c.lower() == "close")
closes_all = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values,
                       index=pd.to_datetime(df[dcol]).dt.normalize()).dropna().sort_index()

combos = [
    # (years_td, rv, fwd, pct)
    (504, 20, 30, 252),   # defaults, 2y
    (252, 20, 30, 252),   # 1y
    (252, 5, 5, 60),      # noisiest
    (252, 120, 90, 252),  # smoothest (max rv, max fwd), 1y
    (504, 120, 90, 252),  # smoothest, 2y
    (252, 60, 90, 252),
    (252, 120, 30, 252),
    (1260, 20, 30, 252),  # 5y default windows
]
print(f"{'n_td':>5} {'rv':>4} {'fwd':>4} | {'win':>4} {'err':>4} {'sl>=1':>6} {'out':>5} {'empty':>5} {'dupHi':>5} | slope[min,max]            inter[min,max]         min(nHi,nLo)")
for n_td, rv_w, fwd_w, pct_w in combos:
    warm = int(rv_w * 1.6) + 15
    rows = []
    errs = 0
    step = 42
    for end in range(n_td + warm, len(closes_all) + 1, step):
        win = closes_all.iloc[end - n_td - warm:end]
        rv = ivx.compute_realized_vol(win, rv_w)
        pct = ivx.compute_percentile_series(rv, pct_w)
        d = pd.DataFrame({"close": win, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"]).iloc[-n_td:]
        try:
            a = ivx.analyze_forward_vol(d["vol"], forward_window=fwd_w, percentile=d["vol_percentile"])
        except ValueError:
            errs += 1
            continue
        lo, hi = float(a["df"]["current_vol"].min()), float(a["df"]["current_vol"].max())
        inter = a["intersection"]
        dup = (a["reg_high"] is not None and abs(a["reg_high"]["slope"] - a["reg_diff"]["slope"]) < 1e-12)
        rows.append((a["reg_forward"]["slope"], inter, not (lo < inter < hi),
                     a["n_high"] == 0 or a["n_low"] == 0, dup, min(a["n_high"], a["n_low"])))
    r = pd.DataFrame(rows, columns=["slope", "inter", "out", "empty_reg", "dup", "minn"])
    print(f"{n_td:>5} {rv_w:>4} {fwd_w:>4} | {len(r):>4} {errs:>4} {(r.slope>=1).sum():>6} {r.out.sum():>5} {r.empty_reg.sum():>5} {r.dup.sum():>5} | "
          f"[{r.slope.min():.3f},{r.slope.max():.3f}]  [{r.inter.min():.3f},{r.inter.max():.3f}]   {r.minn.min()}")
    if r.out.any():
        bad = r[r.out]
        print("     out-of-range cases: slope=", np.round(bad.slope.values, 3)[:10], " inter=", np.round(bad.inter.values, 3)[:10])
