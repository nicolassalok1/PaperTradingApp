"""p4 repro — regime-split-out-of-range (offline, deterministic).

Checks:
  (1) halt 300d + 100d noisy closes -> intersection = intercept/(1-slope) outside
      [min, max] of current_vol, one regime empty, reg_high == reg_diff.
  (2) Is it reachable WITHOUT a halt ? Smoothly trending vol (log-vol drifting up
      over the sample) gives slope ~ 1 -> intersection far outside the data range.
  (3) Guard |1-slope| > 1e-12 : does it avoid the explosion near slope ~ 1 ?
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A

def report(res, label):
    df = res["df"]; lo, hi = df["current_vol"].min(), df["current_vol"].max()
    same = res["reg_high"] is not None and all(
        abs(res["reg_high"][k] - res["reg_diff"][k]) < 1e-12 for k in ("slope", "intercept", "r2"))
    same_low = res["reg_low"] is not None and all(
        abs(res["reg_low"][k] - res["reg_diff"][k]) < 1e-12 for k in ("slope", "intercept", "r2"))
    print(f"[{label}] slope1={res['reg_forward']['slope']:.4f} intercept={res['reg_forward']['intercept']:.4f} "
          f"intersection={res['intersection']:.4f} range=[{lo:.4f},{hi:.4f}] in_range={lo < res['intersection'] < hi} "
          f"n_high={res['n_high']} n_low={res['n_low']} reg_high_is_reg_diff={same} reg_low_is_reg_diff={same_low}")

# (1) halt + noisy
idx = pd.bdate_range("2023-01-02", periods=400)
rng = np.random.default_rng(1)
px = np.concatenate([np.full(300, 50.0), 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))])
rv = A.compute_realized_vol(pd.Series(px, index=idx), 20).dropna()
report(A.analyze_forward_vol(rv, forward_window=30), "1: halt300+noisy100")

# (2) smoothly trending vol, no halt: vol series directly (what analyze_forward_vol consumes)
n = 500
t = np.arange(n)
for drift, label in ((0.0020, "2a: log-vol drift up"), (-0.0020, "2b: log-vol drift down")):
    rng = np.random.default_rng(7)
    logv = np.log(0.15) + drift * t + np.cumsum(rng.normal(0, 0.01, n))
    vol = pd.Series(np.exp(logv), index=pd.bdate_range("2023-01-02", periods=n))
    report(A.analyze_forward_vol(vol, forward_window=30), label)

# (2c) realistic RV from a price path with a slowly rising vol (GARCH-ish persistent trend)
rng = np.random.default_rng(3)
sig = 0.006 * np.exp(0.0025 * np.arange(520))  # daily sigma 0.6% -> 2.2% over 2y
px = 100 * np.exp(np.cumsum(rng.normal(0, sig)))
rv = A.compute_realized_vol(pd.Series(px, index=pd.bdate_range("2023-01-02", periods=520)), 20).dropna()
report(A.analyze_forward_vol(rv, forward_window=30), "2c: RV from price path with rising sigma")

# (2d) scan: how often is intersection out of range on 200 OU mean-reverting log-vol paths (finder's claim: 0)
out = 0
for seed in range(200):
    rng = np.random.default_rng(seed)
    x = np.empty(500); x[0] = np.log(0.2)
    for i in range(1, 500):
        x[i] = x[i-1] + 0.02 * (np.log(0.2) - x[i-1]) + rng.normal(0, 0.05)
    vol = pd.Series(np.exp(x), index=pd.bdate_range("2023-01-02", periods=500))
    res = A.analyze_forward_vol(vol, forward_window=30)
    lo, hi = res["df"]["current_vol"].min(), res["df"]["current_vol"].max()
    if not (lo < res["intersection"] < hi): out += 1
print(f"[2d] OU log-vol (theta=0.02): {out}/200 out-of-range intersections")
# weaker mean reversion (theta=0.002) -> closer to random walk -> slope -> 1
out = 0; empty = 0
for seed in range(200):
    rng = np.random.default_rng(seed)
    x = np.empty(500); x[0] = np.log(0.2)
    for i in range(1, 500):
        x[i] = x[i-1] + 0.002 * (np.log(0.2) - x[i-1]) + rng.normal(0, 0.03)
    vol = pd.Series(np.exp(x), index=pd.bdate_range("2023-01-02", periods=500))
    res = A.analyze_forward_vol(vol, forward_window=30)
    lo, hi = res["df"]["current_vol"].min(), res["df"]["current_vol"].max()
    if not (lo < res["intersection"] < hi): out += 1
    if res["n_low"] == 0 or res["n_high"] == 0: empty += 1
print(f"[2e] near-random-walk log-vol (theta=0.002): {out}/200 out-of-range intersections, {empty}/200 with an empty regime")

# (3) guard vs slope ~ 1 : intercept 0.01, slope 1 - 1e-6 -> intersection 1e4
slope = 1 - 1e-6; intercept = 0.01
print(f"[3] guard |1-slope|>1e-12 with slope={slope}: passes -> intersection={intercept/(1-slope):.1f}")
