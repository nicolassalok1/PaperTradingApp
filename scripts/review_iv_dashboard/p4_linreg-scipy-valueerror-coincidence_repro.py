"""p4 repro — linreg-scipy-valueerror-coincidence (offline, deterministic).

Independent check of:
  (a) constant closes -> RV == 0 everywhere -> analyze_forward_vol raises the *scipy*
      ValueError (English) which service.py L586 catches as "analysis impossible";
  (b) closes with <=2 distinct RV levels -> global regression OK but the per-regime
      regression has identical x -> ValueError -> the whole analysis is lost;
  (c) how reachable is (b) with "realistic-ish" illiquid data: long flat stretches
      (RV==0 windows) mixed with noisy stretches -> does the LOW regime become all-zero ?
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy import stats
from app.model.iv_dashboard import analytics as A

idx = pd.bdate_range("2024-01-02", periods=400)

def run_analysis(closes: pd.Series, label: str):
    rv = A.compute_realized_vol(closes, 20)
    pct = A.compute_percentile_series(rv, 252)
    sdf = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"])
    print(f"\n[{label}] n_rv={len(sdf)} rv_nunique={sdf['vol'].nunique()} rv_levels={sorted(sdf['vol'].round(6).unique())[:5]}")
    try:
        res = A.analyze_forward_vol(sdf["vol"], forward_window=30, percentile=sdf["vol_percentile"])
        print("   analysis OK: slope1=%.4f inter=%.4f n_high=%d n_low=%d reg_high=%s reg_low=%s" % (
            res["reg_forward"]["slope"], res["intersection"], res["n_high"], res["n_low"],
            res["reg_high"] is not None, res["reg_low"] is not None))
        return "ok"
    except ValueError as exc:
        print(f"   ValueError caught (same clause as service.py L586): {exc!r}")
        return str(exc)

# (a) constant closes
closes_const = pd.Series(10.0, index=idx)
run_analysis(closes_const, "a: constant closes")
# percentile / regime display with all-zero vol (what the service computes)
rv = A.compute_realized_vol(closes_const, 20); pct = A.compute_percentile_series(rv, 252)
print("   current_vol=%r current_pct=%r regime=%s" % (float(rv.iloc[-1]), float(pct.iloc[-1]), A.classify_regime(float(pct.iloc[-1]))["label"]))

# (b) closes alternating 1.00/1.01 every 7 days
vals = np.where((np.arange(400) // 7) % 2 == 0, 1.00, 1.01)
run_analysis(pd.Series(vals, index=idx), "b: 1.00/1.01 alternating every 7d")

# (b') alternating every day (10.00 / 10.01) -> nunique of RV ?
vals2 = np.where(np.arange(400) % 2 == 0, 10.00, 10.01)
run_analysis(pd.Series(vals2, index=idx), "b': 10.00/10.01 alternating daily")

# (c) illiquid stock: flat stretches (identical closes >= 20d) mixed with noisy stretches
hits = 0; total = 0; details = []
for seed in range(200):
    rng = np.random.default_rng(seed)
    n = 400
    px = np.empty(n); px[0] = 5.0
    i = 1
    while i < n:
        if rng.random() < 0.5:
            L = int(rng.integers(20, 60)); px[i:i+L] = px[i-1]; i += L
        else:
            L = int(rng.integers(10, 60)); seg = px[i-1]*np.exp(np.cumsum(rng.normal(0, 0.02, L))); px[i:i+L] = seg[:n-i]; i += L
    closes = pd.Series(px[:n], index=idx)
    rv = A.compute_realized_vol(closes, 20)
    sdf = pd.DataFrame({"vol": rv}).dropna()
    if len(sdf) < 30: continue
    total += 1
    try:
        A.analyze_forward_vol(sdf["vol"], forward_window=30)
    except ValueError as exc:
        if "identical" in str(exc):
            hits += 1
            if len(details) < 3:
                details.append((seed, str(exc), int((sdf['vol']==0).sum()), len(sdf)))
print(f"\n[c] illiquid-stock sims: {hits}/{total} runs raise the scipy 'identical x' ValueError (whole analysis lost)")
for d in details: print("   e.g.", d)

# sanity: does scipy 1.11.4 really raise for identical x?
try:
    stats.linregress(np.zeros(12), np.arange(12.0))
except ValueError as exc:
    print("\nscipy.linregress identical-x ->", repr(exc))
