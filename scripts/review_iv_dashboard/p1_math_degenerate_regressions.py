"""Probe: degenerate inputs to _linreg / analyze_forward_vol (no network)."""
import sys, warnings
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from scipy import stats
from app.model.iv_dashboard import analytics as A


def run(name, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
            print(f"[{name}] OK ->", out)
        except Exception as e:
            print(f"[{name}] RAISE {type(e).__name__}: {e}")
        for x in w:
            print(f"   warning: {x.category.__name__}: {x.message}")


idx = pd.bdate_range("2024-01-01", periods=400)

# 1. constant vol series -> all x identical
const = pd.Series(0.2, index=idx)
run("constant-vol analyze_forward_vol", lambda: {k: v for k, v in A.analyze_forward_vol(const).items() if k != "df"})

# 2. raw linregress behaviours
run("linregress 2 pts", lambda: stats.linregress([1.0, 2.0], [1.0, 2.0])._asdict())
run("linregress 1 pt", lambda: stats.linregress([1.0], [1.0])._asdict())
run("linregress NaN", lambda: stats.linregress([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])._asdict())
run("linregress inf", lambda: stats.linregress([1.0, 2.0, np.inf], [1.0, 2.0, 3.0])._asdict())
run("linregress identical x", lambda: stats.linregress([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])._asdict())
run("linregress identical y", lambda: stats.linregress([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])._asdict())

# 3. near-constant: halted name -> closes with few distinct values.
rng = np.random.default_rng(0)
closes = np.concatenate([np.full(300, 1.00), 1.00 * np.exp(np.cumsum(rng.normal(0, 0.02, 100)))])
px = pd.Series(closes, index=idx)
rv = A.compute_realized_vol(px, 20)
print("RV distinct values in flat part:", rv.iloc[:280].dropna().nunique(), " n zeros:", int((rv == 0).sum()))
run("halted-then-noisy analyze_forward_vol", lambda: {k: v for k, v in A.analyze_forward_vol(rv.dropna()).items() if k not in ("df",)})

# 4. regime subset with > 10 identical x but overall non-degenerate
closes2 = np.concatenate([np.full(200, 5.0), 5.0 * np.exp(np.cumsum(rng.normal(0, 0.03, 60)))])
px2 = pd.Series(closes2, index=idx[: len(closes2)])
rv2 = A.compute_realized_vol(px2, 20).dropna()
print("rv2 zeros:", int((rv2 == 0).sum()), "nonzero:", int((rv2 != 0).sum()))
run("flat-then-noisy (low regime all x=0)", lambda: {k: v for k, v in A.analyze_forward_vol(rv2).items() if k not in ("df",)})

# 5. two distinct closes only (tick-size stock): alternate 1.00 / 1.01
closes3 = np.where(np.arange(400) % 2 == 0, 1.00, 1.01)
rv3 = A.compute_realized_vol(pd.Series(closes3, index=idx), 20).dropna()
print("rv3 nunique:", rv3.nunique(), "values:", rv3.unique()[:5])
run("two-close-alternating", lambda: {k: v for k, v in A.analyze_forward_vol(rv3).items() if k not in ("df",)})

# 6. regime subset: identical x in HIGH regime but >10 pts; low regime varied
# Build vol series: 40 noisy low values around 0.1, then 15 identical at 0.5, then 40 noisy low
v6 = np.concatenate([rng.uniform(0.08, 0.12, 60), np.full(15, 0.5), rng.uniform(0.08, 0.12, 60)])
s6 = pd.Series(v6, index=idx[: len(v6)])
run("high regime identical x (15 pts of 0.5)", lambda: {k: v for k, v in A.analyze_forward_vol(s6, forward_window=5).items() if k not in ("df",)})

# 7. nan slope leak: regression where y constant (vol_diff == 0 everywhere is impossible unless const)
# but p-value with n=2 regime? MIN_REGIME_POINTS > 10 so n>=11. Check p-value nan cases with ties:
v7 = np.concatenate([rng.uniform(0.08, 0.12, 40), np.full(12, 0.5), rng.uniform(0.08, 0.12, 40)])
s7 = pd.Series(v7, index=idx[: len(v7)])
run("12 identical high pts", lambda: {k: v for k, v in A.analyze_forward_vol(s7, forward_window=5).items() if k not in ("df",)})
