"""
Orchestrator probe — §4.1 mathematics of app/model/iv_dashboard/analytics.py.

Independent oracle (not the agents' scripts). Offline, deterministic.
Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/orch_probe_math.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.iv_dashboard import analytics as an  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
section("A. linregress on a constant series -> which exception, caught where?")
idx = pd.bdate_range("2025-01-01", periods=120)
const_vol = pd.Series(0.20, index=idx)
try:
    an.analyze_forward_vol(const_vol, forward_window=30)
    print("no exception (unexpected)")
except Exception as exc:  # noqa: BLE001
    print(f"type={type(exc).__name__} msg={exc!s}")
    print("caught by service.py L586 'except ValueError'?", isinstance(exc, ValueError))

# Near-constant: penny stock with 2 distinct closes -> RV has a handful of values
section("B. near-constant RV: closes alternating 1.00/1.01 for 300 days")
closes = pd.Series(np.where(np.arange(300) % 7 == 0, 1.01, 1.00), index=pd.bdate_range("2024-01-01", periods=300))
rv = an.compute_realized_vol(closes, 20)
print("distinct RV values:", rv.dropna().round(6).nunique(), "| n:", rv.dropna().size)
try:
    res = an.analyze_forward_vol(rv, forward_window=30)
    print("OK slope=", round(res["reg_forward"]["slope"], 4), "n_high/n_low=", res["n_high"], res["n_low"],
          "reg_high:", None if res["reg_high"] is None else round(res["reg_high"]["slope"], 4),
          "reg_low:", None if res["reg_low"] is None else round(res["reg_low"]["slope"], 4))
except Exception as exc:  # noqa: BLE001
    print(f"type={type(exc).__name__} msg={exc!s}")

# Regime subset with all-identical x but > 10 points (outer regression fine)
section("C. regime subset identical x (>10 pts) -> raises inside after outer check?")
vals = np.concatenate([np.full(40, 0.10), np.linspace(0.11, 0.40, 60)])
rng = np.random.default_rng(0)
v = pd.Series(vals, index=pd.bdate_range("2025-01-01", periods=100))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        res = an.analyze_forward_vol(v, forward_window=5)
        print("OK intersection=", round(res["intersection"], 4), "n_high/n_low=", res["n_high"], res["n_low"],
              "reg_low=", res["reg_low"] and {k: round(val, 4) if isinstance(val, float) else val for k, val in res["reg_low"].items()})
    except Exception as exc:  # noqa: BLE001
        print(f"type={type(exc).__name__} msg={exc!s}")
    print("warnings:", [str(x.message)[:80] for x in w])

# 2 distinct values in the whole series (x has two levels) -> regime subset constant
section("D. two-level series (0.10 x 60, 0.30 x 60) -> per-regime regression on constant x")
v2 = pd.Series(np.r_[np.full(60, 0.10), np.full(60, 0.30)], index=pd.bdate_range("2025-01-01", periods=120))
try:
    res = an.analyze_forward_vol(v2, forward_window=10)
    print("OK n_high/n_low=", res["n_high"], res["n_low"], "reg_high=", res["reg_high"], "reg_low=", res["reg_low"])
except Exception as exc:  # noqa: BLE001
    print(f"type={type(exc).__name__} msg={exc!s}")

# --------------------------------------------------------------------------- #
section("E. forward construction: which rows are partial means?")
v = pd.Series(np.arange(1.0, 51.0), index=pd.bdate_range("2025-01-01", periods=50))
fw = 10
forward = v.rolling(window=fw, min_periods=1).mean().shift(-fw)
full_mean = v.rolling(window=fw, min_periods=fw).mean().shift(-fw)
partial_rows = forward.index[(forward.notna()) & (full_mean.isna())]
print("n rows with forward defined:", forward.notna().sum(), "| rows where forward is a PARTIAL mean:", len(partial_rows))
print("positions of partial rows:", [int(v.index.get_loc(i)) for i in partial_rows])
print("forward[0] uses v[1..10]? value=", forward.iloc[0], "expected mean(v[1..10])=", v.iloc[1:11].mean(),
      "| but rolling-mean-at-t+fw spans v[t+1..t+fw] => includes v[t+fw]; expected (if defined as mean over next fw days) =", v.iloc[1:11].mean())
print("NOTE: forward[t] = mean(v[t+1 .. t+fw]) — rolling(fw).mean() at t+fw covers [t+1, t+fw]; so partial means happen only when t+fw < fw-1, i.e. never for t>=0 -> check:", len(partial_rows))

# --------------------------------------------------------------------------- #
section("F. percentile comparability: rolling rank(pct) vs percentile_within")
rng = np.random.default_rng(1)
hist = pd.Series(rng.lognormal(np.log(0.15), 0.3, 252))
pct_roll = an.compute_percentile_series(hist, 252).iloc[-1]
pct_within_incl = an.percentile_within(hist, hist.iloc[-1])
pct_within_excl = an.percentile_within(hist.iloc[:-1], hist.iloc[-1])
print(f"rolling rank(pct) last={pct_roll:.4f} | percentile_within(incl self)={pct_within_incl:.4f} | excl self={pct_within_excl:.4f}")
# Ties
ties = pd.Series([0.1] * 251 + [0.1])
print("all-ties: rolling rank(pct)=", round(an.compute_percentile_series(ties, 252).iloc[-1], 4),
      "| percentile_within=", round(an.percentile_within(ties, 0.1), 4))

# --------------------------------------------------------------------------- #
section("G. epistemics: SPY-like RV 12-18% vs IV 17% -> what the tab shows")
rv_hist = pd.Series(rng.uniform(0.12, 0.18, 252))
p = an.percentile_within(rv_hist, 0.17)
print(f"IV=17% inside RV[12..18%] -> percentile={p:.3f} regime={an.classify_regime(p)['label']} signal={an.classify_regime(p)['signal_label']}")
p2 = an.percentile_within(rv_hist, 0.19)
print(f"IV=19% -> percentile={p2:.3f} regime={an.classify_regime(p2)['label']} signal={an.classify_regime(p2)['signal_label']}")

# --------------------------------------------------------------------------- #
section("H. min_periods=60 relaxation: percentile on day 61 of a 1-year history")
short = pd.Series(np.linspace(0.10, 0.30, 61))
print("percentile day 61 (monotonic rising) =", round(an.compute_percentile_series(short, 252).iloc[-1], 4),
      "| day 60 =", round(an.compute_percentile_series(short, 252).iloc[-2], 4),
      "| day 59 NaN? ", bool(np.isnan(an.compute_percentile_series(short, 252).iloc[-3])))

# --------------------------------------------------------------------------- #
section("I. intersection outside data range")
v3 = pd.Series(np.linspace(0.10, 0.20, 80) + rng.normal(0, 0.001, 80), index=pd.bdate_range("2025-01-01", periods=80))
res = an.analyze_forward_vol(v3, forward_window=10)
print("intersection=", round(res["intersection"], 4), "data range=", round(v3.min(), 4), round(v3.max(), 4),
      "n_high/n_low=", res["n_high"], res["n_low"], "reg_high None?", res["reg_high"] is None, "reg_low None?", res["reg_low"] is None)
