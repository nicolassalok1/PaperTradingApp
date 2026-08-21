"""p4 repro — rv-bad-close-silent-drop (offline, deterministic).

One close <= 0 in the middle of a 300-day series. Compare compute_realized_vol on
the clean series vs the corrupted one:
  - how many RV rows vanish, which dates,
  - how many RV values change and by how much (vol points),
  - what the service layer would show (series_df / log) -> any trace of the bad row ?
Also measure the alternative "NaN-propagating" variant sketched in the fix.
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A

idx = pd.bdate_range("2024-01-02", periods=300)
rng = np.random.default_rng(42)
clean = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)

for bad_val in (0.0, -5.0, np.nan, "n/a"):
    bad = clean.copy().astype(object) if isinstance(bad_val, str) else clean.copy()
    t = 150
    bad.iloc[t] = bad_val
    rv_clean = A.compute_realized_vol(clean, 20)
    rv_bad = A.compute_realized_vol(bad, 20)
    missing = rv_clean.index.difference(rv_bad.index)
    common = rv_clean.dropna().index.intersection(rv_bad.dropna().index)
    diff = (rv_bad.loc[common] - rv_clean.loc[common]).abs()
    changed = diff[diff > 1e-12]
    print(f"\n[bad close = {bad_val!r} at {idx[t].date()}]")
    print(f"  len(rv): clean={len(rv_clean)} bad={len(rv_bad)}  missing dates={list(missing.strftime('%Y-%m-%d'))}")
    print(f"  NaN count in rv_bad (beyond warm-up): {int(rv_bad.iloc[20:].isna().sum())}")
    print(f"  RV values changed: {len(changed)}  max |diff| = {changed.max()*100:.2f} vol pts  "
          f"(clean RV level ~ {rv_clean.mean()*100:.1f}%)  first/last changed: {changed.index.min().date()} .. {changed.index.max().date()}")
    # rows per window: what the 'bad' RV at the first changed date actually spans
    d0 = changed.index.min()
    pos = clean.index.get_loc(d0)
    print(f"  at {d0.date()}: window of 20 returns spans closes {clean.index[pos-21].date()} .. {d0.date()} (22 calendar rows) vs 21 for clean")

# fix-sketch variant: keep alignment, NaN-propagate
def rv_nan_variant(closes, window=20):
    px = pd.to_numeric(closes, errors="coerce").astype(float)
    px = px.where(px > 0)
    rets = np.log(px / px.shift(1)).iloc[1:]
    return rets.rolling(window=window, min_periods=window).std() * np.sqrt(252.0)

bad = clean.copy(); bad.iloc[150] = 0.0
rv_v = rv_nan_variant(bad)
print(f"\n[fix variant NaN-propagating] NaN rows beyond warm-up: {int(rv_v.iloc[20:].isna().sum())} "
      f"(vs 0 now, but 2 dates dropped + 19 biased)")
# note what analyze_forward_vol does with the gap: v.dropna() -> forward window positional over the gap
v = rv_v.dropna()
print(f"  analyze_forward_vol on the variant: v.dropna() -> {len(v)} rows, gap skipped positionally by rolling/shift(-fw)")

# does the service leave any trace? (service calls compute_realized_vol then dropna(subset=['vol']))
print("\n[service trace] get_iv_dashboard_data logs only 'N barres daily reçues' + current RV; no count of closes <= 0 (by inspection of service.py L523-545).")
