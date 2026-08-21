"""p4 IMPACT probe (G1_math) -- real cached AAPL history (Stooq, ~10k rows, no network).

S1  regime-split-out-of-range : over sliding 2-year windows (default params
    rv=20, fwd=30, pct=252), how often is slope>=1 / intersection outside
    [min,max] of current_vol / a regime empty / reg_high == reg_diff ?
S2  rv-bad-close-silent-drop  : inject ONE zero close in a real 2y window ->
    how many RV values move, by how much (vol points), and what would the two
    candidate fixes do (NaN-hole fix from the finding vs filter-first) ?
S3  linreg-scipy-valueerror-coincidence : confirm scipy 1.11.4 raises (not
    warns) on identical x, and that a REAL 2y window never gets near
    nunique<2 at the global or per-regime level.
S4  duplicate-date-crash : confirm the exact exception + that dedup keep='last'
    before the DataFrame build is sufficient.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, WORKTREE)

from app.model.iv_dashboard import analytics as ivx  # noqa: E402

RV_W, FWD_W, PCT_W = 20, 30, 252
TWO_Y = 504  # trading days


def load_aapl() -> pd.Series:
    cands = glob.glob(os.path.join(WORKTREE, "cache", "**", "stooq_aapl*.csv"), recursive=True)
    cands += glob.glob(os.path.join(WORKTREE, "..", "..", "..", "cache", "**", "stooq_aapl*.csv"), recursive=True)
    df = pd.read_csv(cands[0])
    dcol = next(c for c in df.columns if c.lower() == "date")
    ccol = next(c for c in df.columns if c.lower() == "close")
    s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol]).dt.normalize())
    return s.dropna().sort_index()


def run_pipeline(closes: pd.Series):
    rv = ivx.compute_realized_vol(closes, RV_W)
    pct = ivx.compute_percentile_series(rv, PCT_W)
    df = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"])
    df = df.iloc[-TWO_Y:]
    return df, ivx.analyze_forward_vol(df["vol"], forward_window=FWD_W, percentile=df["vol_percentile"])


closes_all = load_aapl()
print(f"AAPL stooq rows={len(closes_all)} {closes_all.index.min().date()} -> {closes_all.index.max().date()}")

# ----------------------------------------------------------------------- S1
print("\n=== S1 regime-split degeneracy on sliding real 2y windows (step 21 td) ===")
warm = RV_W + 15
step = 21
stats_rows = []
for end in range(TWO_Y + warm, len(closes_all) + 1, step):
    win = closes_all.iloc[end - TWO_Y - warm:end]
    try:
        df, a = run_pipeline(win)
    except ValueError as exc:
        stats_rows.append(dict(end=win.index[-1], err=str(exc)))
        continue
    lo, hi = float(df["vol"].min()), float(df["vol"].max())
    inter = a["intersection"]
    s1 = a["reg_forward"]["slope"]
    same = (a["reg_high"] is not None
            and abs(a["reg_high"]["slope"] - a["reg_diff"]["slope"]) < 1e-12
            and abs(a["reg_high"]["intercept"] - a["reg_diff"]["intercept"]) < 1e-12)
    stats_rows.append(dict(end=win.index[-1], slope=s1, inter=inter, lo=lo, hi=hi,
                           out=not (lo < inter < hi), n_high=a["n_high"], n_low=a["n_low"],
                           regime_empty=(a["n_high"] == 0 or a["n_low"] == 0), dup=same,
                           nuniq=int(df["vol"].nunique()), n=len(df)))
st = pd.DataFrame(stats_rows)
ok = st[st.get("err").isna()] if "err" in st else st
print(f"windows={len(st)} errors={len(st)-len(ok)}")
print(f"slope>=1          : {(ok.slope >= 1).sum():4d} / {len(ok)}  ({(ok.slope >= 1).mean():.1%})")
print(f"intersection out  : {ok.out.sum():4d} / {len(ok)}  ({ok.out.mean():.1%})")
print(f"a regime empty    : {ok.regime_empty.sum():4d}")
print(f"reg_high==reg_diff: {ok.dup.sum():4d}")
print(f"slope range [{ok.slope.min():.3f}, {ok.slope.max():.3f}]  "
      f"intersection range [{ok.inter.min():.3f}, {ok.inter.max():.3f}]")
print(f"min nunique(vol) over windows = {ok.nuniq.min()} (n={ok.n.min()}..{ok.n.max()})")
near1 = ok[(ok.slope - 1).abs() < 0.05]
print(f"windows with |slope-1|<0.05 : {len(near1)} ; of which intersection out of range: {int(near1.out.sum())}")
if ok.out.any():
    print(ok[ok.out][["end", "slope", "inter", "lo", "hi", "n_high", "n_low"]].to_string(index=False))
# how far inside the range does the split land, typically? (is it a 'sane' split)
frac = ((ok.inter - ok.lo) / (ok.hi - ok.lo)).clip(-1, 2)
print(f"split position as fraction of [lo,hi]: q10={frac.quantile(.1):.2f} median={frac.median():.2f} q90={frac.quantile(.9):.2f}")
print(f"min(n_high, n_low) : min={ok[['n_high','n_low']].min(axis=1).min()}  q10={ok[['n_high','n_low']].min(axis=1).quantile(.1):.0f}")

# ----------------------------------------------------------------------- S2
print("\n=== S2 one zero close injected into the latest real 2y window ===")
win = closes_all.iloc[-(TWO_Y + warm):].copy()
base_rv = ivx.compute_realized_vol(win, RV_W).dropna()
bad = win.copy()
j = len(bad) // 2
bad.iloc[j] = 0.0
cur_rv = ivx.compute_realized_vol(bad, RV_W).dropna()
common = base_rv.index.intersection(cur_rv.index)
delta = (cur_rv.loc[common] - base_rv.loc[common]).abs()
print(f"CURRENT code : rv len {len(base_rv)} -> {len(cur_rv)} (dates lost={len(base_rv)-len(cur_rv)}), "
      f"values changed={(delta > 1e-12).sum()}, max |dRV|={delta.max()*100:.2f} vol pts, "
      f"RV around hole is NaN? {bool(cur_rv.reindex(base_rv.index).isna().sum() > 2)}")
print(f"  lost return t-1->t+1 : {np.log(win.iloc[j+1]/win.iloc[j-1]):+.4%} (real 2-day move at that date)")

# finding's fix: keep alignment, NaN hole, min_periods=window
px = bad.where(bad > 0)
rets_nan = np.log(px / px.shift(1)).iloc[1:]
rv_nanfix = (rets_nan.rolling(RV_W, min_periods=RV_W).std() * np.sqrt(252.0))
rv_nanfix_valid = rv_nanfix.iloc[RV_W - 1:]
print(f"FINDING fix  : NaN RV days created around the hole = {int(rv_nanfix_valid.isna().sum())} "
      f"(one bad row blanks {RV_W+1} RV days; series_df.dropna then drops them)")

# alternative: filter-first (keep t-1 -> t+1 as a 2-day return)
px2 = bad[bad > 0]
rets_ff = np.log(px2 / px2.shift(1)).dropna()
rv_ff = rets_ff.rolling(RV_W, min_periods=RV_W).std() * np.sqrt(252.0)
rv_ff = rv_ff.dropna()
common2 = base_rv.index.intersection(rv_ff.index)
d2 = (rv_ff.loc[common2] - base_rv.loc[common2]).abs()
print(f"FILTER-FIRST : rv len {len(rv_ff)} (dates lost={len(base_rv)-len(rv_ff)}), "
      f"values changed={(d2 > 1e-12).sum()}, max |dRV|={d2.max()*100:.2f} vol pts (2-day return kept, no hole)")

# ----------------------------------------------------------------------- S3
print("\n=== S3 scipy identical-x behaviour + real-data distance from the trigger ===")
from scipy import stats  # noqa: E402
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats.linregress(np.ones(12), np.arange(12.0))
    print("identical x -> NO exception (finding would be wrong)")
except Exception as exc:  # noqa: BLE001
    print(f"identical x -> {type(exc).__name__}: {exc}")
df, a = run_pipeline(closes_all.iloc[-(TWO_Y + warm):])
for key, mask in (("high", df.iloc[-len(a['df']):]["vol"] > a["intersection"]),):
    pass
hm = a["df"]["current_vol"] > a["intersection"]
print(f"latest real 2y: nunique(vol)={a['df']['current_vol'].nunique()} / n={len(a['df'])}, "
      f"nunique high regime={a['df'].loc[hm,'current_vol'].nunique()}, low={a['df'].loc[~hm,'current_vol'].nunique()}")
print(f"  min nunique over all {len(ok)} sliding windows = {ok.nuniq.min()}  (trigger needs <2 globally, or a regime of >10 identical values)")

# ----------------------------------------------------------------------- S4
print("\n=== S4 duplicate date: exact failure + dedup sufficiency ===")
dupd = win.copy()
dupd = pd.concat([dupd, dupd.iloc[[100]]]).sort_index()
rv_d = ivx.compute_realized_vol(dupd, RV_W)
pct_d = ivx.compute_percentile_series(rv_d, PCT_W)
try:
    pd.DataFrame({"close": dupd, "vol": rv_d, "vol_percentile": pct_d})
    print("no exception (finding wrong)")
except Exception as exc:  # noqa: BLE001
    print(f"DataFrame build -> {type(exc).__name__}: {exc}")
print(f"  log-return injected at the duplicate: {float(np.log(dupd/dupd.shift(1)).iloc[101]):.6f} (artificial 0)")
dd = dupd[~dupd.index.duplicated(keep='last')]
rv_dd = ivx.compute_realized_vol(dd, RV_W)
s = pd.DataFrame({"close": dd, "vol": rv_dd, "vol_percentile": ivx.compute_percentile_series(rv_dd, PCT_W)})
print(f"  after dedup keep='last': build OK, len={len(s)}, RV identical to clean: "
      f"{np.allclose(rv_dd.dropna().values, base_rv.values)}")
