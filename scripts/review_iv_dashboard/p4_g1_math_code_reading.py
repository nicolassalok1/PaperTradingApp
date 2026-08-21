"""Phase-4 skeptic probe (code-reading lens) for group G1_math.

Re-measures the four findings against the pinned libs, without network.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.model.iv_dashboard import analytics  # noqa: E402

print("versions:", np.__version__, pd.__version__)
dates = pd.bdate_range("2024-01-01", periods=300)


def sep(t):
    print("\n" + "=" * 8, t)


# --------------------------------------------------------------------- #
sep("F1 linreg — constant closes")
closes = pd.Series(10.0, index=dates)
rv = analytics.compute_realized_vol(closes, 20)
pct = analytics.compute_percentile_series(rv, 252)
print("rv nunique:", rv.dropna().nunique(), "last rv:", rv.dropna().iloc[-1],
      "last pct:", round(float(pct.dropna().iloc[-1]), 4),
      "regime:", analytics.classify_regime(pct.dropna().iloc[-1])["label"])
try:
    analytics.analyze_forward_vol(rv.dropna(), forward_window=30, percentile=pct)
    print("no exception")
except ValueError as exc:
    print("ValueError caught as in service L586:", repr(str(exc)))
except Exception as exc:  # noqa: BLE001
    print("OTHER exception type (would crash view L168):", type(exc).__name__, exc)

sep("F1 linreg — 2 distinct RV levels (global passes, regime fails?)")
px = np.where((np.arange(300) // 7) % 2 == 0, 1.00, 1.01)
closes2 = pd.Series(px, index=dates)
rv2 = analytics.compute_realized_vol(closes2, 20).dropna()
print("rv nunique:", rv2.nunique(), "n:", len(rv2), "levels:", sorted(rv2.round(6).unique())[:4])
try:
    res = analytics.analyze_forward_vol(rv2, forward_window=30)
    print("no exception; n_high/n_low:", res["n_high"], res["n_low"])
except ValueError as exc:
    print("ValueError:", repr(str(exc)))

sep("F1 linreg — realistic: 60d halt in the middle of a noisy series, 50 seeds")
hits = 0
for seed in range(50):
    r = np.random.default_rng(seed)
    rets = r.normal(0, 0.01, 300)
    rets[120:180] = 0.0
    c = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates)
    v = analytics.compute_realized_vol(c, 20).dropna()
    try:
        analytics.analyze_forward_vol(v, forward_window=30)
    except ValueError as exc:
        hits += 1
print("ValueError hits:", hits, "/ 50")

# --------------------------------------------------------------------- #
sep("F2 regime split — halt then activity (slope > 1)")
r = np.random.default_rng(0)
rets = np.concatenate([np.zeros(300), r.normal(0, 0.02, 100)])
d2 = pd.bdate_range("2023-01-01", periods=400)
c = pd.Series(100 * np.exp(np.cumsum(rets)), index=d2)
v = analytics.compute_realized_vol(c, 20).dropna()
res = analytics.analyze_forward_vol(v, forward_window=30)
print("slope1 %.4f intercept %.4f intersection %.4f" % (
    res["reg_forward"]["slope"], res["reg_forward"]["intercept"], res["intersection"]))
print("vol range [%.4f, %.4f]" % (res["df"]["current_vol"].min(), res["df"]["current_vol"].max()))
print("n_high", res["n_high"], "n_low", res["n_low"], "reg_low", res["reg_low"])
print("reg_high == reg_diff:", res["reg_high"] == res["reg_diff"])

sep("F2 regime split — slope ~ 1 (1 - 1e-9) passes the 1e-12 guard?")
slope = 1 - 1e-9
print("abs(1-slope) > 1e-12:", abs(1 - slope) > 1e-12,
      "-> intersection = 0.01/(1-slope) =", 0.01 / (1 - slope))

# --------------------------------------------------------------------- #
sep("F3 bad close — one 0.0 close in 300")
r = np.random.default_rng(1)
c_ok = pd.Series(100 * np.exp(np.cumsum(r.normal(0, 0.01, 300))), index=dates)
c_bad = c_ok.copy()
c_bad.iloc[150] = 0.0
rets_ok = analytics.compute_log_returns(c_ok)
rets_bad = analytics.compute_log_returns(c_bad)
print("len rets ok/bad:", len(rets_ok), len(rets_bad))
missing = rets_ok.index.difference(rets_bad.index)
print("missing return dates:", [str(d.date()) for d in missing], "(= J and J+1)")
print("true jump J-1->J+1 in ok series: %.5f ; sum of the two dropped rets: %.5f"
      % (np.log(c_ok.iloc[151] / c_ok.iloc[149]), rets_ok.iloc[149] + rets_ok.iloc[150]))
rv_ok = analytics.compute_realized_vol(c_ok, 20)
rv_bad = analytics.compute_realized_vol(c_bad, 20)
print("len rv ok/bad:", len(rv_ok), len(rv_bad), "NaN in rv_bad:", int(rv_bad.isna().sum()),
      "(only the 19 warm-up)")
common = rv_ok.index.intersection(rv_bad.index)
diff = (rv_ok.loc[common] - rv_bad.loc[common]).abs()
print("rows with rv changed:", int((diff > 1e-12).sum()),
      "max abs diff (vol pts): %.3f" % (diff.max() * 100))
c_neg = c_ok.copy()
c_neg.iloc[150] = -5.0
print("negative close -> same len:", len(analytics.compute_log_returns(c_neg)) == len(rets_bad))

# --------------------------------------------------------------------- #
sep("F4 duplicate date — service L534 construct")
c_dup = pd.concat([c_ok.iloc[:151], c_ok.iloc[150:151], c_ok.iloc[151:]])
print("len", len(c_dup), "unique dates", c_dup.index.nunique())
rv_d = analytics.compute_realized_vol(c_dup, 20)
pct_d = analytics.compute_percentile_series(rv_d, 252)
rets_d = analytics.compute_log_returns(c_dup)
print("return at dup row:", rets_d.loc[rets_d.index == c_ok.index[150]].round(8).tolist())
try:
    pd.DataFrame({"close": c_dup, "vol": rv_d, "vol_percentile": pct_d})
    print("no exception")
except Exception as exc:  # noqa: BLE001
    print("exception:", type(exc).__name__, "-", exc)
try:
    pd.DataFrame({"a": c_dup, "b": c_dup})
    print("identical dup indexes: no raise (union only triggered because rv lost row 1)")
except Exception as exc:  # noqa: BLE001
    print("identical dup indexes raise too:", exc)
svc = (ROOT / "app/model/iv_dashboard/service.py").read_text(encoding="utf-8")
print("'duplicated' in service.py:", "duplicated" in svc, "| 'drop_duplicates':", "drop_duplicates" in svc)
md = (ROOT / "app/model/market_data/market_data.py").read_text(encoding="utf-8")
print("'duplicated' in market_data.py:", "duplicated" in md, "| 'drop_duplicates':", "drop_duplicates" in md)
