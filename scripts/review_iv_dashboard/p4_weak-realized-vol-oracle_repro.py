"""p4 skeptic repro: does test_realized_vol_known_magnitude / warmup discriminate?"""
import sys

sys.path.insert(
    0,
    r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca",
)
import numpy as np
import pandas as pd

from app.model.iv_dashboard import analytics as ivx

# --- replicate test_realized_vol_known_magnitude exactly
n = 260
log_rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
closes = pd.Series(
    100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])),
    index=pd.bdate_range("2024-01-01", periods=n + 1),
)
rv = ivx.compute_realized_vol(closes, window=20).dropna()
actual = float(rv.iloc[-1])
expected_test = 0.01 * np.sqrt(252)
print(
    f"actual(impl)={actual:.6f} expected_in_test={expected_test:.6f} "
    f"rel_err={(actual-expected_test)/expected_test*100:.4f}%"
)
print(
    "sample std exact ddof=1 of alternating +-1% over 20 =",
    0.01 * np.sqrt(20 / 19),
    "-> annualized",
    0.01 * np.sqrt(20 / 19) * np.sqrt(252),
)

# --- which alternative implementations would ALSO pass rel=0.05?
rets = ivx.compute_log_returns(closes)
simple = (closes / closes.shift(1) - 1).dropna()
variants = {
    "impl ddof=1 sqrt252": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(252),
    "ddof=0 sqrt252": rets.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252),
    "ddof=1 sqrt256": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(256),
    "ddof=1 sqrt260": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(260),
    "ddof=1 sqrt365": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(365),
    "simple returns ddof=1 sqrt252": simple.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(252),
    "window=19 ddof=1": rets.rolling(19).std(ddof=1).iloc[-1] * np.sqrt(252),
    "window=21 ddof=1": rets.rolling(21).std(ddof=1).iloc[-1] * np.sqrt(252),
    "window=10 ddof=1": rets.rolling(10).std(ddof=1).iloc[-1] * np.sqrt(252),
    "window=30 ddof=1": rets.rolling(30).std(ddof=1).iloc[-1] * np.sqrt(252),
    "window=100 ddof=1": rets.rolling(100).std(ddof=1).iloc[-1] * np.sqrt(252),
    "ddof=0 sqrt260": rets.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(260),
    "ddof=0 sqrt270": rets.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(270),
    "ddof=1 sqrt240": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(240),
}
for k, v in variants.items():
    rel = abs(v - expected_test) / expected_test
    print(f"  {k:32s} value={v:.6f} rel={rel*100:6.3f}%  passes(rel=0.05)={rel<=0.05}")

# --- proposed fix oracle: does it reject the variants?
exp_fix = 0.01 * np.sqrt(20 / 19) * np.sqrt(252)
print(
    "\nproposed fix expected =",
    exp_fix,
    " impl passes rel=1e-9:",
    abs(actual - exp_fix) / exp_fix <= 1e-9,
    " rel:",
    abs(actual - exp_fix) / exp_fix,
)
for k, v in variants.items():
    print(f"  fix rejects {k:32s}: {abs(v-exp_fix)/exp_fix > 1e-9}")

# --- warm-up test
closes2 = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
rv2 = ivx.compute_realized_vol(closes2, window=20)
lead = int(rv2.isna().cumprod().sum())
print(
    f"\nwarmup: len(rv)={len(rv2)} leading NaNs={lead} test asserts iloc[:{20-2}] "
    f"(={20-2} rows); total NaNs={int(rv2.isna().sum())}"
)
rv19 = ivx.compute_realized_vol(closes2, window=19)
print(
    "window=19 -> leading NaNs",
    int(rv19.isna().cumprod().sum()),
    "; current test passes:",
    bool(rv19.iloc[:18].isna().all()),
)
rv18 = ivx.compute_realized_vol(closes2, window=18)
print(
    "window=18 -> leading NaNs",
    int(rv18.isna().cumprod().sum()),
    "; current test passes:",
    bool(rv18.iloc[:18].isna().all()),
)
rets2 = ivx.compute_log_returns(closes2)
rv_mp = rets2.rolling(20, min_periods=2).std() * np.sqrt(252)
print(
    "min_periods=2 regression -> leading NaNs",
    int(rv_mp.isna().cumprod().sum()),
    "; current test passes:",
    bool(rv_mp.iloc[:18].isna().all()),
)
print(
    "fix asserts on impl: ",
    bool(rv2.iloc[:19].isna().all()) and bool(rv2.iloc[19:].notna().all()),
)
print(
    "fix on window=19: ",
    bool(rv19.iloc[:19].isna().all()) and bool(rv19.iloc[19:].notna().all()),
)
print(
    "fix on min_periods=2: ",
    bool(rv_mp.iloc[:19].isna().all()) and bool(rv_mp.iloc[19:].notna().all()),
)
