"""
Probe: how discriminating are the assertions of tests/test_iv_dashboard_analytics.py?
 - test_realized_vol_known_magnitude: rel=0.05 — would ddof=0 / sqrt(256) / sqrt(260) pass?
 - test_realized_vol_warmup_is_nan: how many leading NaNs really exist vs the 18 asserted?
 - test_analyze_forward_vol_mean_reverting_series: seed determinism (20 runs) and slope margin.
 - test_percentile_within_bounds_and_nan: the actual mid value.
 - test_controller_clamps_and_normalizes: does the controller swallow service exceptions? (it must not)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.model.iv_dashboard import analytics as ivx  # noqa: E402

# ---- 1. known magnitude tolerance -------------------------------------------------
n = 260
log_rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
closes = pd.Series(
    100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])),
    index=pd.bdate_range("2024-01-01", periods=n + 1),
)
rv = ivx.compute_realized_vol(closes, window=20).dropna()
expected = 0.01 * np.sqrt(252)
actual = float(rv.iloc[-1])
print(f"[known_magnitude] actual={actual:.6f} expected={expected:.6f} rel_err={abs(actual/expected-1):.4%} (tol 5%)")
rets = np.log(closes / closes.shift(1)).dropna()
variants = {
    "ddof=1 * sqrt(252) (impl)": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(252),
    "ddof=0 * sqrt(252)": rets.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252),
    "ddof=1 * sqrt(256)": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(256),
    "ddof=1 * sqrt(260)": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(260),
    "ddof=1 * sqrt(365)": rets.rolling(20).std(ddof=1).iloc[-1] * np.sqrt(365),
    "simple returns ddof=1 * sqrt(252)": closes.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252),
}
for k, v in variants.items():
    print(f"   {k:38s} -> {v:.6f}  rel_err={abs(v/expected-1):.4%}  passes_rel_0.05={abs(v/expected-1) <= 0.05}")

# ---- 2. warm-up NaN count ----------------------------------------------------------
closes2 = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
rv2 = ivx.compute_realized_vol(closes2, window=20)
lead_nan = int(rv2.isna().values.argmin()) if rv2.notna().any() else len(rv2)
print(f"[warmup] len(rv)={len(rv2)} leading NaNs={lead_nan} ; test asserts only first {20-2} rows NaN")

# ---- 3. determinism + slope margin -------------------------------------------------
def _mean_reverting_vol(n: int = 700, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    vol[0] = 0.20
    for i in range(1, n):
        vol[i] = vol[i - 1] + 0.15 * (0.20 - vol[i - 1]) + rng.normal(0.0, 0.01)
    vol = np.clip(vol, 0.05, 0.9)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(vol, index=idx)


slopes = set()
for _ in range(20):
    res = ivx.analyze_forward_vol(_mean_reverting_vol(), forward_window=30)
    slopes.add((round(res["reg_forward"]["slope"], 12), round(res["reg_diff"]["slope"], 12)))
print(f"[determinism] distinct (slope1, slope2) over 20 runs: {len(slopes)} -> {slopes}")
res = ivx.analyze_forward_vol(_mean_reverting_vol(), forward_window=30)
print(f"   slope1={res['reg_forward']['slope']:.4f} (<1 margin {1-res['reg_forward']['slope']:.4f}) "
      f"slope2={res['reg_diff']['slope']:.4f} r2={res['reg_forward']['r2']:.4f} "
      f"n_high={res['n_high']} n_low={res['n_low']} reg_high={'ok' if res['reg_high'] else None} reg_low={'ok' if res['reg_low'] else None}")
# seed sweep: would slope1<1 hold for other seeds? (robustness of the oracle, not of the test)
flips = [s for s in range(50) if ivx.analyze_forward_vol(_mean_reverting_vol(seed=s), forward_window=30)["reg_forward"]["slope"] >= 1.0]
print(f"   seeds in 0..49 with slope1>=1: {flips}")

# ---- 4. percentile mid -------------------------------------------------------------
hist = pd.Series(np.linspace(0.1, 0.3, 100))
print(f"[percentile_within] mid(0.2)={ivx.percentile_within(hist, 0.2):.4f} (asserted 0.4<x<0.6)")

# ---- 5. controller: does it swallow service exceptions? ----------------------------
from app.controller import iv_dashboard_controller as ctrl  # noqa: E402

def _boom(symbol, **kw):
    raise RuntimeError("service down")

orig = ctrl._svc.get_iv_dashboard_data
ctrl._svc.get_iv_dashboard_data = _boom
try:
    ctrl.get_iv_analysis("SPY")
    print("[controller] swallowed the exception (unexpected)")
except RuntimeError as exc:
    print(f"[controller] propagates service exception as-is: {exc!r} (no test covers this)")
finally:
    ctrl._svc.get_iv_dashboard_data = orig

# ---- 6. constant-vol series: analyze_forward_vol with enough points ---------------
try:
    r = ivx.analyze_forward_vol(pd.Series(np.full(120, 0.2)), forward_window=30)
    print(f"[constant vol, 90 pts] slope1={r['reg_forward']['slope']} r2={r['reg_forward']['r2']} intersection={r['intersection']} (no test)")
except Exception as exc:  # noqa: BLE001
    print(f"[constant vol, 90 pts] RAISES {type(exc).__name__}: {exc} (no test)")
