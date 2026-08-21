"""G9 skeptic probe (impact lens): realized-vol oracle strength + warm-up NaN count.

Independent of the phase-1 script: recomputes the test fixture and checks
(1) actual vs expected, (2) which wrong conventions the rel=0.05 tolerance lets
through, (3) whether the proposed exact oracle (rel=1e-9) is numerically
achievable, (4) the real downstream impact of a ddof drift on the displayed
numbers (RV, IV-RV spread, percentile/regime).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")

from app.model.iv_dashboard import analytics as ivx  # noqa: E402

# --- fixture identical to tests/test_iv_dashboard_analytics.py::test_realized_vol_known_magnitude
n = 260
log_rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
closes = pd.Series(
    100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])),
    index=pd.bdate_range("2024-01-01", periods=n + 1),
)
rv = ivx.compute_realized_vol(closes, window=20).dropna()
actual = float(rv.iloc[-1])
expected_test = 0.01 * np.sqrt(252)
print(f"actual={actual:.6f} expected(test)={expected_test:.6f} rel_err={abs(actual/expected_test-1)*100:.4f}%")

# proposed oracle
exp_fix = 0.01 * np.sqrt(20 / 19) * np.sqrt(252)
rel_fix = abs(actual / exp_fix - 1)
print(f"proposed exact oracle={exp_fix:.9f} rel_err={rel_fix:.3e} passes rel=1e-9: {rel_fix <= 1e-9}")

# which wrong conventions would pass rel=0.05 against the test's expected value?
rets = ivx.compute_log_returns(closes)
simple = (closes / closes.shift(1) - 1).dropna()
variants = {
    "impl ddof=1*sqrt(252)": rets.rolling(20).std() * np.sqrt(252),
    "ddof=0*sqrt(252)": rets.rolling(20).std(ddof=0) * np.sqrt(252),
    "ddof=1*sqrt(256)": rets.rolling(20).std() * np.sqrt(256),
    "ddof=1*sqrt(260)": rets.rolling(20).std() * np.sqrt(260),
    "ddof=1*sqrt(365)": rets.rolling(20).std() * np.sqrt(365),
    "simple returns ddof=1": simple.rolling(20).std() * np.sqrt(252),
    "window=19 (off by one)": rets.rolling(19).std() * np.sqrt(252),
    "window=21 (off by one)": rets.rolling(21).std() * np.sqrt(252),
}
for name, s in variants.items():
    v = float(s.dropna().iloc[-1])
    rel = abs(v / expected_test - 1)
    print(f"  {name:28s} last={v:.6f} rel={rel*100:6.3f}%  passes_rel0.05={rel <= 0.05}")

# --- warm-up test
closes30 = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
rv30 = ivx.compute_realized_vol(closes30, window=20)
lead_nan = int(rv30.isna().values.argmin())
print(f"warmup: len={len(rv30)} leading_nans={lead_nan} test_checks_first={20-2} proposed_checks_first={20-1}")
# would a window shortened by one day pass the current assertion?
rv19 = ivx.compute_realized_vol(closes30, window=19)
print(f"  window=19 -> leading_nans={int(rv19.isna().values.argmin())}; current assert (first 18 NaN) passes: {rv19.iloc[:18].isna().all()}")

# --- downstream impact of a ddof=0 drift on displayed numbers (realistic series)
rng = np.random.default_rng(3)
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.011, 760))), index=pd.bdate_range("2023-08-01", periods=760))
rv1 = ivx.compute_realized_vol(px, 20)
rv0 = ivx.compute_log_returns(px).rolling(20).std(ddof=0) * np.sqrt(252)
pct1 = ivx.compute_percentile_series(rv1, 252)
pct0 = ivx.compute_percentile_series(rv0, 252)
iv = 0.20
print(f"ddof drift on a realistic series: RV {rv1.iloc[-1]*100:.2f}% -> {rv0.iloc[-1]*100:.2f}% ; "
      f"spread IV-RV {(iv-rv1.iloc[-1])*100:+.2f} -> {(iv-rv0.iloc[-1])*100:+.2f} pts ; "
      f"RV percentile {pct1.iloc[-1]:.3f} -> {pct0.iloc[-1]:.3f} (scale-invariant) ; "
      f"IV-vs-RV percentile {ivx.percentile_within(rv1.tail(252), iv):.3f} -> {ivx.percentile_within(rv0.tail(252), iv):.3f}")
