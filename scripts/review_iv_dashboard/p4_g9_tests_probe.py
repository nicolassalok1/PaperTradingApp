"""
G9_tests skeptic probe (code-reading lens).

Measures, without touching tracked files:
  A. service.py / tab_iv_dashboard.py statement coverage under the CI selection
     (-m "unit or smoke") -- finding service-zero-unit-tests.
  B. render guard: marker selection + wall time -- finding render-guard-not-in-ci-gate.
  C. tab_iv_dashboard.py coverage under the driver's two AppTest runs (driver
     executed under coverage in a pristine subprocess) -- finding
     render-guard-happy-path-only.
  D. realized-vol oracle arithmetic -- finding weak-realized-vol-oracle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = Path(__file__).with_suffix(".out.txt")
lines: list[str] = []


def say(s: str) -> None:
    print(s)
    lines.append(s)


def run(args: list[str], env: dict | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=str(cwd))


# ---------------------------------------------------------------- A. CI-selection coverage
say("=== A. coverage of iv_dashboard modules under CI selection (-m 'unit or smoke') ===")
cp = run([PY, "-m", "pytest", "-m", "unit or smoke", "-q", "-p", "no:cacheprovider",
          "--cov=app.model.iv_dashboard", "--cov=app.vue.tabs.tab_iv_dashboard", "--cov=app.controller.iv_dashboard_controller",
          "--cov-report=term-missing", "--no-cov-on-fail",
          "tests/test_iv_dashboard_analytics.py", "tests/smoke/test_offline_imports.py"])
for ln in cp.stdout.splitlines():
    if "iv_dashboard" in ln or "passed" in ln or "failed" in ln or "TOTAL" in ln:
        say("  " + ln.rstrip())

# ---------------------------------------------------------------- B. render guard marker + timing
say("=== B. render guard marker selection + timing ===")
cp = run([PY, "-m", "pytest", "-m", "unit or smoke", "--co", "-q", "-p", "no:cacheprovider", "tests"])
hit = [ln for ln in cp.stdout.splitlines() if "iv_dashboard_render" in ln]
say(f"  collected under 'unit or smoke': {hit or 'NONE'}")
cp = run([PY, "-m", "pytest", "-m", "integration", "--co", "-q", "-p", "no:cacheprovider", "tests/integration/test_iv_dashboard_render.py"])
say(f"  collected under 'integration': {[ln for ln in cp.stdout.splitlines() if '::' in ln]}")
cp = run([PY, "-m", "pytest", "-m", "smoke", "--co", "-q", "-p", "no:cacheprovider", "tests/integration/test_app_boot.py", "tests/integration/test_iv_dashboard_render.py"])
say(f"  collected under 'smoke' (integration dir): {[ln for ln in cp.stdout.splitlines() if '::' in ln] or 'NONE'}")
t0 = time.perf_counter()
cp = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--durations=3", "tests/integration/test_iv_dashboard_render.py"])
wall = time.perf_counter() - t0
say(f"  render test run: wall={wall:.2f}s rc={cp.returncode}")
for ln in cp.stdout.splitlines():
    if re.search(r"passed|failed|error|s (setup|call)", ln):
        say("    " + ln.rstrip())
# Also time test_app_boot for the 'widen CI' alternative
t0 = time.perf_counter()
cp = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/integration/test_app_boot.py"])
say(f"  test_app_boot run: wall={time.perf_counter() - t0:.2f}s rc={cp.returncode} tail={cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ''}")

# ---------------------------------------------------------------- C. view coverage under the driver
say("=== C. tab_iv_dashboard.py coverage under the driver's two AppTest runs ===")
driver = ROOT / "tests" / "integration" / "_iv_dashboard_render_driver.py"
rc_file = Path(__file__).with_name("p4_g9_coveragerc.ini")
rc_file.write_text("[run]\nbranch = True\nsource = app.vue.tabs.tab_iv_dashboard\n[report]\nshow_missing = True\n", encoding="utf-8")
datafile = Path(__file__).with_name(".p4_g9_coverage")
env = {k: v for k, v in os.environ.items() if k not in {"APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "OPENAI_API_KEY"}}
env["COVERAGE_FILE"] = str(datafile)
cp = subprocess.run([PY, "-m", "coverage", "run", f"--rcfile={rc_file}", str(driver), str(ROOT)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
res_line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("IVDASH_RESULT ")), None)
say(f"  driver rc={cp.returncode} result={res_line}")
cp2 = subprocess.run([PY, "-m", "coverage", "report", f"--rcfile={rc_file}", "-m"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
for ln in cp2.stdout.splitlines():
    if "tab_iv_dashboard" in ln or ln.startswith("Name") or ln.startswith("TOTAL"):
        say("  " + ln.rstrip())
cp3 = subprocess.run([PY, "-m", "coverage", "json", f"--rcfile={rc_file}", "-o", str(Path(__file__).with_name("p4_g9_cov.json"))],
                     capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ROOT))
try:
    cov = json.loads(Path(__file__).with_name("p4_g9_cov.json").read_text(encoding="utf-8"))
    for fname, d in cov["files"].items():
        if "tab_iv_dashboard" in fname:
            say(f"  missing_lines={d['missing_lines']}")
            say(f"  missing_branches={d.get('missing_branches')}")
except Exception as exc:  # noqa: BLE001
    say(f"  coverage json failed: {exc}")
# Check the specific degraded branches cited by the finder are among the missing lines
cited = {249, 540, 432, 466, 151, 152, 153, 154, 155, 156, 157, 165, 166, 167, 168}
try:
    miss = set(d["missing_lines"])
    say(f"  cited-by-finder lines present in missing set: {sorted(cited & miss)} ; cited but COVERED: {sorted(cited - miss)}")
except Exception:  # noqa: BLE001
    pass
for p in (datafile, rc_file, Path(__file__).with_name("p4_g9_cov.json")):
    try:
        p.unlink()
    except OSError:
        pass

# ---------------------------------------------------------------- D. RV oracle arithmetic
say("=== D. realized-vol oracle arithmetic ===")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from app.model.iv_dashboard import analytics as ivx  # noqa: E402

n = 260
log_rets = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
closes = pd.Series(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])),
                   index=pd.bdate_range("2024-01-01", periods=n + 1))
rv = ivx.compute_realized_vol(closes, window=20).dropna()
actual = float(rv.iloc[-1])
exp_test = 0.01 * np.sqrt(252)
exp_ddof1 = 0.01 * np.sqrt(20 / 19) * np.sqrt(252)
say(f"  actual={actual:.6f} test_expected(ddof0)={exp_test:.6f} rel_err={(actual / exp_test - 1) * 100:.4f}%")
say(f"  ddof1 closed form={exp_ddof1:.6f} rel_err vs actual={(actual / exp_ddof1 - 1):.3e}")
variants = {
    "ddof=1*sqrt(252) [impl]": actual,
    "ddof=0*sqrt(252)": 0.01 * np.sqrt(252),
    "ddof=1*sqrt(256)": 0.01 * np.sqrt(20 / 19) * np.sqrt(256),
    "ddof=1*sqrt(260)": 0.01 * np.sqrt(20 / 19) * np.sqrt(260),
    "ddof=1*sqrt(365)": 0.01 * np.sqrt(20 / 19) * np.sqrt(365),
}
# simple returns variant: compute directly
simple = closes.pct_change().dropna().rolling(20).std().iloc[-1] * np.sqrt(252)
variants["simple returns ddof=1*sqrt(252)"] = float(simple)
# window=19 and window=21 variants (would a window regression pass?)
for w in (15, 19, 21, 25):
    variants[f"window={w} ddof=1*sqrt(252)"] = float(ivx.compute_realized_vol(closes, window=w).dropna().iloc[-1])
for k, v in variants.items():
    rel = abs(v / exp_test - 1)
    say(f"  {k:38s} value={v:.6f} rel_vs_test_expected={rel * 100:6.3f}%  passes rel=0.05: {rel <= 0.05}")

# warm-up test
closes2 = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
rv2 = ivx.compute_realized_vol(closes2, window=20)
lead_nan = int(rv2.isna().values.argmin()) if rv2.notna().any() else len(rv2)
say(f"  warm-up: len(rv)={len(rv2)} leading NaNs={lead_nan} test slices iloc[:{20 - 2}] -> {20 - 2} rows checked; first non-NaN at iloc[{lead_nan}]")
# would window=19 pass the current warm-up test?
rv19 = ivx.compute_realized_vol(closes2, window=19)
say(f"  window=19 -> leading NaNs={int(rv19.isna().values.argmin())}; current assertion iloc[:18].isna().all() = {bool(rv19.iloc[:18].isna().all())}")
rv18 = ivx.compute_realized_vol(closes2, window=18)
say(f"  window=18 -> leading NaNs={int(rv18.isna().values.argmin())}; current assertion iloc[:18].isna().all() = {bool(rv18.iloc[:18].isna().all())}")
say(f"  proposed fix: iloc[:19].isna().all()={bool(rv2.iloc[:19].isna().all())} iloc[19:].notna().all()={bool(rv2.iloc[19:].notna().all())}")
# proposed ddof=1 oracle passes at rel=1e-9?
say(f"  proposed oracle |actual/exp_ddof1-1| = {abs(actual / exp_ddof1 - 1):.2e} (<=1e-9: {abs(actual / exp_ddof1 - 1) <= 1e-9})")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[written] {OUT}")
