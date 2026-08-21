"""p4 skeptic repro: is tests/integration/test_iv_dashboard_render.py selected by the CI gate?"""
import subprocess
import sys
import time

PY = sys.executable
ROOT = r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"


def run(args, label):
    t0 = time.perf_counter()
    cp = subprocess.run(
        [PY, "-m", "pytest", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.perf_counter() - t0
    print(f"=== {label} (exit={cp.returncode}, {dt:.1f}s)")
    return cp


# 1. CI selection -> which iv_dashboard tests get collected?
cp = run(["-m", "unit or smoke", "--co", "-q"], "CI selection: -m 'unit or smoke' --co")
lines = [ln for ln in cp.stdout.splitlines() if "iv_dashboard" in ln]
print("  iv_dashboard items collected under CI selection:", len(lines))
for ln in lines:
    print("   ", ln)
print(
    "  render guard in CI selection:",
    any("test_iv_dashboard_render" in ln for ln in cp.stdout.splitlines()),
)
total = [ln for ln in cp.stdout.splitlines() if "selected" in ln or "tests collected" in ln or "deselected" in ln]
print("  summary:", total)

# 2. integration selection
cp = run(["-m", "integration", "--co", "-q", "tests/integration/test_iv_dashboard_render.py"], "-m integration --co")
print("  items:", [ln for ln in cp.stdout.splitlines() if "::" in ln])

# 3. Does the render guard actually run offline and how long? (the CI has --disable-socket from addopts)
cp = run(["tests/integration/test_iv_dashboard_render.py", "-q", "--durations=3", "-p", "no:cacheprovider"], "run render guard alone")
print(cp.stdout[-1500:])
print("stderr tail:", cp.stderr[-500:])

# 4. Would adding the smoke marker make it collected by the CI selection? (simulate with -m expression on the file)
cp = run(
    ["-m", "unit or smoke or integration", "--co", "-q", "tests/integration/test_iv_dashboard_render.py"],
    "widened selection --co",
)
print("  items:", [ln for ln in cp.stdout.splitlines() if "::" in ln])

# 5. how long would the whole integration folder take if the gate were widened? (collection only + count)
cp = run(["-m", "integration", "--co", "-q"], "all integration tests --co")
print("  n integration items:", len([ln for ln in cp.stdout.splitlines() if "::" in ln]))
print("  files:", sorted({ln.split("::")[0] for ln in cp.stdout.splitlines() if "::" in ln}))
