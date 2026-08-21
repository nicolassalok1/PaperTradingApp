"""
Phase-4 skeptic probe, group G8_arch (code-reading lens). Offline, read-only.

Reproduces the measurements behind three verdicts:
  1. forbid-streamlit-script-trips-on-docstring
  2. controller-bounds-duplicated-silent-clamp
  3. smoke-tab-count-and-inventory
Run from the worktree root with the repo venv python.
"""
from __future__ import annotations

import ast
import importlib
import logging
import pkgutil
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PY = sys.executable


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 1. forbid script
section("1. scripts/precommit_forbid_streamlit.py")
r = subprocess.run([PY, "scripts/precommit_forbid_streamlit.py"], cwd=ROOT, capture_output=True, text=True)
print("rc =", r.returncode, "|", r.stdout.strip())
svc = ROOT / "app/model/iv_dashboard/service.py"
src = svc.read_text(encoding="utf8")
tree = ast.parse(src)
real = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import) and any(a.name.startswith("streamlit") for a in n.names):
        real.append(n)
    if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("streamlit"):
        real.append(n)
print("AST real streamlit imports in service.py:", len(real))
hits = [i + 1 for i, l in enumerate(src.splitlines()) if "import streamlit" in l.lower()]
print("textual 'import streamlit' hit lines:", hits)
print("docstring-only hit:", hits == [15] and not real)
r2 = subprocess.run([PY, "scripts/check_mvc_integrity.py"], cwd=ROOT, capture_output=True, text=True)
print("check_mvc_integrity rc =", r2.returncode)
ci = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf8")
print("forbid script wired in CI:", "precommit_forbid_streamlit" in ci)

# ---------------------------------------------------------------- 2. clamp
section("2. controller clamp vs view bounds")
ctrl = importlib.import_module("app.controller.iv_dashboard_controller")
captured = {}
with mock.patch.object(ctrl._svc, "get_iv_dashboard_data", side_effect=lambda s, **kw: captured.update(kw) or {}):
    with mock.patch.object(logging.Logger, "warning") as warn:
        ctrl.get_iv_analysis("spy", years=0.1, rv_window=2, forward_window=500, percentile_window=30)
        print("kwargs passed to model:", {k: captured[k] for k in ("years", "rv_window", "forward_window", "percentile_window")})
        print("logger.warning calls during clamp:", warn.call_count)
view_src = (ROOT / "app/vue/tabs/tab_iv_dashboard.py").read_text(encoding="utf8")
view_bounds = re.findall(r"min_value=(\d+),\s*max_value=(\d+)", view_src, flags=re.S)
ctrl_src = (ROOT / "app/controller/iv_dashboard_controller.py").read_text(encoding="utf8")
ctrl_bounds = re.findall(r"_clamp_(?:int|float)\(\w+, [\d.]+, ([\d.]+), ([\d.]+)\)", ctrl_src)
print("view  bounds:", view_bounds)
print("ctrl  bounds:", ctrl_bounds)
print("shared constant between view and controller:", "BOUNDS" in view_src)
print("payload keys echoed by view labels:", sorted(set(re.findall(r"result\.get\('(\w+_window)'\)", view_src))))

# ---------------------------------------------------------------- 3. smoke inventory
section("3. smoke test inventory")
smoke = (ROOT / "tests/smoke/test_offline_imports.py").read_text(encoding="utf8")
print("test name:", re.search(r"def (test_all_\w+_tabs_present)", smoke).group(1),
      "| asserted count:", re.search(r"len\(mods\) == (\d+)", smoke).group(1))
tabs = [m.name for m in pkgutil.iter_modules([str(ROOT / "app/vue/tabs")]) if m.name.startswith("tab_")]
print("worktree tab_ modules:", len(tabs))
ctrls_fs = sorted(p.stem for p in (ROOT / "app/controller").glob("*_controller.py"))
ctrls_listed = re.findall(r'"(\w+_controller)"', smoke.split("CONTROLLERS = [")[1].split("]")[0])
print("controllers on disk not in CONTROLLERS:", sorted(set(ctrls_fs) - set(ctrls_listed)))
main_smoke = subprocess.run(["git", "show", "origin/main:tests/smoke/test_offline_imports.py"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf8").stdout
print("origin/main test name:", re.search(r"def (test_all_\w+_tabs_present)", main_smoke).group(1),
      "| count:", re.search(r"len\(mods\) == (\d+)", main_smoke).group(1),
      "| kalman_controller listed:", "kalman_controller" in main_smoke)
mt = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "origin/main", "HEAD"], cwd=ROOT,
                    capture_output=True, text=True)
print("merge-tree conflicts (origin/main <- HEAD):", [l for l in mt.stdout.splitlines()[1:] if l.strip()])
