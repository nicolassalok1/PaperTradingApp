"""
Phase-4 skeptic probe — group G8_arch, lens IMPACT & SEVERITY.

Measures (offline, no network, read-only):
  1. forbid-streamlit-script-trips-on-docstring
     - rc of the legacy textual gate vs the canonical AST gate
     - whether the legacy script is wired into CI / hooks / runners at all
  2. controller-bounds-duplicated-silent-clamp
     - the view widgets already enforce the same bounds as the controller
     - the payload echoes the *clamped* values and the view labels read them
     - same convention already used by hedger_v2_controller
  3. smoke-tab-count-and-inventory
     - tab counts worktree vs main checkout, overlap of the edited hunk
     - tab module imports the controller at module level (transitive coverage)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

WT = Path("C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
MAIN = Path("C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix")
PY = sys.executable

sys.path.insert(0, str(WT))


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 1. gate
section("1. forbid-streamlit gate")
legacy = subprocess.run([PY, "scripts/precommit_forbid_streamlit.py"], cwd=WT, capture_output=True, text=True)
canon = subprocess.run([PY, "scripts/check_mvc_integrity.py"], cwd=WT, capture_output=True, text=True)
print("legacy textual gate rc =", legacy.returncode, "|", legacy.stdout.strip())
print("canonical AST gate rc =", canon.returncode, "|", canon.stdout.strip()[:80])
wired = []
for f in [".github/workflows/tests.yml", "run_me.ps1", "run_tests.ps1", "pyproject.toml", "README.md"]:
    p = WT / f
    if p.exists() and "forbid_streamlit" in p.read_text(encoding="utf-8", errors="ignore"):
        wired.append(f)
print("legacy script referenced by:", wired or "NOTHING (orphan script)")
hooks = [p.name for p in (MAIN / ".git" / "hooks").glob("*") if not p.name.endswith(".sample")]
print("git hooks present:", hooks)
# Only the docstring line matches; no code import
svc = (WT / "app/model/iv_dashboard/service.py").read_text(encoding="utf-8")
code_imports = re.findall(r"^\s*(?:import|from)\s+streamlit", svc, flags=re.M)
print("real streamlit import statements in service.py:", len(code_imports))

# ---------------------------------------------------------------- 2. clamp
section("2. controller clamp impact")
from app.controller import iv_dashboard_controller as ctrl  # noqa: E402

captured = {}


def _fake(sym, **kw):
    captured.update(kw)
    return {"symbol": sym, **{k: kw[k] for k in ("years", "rv_window", "forward_window", "percentile_window")}}


with mock.patch.object(ctrl._svc, "get_iv_dashboard_data", _fake):
    out = ctrl.get_iv_analysis("spy", years=0.1, rv_window=2, forward_window=500, percentile_window=30)
print("out-of-range call ->", {k: captured[k] for k in ("years", "rv_window", "forward_window", "percentile_window")})
print("payload echoes clamped values:", all(out[k] == captured[k] for k in ("rv_window", "forward_window", "percentile_window")))

view = (WT / "app/vue/tabs/tab_iv_dashboard.py").read_text(encoding="utf-8")
view_bounds = re.findall(r"min_value=(\d+),\s*max_value=(\d+)", view)
view_bounds += re.findall(r"min_value=(\d+),\n\s*max_value=(\d+)", view)
print("view widget bounds (min,max):", view_bounds)
ctrl_src = (WT / "app/controller/iv_dashboard_controller.py").read_text(encoding="utf-8")
print("controller bounds:", re.findall(r"_clamp_(?:int|float)\(\w+, [\d.]+, ([\d.]+), ([\d.]+)\)", ctrl_src))
print("labels reading payload:", [m.strip() for m in re.findall(r"result\.get\('(?:rv|percentile)_window'\)", view)])
print("DURATION choices:", re.search(r"_DURATION_CHOICES = (\{.*\})", view).group(1))
hedger = (WT / "app/controller/hedger_v2_controller.py").read_text(encoding="utf-8")
print("hedger_v2_controller silent clamps:", len(re.findall(r"max\(\d[\d_]*, min\(", hedger)))

# ---------------------------------------------------------------- 3. smoke
section("3. smoke tab-count")
wt_tabs = sorted(p.name for p in (WT / "app/vue/tabs").glob("tab_*.py"))
main_tabs = sorted(p.name for p in (MAIN / "app/vue/tabs").glob("tab_*.py"))
print("worktree tabs:", len(wt_tabs), "| main checkout tabs:", len(main_tabs), "| union:", len(set(wt_tabs) | set(main_tabs)))
wt_smoke = (WT / "tests/smoke/test_offline_imports.py").read_text(encoding="utf-8").splitlines()
main_smoke = (MAIN / "tests/smoke/test_offline_imports.py").read_text(encoding="utf-8").splitlines()
print("worktree test name/assert:", [l.strip() for l in wt_smoke if "tabs_present" in l or "len(mods) ==" in l])
print("main test name/assert:", [l.strip() for l in main_smoke if "tabs_present" in l or "len(mods) ==" in l])
print("main CONTROLLERS has kalman:", any("kalman_controller" in l for l in main_smoke))
print("worktree CONTROLLERS has iv_dashboard:", any("iv_dashboard_controller" in l for l in wt_smoke))
print("tab imports controller at module level:", bool(re.search(r"^from app\.controller import iv_dashboard_controller", view, re.M)))
