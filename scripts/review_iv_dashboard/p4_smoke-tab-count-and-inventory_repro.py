"""p4 repro — smoke test: stale name, pinned count, CONTROLLERS inventory drift,
and what happens to the pin once the sibling branch (main checkout) merges.

Filesystem-only reads (no git commands), offline, deterministic.
"""
from __future__ import annotations

import ast
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]                       # this worktree
MAIN = Path("C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix")  # sibling checkout
PY = sys.executable


def tabs(root: Path) -> list[str]:
    d = root / "app" / "vue" / "tabs"
    return sorted(m.name for m in pkgutil.iter_modules([str(d)]) if m.name.startswith("tab_"))


def controllers_on_disk(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "app" / "controller").glob("*_controller.py"))


def controllers_listed(root: Path) -> list[str]:
    src = (root / "tests/smoke/test_offline_imports.py").read_text(encoding="utf8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CONTROLLERS" for t in node.targets):
            return sorted(ast.literal_eval(node.value))
    return []


def pin(root: Path) -> tuple[str, int]:
    src = (root / "tests/smoke/test_offline_imports.py").read_text(encoding="utf8")
    name = re.search(r"def (test_all_\w+_tabs_present)", src).group(1)
    n = int(re.search(r"assert len\(mods\) == (\d+)", src).group(1))
    return name, n


wt_tabs, main_tabs = tabs(WT), tabs(MAIN)
print("[worktree] tabs =", len(wt_tabs), wt_tabs)
print("[main    ] tabs =", len(main_tabs), main_tabs)
print("[worktree] test name / pin :", pin(WT))
print("[main    ] test name / pin :", pin(MAIN))

union = sorted(set(wt_tabs) | set(main_tabs))
print("[merge   ] union of tabs   =", len(union), "-> worktree pin holds:", len(union) == pin(WT)[1],
      "| main pin holds:", len(union) == pin(MAIN)[1])

# CONTROLLERS inventory
for label, root in (("worktree", WT), ("main", MAIN)):
    disk, listed = controllers_on_disk(root), controllers_listed(root)
    print(f"[{label:8}] controllers on disk={len(disk)} listed={len(listed)} missing={sorted(set(disk)-set(listed))}")

# Does the IV controller get imported at all by the smoke file (transitively via the tab)?
chk = subprocess.run(
    [PY, "-c",
     "import sys, importlib; importlib.import_module('app.vue.tabs.tab_iv_dashboard'); "
     "print('app.controller.iv_dashboard_controller' in sys.modules)"],
    cwd=WT, capture_output=True, text=True,
)
print("[transitive] tab import loads iv_dashboard_controller:", chk.stdout.strip(), chk.stderr.strip()[-150:])

# Does the worktree smoke file actually pass right now?
res = subprocess.run([PY, "-m", "pytest", "tests/smoke/test_offline_imports.py", "-q", "-p", "no:cacheprovider"],
                     cwd=WT, capture_output=True, text=True)
print("[pytest  ] rc =", res.returncode, "|", res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-300:])

# Is the function name stale? (name says 'ten', asserts something else)
name, n = pin(WT)
words = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13}
word = name.split("_")[2]
print(f"[name    ] '{name}' says {words.get(word)} but asserts {n} -> stale: {words.get(word) != n}")

# Was the pin pre-existing (i.e. not introduced by this diff)? compare with the test_app_boot stance.
boot = (WT / "tests/integration/test_app_boot.py").read_text(encoding="utf8")
print("[policy  ] test_app_boot says count 'deliberately not pinned':", "deliberately not pinned" in boot)
