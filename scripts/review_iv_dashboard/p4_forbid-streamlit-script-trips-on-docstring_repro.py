"""p4 repro — does scripts/precommit_forbid_streamlit.py go red on the docstring of
app/model/iv_dashboard/service.py, and is there any *real* streamlit import?

Offline, deterministic. Run from repo root with the venv interpreter.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# 1) Run the gate script exactly as the repo would (cwd = repo root).
proc = subprocess.run(
    [PY, "scripts/precommit_forbid_streamlit.py"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
print("[gate] rc =", proc.returncode)
print("[gate] stdout =", proc.stdout.strip())

# 2) Independent oracle: text scan vs AST scan over every app/model/**/*.py.
text_hits, ast_hits = [], []
for py in sorted((ROOT / "app" / "model").rglob("*.py")):
    src = py.read_text(encoding="utf8", errors="ignore")
    low = src.lower()
    if "import streamlit" in low or "from streamlit" in low:
        text_hits.append(py.relative_to(ROOT).as_posix())
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "streamlit" for a in node.names):
                ast_hits.append(py.relative_to(ROOT).as_posix())
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "streamlit":
                ast_hits.append(py.relative_to(ROOT).as_posix())

print("[text scan] files containing 'import streamlit'/'from streamlit':", text_hits)
print("[ast scan ] files with a real streamlit import            :", ast_hits)

# 3) Where exactly is the text hit, and is it inside the module docstring?
svc = ROOT / "app" / "model" / "iv_dashboard" / "service.py"
src = svc.read_text(encoding="utf8")
tree = ast.parse(src)
doc = ast.get_docstring(tree) or ""
for i, line in enumerate(src.splitlines(), 1):
    if "import streamlit" in line.lower():
        print(f"[service.py] line {i}: {line.strip()!r}  in_docstring={line.strip() in doc}")

# 4) Is streamlit loaded when importing the model service?
chk = subprocess.run(
    [PY, "-c",
     "import sys; import app.model.iv_dashboard.service; "
     "print('streamlit' in sys.modules)"],
    cwd=ROOT, capture_output=True, text=True,
)
print("[import service] streamlit in sys.modules ->", chk.stdout.strip(), chk.stderr.strip()[-200:])

# 5) Is the gate script wired anywhere (CI / hooks / ps1)?
wired = []
for f in [ROOT / ".github" / "workflows" / "tests.yml", ROOT / "run_tests.ps1", ROOT / "run_me.ps1"]:
    if f.exists() and "forbid_streamlit" in f.read_text(encoding="utf8", errors="ignore"):
        wired.append(f.name)
print("[wiring] files referencing precommit_forbid_streamlit:", wired or "NONE")

# Verdict helper
false_positive = proc.returncode != 0 and not ast_hits
print("[verdict] gate red without any real import (false positive):", false_positive)
