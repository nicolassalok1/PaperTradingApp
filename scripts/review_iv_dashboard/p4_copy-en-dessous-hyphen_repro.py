"""p4 repro — copy-en-dessous-hyphen.

Checks (offline):
1. `classify_regime` returns the label "EN-DESSOUS DE LA MOYENNE" for 0.2 < p <= 0.4 (the
   bucket that the view renders verbatim through `_chip("Régime courant", regime["label"])`).
2. The share of the percentile range that hits this bucket (impact claim: ~20 %).
3. Which tracked tests compare that exact string (blast radius of the fix).
4. The sibling labels for consistency (AU-DESSUS keeps its hyphen — correct French).
"""
from __future__ import annotations

import re
import subprocess
import sys

sys.path.insert(
    0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"
)
ROOT = r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"

from app.model.iv_dashboard import analytics as A

labels = {p: A.classify_regime(p)["label"] for p in (0.1, 0.2, 0.21, 0.3, 0.4, 0.41, 0.6, 0.61, 0.8, 0.81)}
for p, lab in labels.items():
    print(f"p={p:<5} -> {lab}")

hit = [p for p in [i / 1000 for i in range(0, 1001)] if A.classify_regime(p)["label"] == "EN-DESSOUS DE LA MOYENNE"]
print(f"\nshare of p in [0,1] (step 1e-3) hitting the hyphenated label: {len(hit)/1001:.1%}  (range {min(hit)}..{max(hit)})")

# view renders label verbatim?
src = open(ROOT + "/app/vue/tabs/tab_iv_dashboard.py", encoding="utf-8").read()
print("view uses regime.get('label') verbatim:", 'str(regime.get("label", "N/A"))' in src)

# all hyphenated 'EN-DESSOUS' occurrences in tracked files (code + tests)
out = subprocess.run(
    ["git", "grep", "-n", "-i", "EN-DESSOUS"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
)
print("\ngit grep -n -i 'EN-DESSOUS':")
print(out.stdout or "(none)")
