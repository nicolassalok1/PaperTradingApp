"""p4 skeptic repro — iv-history-filename-unsanitized-symbol.

Offline, deterministic. Measures what _iv_history_path() / record_iv_observation()
do with separator-bearing symbols, whether the written file stays under the
(temp) IVHistory dir, and whether load_iv_history() can read outside it.
Also checks the reachability gate: record_iv_observation is only invoked after
fetch_current_atm_iv() returned an info dict (service.py L566-L568).
"""
from __future__ import annotations

import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.iv_dashboard import service  # noqa: E402

INFO = {"iv": 0.16, "dte": 30, "n_contracts": 8, "method": "greeks Alpaca", "spot": 645.0}
SYMBOLS = ["SPY", "BRK.B", "BRK/B", "A/B", "../evil", "/../../ESCAPE", "..\\..\\ESCAPE2", "A B", "", "CON"]


def under(base: pathlib.Path, p: pathlib.Path) -> bool:
    try:
        p.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    base = pathlib.Path(tempfile.mkdtemp(prefix="p4_sym_")) / "IVHistory"
    base.mkdir()
    service.CACHE_IV_HISTORY_DIR = base
    print(f"sandbox IVHistory = {base}")
    print(f"same regex as options/logic.py:1107 -> {[re.sub(r'[^A-Za-z0-9._-]', '_', s.strip().upper()) or 'SYMBOL' for s in SYMBOLS]}")
    for s in SYMBOLS:
        p = service._iv_history_path(s)
        print(f"\nsymbol {s!r:18} -> path {str(p).replace(str(base), '<IVH>')}  under IVHistory (lexical resolve): {under(base, p)}")
        service.record_iv_observation(s, INFO)
        written = sorted(str(q.relative_to(base.parent)) for q in base.parent.rglob("*") if q.is_file())
        print(f"   files now under sandbox root: {written}")
        rows = len(service.load_iv_history(s))
        print(f"   load_iv_history rows = {rows}")

    # Reachability gate: in get_iv_dashboard_data the write happens only after
    # fetch_current_atm_iv() returned a non-None info (needs spot + Alpaca snapshots).
    src = (ROOT / "app/model/iv_dashboard/service.py").read_text(encoding="utf-8").splitlines()
    for i in range(560, 578):
        if "record_iv_observation" in src[i] or "fetch_current_atm_iv" in src[i] or "load_iv_history" in src[i]:
            print(f"L{i+1}: {src[i].strip()}")
    print(f"os.name={os.name} (Windows normalises '..' lexically before touching the filesystem)")
