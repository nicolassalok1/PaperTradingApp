"""
MVC import-matrix test — delegates to the SINGLE source of truth
(`scripts/check_mvc_integrity.py`) so the test can never drift from, or weaken,
the gate. The canonical rule is documented in `docs/mvc_import_matrix.md`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_mvc_integrity.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_mvc_integrity", _CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mvc_rules_hold():
    checker = _load_checker()
    app_root = _REPO_ROOT / "app"
    violations = checker.check_mvc_integrity(app_root)
    assert not violations, "MVC import-matrix violations:\n" + "\n".join(
        f" - [{code}] {Path(path).relative_to(_REPO_ROOT)}: {msg}"
        for code, path, msg in violations
    )
