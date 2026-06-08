"""The MVC gate must see relative AND dynamic cross-layer imports.

Regression guard for the ultracode-verification findings (parse_imports ignored
node.level and importlib/__import__).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_mvc_integrity.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_mvc_integrity", _CHECKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path: Path, rel: str, src: str) -> Path:
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(src, encoding="utf-8")
    return f


def test_relative_cross_layer_import_is_resolved(tmp_path):
    checker = _load_checker()
    # app/vue/tabs/foo.py : `from ...model import options` -> app.model.options
    f = _write(tmp_path, "app/vue/tabs/foo.py", "from ...model import options\n")
    imps = checker.parse_imports(f, tmp_path)
    assert any(m.startswith("app.model") for m in imps), imps


def test_relative_view_import_flagged_by_full_check(tmp_path):
    checker = _load_checker()
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/vue/__init__.py", "")
    _write(tmp_path, "app/vue/tabs/__init__.py", "")
    _write(tmp_path, "app/vue/tabs/bad.py", "from ...model import options\n")
    violations = checker.check_mvc_integrity(tmp_path / "app")
    codes = {c for c, _p, _m in violations}
    assert "view-model-import" in codes, violations


def test_dynamic_literal_import_is_detected(tmp_path):
    checker = _load_checker()
    f = _write(
        tmp_path,
        "app/utils/dyn.py",
        "import importlib\nimportlib.import_module('streamlit')\n",
    )
    imps = checker.parse_imports(f, tmp_path)
    assert "streamlit" in imps, imps
