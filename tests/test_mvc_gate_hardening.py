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


def test_from_import_name_resolves_submodule(tmp_path):
    # `from app import vue` and `from .. import vue` both pull in app.vue.
    checker = _load_checker()
    f1 = _write(tmp_path, "app/model/a.py", "from app import vue\n")
    assert "app.vue" in checker.parse_imports(f1, tmp_path)
    f2 = _write(tmp_path, "app/model/sub/b.py", "from ... import vue\n")
    assert "app.vue" in checker.parse_imports(f2, tmp_path)


def test_from_import_name_flagged_as_model_view_violation(tmp_path):
    checker = _load_checker()
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "app/model/__init__.py", "")
    _write(tmp_path, "app/model/bad.py", "from app import vue\n")
    violations = checker.check_mvc_integrity(tmp_path / "app")
    assert "model-view-import" in {c for c, _p, _m in violations}, violations


def test_non_importlib_import_module_not_false_positive(tmp_path):
    # A method named import_module on some other object must NOT be treated as an import.
    checker = _load_checker()
    f = _write(tmp_path, "app/utils/x.py", "loader.import_module('app.vue.tabs')\n")
    assert "app.vue.tabs" not in checker.parse_imports(f, tmp_path)
