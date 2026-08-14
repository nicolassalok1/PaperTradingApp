"""D3 — no `from ... import *` under app/, and every options module still imports.

53 modules pulled their whole namespace from
`app.vue.components.options.controller_bridge` with a star import. That hides which
symbol comes from where, defeats every static check, and is the reason the package
carries a real import cycle: controller_bridge imports plot_limits / ui_helpers /
bridge_context / bridge_render from its own package, whose __init__ re-exports
controller_bridge with a star — hence the deferred `# noqa: E402` imports at the
bottom of the bridge.

Replacing the stars with explicit named imports must not change a single symbol, so
this file guards the migration from both ends: nothing may still use a star, and
every module must keep importing cleanly offline.
"""

from __future__ import annotations

import ast
import importlib
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "app"
OPTIONS_PKG = APP / "vue" / "components" / "options"


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# app.vue.components.options.ui_main imports app.vue.pages.shared_ui, and app/vue/pages/
# does not exist — the module cannot be imported at all, and nothing in the repo imports
# it. Proven dead code, removed under D4; excluded here so this guard stays meaningful
# instead of red for an unrelated reason.
_UNIMPORTABLE = {"app.vue.components.options.ui_main"}


def _options_modules() -> list[str]:
    out = []
    for py in sorted(OPTIONS_PKG.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        name = _module_name(py)
        if name in _UNIMPORTABLE:
            continue
        out.append(name)
    return out


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _blocked(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("Network access during import is forbidden (offline smoke test).")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=True)
    yield


@pytest.mark.smoke
def test_no_star_import_anywhere_under_app():
    offenders: list[str] = []
    for py in sorted(APP.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - would fail elsewhere anyway
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                offenders.append(f"{py.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    assert offenders == [], f"star imports left under app/: {offenders}"


@pytest.mark.smoke
@pytest.mark.parametrize("modname", _options_modules())
def test_every_options_module_still_imports_offline(modname):
    importlib.import_module(modname)
