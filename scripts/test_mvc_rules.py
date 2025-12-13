from __future__ import annotations

import ast
from pathlib import Path


def _imports_in(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([a.name for a in node.names if a.name])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_mvc_rules_hold():
    repo_root = Path(__file__).resolve().parent.parent
    app_root = repo_root / "app"

    for py in app_root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(app_root).as_posix()
        layer = rel.split("/", 1)[0] if "/" in rel else rel
        mods = _imports_in(py)

        if layer == "model":
            assert not any(m.startswith("streamlit") for m in mods), f"Model imports streamlit: {py}"
            assert not any(m.startswith("app.vue") for m in mods), f"Model imports view: {py}"
            assert not any(
                m.startswith("app.controller") for m in mods
            ), f"Model imports controller: {py}"

        if layer == "controller":
            assert not any(m.startswith("streamlit") for m in mods), f"Controller imports streamlit: {py}"
            assert not any(m.startswith("app.vue") for m in mods), f"Controller imports view: {py}"

        if layer == "vue":
            assert not any(m.startswith("app.model") for m in mods), f"View imports model: {py}"
            assert not any(m.startswith("app.utils") for m in mods), f"View imports utils: {py}"

