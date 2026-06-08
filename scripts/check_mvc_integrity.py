# scripts/check_mvc_integrity.py

"""
Check basic MVC integrity rules for the app/ package.

Rules enforced:
- MODEL (app/model):
    * must NOT import streamlit
    * must NOT import app.vue
    * must NOT import app.controller
- VIEW (app/vue):
    * must NOT import app.model.*
    * must NOT import app.utils.* (UI ne parle pas aux utils directement)
    * may import app.controller.* or local vue modules
- CONTROLLER (app/controller):
    * must NOT import streamlit
    * must NOT import app.vue.*
- UTILS (app/utils):
    * must NOT import streamlit
    * must NOT import app.model.*, app.controller.*, app.vue.*

Additionally:
- app/model/options/* must not contain any "streamlit" references at all.

Exit code:
- 0 if no violations
- 1 if at least one violation
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT_MARKER = "app"  # we assume this script lives under <repo>/scripts/


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / REPO_ROOT_MARKER).is_dir():
            return parent
    raise RuntimeError("Could not find repo root containing 'app/' folder.")


def iter_py_files(base: Path) -> List[Path]:
    return [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]


def _module_name(path: Path, repo_root: Path) -> str:
    """Dotted module name of a file, e.g. app/vue/tabs/foo.py -> app.vue.tabs.foo."""
    rel = path.relative_to(repo_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_base(node: "ast.ImportFrom", pkg_parts: List[str]) -> str | None:
    """
    Resolve the base package/module of a `from X import ...` to an absolute dotted
    path, handling relative levels. Returns None if it cannot be resolved.
    """
    level = node.level or 0
    if level <= 0:
        return node.module  # absolute: `from app.model import x`
    # level=1 -> current package; level=2 -> parent; etc.
    keep = len(pkg_parts) - (level - 1)
    if keep < 0:
        keep = 0
    base_parts = list(pkg_parts[:keep])
    if node.module:
        base_parts += node.module.split(".")
    return ".".join(base_parts) or None


def parse_imports(path: Path, repo_root: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        # Treat parse failures as a violation
        return ["<SYNTAX_ERROR>"]

    pkg_parts = _module_name(path, repo_root).split(".")[:-1]  # this file's package
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports (node.level) to absolute modules so that
            # `from ..model import x` is checked like `from app.model import x`.
            base = _resolve_base(node, pkg_parts)
            if base:
                imports.append(base)
                # `from BASE import name` also pulls in BASE.name, which may be a
                # subpackage/module (e.g. `from app import vue`, `from .. import vue`).
                for alias in node.names:
                    if alias.name and alias.name != "*":
                        imports.append(f"{base}.{alias.name}")
        elif isinstance(node, ast.Call):
            # Best-effort: dynamic imports with an ABSOLUTE STRING LITERAL target.
            # Matches `importlib.import_module("...")` and `__import__("...")`.
            # Computed/relative dynamic imports cannot be resolved statically.
            fn = node.func
            is_importlib = (
                isinstance(fn, ast.Attribute)
                and fn.attr == "import_module"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "importlib"
            )
            is_dunder = isinstance(fn, ast.Name) and fn.id == "__import__"
            if (is_importlib or is_dunder) and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    imports.append(a0.value)
    return imports


def layer_of(path: Path, app_root: Path) -> str:
    rel = path.relative_to(app_root)
    parts = rel.parts
    if not parts:
        return "unknown"
    if parts[0] == "model":
        return "model"
    if parts[0] == "vue":
        return "view"
    if parts[0] == "controller":
        return "controller"
    if parts[0] == "utils":
        return "utils"
    return "other"


def check_mvc_integrity(app_root: Path) -> List[Tuple[str, Path, str]]:
    violations: List[Tuple[str, Path, str]] = []

    repo_root = app_root.parent
    py_files = iter_py_files(app_root)
    for f in py_files:
        lyr = layer_of(f, app_root)
        imports = parse_imports(f, repo_root)

        # Quick text scan for streamlit in model/options/*
        if lyr == "model" and "options" in f.parts:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if "streamlit" in text:
                violations.append(
                    ("model-options-streamlit", f, "Found 'streamlit' in options model file")
                )

        for mod in imports:
            # Global rules
            if lyr == "model":
                if mod.startswith("streamlit"):
                    violations.append(("model-streamlit", f, f"Imports forbidden: {mod}"))
                if mod.startswith("app.vue"):
                    violations.append(("model-view-import", f, f"Imports forbidden: {mod}"))
                if mod.startswith("app.controller"):
                    violations.append(("model-controller-import", f, f"Imports forbidden: {mod}"))

            elif lyr == "view":
                if mod.startswith("app.model"):
                    violations.append(("view-model-import", f, f"Imports forbidden: {mod}"))
                if mod.startswith("app.utils"):
                    violations.append(("view-utils-import", f, f"Imports forbidden: {mod}"))

            elif lyr == "controller":
                if mod.startswith("streamlit"):
                    violations.append(("controller-streamlit", f, f"Imports forbidden: {mod}"))
                if mod.startswith("app.vue"):
                    violations.append(("controller-view-import", f, f"Imports forbidden: {mod}"))

            elif lyr == "utils":
                if mod.startswith("streamlit"):
                    violations.append(("utils-streamlit", f, f"Imports forbidden: {mod}"))
                if (
                    mod.startswith("app.model")
                    or mod.startswith("app.controller")
                    or mod.startswith("app.vue")
                ):
                    violations.append(("utils-layer-import", f, f"Imports forbidden: {mod}"))

            elif lyr == "other":
                # Files directly under app/ or in an unrecognized subdir are not the
                # view; they must stay Streamlit-free and must not import the view.
                if mod.startswith("streamlit"):
                    violations.append(("other-streamlit", f, f"Imports forbidden: {mod}"))
                if mod.startswith("app.vue"):
                    violations.append(("other-view-import", f, f"Imports forbidden: {mod}"))

    return violations


def main() -> None:
    repo_root = find_repo_root()
    app_root = repo_root / "app"

    violations = check_mvc_integrity(app_root)

    if not violations:
        print("[MVC] OK — no integrity violations detected.")
        sys.exit(0)

    print("[MVC] VIOLATIONS DETECTED:")
    for code, path, msg in violations:
        print(f" - [{code}] {path}: {msg}")
    sys.exit(1)


if __name__ == "__main__":
    main()
