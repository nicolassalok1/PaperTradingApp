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


def parse_imports(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        # Treat parse failures as a violation
        return ["<SYNTAX_ERROR>"]
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
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

    py_files = iter_py_files(app_root)
    for f in py_files:
        lyr = layer_of(f, app_root)
        imports = parse_imports(f)

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
