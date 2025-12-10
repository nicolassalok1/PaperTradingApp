# scripts/scan_imports.py

"""
Scan Python imports under app/ and build a simple dependency report.

Features:
- Map each module (app.*) to the modules it imports.
- Compute inbound dependencies to detect "orphans" (no one imports them).
- Print:
    * adjacency list (who imports what)
    * list of modules with zero inbound edges (possible dead code)

This is a heuristic tool, NOT a perfect static analyzer.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT_MARKER = "app"


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / REPO_ROOT_MARKER).is_dir():
            return parent
    raise RuntimeError("Could not find repo root containing 'app/' folder.")


def iter_py_files(base: Path) -> List[Path]:
    return [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]


def module_name_from_path(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root)
    return rel.with_suffix("").as_posix().replace("/", ".")


def parse_imports(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods


def build_dependency_graph(
    app_root: Path, repo_root: Path
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
        outgoing: module -> set(imported_modules)
        incoming: module -> set(modules_that_import_it)
    Only modules under "app." are tracked fully; others appear as external.
    """
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    incoming: Dict[str, Set[str]] = defaultdict(set)

    py_files = iter_py_files(app_root)
    all_modules: Set[str] = set()

    for f in py_files:
        mod_name = module_name_from_path(f, repo_root)
        all_modules.add(mod_name)
        imports = parse_imports(f)
        for im in imports:
            outgoing[mod_name].add(im)

    # Build incoming map only for app.* modules
    for src, targets in outgoing.items():
        for dst in targets:
            if dst.startswith("app."):
                incoming[dst].add(src)

    # Ensure all app.* modules appear in incoming/outgoing maps
    for m in all_modules:
        outgoing.setdefault(m, set())
        incoming.setdefault(m, set())

    return outgoing, incoming


def main() -> None:
    repo_root = find_repo_root()
    app_root = repo_root / "app"

    outgoing, incoming = build_dependency_graph(app_root, repo_root)

    print("=== IMPORT GRAPH (app.* modules) ===")
    for mod in sorted(outgoing):
        targets = sorted(t for t in outgoing[mod] if t.startswith("app."))
        print(f"{mod}:")
        if targets:
            for t in targets:
                print(f"  -> {t}")
        else:
            print("  (no app.* imports)")
        print()

    print("=== POSSIBLE ORPHANS (no inbound app.* imports) ===")
    # Whitelist of obvious "entry" modules that can be top-level without inbound deps
    whitelist_prefixes = [
        "app.vue.main_app",
        "app.vue",
        "app.controller",
        "app.model.__init__",
        "app.utils",
    ]
    for mod in sorted(incoming):
        if not incoming[mod] and mod.startswith("app."):
            if any(mod == w or mod.startswith(w + ".") for w in whitelist_prefixes):
                continue
            print(f"  {mod}")
    print("=== END ===")


if __name__ == "__main__":
    main()
