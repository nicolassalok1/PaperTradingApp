"""
Every `__main__`-guarded entry point under `app/` must be importable on its own.

These scripts are launched by hand, one at a time, in a fresh interpreter. Nothing
else imports them, so nothing else proves they still load — and the dead-code
method inherited from E4 treats them as *roots* of the import graph, which is only
sound if they can actually be imported.

`app/model/market_data/scripts/update_balance.py` could not. Importing it first
walked `settlement -> buy_sell -> dashboard.cache -> dashboard/__init__ ->
dashboard.service -> settlement`, and the last hop hit a partially initialised
module: `ImportError: cannot import name 'process_matured_forwards'`. The forward
settlement script failed on its very first line, every run.

Import *order* is the whole bug, which is why each entry point is imported with the
`app.*` namespace purged. Sharing an interpreter hides it: `tests/integration/
test_portfolio_cli_spine.py` imports `app.model.dashboard.cache` before
`app.model.portfolio.settlement`, which primes the cycle from the working end, so
those tests passed against code whose real entry point was broken. Purging is
enough — the cycle lives entirely inside `app.*`, and third-party packages stay
cached, so this costs a fraction of a real subprocess.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"

_MAIN_GUARD = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""", re.MULTILINE)


def _entry_points() -> list[str]:
    """Every module under `app/` that can be run as a script."""
    found = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if _MAIN_GUARD.search(path.read_text(encoding="utf-8", errors="ignore")):
            rel = path.relative_to(REPO_ROOT).with_suffix("")
            found.append(rel.as_posix().replace("/", "."))
    return found


@pytest.fixture
def purged_app_modules():
    """Hand each import a clean `app.*` namespace, then put the session back.

    Restoring matters: other tests hold references to the module objects that were
    live before this ran, and leaving replacements behind would give them a second,
    inconsistent copy of the package.
    """
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}

    def _purge():
        for name in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
            del sys.modules[name]

    _purge()
    yield
    _purge()
    sys.modules.update(saved)


@pytest.mark.smoke
@pytest.mark.parametrize("modname", _entry_points())
def test_entry_point_imports_on_its_own(modname, purged_app_modules):
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "app" or missing.startswith("app."):
            raise
        # An optional heavy dependency that this environment does not ship — e.g.
        # `torch`, needed only by the offline training scripts. Not a code defect,
        # and narrow on purpose: anything failing to resolve inside `app.` is.
        pytest.skip(f"{modname} needs the optional dependency {missing!r}")


@pytest.mark.smoke
def test_the_scan_actually_finds_the_cli_scripts():
    """Guards the guard: a discovery regex that silently matches nothing would
    make every case above vacuous."""
    found = _entry_points()
    assert "app.model.market_data.scripts.update_balance" in found, found
    assert "app.model.market_data.scripts.update_portfolio_value" in found, found
