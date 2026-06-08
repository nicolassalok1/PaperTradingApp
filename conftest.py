from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# scripts/test_all.py is a MANUAL connectivity script (hits Alpaca over the network,
# no `test_*` functions). Keep it out of the automated suite.
collect_ignore = ["scripts/test_all.py"]

# Every test must declare a category — no silent exclusion by `-m` selection.
_KNOWN_MARKERS = {"unit", "smoke", "characterization", "integration", "slow"}


def pytest_collection_modifyitems(config, items):
    unmarked = [
        item.nodeid
        for item in items
        if not (_KNOWN_MARKERS & {m.name for m in item.iter_markers()})
    ]
    if unmarked:
        raise pytest.UsageError(
            "Unmarked tests found (policy: every test needs a known marker "
            f"{sorted(_KNOWN_MARKERS)}):\n  " + "\n  ".join(unmarked)
        )
