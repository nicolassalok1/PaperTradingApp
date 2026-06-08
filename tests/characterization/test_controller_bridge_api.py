"""
Step-0 safety net — CHARACTERIZATION SNAPSHOT of controller_bridge's PUBLIC API.

Captures the public surface (``__all__`` + signatures of public callables) of
``app.vue.components.options.controller_bridge`` into a baseline JSON, BEFORE any
refactor touches it. The Step-6 refactor (god-object split, killing ``import *``)
must keep this snapshot green — the public façade stays stable.

First run with no baseline: the baseline is written and the test passes (it records
the current behavior). Subsequent runs compare against it. To intentionally update
the baseline after an approved change: delete the JSON and re-run.

Run: conda run -n papertrading python -m pytest tests/characterization -q
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BRIDGE = "app.vue.components.options.controller_bridge"
BASELINE = Path(__file__).resolve().parent / "controller_bridge_api.json"


def _public_api() -> dict:
    mod = importlib.import_module(BRIDGE)
    names = sorted(getattr(mod, "__all__", []))
    api: dict = {"all": names, "callables": {}}
    for name in names:
        obj = getattr(mod, name, None)
        if callable(obj):
            try:
                api["callables"][name] = str(inspect.signature(obj))
            except (TypeError, ValueError):
                api["callables"][name] = "<no-signature>"
    return api


@pytest.mark.characterization
def test_controller_bridge_public_api_stable():
    current = _public_api()

    if not BASELINE.exists():
        BASELINE.write_text(
            json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pytest.skip(f"Baseline created at {BASELINE.name} — re-run to compare.")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    missing = sorted(set(baseline["all"]) - set(current["all"]))
    added = sorted(set(current["all"]) - set(baseline["all"]))
    changed = {
        n: (baseline["callables"][n], current["callables"].get(n))
        for n in baseline["callables"]
        if n in current["callables"]
        and baseline["callables"][n] != current["callables"][n]
    }

    assert not missing, f"Public symbols REMOVED from controller_bridge: {missing}"
    assert not changed, f"Public signatures CHANGED: {changed}"
    # `added` is allowed (extending the façade is fine); surfaced for awareness.
    if added:
        print(f"[characterization] new public symbols (allowed): {added}")
