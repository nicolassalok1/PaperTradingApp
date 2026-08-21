"""
Step-0 safety net — OFFLINE IMPORT SMOKE.

Imports every tab + controller + the options controller_bridge with:
  - no API keys in the environment,
  - network access blocked (any socket connect raises),
  - no training / no external client construction expected at import time.

This is a non-regression guard: it must stay green after every reliability step.
It deliberately scopes "side effects" to NETWORK calls (the dangerous kind).
The benign streamlit-stub side effect in controller_bridge is allowed.

Run: conda run -n papertrading python -m pytest tests/smoke -q
"""

from __future__ import annotations

import importlib
import pkgutil
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keys that must NOT be required for a clean import.
_SECRET_ENV_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "OPENAI_API_KEY",
]

CONTROLLERS = [
    "calibration_controller",
    "dashboard_v2_controller",
    "hedger_v2_controller",
    "options_controller",
    "portfolio_and_risk_controller",
    "trading_controller",
    "yieldcurve_controller",
]

BRIDGE = "app.vue.components.options.controller_bridge"


def _tab_modules() -> list[str]:
    tabs_dir = REPO_ROOT / "app" / "vue" / "tabs"
    return [
        f"app.vue.tabs.{m.name}"
        for m in pkgutil.iter_modules([str(tabs_dir)])
        if m.name.startswith("tab_")
    ]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Strip secrets and block all outbound network for the duration of the test."""
    for k in _SECRET_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    def _blocked(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError(
            "Network access during import is forbidden (offline smoke test)."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=True)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=True)
    yield


@pytest.mark.smoke
@pytest.mark.parametrize("modname", _tab_modules())
def test_tab_imports_offline(modname):
    importlib.import_module(modname)


@pytest.mark.smoke
@pytest.mark.parametrize("ctrl", CONTROLLERS)
def test_controller_imports_offline(ctrl):
    importlib.import_module(f"app.controller.{ctrl}")


@pytest.mark.smoke
def test_controller_bridge_imports_offline():
    importlib.import_module(BRIDGE)


@pytest.mark.smoke
def test_all_ten_tabs_present():
    # Locks the tab-count invariant. 11 from the original audit, +1 for the
    # "🧪 Exercices" tab, then -2 when "🤖 Bots" and "🧪 Exercices" were retired (H1),
    # +1 for the "🌡️ Vol Implicite" tab (IV dashboard on Alpaca data).
    mods = _tab_modules()
    assert len(mods) == 11, mods
    assert "app.vue.tabs.tab_iv_dashboard" in mods, mods
    assert "app.vue.tabs.tab_bots" not in mods, mods
    assert "app.vue.tabs.tab_exercices" not in mods, mods
