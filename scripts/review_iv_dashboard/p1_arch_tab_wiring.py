"""Probe: does main_app's autodiscovery resolve tab_iv_dashboard to the label used in TAB_GROUPS?

Offline: strips Alpaca keys and blocks sockets before importing main_app.
"""
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
    os.environ.pop(k, None)


def _blocked(*a, **k):
    raise RuntimeError("network blocked by probe")


socket.socket.connect = _blocked  # type: ignore[assignment]
socket.create_connection = _blocked  # type: ignore[assignment]

from app.vue import main_app  # noqa: E402

tabs = main_app.autodiscover_tabs()
labels = main_app.ordered_tab_labels(tabs)
print("discovered labels :", sorted(tabs))
print("ordered top-level :", labels)
print("iv tab discovered :", "🌡️ Vol Implicite" in tabs)
print("iv tab in order   :", "🌡️ Vol Implicite" in labels, "position", labels.index("🌡️ Vol Implicite") if "🌡️ Vol Implicite" in labels else None)
fn = tabs.get("🌡️ Vol Implicite")
print("render fn         :", getattr(fn, "__module__", None), getattr(fn, "__name__", None))
print("override == TAB_LABEL :", main_app.DEFAULT_LABEL_OVERRIDES["tab_iv_dashboard"] == sys.modules["app.vue.tabs.tab_iv_dashboard"].TAB_LABEL)
print("n top-level tabs  :", len(labels), "| n discovered", len(tabs))
