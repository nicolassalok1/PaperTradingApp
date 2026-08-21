"""
Render guard for the 🌡️ Vol Implicite tab.

Runs `AppTest` in a pristine subprocess for the same reason as
`test_app_boot.py`: once any in-process test imports
`app/vue/components/options/controller_bridge.py`, the real `streamlit` module
is permanently stubbed (`st.columns` & co.), so in-process AppTest results
depend on test order. A clean interpreter makes this order-independent.

The child (`_iv_dashboard_render_driver.py`) blocks outbound sockets itself
(`--disable-socket` does not reach it), then renders the tab twice: with a
seeded analysis payload (three Plotly charts + metrics must appear) and
without one (info placeholder only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().parent / "_iv_dashboard_render_driver.py"

_RESULT_MARKER = "IVDASH_RESULT "


@pytest.fixture(scope="module")
def render_result():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "OPENAI_API_KEY"}
    }
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, str(DRIVER), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
        cwd=str(REPO_ROOT),
    )
    line = next(
        (
            ln[len(_RESULT_MARKER) :]
            for ln in completed.stdout.splitlines()
            if ln.startswith(_RESULT_MARKER)
        ),
        None,
    )
    assert line is not None, (
        "the render process produced no result line\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return json.loads(line)


def test_tab_renders_seeded_payload_without_raising(render_result):
    assert not render_result["seeded"]["exceptions"], render_result["seeded"]["exceptions"]


def test_tab_shows_three_charts_and_metrics(render_result):
    assert render_result["seeded"]["n_charts"] == 3, render_result["seeded"]
    assert render_result["seeded"]["n_metrics"] >= 3, render_result["seeded"]


def test_tab_shows_no_mean_reversion_signal_on_iv_percentile(render_result):
    """Review M2: the IV-within-RV percentile is shown, but never dressed as a regime signal."""
    seeded = render_result["seeded"]
    assert not seeded["has_iv_signal_chip"], seeded
    assert seeded["has_vrp_caption"], seeded


def test_tab_empty_state_shows_placeholder(render_result):
    empty = render_result["empty"]
    assert not empty["exceptions"], empty["exceptions"]
    assert empty["has_info"], empty
    assert empty["n_charts"] == 0, empty
