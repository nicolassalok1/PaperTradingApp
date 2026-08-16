"""The Options tab's global parameters belong to the user, not to a sibling tab.

`main_app` renders every top-level tab on every rerun, in `TAB_GROUPS` order —
Calibration avancée before Options. Anything Calibration writes to the shared
`common_*` keys on a plain render therefore lands *before* the Options widgets are
instantiated, and silently resets whatever the user typed. The review saw it live:
Yield-Curve toggle off, r set to 0.05, rerun -> 0.022 again (the YC rate at 2y).

Measured in a clean interpreter for the same reason as `test_app_boot.py`: the
options bridge stubs the real streamlit module at import when no ScriptRunContext
exists, and that stub is process-wide.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ENTRY = REPO_ROOT / "streamlit_app.py"

_SECRET_ENV_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "OPENAI_API_KEY",
]

_RESULT_MARKER = "PARAMS_RESULT "

# No ticker is typed, so the 25 option panels skip themselves and each rerun stays
# cheap; the global-parameter expander is rendered regardless.
_DRIVER = """
import json, socket, sys

repo_root, app_entry = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is forbidden in this test")


socket.socket.connect = _blocked
socket.create_connection = _blocked

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(app_entry, default_timeout=300)
app.run()

toggle = [w for w in app.toggle if w.key == "opt_use_yield_curve_rate"][0]
toggle.set_value(False)
app.run()

rate = [w for w in app.number_input if w.key == "common_rate_value"][0]
rate.set_value(0.05)
app.run()
after_set = float(app.session_state["common_rate_value"])

q = [w for w in app.number_input if w.key == "d_common"][0]
q.set_value(0.03)
app.run()
app.run()  # one more plain rerun: nothing else touched

print({marker!r} + json.dumps({{
    "exceptions": [str(e.value) for e in app.exception],
    "rate_after_set": after_set,
    "rate_after_reruns": float(app.session_state["common_rate_value"]),
    "q_after_reruns": float(app.session_state["d_common"]),
    "yc_toggle": bool(app.session_state["opt_use_yield_curve_rate"]),
}}))
""".format(marker=_RESULT_MARKER)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def outcome():
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS}
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(REPO_ROOT), str(APP_ENTRY)],
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
        "the driver produced no result line\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return json.loads(line)


def test_manual_rate_survives_reruns(outcome):
    assert not outcome["exceptions"], outcome["exceptions"]
    assert outcome["yc_toggle"] is False
    assert outcome["rate_after_set"] == pytest.approx(0.05)
    assert outcome["rate_after_reruns"] == pytest.approx(0.05)


def test_dividend_yield_survives_reruns(outcome):
    assert outcome["q_after_reruns"] == pytest.approx(0.03)
