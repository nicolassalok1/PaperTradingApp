"""
Permanent boot guard for the Streamlit app.

Pass 3 booted the real app once, by hand, and dumped `sys.modules` to prove which
modules load at runtime — the evidence that let E4 delete nine of them. Nothing
kept that proof alive: no test boots the app, so a tab that starts raising on
import or on first render now fails in front of a user, not in CI. This restores
the boot as a standing check instead of a one-shot measurement.

Why a subprocess, and not `AppTest` in-process
----------------------------------------------
`app/vue/components/options/controller_bridge.py` calls `_bootstrap_fake_streamlit()`
at import time. With no `ScriptRunContext` around — i.e. under plain pytest — it
overwrites `st.write`, `st.markdown`, `st.columns`, `st.session_state` and a dozen
more on the *real* `streamlit` module with no-op lambdas, and never restores them.
That is deliberate (it lets the option panels run headless) but it is process-wide
and permanent: once any test imports the bridge, every later `AppTest` boot in the
same interpreter renders against a gutted streamlit and reports failures that say
nothing about the app. In isolation the boot passes; behind `tests/smoke/
test_offline_imports.py` it does not.

Booting is a whole-process property, so it is measured in a clean process. That
also makes this test order-independent, which the in-process version was not.

What is asserted is derived from intent, never recorded from a previous run: the
app raises nothing, every discovered tab module reaches the screen, and the only
errors printed are the missing-credentials degradation. The tab *count* is
deliberately not pinned — it is an output of the code, not a contract.
"""

from __future__ import annotations

import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ENTRY = REPO_ROOT / "streamlit_app.py"
TABS_DIR = REPO_ROOT / "app" / "vue" / "tabs"

# No `.env` in the repo puts the app in offline mode; strip any inherited keys so
# a developer's shell cannot turn this into a live Alpaca call.
_SECRET_ENV_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "OPENAI_API_KEY",
]

# Offline boot cannot reach the broker, and the app says so. Anything else on the
# error channel is a real regression.
_CREDENTIAL_DEGRADATION = "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set"

_RESULT_MARKER = "BOOT_RESULT "

# `--disable-socket` is a pytest plugin and does not reach the child, so the child
# blocks outbound network itself before importing anything from the app.
_BOOT_DRIVER = """
import json, socket, sys

repo_root, app_entry = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is forbidden in the boot guard")


socket.socket.connect = _blocked
socket.create_connection = _blocked

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(app_entry, default_timeout=300)
app.run()

print({marker!r} + json.dumps({{
    "exceptions": [str(e.value) for e in app.exception],
    "errors": [str(e.value) for e in app.error],
    "tabs": len(app.tabs),
    "tab_modules": sorted(m for m in sys.modules if m.startswith("app.vue.tabs.tab_")),
}}))
""".format(marker=_RESULT_MARKER)

pytestmark = pytest.mark.integration


def _tab_module_count() -> int:
    return len(
        [m for m in pkgutil.iter_modules([str(TABS_DIR)]) if m.name.startswith("tab_")]
    )


@pytest.fixture(scope="module")
def boot():
    """Boot the app once, in a pristine interpreter, and return its summary."""
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS}
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, "-c", _BOOT_DRIVER, str(REPO_ROOT), str(APP_ENTRY)],
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
        "the boot process produced no result line\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return json.loads(line)


def test_the_app_boots_without_raising(boot):
    assert not boot["exceptions"], boot["exceptions"]


def test_every_discovered_tab_reaches_the_screen(boot):
    """`autodiscover_tabs` imports `app/vue/tabs/tab_*.py` by importlib + pkgutil.

    Static analysis cannot see those edges, so any dead-code sweep has to treat the
    loader as a root — the assumption E4's deletions rest on. Asserting on rendered
    tabs keeps this honest: a `sys.modules` check alone would pass even if the boot
    rendered nothing, since other tests import the tab modules anyway.
    """
    expected = _tab_module_count()
    assert len(boot["tab_modules"]) == expected, boot["tab_modules"]
    assert boot["tabs"] >= expected


def test_offline_boot_complains_only_about_missing_credentials(boot):
    """Booting with no keys degrades; it must not fail for any other reason."""
    unexpected = [e for e in boot["errors"] if _CREDENTIAL_DEGRADATION not in e]
    assert not unexpected, unexpected
