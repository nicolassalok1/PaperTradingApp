"""Regression: Dashboard v2 tab degrades gracefully on Alpaca data failure [E02].

Before the fix, an Alpaca auth/API error (e.g. revoked credentials — which the engine's
offline fallback does NOT catch, since it only triggers on empty/"dummy" keys) propagated
out of ``render_tab()`` as a raw exception. The top-level handler in ``main_app`` then
rendered it as a red ``st.exception`` traceback box — unlike every other Alpaca tab, which
catches the error and shows a clean ``st.error``. This test locks the consistent behaviour.

Run: conda run -n papertrading python -m pytest tests/test_dashboard_v2_graceful.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.mark.unit
def test_dashboard_renders_clean_error_when_account_fetch_fails(monkeypatch):
    from app.vue.tabs import tab_dashboard_v2 as tab

    def _boom():
        # Mirrors an alpaca APIError on revoked credentials.
        raise RuntimeError('{"message": "unauthorized."}')

    # The account summary is the first eager Alpaca-backed call in render_tab().
    monkeypatch.setattr(tab.ctrl, "get_account_summary", _boom, raising=True)

    errors: list[str] = []
    monkeypatch.setattr(
        tab.st, "error", lambda msg, *a, **k: errors.append(str(msg)), raising=True
    )
    # A raw red traceback box must never be produced by the tab itself.
    monkeypatch.setattr(
        tab.st,
        "exception",
        lambda *a, **k: pytest.fail("render_tab must not call st.exception"),
        raising=True,
    )

    # Must NOT raise — graceful degradation, unlike the pre-fix behaviour.
    tab.render_tab()

    assert errors, "expected a clean st.error message"
    assert any("Unable to load account" in e for e in errors), errors
