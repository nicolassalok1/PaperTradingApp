"""Paper fail-closed guard — PLAN.md step 4 trading safety."""

from __future__ import annotations

import pytest

from app.utils.trading_guard import (
    PAPER_BASE_URL,
    enforce_paper_endpoint,
    is_paper_endpoint,
)

pytestmark = pytest.mark.unit

_LIVE_URL = "https://api.alpaca.markets"


def test_empty_defaults_to_paper():
    assert enforce_paper_endpoint(None) == PAPER_BASE_URL
    assert enforce_paper_endpoint("") == PAPER_BASE_URL
    assert is_paper_endpoint(enforce_paper_endpoint("   "))


def test_paper_url_passes_through():
    assert enforce_paper_endpoint(PAPER_BASE_URL) == PAPER_BASE_URL


def test_live_url_blocked_without_optin(monkeypatch):
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="fail-closed"):
        enforce_paper_endpoint(_LIVE_URL)


def test_live_url_allowed_with_explicit_param():
    assert enforce_paper_endpoint(_LIVE_URL, allow_live=True) == _LIVE_URL


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_live_url_allowed_with_env_optin(monkeypatch, val):
    monkeypatch.setenv("ALPACA_ALLOW_LIVE", val)
    assert enforce_paper_endpoint(_LIVE_URL) == _LIVE_URL


@pytest.mark.parametrize("val", ["0", "false", "", "nope"])
def test_live_url_blocked_with_falsey_env(monkeypatch, val):
    monkeypatch.setenv("ALPACA_ALLOW_LIVE", val)
    with pytest.raises(RuntimeError):
        enforce_paper_endpoint(_LIVE_URL)
