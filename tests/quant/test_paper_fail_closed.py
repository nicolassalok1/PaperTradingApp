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


# --------------------------------------------------------------------------- #
# Every ORDER-capable Alpaca client must route through the guard: a live        #
# endpoint without ALPACA_ALLOW_LIVE must fail-closed at construction time.     #
# --------------------------------------------------------------------------- #
def _build_portfolio_client():
    from app.model.portfolio_allocation.engine import AlpacaPortfolioClient

    return AlpacaPortfolioClient()


def _build_hedger_client():
    from app.model.hedger_v2.alpaca_client import AlpacaHedgerClient

    return AlpacaHedgerClient()


def _build_orders_keys():
    from app.model.alpaca_orders.service import AlpacaKeys

    return AlpacaKeys.from_env()


def _build_spot_keys():
    from app.model.alpaca_spot.service import AlpacaKeys

    return AlpacaKeys.from_env()


_ORDER_CLIENT_BUILDERS = [
    _build_portfolio_client,
    _build_hedger_client,
    _build_orders_keys,
    _build_spot_keys,
]


@pytest.mark.parametrize("builder", _ORDER_CLIENT_BUILDERS)
def test_order_clients_failclose_on_live_without_optin(monkeypatch, builder):
    monkeypatch.setenv("APCA_API_KEY_ID", "dummy-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "dummy-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", _LIVE_URL)
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="fail-closed|blocked"):
        builder()


@pytest.mark.parametrize("builder", _ORDER_CLIENT_BUILDERS)
def test_order_clients_ok_on_paper(monkeypatch, builder):
    # Paper endpoint + dummy keys must NOT raise the live guard (offline is fine).
    monkeypatch.setenv("APCA_API_KEY_ID", "dummy-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "dummy-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_BASE_URL)
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)
    builder()  # must not raise the fail-closed guard


def test_rebalance_controller_blocks_live_without_optin(monkeypatch):
    monkeypatch.setenv("APCA_API_BASE_URL", _LIVE_URL)
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)
    from app.controller import portfolio_and_risk_controller as ctrl

    out = ctrl.execute_rebalance("equal_weight")
    assert out["executed"] is False
    assert out["status"] == "blocked_live_endpoint"
