"""C3 — the paper/live guard must be applied at EVERY Alpaca client construction site.

``app.utils.trading_guard`` is correct (it compares the exact hostname) and is already
covered by ``tests/quant/test_paper_fail_closed.py``. What is tested HERE is that the
call sites actually route through it, instead of re-implementing a ``"paper" in url``
substring test of their own.

ORACLE (independent of the application code): the *authority* of a URL, as defined by
RFC 3986 and parsed by :func:`urllib.parse.urlparse`, is what a HTTP client connects to.
For each URL below the real host is a live / third-party host, never the paper host:

    https://paper-api.alpaca.markets@api.alpaca.markets  -> api.alpaca.markets   (userinfo)
    https://api.alpaca.markets/?env=paper                -> api.alpaca.markets   (query)
    https://api.alpaca.markets#paper                     -> api.alpaca.markets   (fragment)
    https://api-paper.evil.example                       -> api-paper.evil.example (look-alike)

Every one of them *contains* the substring "paper", so any ``"paper" in url`` check
classifies them as paper trading and lets the caller reach a live endpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.unit

PAPER_HOST = "paper-api.alpaca.markets"

# URLs whose real host is NOT the paper host but whose text contains "paper".
HOSTILE_URLS = [
    "https://paper-api.alpaca.markets@api.alpaca.markets",
    "https://api.alpaca.markets/?env=paper",
    "https://api.alpaca.markets#paper",
    "https://api-paper.evil.example",
]

# Repo root = tests/quant/<file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _set_live_like_env(monkeypatch, url: str) -> None:
    """Credentials that pass the app's own 'dummy key' filters, plus a hostile base URL.

    Deliberately NOT shaped like a real Alpaca key (no PK/AK prefix), so
    scripts/scan_secrets.py stays green on this file.
    """
    monkeypatch.setenv("APCA_API_KEY_ID", "unit-test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", url)
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)


# --------------------------------------------------------------------------- #
# Oracle sanity: these URLs really are not the paper endpoint.                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_hostile_urls_do_not_resolve_to_the_paper_host(url):
    assert urlparse(url).hostname != PAPER_HOST
    # ...yet a substring check would wave every one of them through.
    assert "paper" in url.lower()


# --------------------------------------------------------------------------- #
# C3.1 grid_bot — submits real orders, must refuse a non-paper endpoint.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_grid_bot_refuses_hostile_base_url(monkeypatch, url):
    from app.model.bots import grid_bot
    from app.model.bots.storage import GridBotConfig

    _set_live_like_env(monkeypatch, url)
    # Keep the run offline and deterministic: the grid only needs a reference price.
    monkeypatch.setattr(grid_bot, "fetch_spot_price", lambda symbol: 100.0, raising=True)

    cfg = GridBotConfig(symbol="AAPL", enabled=True, qty=1.0, n_levels=3, step_pct=0.05,
                        dry_run=False)
    report = grid_bot.run_grid_bot_once(cfg, allow_submit=True, allow_live=False)

    assert report.to_submit_prices, "test setup: the grid must have orders to submit"
    assert any("Refusing to submit orders" in e for e in report.errors), report.errors
    assert report.submitted_orders == []


def test_grid_bot_accepts_the_real_paper_endpoint(monkeypatch):
    """Control: the same run against the true paper endpoint raises no refusal."""
    from app.model.bots import grid_bot
    from app.model.bots.storage import GridBotConfig

    _set_live_like_env(monkeypatch, f"https://{PAPER_HOST}")
    monkeypatch.setattr(grid_bot, "fetch_spot_price", lambda symbol: 100.0, raising=True)
    # Stay fully offline: only the paper/live classification is under test here.
    monkeypatch.setattr(grid_bot, "_try_orders_service", lambda: (None, "offline"), raising=True)

    cfg = GridBotConfig(symbol="AAPL", enabled=True, qty=1.0, n_levels=3, step_pct=0.05,
                        dry_run=False)
    report = grid_bot.run_grid_bot_once(cfg, allow_submit=True, allow_live=False)

    assert not any("Refusing to submit orders" in e for e in report.errors), report.errors


# --------------------------------------------------------------------------- #
# C3.2 risk_management — builds a TradingClient.                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_risk_engine_refuses_hostile_base_url(monkeypatch, url):
    from app.model.risk_management.engine import AlpacaKeys, RiskEngine

    _set_live_like_env(monkeypatch, url)

    with pytest.raises(RuntimeError, match="fail-closed|blocked"):
        AlpacaKeys.from_env()

    # And the engine must end up with no trading client at all.
    engine = RiskEngine()
    assert engine.offline is True
    assert engine.trading_client is None


# --------------------------------------------------------------------------- #
# C3.3 dashboard_v2 — builds a TradingClient.                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_dashboard_v2_client_refuses_hostile_base_url(monkeypatch, url):
    from app.model.dashboard_v2.engine import DashboardV2Client

    _set_live_like_env(monkeypatch, url)

    with pytest.raises(RuntimeError, match="fail-closed|blocked"):
        DashboardV2Client()


# --------------------------------------------------------------------------- #
# C3.4 options.logic — feeds AlpacaREST(...) at logic.py:906.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_options_logic_credentials_refuse_hostile_base_url(monkeypatch, url):
    from app.model.options.logic import _load_alpaca_credentials

    _set_live_like_env(monkeypatch, url)

    with pytest.raises(RuntimeError, match="fail-closed|blocked"):
        _load_alpaca_credentials()


# --------------------------------------------------------------------------- #
# C3.5 market_data — also feeds AlpacaREST(...) (make_alpaca_client).            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_market_data_credentials_refuse_hostile_base_url(monkeypatch, url):
    from app.model.market_data.market_data import _alpaca_credentials

    _set_live_like_env(monkeypatch, url)

    with pytest.raises(RuntimeError, match="fail-closed|blocked"):
        _alpaca_credentials()


# --------------------------------------------------------------------------- #
# Structural: no Alpaca client may be built by a module that ignores the guard. #
# --------------------------------------------------------------------------- #
_CLIENT_CTOR = re.compile(r"\b(?:TradingClient|AlpacaREST)\s*\(")


def test_every_module_building_an_alpaca_client_imports_the_guard():
    offenders: list[str] = []
    builders: list[str] = []
    for py in sorted((REPO_ROOT / "app").rglob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        if not _CLIENT_CTOR.search(src):
            continue
        rel = py.relative_to(REPO_ROOT).as_posix()
        builders.append(rel)
        if "app.utils.trading_guard" not in src:
            offenders.append(rel)

    assert builders, "test setup: no Alpaca client construction site found at all"
    assert offenders == [], (
        "these modules build an Alpaca client without importing app.utils.trading_guard: "
        f"{offenders}"
    )
