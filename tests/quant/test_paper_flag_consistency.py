"""D1 — the `paper=` flag handed to TradingClient must agree with the guard.

`app.utils.trading_guard.enforce_paper_endpoint` correctly refuses a live endpoint
unless ALPACA_ALLOW_LIVE is set. But once the operator DOES opt in, the four already
guarded modules re-derive the client's `paper=` flag with a substring test:

    is_paper = "paper" in (base_url or "").lower()

So with ALPACA_ALLOW_LIVE=1 and a live URL that merely contains the word "paper",
the guard lets the URL through (correct — explicit opt-in) and the client is then
built with `paper=True` while it talks to the LIVE host. The two disagree, and the
client is the one that opens the socket.

ORACLE (independent of the application code): the authority parsed by
`urllib.parse.urlparse` is what the HTTP client connects to. None of the URLs below
resolves to the paper host, therefore `paper=` must be False for all of them.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.unit

PAPER_HOST = "paper-api.alpaca.markets"
PAPER_URL = f"https://{PAPER_HOST}"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Live (or third-party) URLs whose text contains "paper".
LIVE_URLS_CONTAINING_PAPER = [
    "https://paper-api.alpaca.markets@api.alpaca.markets",
    "https://api.alpaca.markets/?env=paper",
    "https://api.alpaca.markets#paper",
    "https://api-paper.evil.example",
]

# (id, module path, factory attribute) — every module that builds a TradingClient
# from an operator-supplied base URL.
ORDER_CLIENTS = [
    ("alpaca_orders", "app.model.alpaca_orders.service", "AlpacaOrdersService"),
    ("alpaca_spot", "app.model.alpaca_spot.service", "AlpacaSpotService"),
    ("hedger_v2", "app.model.hedger_v2.alpaca_client", "AlpacaHedgerClient"),
    ("portfolio_allocation", "app.model.portfolio_allocation.engine", "AlpacaPortfolioClient"),
]
_IDS = [c[0] for c in ORDER_CLIENTS]


class _RecordingTradingClient:
    """Stands in for alpaca-py's TradingClient and records how it was built."""

    def __init__(self):
        self.kwargs: dict | None = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return object()


def _build_with_recorder(monkeypatch, modpath: str, factory: str, url: str):
    monkeypatch.setenv("APCA_API_KEY_ID", "unit-test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", url)

    mod = importlib.import_module(modpath)
    recorder = _RecordingTradingClient()
    monkeypatch.setattr(mod, "TradingClient", recorder, raising=True)
    # Keep every other client offline — only the paper flag is under test.
    monkeypatch.setattr(
        mod, "StockHistoricalDataClient", lambda *a, **k: object(), raising=False
    )
    getattr(mod, factory)()
    assert recorder.kwargs is not None, f"{modpath}: TradingClient was never built"
    assert "paper" in recorder.kwargs, f"{modpath}: no `paper=` kwarg passed"
    return recorder.kwargs["paper"]


@pytest.mark.parametrize("url", LIVE_URLS_CONTAINING_PAPER)
def test_urls_are_live_hosts_despite_containing_paper(url):
    """Oracle sanity: the authority is a live/third-party host in every case."""
    assert urlparse(url).hostname != PAPER_HOST
    assert "paper" in url.lower()


@pytest.mark.parametrize("url", LIVE_URLS_CONTAINING_PAPER)
@pytest.mark.parametrize(("_id", "modpath", "factory"), ORDER_CLIENTS, ids=_IDS)
def test_paper_flag_is_false_on_a_live_host_even_with_optin(
    monkeypatch, _id, modpath, factory, url
):
    # Explicit opt-in: the guard is *supposed* to let the URL through here.
    monkeypatch.setenv("ALPACA_ALLOW_LIVE", "1")
    assert _build_with_recorder(monkeypatch, modpath, factory, url) is False


@pytest.mark.parametrize(("_id", "modpath", "factory"), ORDER_CLIENTS, ids=_IDS)
def test_paper_flag_is_true_on_the_real_paper_endpoint(monkeypatch, _id, modpath, factory):
    """Control: the genuine paper endpoint must still be flagged as paper."""
    monkeypatch.delenv("ALPACA_ALLOW_LIVE", raising=False)
    assert _build_with_recorder(monkeypatch, modpath, factory, PAPER_URL) is True


_SUBSTRING_IDIOM = re.compile(r'"paper"\s+in\b')


def test_no_module_classifies_an_endpoint_by_substring():
    """The exact-hostname guard is the single source of truth — no local re-implementation."""
    offenders: list[str] = []
    for py in sorted((REPO_ROOT / "app").rglob("*.py")):
        for lineno, line in enumerate(py.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if _SUBSTRING_IDIOM.search(line):
                offenders.append(f"{py.relative_to(REPO_ROOT).as_posix()}:{lineno}")
    assert offenders == [], (
        "these lines classify an Alpaca endpoint with a substring instead of "
        f"app.utils.trading_guard.is_paper_endpoint: {offenders}"
    )
