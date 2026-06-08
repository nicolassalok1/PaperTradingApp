"""
Fail-closed trading guard (pure utils — no view/model/controller deps).

PaperTradingApp is a PAPER trading app. Hitting Alpaca's LIVE endpoint must be
impossible by accident: the default is always paper, and a live endpoint is only
allowed when the operator opts in EXPLICITLY via the ALPACA_ALLOW_LIVE env flag.
Anything ambiguous fails closed (defaults to paper).
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_PAPER_HOST = "paper-api.alpaca.markets"
_LIVE_OPT_IN_ENV = "ALPACA_ALLOW_LIVE"
_TRUTHY = {"1", "true", "yes", "on"}


def _hostname(base_url: str | None) -> str:
    u = (base_url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "//" + u  # let urlparse treat a bare host as netloc
    return (urlparse(u).hostname or "").lower()


def is_paper_endpoint(base_url: str | None) -> bool:
    # Exact hostname match (not substring): defeats the userinfo trick
    # https://paper-api.alpaca.markets@api.alpaca.markets whose real host is live.
    return _hostname(base_url) == _PAPER_HOST


def live_opt_in_enabled() -> bool:
    return os.getenv(_LIVE_OPT_IN_ENV, "").strip().lower() in _TRUTHY


def enforce_paper_endpoint(base_url: str | None, *, allow_live: bool | None = None) -> str:
    """
    Return a safe Alpaca base URL or raise.

    - Empty/None  -> default PAPER endpoint (fail-closed).
    - Paper URL   -> returned as-is.
    - Live URL    -> allowed ONLY if `allow_live` is True, or (when None) the
                     ALPACA_ALLOW_LIVE env flag is explicitly truthy. Otherwise
                     raises RuntimeError.
    """
    url = (base_url or "").strip()
    if not url:
        return PAPER_BASE_URL
    if is_paper_endpoint(url):
        return url

    allowed = live_opt_in_enabled() if allow_live is None else bool(allow_live)
    if not allowed:
        raise RuntimeError(
            "Live Alpaca endpoint is blocked (fail-closed). This is a paper-trading "
            f"app. To use a live endpoint, set {_LIVE_OPT_IN_ENV}=1 explicitly."
        )
    return url
