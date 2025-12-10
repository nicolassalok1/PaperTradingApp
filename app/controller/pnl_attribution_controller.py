"""
Controller for PnL attribution.
"""

from __future__ import annotations

from typing import Any, Dict

from app.model.dashboard_v2.engine import DashboardV2Client
from app.model.pnl_attribution.engine import compute_pnl_attribution

_CLIENT: DashboardV2Client | None = None


def _client() -> DashboardV2Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = DashboardV2Client()
    return _CLIENT


def get_pnl_attribution() -> Dict[str, Any]:
    return compute_pnl_attribution(_client())


__all__ = ["get_pnl_attribution"]
