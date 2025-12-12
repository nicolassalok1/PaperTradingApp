"""
Controller for Dashboard v2 (Alpaca overview + PnL attribution).
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.model.dashboard_v2.engine import DashboardV2Client
from app.model.pnl_attribution.engine import compute_pnl_attribution

_CLIENT: DashboardV2Client | None = None


def get_client() -> DashboardV2Client:
    """Singleton accessor for DashboardV2Client (supports offline fallback)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = DashboardV2Client()
    return _CLIENT


def get_account_summary() -> Dict[str, float]:
    return get_client().get_account_summary()


def get_spot_positions() -> List[Dict[str, Any]]:
    return get_client().get_spot_positions()


def get_option_positions() -> List[Dict[str, Any]]:
    return get_client().get_option_positions()


def get_placeholder_positions() -> Dict[str, List[Dict[str, Any]]]:
    return get_client().load_placeholders()


def get_equity_curve(lookback_days: int = 60) -> List[Dict[str, Any]]:
    return get_client().get_equity_curve(lookback_days=lookback_days)


def get_pnl_timeseries(lookback_days: int = 60) -> List[Dict[str, Any]]:
    return get_client().get_pnl_timeseries(lookback_days=lookback_days)


def get_trade_history(limit: int = 200, days_back: int = 30) -> List[Dict[str, Any]]:
    return get_client().get_trade_history(limit=limit, days_back=days_back)


def get_live_risk_snapshot(lookback_days: int = 60) -> Dict[str, Any]:
    return get_client().get_live_risk_snapshot(lookback_days=lookback_days)


def get_drawdowns(lookback_days: int = 90) -> Dict[str, Any]:
    eq_curve = get_equity_curve(lookback_days)
    equity_series = [pt.get("equity", 0.0) for pt in eq_curve]
    drawdowns = get_client().compute_drawdowns(equity_series)
    drawdowns["dates"] = [pt.get("date") for pt in eq_curve]
    return drawdowns


def get_volatility(lookback_days: int = 90, window: int = 20) -> List[Dict[str, Any]]:
    eq_curve = get_equity_curve(lookback_days)
    eq_series = [pt.get("equity", 0.0) for pt in eq_curve]
    returns = []
    for i in range(1, len(eq_series)):
        prev = eq_series[i - 1]
        curr = eq_series[i]
        ret = (curr - prev) / prev if prev else 0.0
        returns.append(ret)
    vols = get_client().compute_volatility(returns, window=window)
    vol_points: List[Dict[str, Any]] = []
    # align vols with dates (skip first point due to returns len = n-1)
    for d, v in zip([pt.get("date") for pt in eq_curve][1:], vols):
        vol_points.append({"date": d, "vol": v})
    return vol_points


def get_exposure_by_symbol() -> List[Dict[str, Any]]:
    return get_client().get_exposure_by_symbol()


def get_exposure_by_sector() -> List[Dict[str, Any]]:
    return get_client().get_exposure_by_sector()


def get_pnl_attribution() -> Dict[str, Any]:
    """PnL attribution for the current dashboard universe."""
    return compute_pnl_attribution(get_client())


__all__ = [
    "get_account_summary",
    "get_spot_positions",
    "get_option_positions",
    "get_placeholder_positions",
    "get_equity_curve",
    "get_pnl_timeseries",
    "get_trade_history",
    "get_live_risk_snapshot",
    "get_drawdowns",
    "get_volatility",
    "get_exposure_by_symbol",
    "get_exposure_by_sector",
    "get_pnl_attribution",
]
