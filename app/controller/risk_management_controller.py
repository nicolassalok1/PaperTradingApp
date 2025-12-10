"""
Controller for risk management dashboard (thin wrappers over model engine).
"""

from __future__ import annotations

from typing import Any

from app.model import risk_management as engine


def get_account() -> dict[str, Any]:
    return engine.get_account_snapshot()


def get_positions() -> list[dict[str, Any]]:
    return engine.get_positions_summary()


def get_risk_summary(confidence: float = 0.95) -> dict[str, Any]:
    positions = get_positions()
    exposure = engine.compute_exposure()
    net_exposure = engine.compute_net_exposure()
    unrealized_pnl_total = engine.compute_unrealized_pnl_total()
    per_position_metrics = [
        engine.compute_position_risk_metrics(pos.get("symbol")) for pos in positions if pos.get("symbol")
    ]
    var_lite = engine.compute_var_lite(confidence=confidence)
    alerts = engine.trigger_alerts()
    pnl_series = engine.compute_portfolio_pnl_series()

    return {
        "exposure": exposure,
        "net_exposure": net_exposure,
        "unrealized_pnl_total": unrealized_pnl_total,
        "per_position_metrics": [m for m in per_position_metrics if m],
        "var_lite": var_lite,
        "alerts": alerts,
        "pnl_series": pnl_series,
    }


__all__ = [
    "get_account",
    "get_positions",
    "get_risk_summary",
]
