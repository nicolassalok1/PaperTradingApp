"""
Simple PnL attribution by symbol.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.model.dashboard_v2.engine import DashboardV2Client, _to_dict


def get_symbol_pnl_breakdown(positions: List[Dict[str, Any]], total_portfolio_value: float) -> List[Dict[str, Any]]:
    total_portfolio_value = float(total_portfolio_value or 0.0)
    breakdown: List[Dict[str, Any]] = []
    for p in positions:
        qty = float(p.get("qty", 0.0) or 0.0)
        mv = float(p.get("market_value", 0.0) or 0.0)
        avg_entry = float(p.get("avg_entry_price", 0.0) or 0.0)
        cost_basis = avg_entry * qty
        unrealized = mv - cost_basis
        weight = (mv / total_portfolio_value) if total_portfolio_value else 0.0
        breakdown.append(
            {
                "symbol": p.get("symbol"),
                "asset_class": p.get("asset_class"),
                "qty": qty,
                "market_value": mv,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized,
                "weight_portfolio": weight,
            }
        )
    return breakdown


def compute_pnl_attribution(client: DashboardV2Client) -> Dict[str, Any]:
    spot = client.get_spot_positions()
    opt = client.get_option_positions()
    all_positions = spot + opt
    summary = client.get_account_summary()
    total_pv = summary.get("portfolio_value", 0.0)
    by_symbol = get_symbol_pnl_breakdown(all_positions, total_pv)
    total_unrealized = sum(item.get("unrealized_pnl", 0.0) or 0.0 for item in by_symbol)
    total_realized = summary.get("realized_pl_total", 0.0)
    return {
        "by_symbol": by_symbol,
        "total_unrealized": total_unrealized,
        "total_realized": total_realized,
    }


__all__ = ["compute_pnl_attribution", "get_symbol_pnl_breakdown"]
