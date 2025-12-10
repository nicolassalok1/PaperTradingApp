"""
Controller for Alpaca-based portfolio allocation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.model.portfolio_allocation.engine import (
    AlpacaPortfolioClient,
    compute_rebalance_orders,
    compute_returns_matrix,
    eigen_portfolio_optimize,
    execute_rebalance_orders,
    get_current_portfolio,
    markowitz_optimize,
    risk_parity_optimize,
)

_CLIENT: AlpacaPortfolioClient | None = None


def _client() -> AlpacaPortfolioClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlpacaPortfolioClient()
    return _CLIENT


def get_portfolio_snapshot() -> Dict[str, Any]:
    client = _client()
    snapshot = get_current_portfolio(client)
    return {
        "equity": snapshot.get("equity", 0.0),
        "cash": snapshot.get("cash", 0.0),
        "symbols": snapshot.get("symbols", []),
        "current_weights": snapshot.get("weights", []),
        "market_values": snapshot.get("market_values", []),
    }


def get_available_symbols() -> List[str]:
    snap = get_portfolio_snapshot()
    return snap.get("symbols", [])


def compute_allocation(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    client = _client()
    symbols = get_available_symbols()
    returns_data = compute_returns_matrix(client, symbols, lookback_days)
    returns = returns_data.get("returns", [])

    if method == "markowitz_min_var":
        weights = markowitz_optimize(returns, mode="min_var")
    elif method == "markowitz_max_sharpe":
        weights = markowitz_optimize(returns, mode="max_sharpe")
    elif method == "risk_parity":
        weights = risk_parity_optimize(returns)
    elif method == "eigen":
        weights = eigen_portfolio_optimize(returns)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "symbols": returns_data.get("symbols", symbols),
        "target_weights": list(weights),
    }


def generate_rebalance_plan(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    client = _client()
    current = get_current_portfolio(client)
    target = compute_allocation(method, lookback_days)
    target_for_orders = {
        "symbols": target.get("symbols", []),
        "weights": target.get("target_weights", []),
        "method": target.get("method", ""),
    }
    orders = compute_rebalance_orders(current, target_for_orders, client)
    return {
        "current": current,
        "target": target_for_orders,
        "orders": orders,
    }


def execute_rebalance(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    client = _client()
    plan = generate_rebalance_plan(method, lookback_days)
    executions = execute_rebalance_orders(client, plan.get("orders", []))
    return {"plan": plan, "executions": executions}


__all__ = [
    "get_portfolio_snapshot",
    "get_available_symbols",
    "compute_allocation",
    "generate_rebalance_plan",
    "execute_rebalance",
]
