"""
Unified controller for the 📊 Portfolio & Risk tab.

This module aggregates:
- Risk management metrics (exposure, VaR-lite, per-position metrics, PnL series)
- Portfolio allocation & rebalance tools (Markowitz, risk parity, eigen-portfolios)

The goal is to provide a single, thin controller per tab while keeping all
business logic inside the model layer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.model import risk_management as risk_engine
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
    """
    Lazy singleton for the AlpacaPortfolioClient used by allocation routines.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlpacaPortfolioClient()
    return _CLIENT


# --- Risk management helpers -------------------------------------------------

def get_account() -> Dict[str, Any]:
    """
    Alpaca account snapshot used for the portfolio & risk dashboard.
    """
    return risk_engine.get_account_snapshot()


def get_positions() -> List[Dict[str, Any]]:
    """
    Aggregated positions summary from the risk engine.
    """
    return risk_engine.get_positions_summary()


def get_positions_breakdown() -> Dict[str, List[Dict[str, Any]]]:
    """
    Raw positions split into equities vs options for display (does not affect equity-only metrics).
    """
    positions = risk_engine.get_positions_full()
    equities: List[Dict[str, Any]] = []
    options: List[Dict[str, Any]] = []
    for p in positions:
        cls = str(p.get("asset_class", "")).lower()
        if "option" in cls:
            options.append(p)
        elif "equity" in cls:
            equities.append(p)
    return {"equities": equities, "options": options}


def get_risk_summary(confidence: float = 0.95) -> Dict[str, Any]:
    """
    Build a compact risk summary dictionary for the view layer.

    Contains:
    - exposure / net_exposure
    - unrealized_pnl_total
    - per_position_metrics
    - var_lite
    - alerts
    - pnl_series
    """
    positions = get_positions()
    exposure = risk_engine.compute_exposure()
    net_exposure = risk_engine.compute_net_exposure()
    unrealized_pnl_total = risk_engine.compute_unrealized_pnl_total()
    per_position_metrics = [
        risk_engine.compute_position_risk_metrics(pos.get("symbol"))
        for pos in positions
        if pos.get("symbol")
    ]
    var_lite = risk_engine.compute_var_lite(confidence=confidence)
    alerts = risk_engine.trigger_alerts()
    pnl_series = risk_engine.compute_portfolio_pnl_series()

    return {
        "exposure": exposure,
        "net_exposure": net_exposure,
        "unrealized_pnl_total": unrealized_pnl_total,
        "per_position_metrics": [m for m in per_position_metrics if m],
        "var_lite": var_lite,
        "alerts": alerts,
        "pnl_series": pnl_series,
    }


# --- Allocation & rebalance helpers ------------------------------------------

def get_portfolio_snapshot() -> Dict[str, Any]:
    """
    Snapshot of the current Alpaca portfolio (equity, cash, weights).
    """
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
    """
    List of tradable symbols present in the current portfolio snapshot.
    """
    snap = get_portfolio_snapshot()
    return snap.get("symbols", [])


def compute_allocation(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    """
    Compute target portfolio weights given a risk model and lookback window.
    """
    client = _client()
    symbols = get_available_symbols()
    symbols = list(symbols or [])
    if not symbols:
        return {"method": method, "symbols": [], "target_weights": []}

    returns_data = compute_returns_matrix(client, symbols, lookback_days)
    returns = returns_data.get("returns", [])
    symbols = list(returns_data.get("symbols", symbols) or symbols)

    # If we have no return history, fall back to equal weights to avoid UI errors.
    returns_empty = False
    try:
        import numpy as _np

        if returns is None:
            returns_empty = True
        elif isinstance(returns, _np.ndarray):
            returns_empty = returns.size == 0
        else:
            returns_empty = len(returns) == 0  # type: ignore[arg-type]
    except Exception:
        returns_empty = True

    if returns_empty:
        n = len(symbols)
        return {
            "method": method,
            "symbols": symbols,
            "target_weights": [1.0 / n] * n if n > 0 else [],
        }

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

    # Ensure alignment between symbols and weights
    if len(weights) != len(symbols):
        n = len(symbols)
        weights = [1.0 / n] * n if n > 0 else []

    return {
        "method": method,
        "symbols": symbols,
        "target_weights": list(weights),
    }


def generate_rebalance_plan(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    """
    Compute a rebalance plan (desired trades) to move from current portfolio
    weights to the target allocation produced by compute_allocation().
    """
    client = _client()
    current = get_current_portfolio(client)
    target = compute_allocation(method, lookback_days)
    target_for_orders = {
        "symbols": target.get("symbols", []),
        "weights": target.get("target_weights", []),
        "target_weights": target.get("target_weights", []),
        "method": target.get("method", ""),
    }
    orders = compute_rebalance_orders(current, target_for_orders, client)
    return {
        "current": current,
        "target": target_for_orders,
        "orders": orders,
    }


def execute_rebalance(method: str, lookback_days: int = 60) -> Dict[str, Any]:
    """
    Execute the rebalance plan on Alpaca and return execution details.
    """
    client = _client()
    plan = generate_rebalance_plan(method, lookback_days)
    executions = execute_rebalance_orders(client, plan.get("orders", []))
    return {"plan": plan, "executions": executions}


__all__ = [
    # Risk metrics / exposure
    "get_account",
    "get_positions",
    "get_positions_breakdown",
    "get_risk_summary",
    # Allocation tools
    "get_portfolio_snapshot",
    "get_available_symbols",
    "compute_allocation",
    "generate_rebalance_plan",
    "execute_rebalance",
]

