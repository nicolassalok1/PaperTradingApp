"""
Buy/Sell controller.
Thin wrappers over trading services for the view layer.
"""

from __future__ import annotations

from app.model.trading.service import (
    append_trade_log,
    buy_asset,
    clear_closing_history_cache,
    compute_spot_totals_with_price,
    fetch_closing_history,
    get_market_price,
    load_trades_log,
    sell_asset,
    trade_spot_with_fallback,
    floor_2,
    floor_4,
)
from app.model.portfolio.positions import (
    load_portfolio_default as load_portfolio,
    save_portfolio_default as save_portfolio,
)
from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache

__all__ = [
    "append_trade_log",
    "floor_2",
    "floor_4",
    "buy_asset",
    "clear_closing_history_cache",
    "compute_spot_totals_with_price",
    "fetch_closing_history",
    "get_market_price",
    "load_portfolio",
    "load_trades_log",
    "load_dashboard_cache",
    "save_dashboard_cache",
    "save_portfolio",
    "sell_asset",
    "trade_spot_with_fallback",
]
