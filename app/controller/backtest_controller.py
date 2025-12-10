"""
Backtest controller.
Exposes model-layer helpers for trading systems/backtests to the view.
"""

from __future__ import annotations

from app.model.backtesting.engine import (
    add_trading_system,
    auto_execute_trading_levels,
    remove_trading_system,
    set_trading_system_status,
)
from app.model.market_data.realtime import get_data
from app.model.market_data.market_data import (
    clear_closing_history_cache,
    load_or_fetch_closing_history,
    fetch_spot_price,
)
from app.model.trading.systems import (
    load_equities,
    load_ts_exec_log,
    save_equities,
    save_ts_exec_log,
)
from app.model.portfolio.positions import load_portfolio_default, save_portfolio_default
from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache
from app.model.trading.service import (
    buy_asset_with_balance,
    sell_asset_with_balance,
    update_dashboard_balance,
    floor_2,
    floor_4,
)
from app.utils.paths import CACHE_CSV_DIR

__all__ = [
    "add_trading_system",
    "auto_execute_trading_levels",
    "remove_trading_system",
    "set_trading_system_status",
    "get_data",
    "clear_closing_history_cache",
    "load_or_fetch_closing_history",
    "buy_asset_with_balance",
    "sell_asset_with_balance",
    "load_equities",
    "save_equities",
    "load_ts_exec_log",
    "save_ts_exec_log",
    "load_portfolio_default",
    "save_portfolio_default",
    "load_dashboard_cache",
    "save_dashboard_cache",
    "update_dashboard_balance",
    "floor_4",
    "floor_2",
    "fetch_spot_price",
    "CACHE_CSV_DIR",
]
