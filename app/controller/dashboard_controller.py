"""
Dashboard controller.
Thin wrappers over model.dashboard services for the view layer.
"""

from __future__ import annotations

from app.model.dashboard import (
    DASHBOARD_VARS_FILE,
    auto_execute_trading_levels,
    chatgpt_response,
    collect_dashboard_tickers,
    compute_forward_pnl,
    compute_spot_pnl,
    dashboard_cache_last_refresh,
    dashboard_price,
    fetch_spot_price,
    load_dashboard_cache,
    load_expired_options,
    load_forwards_data,
    load_options,
    load_portfolio,
    load_sell_systems,
    load_trades_log,
    log_trade,
    refresh_all_spots_pipeline,
    refresh_dashboard_cache,
    refresh_spot_prices_with_systems,
    reset_dashboard,
    save_dashboard_cache,
    save_equities,
    save_expired_options,
    save_forwards_data,
    save_options,
    save_portfolio,
)
from app.model.trading.systems import load_equities, load_ts_exec_log, save_ts_exec_log
from app.model.market_data.history import load_or_fetch_closing_history
from app.model.trading.execution import compute_spot_totals

__all__ = [
    "DASHBOARD_VARS_FILE",
    "chatgpt_response",
    "collect_dashboard_tickers",
    "compute_forward_pnl",
    "compute_spot_pnl",
    "dashboard_cache_last_refresh",
    "dashboard_price",
    "fetch_spot_price",
    "load_dashboard_cache",
    "load_equities",
    "load_expired_options",
    "load_forwards_data",
    "load_options",
    "load_portfolio",
    "load_trades_log",
    "load_ts_exec_log",
    "refresh_all_spots_pipeline",
    "refresh_dashboard_cache",
    "refresh_spot_prices_with_systems",
    "reset_dashboard",
    "save_dashboard_cache",
    "save_equities",
    "save_expired_options",
    "save_forwards_data",
    "save_options",
    "save_portfolio",
    "save_ts_exec_log",
    "load_sell_systems",
    "log_trade",
    "auto_execute_trading_levels",
    "load_or_fetch_closing_history",
    "compute_spot_totals",
]
