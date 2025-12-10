"""
Portfolio services (no UI).
Provides order application helpers used by controllers/views.
"""

from __future__ import annotations

import time
from typing import Tuple

import pandas as pd

from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache
from app.model.portfolio.positions import load_portfolio_default, save_portfolio_default
from app.model.portfolio.stats import compute_eigen_orders
from app.model.trading.buy_sell import buy_asset as core_buy_asset, sell_asset as core_sell_asset
from app.model.trading.execution import execute_orders_with_trade_functions
from app.utils.math_utils import floor_4


def apply_orders(
    orders_df: pd.DataFrame,
    latest_prices: pd.Series,
) -> Tuple[int, list]:
    """
    Execute orders using core buy/sell functions and update dashboard cache timestamps.
    """

    def _update_cache(portfolio: dict, realized_pnl: float, cash_delta: float = 0.0) -> None:
        cache = load_dashboard_cache()
        cache["last_refresh"] = cache.get("last_refresh")
        save_dashboard_cache(cache)

    def _buy_asset(symbol: str, qty: float, price: float):
        return core_buy_asset(
            symbol,
            qty,
            price,
            load_portfolio=load_portfolio_default,
            save_portfolio=save_portfolio_default,
            update_cache_fn=_update_cache,
            floor_fn=floor_4,
            time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
            source="rebalance",
        )

    def _sell_asset(symbol: str, qty: float, price: float):
        return core_sell_asset(
            symbol,
            qty,
            price,
            load_portfolio=load_portfolio_default,
            save_portfolio=save_portfolio_default,
            update_cache_fn=_update_cache,
            floor_fn=floor_4,
            time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
            source="rebalance",
        )

    executed, skipped = execute_orders_with_trade_functions(
        orders_df,
        latest_prices,
        buy_fn=_buy_asset,
        sell_fn=_sell_asset,
    )
    return executed, skipped


__all__ = ["apply_orders", "compute_eigen_orders"]
