"""
Trading service utilities (no UI).
Provides controller-facing helpers for spot pricing and trade execution wrappers.
"""

from __future__ import annotations

import time
from typing import Tuple

from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache
from app.model.market_data import market_data
from app.model.market_data.realtime import get_data
from app.model.market_data.history import clear_closing_history_cache, load_or_fetch_closing_history
from app.model.trading.buy_sell import (
    buy_asset as core_buy_asset,
    sell_asset as core_sell_asset,
    append_trade_log,
)
from app.model.trading.execution import compute_spot_totals, trade_spot
from app.model.trading.logs import load_trades_log
from app.model.portfolio.positions import load_portfolio_default, save_portfolio_default
from app.utils.math_utils import floor_2, floor_4


def get_market_price(symbol: str, fallback: float = 0.0) -> float:
    try:
        spot = market_data.fetch_spot_price(symbol)
        return float(spot) if spot is not None else float(fallback)
    except Exception:
        try:
            return float(get_data(symbol).get("price", fallback) or fallback)
        except Exception:
            return float(fallback)


def trade_spot_with_fallback(symbol: str, fallback: float = 0.0) -> float:
    def _dashboard_price(sym: str, fb: float = 0.0) -> float:
        return get_market_price(sym, fb)

    return trade_spot(
        symbol,
        fallback,
        dashboard_price_fn=_dashboard_price,
        get_data_fn=get_data,
        floor_fn=floor_4,
    )


def compute_spot_totals_with_price(portfolio: dict) -> Tuple[float, float, float]:
    def _dashboard_price(sym: str, fb: float = 0.0) -> float:
        return get_market_price(sym, fb)

    return compute_spot_totals(portfolio, _dashboard_price)


def update_cache_last_refresh(
    portfolio: dict, realized_pnl: float, cash_delta: float = 0.0
) -> None:
    cache = load_dashboard_cache()
    cache["last_refresh"] = cache.get("last_refresh")
    save_dashboard_cache(cache)


def update_dashboard_balance(cash_delta: float) -> None:
    cache = load_dashboard_cache()
    balance = float(cache.get("balance", 0.0) or 0.0) + float(cash_delta or 0.0)
    cache["balance"] = balance
    save_dashboard_cache(cache)


def buy_asset(symbol: str, quantity: float, price: float, source: str = "manual"):
    return core_buy_asset(
        symbol,
        quantity,
        price,
        load_portfolio=load_portfolio_default,
        save_portfolio=save_portfolio_default,
        update_cache_fn=update_cache_last_refresh,
        floor_fn=floor_4,
        time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
    )


def sell_asset(symbol: str, quantity: float, price: float, source: str = "manual"):
    return core_sell_asset(
        symbol,
        quantity,
        price,
        load_portfolio=load_portfolio_default,
        save_portfolio=save_portfolio_default,
        update_cache_fn=update_cache_last_refresh,
        floor_fn=floor_4,
        time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
    )


def buy_asset_with_balance(
    symbol: str, quantity: float, price: float, source: str, meta: dict | None = None
):
    def _update_cache(portfolio: dict, realized_pnl: float, cash_delta: float) -> None:
        update_dashboard_balance(cash_delta)

    return core_buy_asset(
        symbol,
        quantity,
        price,
        load_portfolio=load_portfolio_default,
        save_portfolio=save_portfolio_default,
        update_cache_fn=_update_cache,
        floor_fn=floor_4,
        time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
        meta=meta,
    )


def sell_asset_with_balance(
    symbol: str, quantity: float, price: float, source: str, meta: dict | None = None
):
    def _update_cache(portfolio: dict, realized_pnl: float, cash_delta: float) -> None:
        update_dashboard_balance(cash_delta)

    return core_sell_asset(
        symbol,
        quantity,
        price,
        load_portfolio=load_portfolio_default,
        save_portfolio=save_portfolio_default,
        update_cache_fn=_update_cache,
        floor_fn=floor_4,
        time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
        source=source,
        meta=meta,
    )


# Expose history loader for controllers
fetch_closing_history = load_or_fetch_closing_history


__all__ = [
    "append_trade_log",
    "buy_asset",
    "clear_closing_history_cache",
    "compute_spot_totals_with_price",
    "fetch_closing_history",
    "get_market_price",
    "load_trades_log",
    "update_dashboard_balance",
    "sell_asset",
    "buy_asset_with_balance",
    "sell_asset_with_balance",
    "trade_spot_with_fallback",
    "floor_2",
    "floor_4",
]
