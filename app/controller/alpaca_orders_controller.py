"""
Controller for Alpaca advanced orders (limit/stop/TP/bracket).
"""

from __future__ import annotations

from typing import Any

from app.model.alpaca_orders.service import AlpacaOrdersService

_SERVICE: AlpacaOrdersService | None = None


def _get_service() -> AlpacaOrdersService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AlpacaOrdersService()
    return _SERVICE


def get_account() -> dict[str, Any]:
    return _get_service().get_account()


def get_positions() -> list[dict[str, Any]]:
    return _get_service().get_positions()


def get_open_orders() -> list[dict[str, Any]]:
    return _get_service().get_open_orders()


def create_limit_order(symbol: str, qty: float, price: float, side: str):
    return _get_service().submit_limit_order(symbol, qty, price, side)


def create_stop_loss(symbol: str, qty: float, stop_price: float, side: str):
    return _get_service().submit_stop_loss(symbol, qty, stop_price, side)


def create_take_profit(symbol: str, qty: float, take_profit_price: float, side: str):
    return _get_service().submit_take_profit(symbol, qty, take_profit_price, side)


def create_stop_limit(symbol: str, qty: float, stop_price: float, limit_price: float, side: str):
    return _get_service().submit_stop_limit(symbol, qty, stop_price, limit_price, side)


def create_bracket_order(
    symbol: str,
    qty: float,
    entry_price: float,
    stop_price: float,
    take_profit_price: float,
    side: str,
):
    return _get_service().submit_bracket_order(
        symbol,
        qty,
        entry_price,
        stop_price,
        take_profit_price,
        side,
    )


__all__ = [
    "get_account",
    "get_positions",
    "get_open_orders",
    "create_limit_order",
    "create_stop_loss",
    "create_take_profit",
    "create_stop_limit",
    "create_bracket_order",
]
