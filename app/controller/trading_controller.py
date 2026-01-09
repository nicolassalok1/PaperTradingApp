"""
Unified controller for the 💹 Trading tab (spot + advanced orders).
"""

from __future__ import annotations

from typing import Any, List, Dict

from app.model.alpaca_spot.service import AlpacaSpotService
from app.model.alpaca_orders.service import AlpacaOrdersService

_SPOT_SERVICE: AlpacaSpotService | None = None
_ORDERS_SERVICE: AlpacaOrdersService | None = None


def _spot_service() -> AlpacaSpotService:
    global _SPOT_SERVICE
    if _SPOT_SERVICE is None:
        _SPOT_SERVICE = AlpacaSpotService()
    return _SPOT_SERVICE


def _orders_service() -> AlpacaOrdersService:
    global _ORDERS_SERVICE
    if _ORDERS_SERVICE is None:
        _ORDERS_SERVICE = AlpacaOrdersService()
    return _ORDERS_SERVICE


# --- Spot account & market orders -----------------------------------------

def get_spot_account() -> Dict[str, Any]:
    return _spot_service().get_account()


def get_spot_positions() -> List[Dict[str, Any]]:
    return _spot_service().get_positions()


def get_spot_open_orders() -> List[Dict[str, Any]]:
    return _spot_service().get_open_orders()


def get_price_history(symbol: str, timeframe: str = "1Day", limit: int = 100):
    return _spot_service().get_price_history(symbol, timeframe=timeframe, limit=limit)


def send_spot_market_order(symbol: str, qty: float, side: str):
    return _spot_service().send_market_order(symbol, qty, side)


# --- Advanced orders (same account / positions but via AlpacaOrdersService) ---

def get_orders_account() -> Dict[str, Any]:
    return _orders_service().get_account()


def get_orders_positions() -> List[Dict[str, Any]]:
    return _orders_service().get_positions()


def get_open_advanced_orders() -> List[Dict[str, Any]]:
    return _orders_service().get_open_orders()


def create_limit_order(symbol: str, qty: float, price: float, side: str):
    return _orders_service().submit_limit_order(symbol, qty, price, side)


def create_stop_loss(symbol: str, qty: float, stop_price: float, side: str):
    return _orders_service().submit_stop_loss(symbol, qty, stop_price, side)


def create_take_profit(symbol: str, qty: float, take_profit_price: float, side: str):
    return _orders_service().submit_take_profit(symbol, qty, take_profit_price, side)


def create_stop_limit(symbol: str, qty: float, stop_price: float, limit_price: float, side: str):
    return _orders_service().submit_stop_limit(symbol, qty, stop_price, limit_price, side)


def create_bracket_order(
    symbol: str,
    qty: float,
    entry_price: float,
    stop_price: float,
    take_profit_price: float,
    side: str,
):
    return _orders_service().submit_bracket_order(
        symbol,
        qty,
        entry_price,
        stop_price,
        take_profit_price,
        side,
    )


# --- Options trading helpers (Alpaca options via AlpacaOrdersService) ---

def get_option_positions() -> List[Dict[str, Any]]:
    positions = _orders_service().get_positions()
    if not positions:
        return []
    return [
        pos
        for pos in positions
        if str(pos.get("asset_class") or "").lower() in {"option", "us_option"}
    ]


def get_open_option_orders() -> List[Dict[str, Any]]:
    orders = _orders_service().get_open_orders()
    if not orders:
        return []
    return [
        order
        for order in orders
        if str(order.get("asset_class") or "").lower() in {"option", "us_option"}
    ]


def create_option_market_order(option_symbol: str, qty: float, side: str):
    return _orders_service().submit_option_market_order(option_symbol, qty, side)


def create_option_limit_order(option_symbol: str, qty: float, limit_price: float, side: str):
    return _orders_service().submit_option_limit_order(option_symbol, qty, limit_price, side)


def get_orders_clock() -> Dict[str, Any]:
    return _orders_service().get_clock()


__all__ = [
    # Spot
    "get_spot_account",
    "get_spot_positions",
    "get_spot_open_orders",
    "get_price_history",
    "send_spot_market_order",
    # Advanced orders
    "get_orders_account",
    "get_orders_positions",
    "get_open_advanced_orders",
    "create_limit_order",
    "create_stop_loss",
    "create_take_profit",
    "create_stop_limit",
    "create_bracket_order",
    # Options
    "get_option_positions",
    "get_open_option_orders",
    "create_option_market_order",
    "create_option_limit_order",
    "get_orders_clock",
]
