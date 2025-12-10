"""
Controller for Alpaca spot trading (thin layer over the model service).
"""

from __future__ import annotations

from typing import Any

from app.model.alpaca_spot.service import AlpacaSpotService

_SERVICE: AlpacaSpotService | None = None


def _get_service() -> AlpacaSpotService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AlpacaSpotService()
    return _SERVICE


def get_account() -> dict[str, Any]:
    return _get_service().get_account()


def get_positions() -> list[dict[str, Any]]:
    return _get_service().get_positions()


def get_open_orders() -> list[dict[str, Any]]:
    return _get_service().get_open_orders()


def get_price_history(symbol: str, timeframe: str = "1Day", limit: int = 100):
    return _get_service().get_price_history(symbol, timeframe=timeframe, limit=limit)


def send_order(symbol: str, qty: float, side: str):
    return _get_service().send_market_order(symbol, qty, side)


__all__ = [
    "get_account",
    "get_positions",
    "get_open_orders",
    "get_price_history",
    "send_order",
]
