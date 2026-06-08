"""
Alpaca advanced orders service (model layer).
Encapsulates all Alpaca API calls for limit/stop/TP/SL/bracket orders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)

try:  # optional depending on alpaca-py version
    from alpaca.trading.requests import BracketOrderRequest
except Exception:  # pragma: no cover - compatibility shim
    BracketOrderRequest = None  # type: ignore

try:  # optional class depending on alpaca-py version
    from alpaca.trading.requests import StopOrderRequest
except Exception:  # pragma: no cover - compatibility shim
    StopOrderRequest = None  # type: ignore

try:  # enum name can differ across versions
    from alpaca.trading.enums import QueryOrderStatus as _QueryOrderStatus
except Exception:  # pragma: no cover - compatibility shim
    _QueryOrderStatus = None

from app.utils.secrets import get_secret
from app.utils.trading_guard import enforce_paper_endpoint


@dataclass
class AlpacaKeys:
    api_key: str
    api_secret: str
    base_url: str

    @classmethod
    def from_env(cls) -> "AlpacaKeys":
        """Load Alpaca credentials from env or Streamlit secrets."""
        api_key = (get_secret("APCA_API_KEY_ID") or "").strip()
        api_secret = (get_secret("APCA_API_SECRET_KEY") or "").strip()
        base_url = enforce_paper_endpoint(get_secret("APCA_API_BASE_URL"))
        if not api_key or not api_secret:
            raise EnvironmentError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)


class AlpacaOrdersService:
    """Service wrapper for Alpaca trading endpoints."""

    def __init__(self, keys: AlpacaKeys | None = None) -> None:
        self.keys = keys or AlpacaKeys.from_env()
        is_paper = "paper" in (self.keys.base_url or "").lower()
        self.trading_client = TradingClient(
            self.keys.api_key,
            self.keys.api_secret,
            paper=is_paper,
            raw_data=True,  # return raw dicts to support asset classes beyond pydantic enums (e.g., us_option)
        )

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return (symbol or "").strip().upper()

    @staticmethod
    def _side(side: str) -> OrderSide:
        return OrderSide.BUY if (side or "").strip().lower() == "buy" else OrderSide.SELL

    @staticmethod
    def _invert_side(side: str) -> OrderSide:
        return OrderSide.SELL if (side or "").strip().lower() == "buy" else OrderSide.BUY

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        try:
            return dict(obj)
        except Exception:
            return {"value": obj}

    def _order_status_open(self) -> Any:
        return getattr(_QueryOrderStatus, "OPEN", None) or "open"

    # --- queries ---------------------------------------------------------
    def get_account(self) -> dict[str, Any]:
        account = self.trading_client.get_account()
        return self._to_dict(account)

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.trading_client.get_all_positions()
        return [self._to_dict(pos) for pos in positions] if positions else []

    def get_open_orders(self) -> list[dict[str, Any]]:
        request = GetOrdersRequest(status=self._order_status_open())
        orders = self.trading_client.get_orders(request)
        return [self._to_dict(order) for order in orders] if orders else []

    def get_clock(self) -> dict[str, Any]:
        """Return Alpaca market clock (is_open/next_open/next_close)."""
        clock = self.trading_client.get_clock()
        return self._to_dict(clock)

    # --- order submission -----------------------------------------------
    def submit_limit_order(self, symbol: str, qty: float, limit_price: float, side: str) -> dict[str, Any]:
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        price_val = float(limit_price or 0.0)
        if not symbol_norm or qty_val <= 0 or price_val <= 0:
            raise ValueError("Symbol, positive quantity, and positive limit price are required")
        req = LimitOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=self._side(side),
            time_in_force=TimeInForce.GTC,
            limit_price=price_val,
        )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_option_market_order(self, symbol: str, qty: float, side: str) -> dict[str, Any]:
        """
        Market order on an options contract (OPRA symbol) using Alpaca options trading.
        """
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        if not symbol_norm or qty_val <= 0:
            raise ValueError("Option symbol and positive quantity are required")
        req = MarketOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=self._side(side),
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_option_limit_order(self, symbol: str, qty: float, limit_price: float, side: str) -> dict[str, Any]:
        """
        Limit order on an options contract (OPRA symbol).
        """
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        price_val = float(limit_price or 0.0)
        if not symbol_norm or qty_val <= 0 or price_val <= 0:
            raise ValueError("Option symbol, positive quantity, and positive limit price are required")
        req = LimitOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=self._side(side),
            time_in_force=TimeInForce.DAY,
            limit_price=price_val,
        )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_stop_loss(self, symbol: str, qty: float, stop_price: float, side: str) -> dict[str, Any]:
        """
        Stop-loss exits the position, so the side is inverted (long -> sell stop, short -> buy stop).
        """
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        stop_val = float(stop_price or 0.0)
        if not symbol_norm or qty_val <= 0 or stop_val <= 0:
            raise ValueError("Symbol, positive quantity, and positive stop price are required")

        exit_side = self._invert_side(side)

        if StopOrderRequest:
            req = StopOrderRequest(
                symbol=symbol_norm,
                qty=qty_val,
                side=exit_side,
                stop_price=stop_val,
                time_in_force=TimeInForce.GTC,
            )
        else:  # pragma: no cover - compatibility fallback
            req = MarketOrderRequest(
                symbol=symbol_norm,
                qty=qty_val,
                side=exit_side,
                time_in_force=TimeInForce.GTC,
                order_class=getattr(OrderClass, "SIMPLE", None),
                stop_loss=StopLossRequest(stop_price=stop_val),
            )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_take_profit(self, symbol: str, qty: float, take_profit_price: float, side: str) -> dict[str, Any]:
        """
        Take-profit also exits the position, so the side is inverted (long -> sell limit, short -> buy limit).
        """
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        tp_val = float(take_profit_price or 0.0)
        if not symbol_norm or qty_val <= 0 or tp_val <= 0:
            raise ValueError("Symbol, positive quantity, and positive take-profit price are required")
        req = LimitOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=self._invert_side(side),
            time_in_force=TimeInForce.GTC,
            limit_price=tp_val,
        )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_stop_limit(
        self, symbol: str, qty: float, stop_price: float, limit_price: float, side: str
    ) -> dict[str, Any]:
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        stop_val = float(stop_price or 0.0)
        limit_val = float(limit_price or 0.0)
        if not symbol_norm or qty_val <= 0 or stop_val <= 0 or limit_val <= 0:
            raise ValueError("Symbol, positive quantity, stop price, and limit price are required")
        req = StopLimitOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=self._side(side),
            time_in_force=TimeInForce.GTC,
            stop_price=stop_val,
            limit_price=limit_val,
        )
        order = self.trading_client.submit_order(req)
        return self._to_dict(order)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        side: str,
    ) -> dict[str, Any]:
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        entry_val = float(entry_price or 0.0)
        stop_val = float(stop_price or 0.0)
        tp_val = float(take_profit_price or 0.0)
        if not symbol_norm or qty_val <= 0 or entry_val <= 0 or stop_val <= 0 or tp_val <= 0:
            raise ValueError("Symbol, positive quantity, entry, stop-loss, and take-profit prices are required")

        tp_req = TakeProfitRequest(limit_price=tp_val)
        sl_req = StopLossRequest(stop_price=stop_val)

        if BracketOrderRequest:
            try:
                req: Any = BracketOrderRequest(
                    symbol=symbol_norm,
                    qty=qty_val,
                    side=self._side(side),
                    time_in_force=TimeInForce.GTC,
                    limit_price=entry_val,
                    take_profit=tp_req,
                    stop_loss=sl_req,
                    order_type=OrderType.LIMIT,
                )
            except TypeError:  # pragma: no cover - fallback for versions without this signature
                req = LimitOrderRequest(
                    symbol=symbol_norm,
                    qty=qty_val,
                    side=self._side(side),
                    time_in_force=TimeInForce.GTC,
                    limit_price=entry_val,
                    order_class=getattr(OrderClass, "BRACKET", None) or "bracket",
                    take_profit=tp_req,
                    stop_loss=sl_req,
                )
        else:  # pragma: no cover - fallback when BracketOrderRequest is unavailable
            req = LimitOrderRequest(
                symbol=symbol_norm,
                qty=qty_val,
                side=self._side(side),
                time_in_force=TimeInForce.GTC,
                limit_price=entry_val,
                order_class=getattr(OrderClass, "BRACKET", None) or "bracket",
                take_profit=tp_req,
                stop_loss=sl_req,
            )

        order = self.trading_client.submit_order(req)
        return self._to_dict(order)


__all__ = [
    "AlpacaKeys",
    "AlpacaOrdersService",
]
