"""
Alpaca spot service (model layer).
Uses alpaca-py TradingClient, StockHistoricalDataClient, and StockDataStream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd
from app.utils.secrets import get_secret
from app.utils.trading_guard import enforce_paper_endpoint

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

try:  # Compatibility across alpaca-py versions
    from alpaca.trading.enums import QueryOrderStatus as _QueryOrderStatus
except Exception:
    _QueryOrderStatus = None


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


class AlpacaSpotService:
    """
    Thin wrapper around alpaca-py clients.
    Keeps the rest of the app insulated from client details.
    """

    def __init__(self, keys: AlpacaKeys | None = None) -> None:
        self.keys = keys or AlpacaKeys.from_env()
        is_paper = "paper" in (self.keys.base_url or "").lower()
        self.trading_client = TradingClient(
            self.keys.api_key,
            self.keys.api_secret,
            paper=is_paper,
            raw_data=True,  # keep raw payloads to handle newer asset classes like us_option
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.keys.api_key,
            secret_key=self.keys.api_secret,
        )
        self._stream: StockDataStream | None = None

    # --- Internal helpers -------------------------------------------------
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return (symbol or "").strip().upper()

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

    @staticmethod
    def _parse_timeframe(timeframe: str | TimeFrame) -> TimeFrame:
        if isinstance(timeframe, TimeFrame):
            return timeframe
        tf_str = (timeframe or "").strip().lower()
        if tf_str in {"1d", "1day", "day", "daily"}:
            return TimeFrame.Day
        if tf_str in {"1h", "1hour", "60m", "hour"}:
            return TimeFrame.Hour
        if tf_str in {"30m", "30min"}:
            return TimeFrame(30, TimeFrameUnit.Minute)
        if tf_str in {"15m", "15min"}:
            return TimeFrame(15, TimeFrameUnit.Minute)
        if tf_str in {"5m", "5min"}:
            return TimeFrame(5, TimeFrameUnit.Minute)
        if tf_str in {"1m", "1min", "minute"}:
            return TimeFrame.Minute
        return TimeFrame.Day

    def _get_stream(self) -> StockDataStream:
        if self._stream is None:
            self._stream = StockDataStream(self.keys.api_key, self.keys.api_secret)
        return self._stream

    def _order_status_open(self) -> Any:
        return getattr(_QueryOrderStatus, "OPEN", None) or "open"

    # --- Public API -------------------------------------------------------
    def get_account(self) -> dict[str, Any]:
        account = self.trading_client.get_account()
        return self._to_dict(account)

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.trading_client.get_all_positions()
        return [self._to_dict(pos) for pos in positions] if positions else []

    def get_open_orders(self) -> list[dict[str, Any]]:
        status_filter = self._order_status_open()
        request = GetOrdersRequest(status=status_filter)
        orders = self.trading_client.get_orders(request)
        return [self._to_dict(order) for order in orders] if orders else []

    def get_price_history(
        self, symbol: str, timeframe: str | TimeFrame = "1Day", limit: int = 100
    ) -> pd.DataFrame:
        symbol_norm = self._normalize_symbol(symbol)
        if not symbol_norm:
            return pd.DataFrame()
        tf_obj = self._parse_timeframe(timeframe)
        limit_int = int(limit or 0)
        limit_int = max(1, min(limit_int, 1000))
        request = StockBarsRequest(
            symbol_or_symbols=symbol_norm,
            timeframe=tf_obj,
            limit=limit_int,
        )
        bars = self.data_client.get_stock_bars(request)
        df = getattr(bars, "df", None)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol_norm, level="symbol")
            except Exception:
                df = df.reset_index()
        df = df.reset_index()
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        return df

    def send_market_order(self, symbol: str, qty: float, side: str) -> dict[str, Any]:
        symbol_norm = self._normalize_symbol(symbol)
        qty_val = float(qty or 0.0)
        if not symbol_norm or qty_val <= 0:
            raise ValueError("Symbol and positive quantity are required")
        side_str = (side or "").strip().lower()
        side_enum = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
        order_request = MarketOrderRequest(
            symbol=symbol_norm,
            qty=qty_val,
            side=side_enum,
            time_in_force=TimeInForce.GTC,
        )
        order = self.trading_client.submit_order(order_request)
        return self._to_dict(order)

    def subscribe_bars(
        self, symbols: Iterable[str], on_bar: Callable[[dict[str, Any]], None]
    ) -> StockDataStream:
        """
        Register a simple bar handler. Caller is responsible for stream.run() and stream.stop().
        """

        async def _handler(bar):
            on_bar(self._to_dict(bar))

        stream = self._get_stream()
        for sym in symbols:
            symbol_norm = self._normalize_symbol(sym)
            if symbol_norm:
                stream.subscribe_bars(_handler, symbol_norm)
        return stream


__all__ = ["AlpacaKeys", "AlpacaSpotService"]
