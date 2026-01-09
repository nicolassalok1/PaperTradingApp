"""
Alpaca client utilities for Hedger v2.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from app.utils.secrets import get_secret


def _to_dict(obj: Any) -> Dict[str, Any]:
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


class AlpacaHedgerClient:
    def __init__(self) -> None:
        api_key = (get_secret("APCA_API_KEY_ID") or "").strip()
        api_secret = (get_secret("APCA_API_SECRET_KEY") or "").strip()
        base_url = (get_secret("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").strip()
        self.offline = False
        if (not api_key or not api_secret) or api_key.lower().startswith("dummy") or api_secret.lower().startswith("dummy"):
            self.offline = True

        self.base_url = base_url
        if self.offline:
            self.trading = None
            self.data = None
        else:
            is_paper = "paper" in (base_url or "").lower()
            self.trading = TradingClient(
                api_key,
                api_secret,
                paper=is_paper,
                raw_data=True,
            )
            self.data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    def get_account(self) -> Dict[str, Any]:
        if self.offline or self.trading is None:
            return {}
        try:
            return _to_dict(self.trading.get_account())
        except Exception:
            return {}

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.offline or self.trading is None:
            positions = []
        else:
            try:
                positions = self.trading.get_all_positions()
            except Exception:
                positions = []
        equities: List[Dict[str, Any]] = []
        options: List[Dict[str, Any]] = []
        for p in positions:
            pdict = _to_dict(p)
            asset_class = str(pdict.get("asset_class", "")).lower()
            if asset_class == "us_equity":
                equities.append(pdict)
            elif asset_class in {"option", "us_option"}:
                options.append(pdict)
        return {"equities": equities, "options": options}

    def get_equity_positions(self) -> List[Dict[str, Any]]:
        return self.get_positions().get("equities", [])

    def get_option_positions(self) -> List[Dict[str, Any]]:
        return self.get_positions().get("options", [])

    def get_latest_price(self, symbol: str) -> float:
        if self.offline or self.data is None:
            return 0.0
        sym = (symbol or "").strip().upper()
        if not sym:
            return 0.0
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Day,
            limit=1,
        )
        bars = self.data.get_stock_bars(req)
        df = getattr(bars, "df", None)
        if df is None or df.empty:
            return 0.0
        if df.index.nlevels > 1:
            try:
                df = df.xs(sym, level="symbol")
            except Exception:
                df = df.reset_index()
        df = df.reset_index()
        price_col = "close" if "close" in df.columns else df.columns[-1]
        return float(df.iloc[-1][price_col])

    def submit_market_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        sym = (symbol or "").strip().upper()
        qty_val = float(qty or 0.0)
        side_enum = OrderSide.BUY if (side or "").lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty_val,
            side=side_enum,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        return _to_dict(order)


__all__ = ["AlpacaHedgerClient"]
