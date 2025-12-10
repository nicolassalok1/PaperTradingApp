"""
State builder for RL Hedger Live v2 (with Greeks).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .greeks_engine import aggregate_portfolio_greeks


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


class LiveStateBuilderV2:
    def __init__(self) -> None:
        api_key = os.getenv("APCA_API_KEY_ID") or ""
        api_secret = os.getenv("APCA_API_SECRET_KEY") or ""
        base_url = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
        if not api_key or not api_secret:
            raise EnvironmentError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")

        is_paper = "paper" in (base_url or "").lower()
        self.base_url = base_url
        self.trading = TradingClient(api_key, api_secret, paper=is_paper)
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    def _latest_price(self, symbol: str) -> float:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=1)
        try:
            bars = self.data.get_stock_bars(req)
        except Exception:
            return 0.0
        df = getattr(bars, "df", None)
        if df is None or df.empty:
            return 0.0
        if df.index.nlevels > 1:
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                df = df.reset_index()
        df = df.reset_index()
        price_col = "close" if "close" in df.columns else df.columns[-1]
        return float(df.iloc[-1][price_col])

    def get_live_state(self, underlying_symbol: str) -> Tuple[List[float], Dict[str, Any]]:
        acc = _to_dict(self.trading.get_account())
        positions = self.trading.get_all_positions()
        underlying = (underlying_symbol or "").strip().upper()
        equity_pos = 0.0
        option_positions = []
        for p in positions:
            pdict = _to_dict(p)
            if str(pdict.get("asset_class", "")).lower() == "us_equity":
                if (pdict.get("symbol") or "").upper() == underlying:
                    equity_pos += float(pdict.get("qty", 0.0) or 0.0)
            elif str(pdict.get("asset_class", "")).lower() == "option":
                option_positions.append(pdict)

        greeks = aggregate_portfolio_greeks(option_positions, underlying_price=self._latest_price(underlying))

        underlying_price = self._latest_price(underlying)
        cash = float(acc.get("cash", 0.0) or 0.0)
        now = datetime.utcnow()
        norm_time = (now.hour * 60 + now.minute) / (24 * 60)

        state_vector = [
            underlying_price,
            equity_pos,
            greeks.get("net_delta", 0.0),
            greeks.get("net_gamma", 0.0),
            greeks.get("net_vega", 0.0),
            greeks.get("net_theta", 0.0),
            cash,
            norm_time,
        ]
        metadata = {
            "underlying": underlying,
            "equity_position": equity_pos,
            "cash": cash,
            "greeks": greeks,
            "time": now.isoformat(),
        }
        return state_vector, metadata


__all__ = ["LiveStateBuilderV2"]
