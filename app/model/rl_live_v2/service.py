"""
Service layer for RL Hedger Live v2 (no UI).

Provides helpers for:
  - live snapshot retrieval
  - RL hedge suggestion
  - live hedge execution via Alpaca
  - simple backtest over recent history
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .alpaca_state_builder import LiveStateBuilderV2
from .rl_backtester import RLBacktester
from .rl_inference import load_latest_agent_v2


_STATE_BUILDER: LiveStateBuilderV2 | None = None
_AGENT = None
_TRADING_CLIENT: TradingClient | None = None
_DATA_CLIENT: StockHistoricalDataClient | None = None


def _state_builder() -> LiveStateBuilderV2:
    global _STATE_BUILDER
    if _STATE_BUILDER is None:
        _STATE_BUILDER = LiveStateBuilderV2()
    return _STATE_BUILDER


def _agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = load_latest_agent_v2()
    return _AGENT


def _trading_client() -> TradingClient:
    global _TRADING_CLIENT
    if _TRADING_CLIENT is None:
        api_key = os.getenv("APCA_API_KEY_ID") or ""
        api_secret = os.getenv("APCA_API_SECRET_KEY") or ""
        base_url = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
        _TRADING_CLIENT = TradingClient(api_key, api_secret, paper="paper" in base_url.lower())
    return _TRADING_CLIENT


def _data_client() -> StockHistoricalDataClient:
    global _DATA_CLIENT
    if _DATA_CLIENT is None:
        api_key = os.getenv("APCA_API_KEY_ID") or ""
        api_secret = os.getenv("APCA_API_SECRET_KEY") or ""
        _DATA_CLIENT = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    return _DATA_CLIENT


def get_live_snapshot() -> Dict[str, Any]:
    sb = _state_builder()
    client = getattr(sb, "trading", None)
    if client is None:
        return {"account": {}, "positions": []}
    try:
        account = client.get_account()
        positions = client.get_all_positions()
    except Exception:
        return {"account": {}, "positions": []}
    return {
        "account": account.dict() if hasattr(account, "dict") else account,
        "positions": [p.dict() if hasattr(p, "dict") else p for p in positions],
    }


def get_rl_suggestion(underlying_symbol: str) -> Dict[str, Any]:
    state_vec, meta = _state_builder().get_live_state(underlying_symbol)
    act = _agent().select_action(state_vec)
    return {"state_vector": state_vec, "metadata": meta, "action": act, "greeks": meta.get("greeks", {})}


def execute_rl_hedge(underlying_symbol: str) -> Dict[str, Any]:
    suggestion = get_rl_suggestion(underlying_symbol)
    action = suggestion.get("action", {})
    side = action.get("side")
    delta_qty = float(action.get("delta_qty", 0.0) or 0.0)
    order_resp = None
    if side in {"buy", "sell"} and abs(delta_qty) > 0:
        req = MarketOrderRequest(
            symbol=underlying_symbol.upper(),
            qty=abs(delta_qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order_resp = _trading_client().submit_order(req)
        order_resp = order_resp.dict() if hasattr(order_resp, "dict") else order_resp
    elif side == "flatten" and abs(delta_qty) > 0:
        req = MarketOrderRequest(
            symbol=underlying_symbol.upper(),
            qty=abs(delta_qty),
            side=OrderSide.BUY if delta_qty > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order_resp = _trading_client().submit_order(req)
        order_resp = order_resp.dict() if hasattr(order_resp, "dict") else order_resp
    return {"suggestion": suggestion, "order": order_resp}


def _fetch_history(underlying_symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
    req = StockBarsRequest(
        symbol_or_symbols=underlying_symbol.upper(),
        timeframe=TimeFrame.Day,
        limit=max(lookback_days, 2),
    )
    bars = _data_client().get_stock_bars(req)
    df = getattr(bars, "df", None)
    if df is None or df.empty:
        return []
    if df.index.nlevels > 1:
        try:
            df = df.xs(underlying_symbol.upper(), level="symbol")
        except Exception:
            df = df.reset_index()
    df = df.reset_index()
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    return df.to_dict(orient="records")


def run_backtest(underlying_symbol: str, lookback_days: int = 60) -> Dict[str, Any]:
    hist = _fetch_history(underlying_symbol, lookback_days)
    agent = _agent()
    backtester = RLBacktester(hist, agent, hedging_params={"hedge_size": 1.0})
    return backtester.run()


__all__ = [
    "get_live_snapshot",
    "get_rl_suggestion",
    "execute_rl_hedge",
    "run_backtest",
]

