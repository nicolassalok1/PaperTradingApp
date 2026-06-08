"""
Alpaca client utilities for Hedger v2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, List

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
try:  # optional feed enum (newer SDKs)
    from alpaca.data.enums import DataFeed  # type: ignore
except Exception:
    DataFeed = None  # type: ignore

from app.utils.paths import CACHE_OHLC_DIR
from app.utils.secrets import get_secret
from app.utils.trading_guard import enforce_paper_endpoint


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
        base_url = enforce_paper_endpoint(get_secret("APCA_API_BASE_URL"))
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
                raw_data=True,  # keep raw data to handle option asset_class values
            )
            self.data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    def is_ready(self) -> bool:
        """Whether live Alpaca calls can be made."""
        return not self.offline and self.trading is not None

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
            if str(pdict.get("asset_class", "")).lower() == "us_equity":
                equities.append(pdict)
            elif "option" in str(pdict.get("asset_class", "")).lower():
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
        req_kwargs = {
            "symbol_or_symbols": sym,
            "timeframe": TimeFrame.Day,
            "limit": 1,
        }
        # Prefer IEX feed when available to avoid SIP subscription errors
        if DataFeed:
            req_kwargs["feed"] = DataFeed.IEX

        def _fetch(params):
            return self.data.get_stock_bars(StockBarsRequest(**params))

        try:
            bars = _fetch(req_kwargs)
        except Exception:
            # Retry without feed hint
            req_kwargs.pop("feed", None)
            try:
                bars = _fetch(req_kwargs)
            except Exception:
                return 0.0
        try:
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
        except Exception:
            return 0.0

    def submit_market_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Alpaca trading client not configured (offline).")
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

    def get_stock_bars_df(
        self,
        symbol: str,
        *,
        timeframe: str | TimeFrame = "1Day",
        lookback_days: int = 180,
        force_refresh: bool = False,
        feed: str | None = "iex",
    ):
        """
        Fetch historical bars for a symbol (cached to disk).

        Returns a pandas DataFrame with a timestamp column and OHLCV columns.
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return pd.DataFrame()

        # Cache path keyed by symbol/timeframe/lookback
        tf_key = str(timeframe).replace("TimeFrame.", "").replace(".", "_")
        cache_path = CACHE_OHLC_DIR / f"alpaca_{sym}_{tf_key}_{int(lookback_days)}d.csv"
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_csv(cache_path, parse_dates=["timestamp"])
            except Exception:
                pass

        if self.offline or self.data is None:
            return pd.DataFrame()

        # Parse timeframe if passed as string (supports "1Day", "1Hour", "1Min")
        tf = timeframe
        if isinstance(tf, str):
            tf_norm = tf.strip().lower()
            if "day" in tf_norm:
                tf = TimeFrame.Day
            elif "hour" in tf_norm:
                tf = TimeFrame.Hour
            else:
                tf = TimeFrame.Minute

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(int(lookback_days), 1) + 5)
        limit = min(max(int(lookback_days) * 2, 50), 5_000)
        req_kwargs = {
            "symbol_or_symbols": sym,
            "timeframe": tf,
            "start": start,
            "end": end,
            "limit": limit,
        }
        feed_to_use = (feed or "").lower().strip()
        if feed_to_use:
            req_kwargs["feed"] = DataFeed.IEX if (DataFeed and feed_to_use == "iex") else feed_to_use

        def _fetch(req_params):
            req = StockBarsRequest(**req_params)
            return self.data.get_stock_bars(req)

        try:
            bars = _fetch(req_kwargs)
        except Exception as exc:
            # Fallback: if SIP not permitted, retry with IEX feed explicitly
            msg = str(exc).lower()
            if "sip" in msg or "subscription does not permit" in msg:
                req_kwargs["feed"] = DataFeed.IEX if DataFeed else "iex"
                try:
                    bars = _fetch(req_kwargs)
                except Exception:
                    return pd.DataFrame()
            else:
                raise

        try:
            df = getattr(bars, "df", None)
            if df is None or df.empty:
                return pd.DataFrame()
            if df.index.nlevels > 1:
                try:
                    df = df.xs(sym, level="symbol")
                except Exception:
                    df = df.reset_index()
            df = df.reset_index()
            # Normalise timestamp column name
            if "timestamp" not in df.columns and "time" in df.columns:
                df = df.rename(columns={"time": "timestamp"})
            try:
                df.to_csv(cache_path, index=False)
            except Exception:
                pass
            return df
        except Exception:
            return pd.DataFrame()


__all__ = ["AlpacaHedgerClient"]
