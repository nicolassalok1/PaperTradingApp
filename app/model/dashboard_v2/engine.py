"""
Dashboard v2 engine: Alpaca aggregation + placeholder loaders.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


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


def _load_placeholders_from_files() -> Dict[str, List[Dict[str, Any]]]:
    base = Path("data")
    files = {
        "forwards": base / "forwards_portfolio.json",
        "options_expired": base / "options_expired.json",
        "options_portfolio": base / "options_portfolio.json",
    }

    def _load(fp: Path) -> List[Dict[str, Any]]:
        try:
            if fp.exists():
                with fp.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    return {key: _load(path) for key, path in files.items()}


class FakeOfflineDashboardClient:
    def get_account_summary(self) -> Dict[str, float]:
        return {
            "equity": 0.0,
            "cash": 0.0,
            "portfolio_value": 0.0,
            "buying_power": 0.0,
            "unrealized_pl_total": 0.0,
            "realized_pl_total": 0.0,
        }

    def get_spot_positions(self) -> List[Dict[str, Any]]:
        return []

    def get_option_positions(self) -> List[Dict[str, Any]]:
        return []

    def load_placeholders(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "forwards": [],
            "options_expired": [],
            "options_portfolio": [],
        }


class _LiveDashboardBackend:
    def __init__(self, trading_client: TradingClient) -> None:
        self.trading = trading_client

    def get_account_summary(self) -> Dict[str, float]:
        acc = _to_dict(self.trading.get_account())
        return {
            "equity": float(acc.get("equity", 0.0) or 0.0),
            "cash": float(acc.get("cash", 0.0) or 0.0),
            "portfolio_value": float(acc.get("portfolio_value", acc.get("equity", 0.0)) or 0.0),
            "buying_power": float(acc.get("buying_power", 0.0) or 0.0),
            "unrealized_pl_total": float(acc.get("unrealized_pl", acc.get("unrealized_pl_total", 0.0)) or 0.0),
            "realized_pl_total": float(acc.get("realized_pl", acc.get("realized_pl_total", 0.0)) or 0.0),
        }

    def get_spot_positions(self) -> List[Dict[str, Any]]:
        positions = self.trading.get_all_positions()
        results: List[Dict[str, Any]] = []
        for p in positions:
            pdict = _to_dict(p)
            if str(pdict.get("asset_class", "")).lower() == "us_equity":
                results.append(pdict)
        return results

    def get_option_positions(self) -> List[Dict[str, Any]]:
        positions = self.trading.get_all_positions()
        results: List[Dict[str, Any]] = []
        for p in positions:
            pdict = _to_dict(p)
            if str(pdict.get("asset_class", "")).lower() == "option":
                results.append(pdict)
        return results

    def load_placeholders(self) -> Dict[str, List[Dict[str, Any]]]:
        return _load_placeholders_from_files()


class DashboardV2Client:
    def __init__(self) -> None:
        api_key = os.getenv("APCA_API_KEY_ID")
        api_secret = os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL")

        self.offline = False
        if not api_key or not api_secret or not base_url:
            self.offline = True
            self.trading = None
            self.data = None
            self.backend = FakeOfflineDashboardClient()
            return

        is_paper = "paper" in (base_url or "").lower()
        base_url = base_url or "https://paper-api.alpaca.markets"
        self.trading = TradingClient(
            api_key,
            api_secret,
            paper=is_paper,
        )
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        self.backend = _LiveDashboardBackend(self.trading)

    def get_account_summary(self) -> Dict[str, float]:
        return self.backend.get_account_summary()

    def get_spot_positions(self) -> List[Dict[str, Any]]:
        return self.backend.get_spot_positions()

    def get_option_positions(self) -> List[Dict[str, Any]]:
        return self.backend.get_option_positions()

    def load_placeholders(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.backend.load_placeholders()

    # --- Extensions for charts, PnL, trades, risk -----------------------
    def _proxy_history(self, lookback_days: int = 60, symbol: str = "SPY") -> pd.DataFrame:
        if self.offline or self.data is None:
            return pd.DataFrame()
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            limit=max(lookback_days, 2),
        )
        try:
            bars = self.data.get_stock_bars(req)
        except Exception:
            return pd.DataFrame()
        df = getattr(bars, "df", None)
        if df is None or df.empty:
            return pd.DataFrame()
        if df.index.nlevels > 1:
            try:
                df = df.xs(symbol, level="symbol")
            except Exception:
                df = df.reset_index()
        df = df.reset_index()
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        return df[["time", "close"]]

    def get_equity_curve(self, lookback_days: int = 60) -> List[Dict[str, Any]]:
        summary = self.get_account_summary()
        current_equity = summary.get("equity", 0.0)
        current_pv = summary.get("portfolio_value", current_equity)
        proxy = self._proxy_history(lookback_days=lookback_days)
        if proxy.empty:
            today = datetime.utcnow().date()
            return [{"date": today, "equity": current_equity, "portfolio_value": current_pv}]
        proxy = proxy.sort_values("time")
        base_price = float(proxy["close"].iloc[-1])
        curve = []
        for _, row in proxy.iterrows():
            px = float(row["close"])
            equity = current_equity * (px / base_price) if base_price else current_equity
            pv = current_pv * (px / base_price) if base_price else current_pv
            curve.append({"date": row["time"].date(), "equity": equity, "portfolio_value": pv})
        return curve

    def get_pnl_timeseries(self, lookback_days: int = 60) -> List[Dict[str, Any]]:
        eq_curve = self.get_equity_curve(lookback_days=lookback_days)
        if not eq_curve:
            return []
        equity0 = eq_curve[0]["equity"]
        pnl_ts: List[Dict[str, Any]] = []
        prev_equity = equity0
        cum = 0.0
        for point in eq_curve:
            equity = point["equity"]
            pnl = equity - prev_equity
            cum = equity - equity0
            pnl_ts.append({"date": point["date"], "pnl": pnl, "cum_pnl": cum})
            prev_equity = equity
        return pnl_ts

    @staticmethod
    def compute_drawdowns(equity_series: List[float]) -> Dict[str, Any]:
        if not equity_series:
            return {"max_drawdown": 0.0, "drawdown_series": []}
        equity = np.array(equity_series, dtype=float)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        return {"max_drawdown": float(drawdowns.min() if len(drawdowns) else 0.0), "drawdown_series": drawdowns.tolist()}

    @staticmethod
    def compute_volatility(returns_series: List[float], window: int = 20) -> List[float]:
        if not returns_series:
            return []
        returns = pd.Series(returns_series)
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        return [None if pd.isna(v) else float(v) for v in vol.tolist()]

    def get_trade_history(self, limit: int = 200, days_back: int = 30) -> List[Dict[str, Any]]:
        if self.offline or self.trading is None:
            return []
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit,
            )
            orders = self.trading.get_orders(req)
        except Exception:
            orders = []
        results: List[Dict[str, Any]] = []
        for o in orders:
            odict = _to_dict(o)
            submitted_at = odict.get("submitted_at") or odict.get("created_at")
            try:
                submitted_dt = pd.to_datetime(submitted_at)
                if submitted_dt.tzinfo is not None:
                    submitted_dt = submitted_dt.tz_convert(None)
            except Exception:
                submitted_dt = None
            if submitted_dt is not None:
                try:
                    if submitted_dt.to_pydatetime() < cutoff:
                        continue
                except Exception:
                    pass
            results.append(
                {
                    "symbol": odict.get("symbol"),
                    "side": odict.get("side"),
                    "qty": float(odict.get("qty", 0.0) or 0.0),
                    "filled_avg_price": float(odict.get("filled_avg_price", 0.0) or 0.0),
                    "submitted_at": submitted_at,
                    "status": odict.get("status"),
                }
            )
        return results

    def get_live_risk_snapshot(self, lookback_days: int = 60) -> Dict[str, Any]:
        spots = self.get_spot_positions()
        opts = self.get_option_positions()
        all_pos = spots + opts
        gross = sum(abs(float(p.get("market_value", 0.0) or 0.0)) for p in all_pos)
        net = sum(float(p.get("market_value", 0.0) or 0.0) for p in all_pos)
        largest_pct = 0.0
        if gross > 0:
            largest_pct = max(abs(float(p.get("market_value", 0.0) or 0.0)) for p in all_pos) / gross

        # Simple VaR-lite using proxy returns
        var_lite = None
        proxy = self._proxy_history(lookback_days=lookback_days)
        if not proxy.empty and len(proxy) > 5:
            returns = proxy["close"].pct_change().dropna()
            if not returns.empty:
                var_lite = float(returns.quantile(0.05)) * gross

        return {
            "gross_exposure": float(gross),
            "net_exposure": float(net),
            "largest_position_pct": float(largest_pct),
            "var_lite": var_lite,
        }

    def get_exposure_by_symbol(self) -> List[Dict[str, Any]]:
        spots = self.get_spot_positions()
        opts = self.get_option_positions()
        summary = self.get_account_summary()
        pv = float(summary.get("portfolio_value", 0.0) or 0.0)
        exposures: List[Dict[str, Any]] = []
        for p in spots + opts:
            mv = float(p.get("market_value", 0.0) or 0.0)
            weight = (mv / pv) if pv else 0.0
            exposures.append(
                {
                    "symbol": p.get("symbol"),
                    "market_value": mv,
                    "weight": weight,
                    "asset_class": p.get("asset_class"),
                }
            )
        return exposures

    def get_exposure_by_sector(self) -> List[Dict[str, Any]]:
        # Placeholder: no sector data available in Alpaca positions by default.
        return []


__all__ = ["DashboardV2Client"]
