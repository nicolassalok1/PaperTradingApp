"""
Risk management computations (model layer, no UI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


@dataclass
class AlpacaKeys:
    api_key: str
    api_secret: str
    base_url: str

    @classmethod
    def from_env(cls) -> "AlpacaKeys":
        api_key = os.getenv("APCA_API_KEY_ID") or ""
        api_secret = os.getenv("APCA_API_SECRET_KEY") or ""
        base_url = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
        if (not api_key or not api_secret) or api_key.lower().startswith("dummy") or api_secret.lower().startswith("dummy"):
            raise EnvironmentError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)


class RiskEngine:
    def __init__(self, keys: AlpacaKeys | None = None) -> None:
        self.offline = False
        try:
            self.keys = keys or AlpacaKeys.from_env()
        except Exception:
            self.offline = True
            self.trading_client = None
            self.data_client = None
            return

        is_paper = "paper" in (self.keys.base_url or "").lower()
        self.trading_client = TradingClient(
            self.keys.api_key,
            self.keys.api_secret,
            paper=is_paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.keys.api_key,
            secret_key=self.keys.api_secret,
        )

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

    def _positions(self) -> list[dict[str, Any]]:
        if self.offline or self.trading_client is None:
            return []
        try:
            positions = self.trading_client.get_all_positions()
        except Exception:
            return []
        return [self._to_dict(p) for p in positions] if positions else []

    # --- Public API -----------------------------------------------------
    def get_account_snapshot(self) -> dict[str, Any]:
        if self.offline or self.trading_client is None:
            acc = {}
        else:
            try:
                account = self.trading_client.get_account()
                acc = self._to_dict(account)
            except Exception:
                acc = {}
        return {
            "equity": float(acc.get("equity", 0.0) or 0.0),
            "cash": float(acc.get("cash", 0.0) or 0.0),
            "portfolio_value": float(acc.get("portfolio_value", acc.get("equity", 0.0)) or 0.0),
            "raw": acc,
        }

    def get_positions_summary(self) -> list[dict[str, Any]]:
        positions = self._positions()
        summary = []
        for pos in positions:
            qty = float(pos.get("qty", 0.0) or 0.0)
            mv = float(pos.get("market_value", 0.0) or 0.0)
            upnl = float(pos.get("unrealized_pl", 0.0) or 0.0)
            summary.append(
                {
                    "symbol": pos.get("symbol"),
                    "qty": qty,
                    "side": pos.get("side"),
                    "market_value": mv,
                    "unrealized_pl": upnl,
                }
            )
        return summary

    def compute_exposure(self, positions: Iterable[dict[str, Any]] | None = None) -> float:
        positions = list(positions) if positions is not None else self.get_positions_summary()
        return float(sum(abs(float(p.get("market_value", 0.0) or 0.0)) for p in positions))

    def compute_net_exposure(self, positions: Iterable[dict[str, Any]] | None = None) -> float:
        positions = list(positions) if positions is not None else self.get_positions_summary()
        net = 0.0
        for pos in positions:
            mv = float(pos.get("market_value", 0.0) or 0.0)
            side = (pos.get("side") or "").lower()
            signed_mv = mv if side == "long" else -abs(mv)
            net += signed_mv
        return float(net)

    def compute_unrealized_pnl_total(self, positions: Iterable[dict[str, Any]] | None = None) -> float:
        positions = list(positions) if positions is not None else self.get_positions_summary()
        return float(sum(float(p.get("unrealized_pl", 0.0) or 0.0) for p in positions))

    def compute_position_risk_metrics(self, symbol: str) -> dict[str, Any]:
        positions = self.get_positions_summary()
        pos = next((p for p in positions if (p.get("symbol") or "").upper() == (symbol or "").upper()), None)
        if not pos:
            return {}
        total_expo = self.compute_exposure(positions)
        total_expo = total_expo if total_expo > 0 else 1.0
        mv = float(pos.get("market_value", 0.0) or 0.0)
        return {
            "symbol": pos.get("symbol"),
            "qty": pos.get("qty"),
            "market_value": mv,
            "unrealized_pl": pos.get("unrealized_pl"),
            "weight_exposure": mv / total_expo,
            "risk_contribution": abs(mv) / total_expo,
        }

    def _load_price_panel(self, symbols: list[str], limit: int = 90) -> pd.DataFrame:
        if self.offline or self.data_client is None:
            return pd.DataFrame()

        symbols_norm = [s.strip().upper() for s in symbols if s]
        if not symbols_norm:
            return pd.DataFrame()

        def _safe_get_df(req: StockBarsRequest) -> pd.DataFrame:
            try:
                bars = self.data_client.get_stock_bars(req)
            except Exception:
                return pd.DataFrame()
            try:
                df = getattr(bars, "df", None)
            except Exception:
                return pd.DataFrame()
            if df is None or getattr(df, "empty", True):
                return pd.DataFrame()
            return df

        req_all = StockBarsRequest(
            symbol_or_symbols=symbols_norm,
            timeframe=TimeFrame.Day,
            limit=min(max(limit, 2), 1000),
        )

        df = _safe_get_df(req_all)
        if df.empty:
            # Fallback: fetch per-symbol (more robust if a single symbol breaks the batch request).
            frames: list[pd.DataFrame] = []
            for sym in symbols_norm:
                req_one = StockBarsRequest(
                    symbol_or_symbols=[sym],
                    timeframe=TimeFrame.Day,
                    limit=min(max(limit, 2), 1000),
                )
                df_one = _safe_get_df(req_one)
                if not df_one.empty:
                    frames.append(df_one)
            if not frames:
                return pd.DataFrame()
            df = pd.concat(frames, axis=0, ignore_index=False)

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        if df is None or df.empty:
            return pd.DataFrame()

        # Ensure we have the expected columns; otherwise return empty to avoid crashing callers.
        if "symbol" not in df.columns or "timestamp" not in df.columns:
            return pd.DataFrame()
        # Ensure we have symbol, timestamp, close columns
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        if "time" not in df.columns or "symbol" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()
        return df

    def compute_portfolio_pnl_series(self, limit: int = 90) -> pd.Series:
        positions = self.get_positions_summary()
        symbols = [p.get("symbol") for p in positions if p.get("symbol")]
        if not symbols:
            return pd.Series(dtype=float)
        qty_map = {p.get("symbol"): float(p.get("qty", 0.0) or 0.0) for p in positions}
        side_map = {p.get("symbol"): (p.get("side") or "").lower() for p in positions}
        df = self._load_price_panel(symbols, limit=limit)
        if df.empty:
            return pd.Series(dtype=float)
        df = df[["time", "symbol", "close"]]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        df["qty"] = df["symbol"].map(qty_map)
        df["side"] = df["symbol"].map(side_map)
        df["signed_qty"] = df.apply(
            lambda row: row["qty"] if str(row["side"]).lower() == "long" else -row["qty"], axis=1
        )
        df["position_value"] = df["close"] * df["signed_qty"]
        portfolio_series = df.pivot_table(index="time", values="position_value", aggfunc="sum").iloc[:, 0]
        # Convert to cumulative PnL relative to first value
        pnl_series = portfolio_series - float(portfolio_series.iloc[0] if not portfolio_series.empty else 0.0)
        return pnl_series

    def compute_var_lite(self, confidence: float = 0.95) -> float:
        positions = self.get_positions_summary()
        symbols = [p.get("symbol") for p in positions if p.get("symbol")]
        df = self._load_price_panel(symbols, limit=120)
        if df.empty:
            return 0.0
        df = df[["time", "symbol", "close"]]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        close_pivot = df.pivot_table(index="time", columns="symbol", values="close").ffill().dropna(how="any")
        if close_pivot.shape[0] < 2:
            return 0.0
        returns = close_pivot.diff().dropna(how="any")
        portfolio_value = self.get_account_snapshot().get("portfolio_value", 0.0)
        if portfolio_value <= 0:
            return 0.0
        agg_returns = returns.mean(axis=1)
        alpha = max(0.0, min(1.0, 1.0 - confidence))
        var = np.quantile(agg_returns, alpha) * portfolio_value
        return float(var)

    def trigger_alerts(self) -> list[str]:
        alerts: list[str] = []
        account = self.get_account_snapshot()
        positions = self.get_positions_summary()
        total_expo = self.compute_exposure(positions)
        net_expo = self.compute_net_exposure(positions)
        unreal = self.compute_unrealized_pnl_total(positions)
        portfolio_value = account.get("portfolio_value", 0.0)
        equity = account.get("equity", 0.0)

        if portfolio_value and equity < 0.95 * portfolio_value:
            alerts.append("Equity drawdown exceeds 5% of portfolio value.")

        if portfolio_value and unreal < -0.02 * portfolio_value:
            alerts.append("Unrealized PnL worse than -2% of portfolio value.")

        if total_expo > 0:
            for pos in positions:
                mv = float(pos.get("market_value", 0.0) or 0.0)
                symbol = pos.get("symbol")
                if abs(mv) / total_expo > 0.40:
                    alerts.append(f"Position {symbol} exceeds 40% of total exposure.")

        if abs(net_expo) > 0.50 * (portfolio_value or total_expo or 0.0):
            alerts.append("Net exposure exceeds 50% of portfolio value.")

        return alerts


_ENGINE = RiskEngine()


def get_account_snapshot() -> dict[str, Any]:
    return _ENGINE.get_account_snapshot()


def get_positions_summary() -> list[dict[str, Any]]:
    return _ENGINE.get_positions_summary()


def compute_exposure() -> float:
    return _ENGINE.compute_exposure()


def compute_net_exposure() -> float:
    return _ENGINE.compute_net_exposure()


def compute_unrealized_pnl_total() -> float:
    return _ENGINE.compute_unrealized_pnl_total()


def compute_position_risk_metrics(symbol: str) -> dict[str, Any]:
    return _ENGINE.compute_position_risk_metrics(symbol)


def compute_var_lite(confidence: float = 0.95) -> float:
    return _ENGINE.compute_var_lite(confidence=confidence)


def trigger_alerts() -> list[str]:
    return _ENGINE.trigger_alerts()


def compute_portfolio_pnl_series(limit: int = 90) -> pd.Series:
    return _ENGINE.compute_portfolio_pnl_series(limit=limit)


__all__ = [
    "get_account_snapshot",
    "get_positions_summary",
    "compute_exposure",
    "compute_net_exposure",
    "compute_unrealized_pnl_total",
    "compute_position_risk_metrics",
    "compute_var_lite",
    "trigger_alerts",
    "compute_portfolio_pnl_series",
]
