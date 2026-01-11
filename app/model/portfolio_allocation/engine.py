"""
Alpaca-backed portfolio allocation engine.
Includes data access, optimizers, and rebalance order generation/execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from app.utils.secrets import get_secret

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


@dataclass
class AlpacaPortfolioClient:
    api_key: str
    api_secret: str
    base_url: str

    def __init__(self) -> None:
        self.api_key = (get_secret("APCA_API_KEY_ID") or "").strip()
        self.api_secret = (get_secret("APCA_API_SECRET_KEY") or "").strip()
        self.base_url = (get_secret("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").strip()
        self.offline = False
        if (not self.api_key or not self.api_secret) or self.api_key.lower().startswith("dummy") or self.api_secret.lower().startswith("dummy"):
            self.offline = True

        if self.offline:
            self.trading = None
            self.data = None
            return

        is_paper = "paper" in (self.base_url or "").lower()
        self.trading = TradingClient(
            self.api_key,
            self.api_secret,
            paper=is_paper,
            raw_data=True,  # raw responses avoid enum mismatches on asset_class
        )
        self.data = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

    def get_account(self) -> Dict[str, Any]:
        if self.offline or self.trading is None:
            return {}
        try:
            account = self.trading.get_account()
            return _to_dict(account)
        except Exception:
            return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        if self.offline or self.trading is None:
            return []
        try:
            positions = self.trading.get_all_positions()
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        for p in positions or []:
            pdict = _to_dict(p)
            if "equity" in str(pdict.get("asset_class", "")).lower():
                results.append(pdict)
        return results

    def get_latest_price(self, symbol: str) -> float:
        symbol_norm = (symbol or "").strip().upper()
        if not symbol_norm:
            return 0.0
        price = 0.0
        if not self.offline and self.data is not None:
            req = StockBarsRequest(
                symbol_or_symbols=symbol_norm,
                timeframe=TimeFrame.Day,
                limit=1,
            )
            try:
                bars = self.data.get_stock_bars(req)
            except Exception:
                bars = None
            if bars is not None:
                df = getattr(bars, "df", None)
                if df is not None and not df.empty:
                    if isinstance(df.index, pd.MultiIndex):
                        try:
                            df = df.xs(symbol_norm, level="symbol")
                        except Exception:
                            df = df.reset_index()
                    df = df.reset_index()
                    price_col = "close" if "close" in df.columns else df.columns[-1]
                    try:
                        price = float(df.iloc[-1][price_col])
                    except Exception:
                        price = 0.0

        if price <= 0:
            try:
                from app.model.market_data.market_data import fetch_ohlc_history

                df_fb = fetch_ohlc_history(symbol_norm, period="3mo", interval="1d")
                if df_fb is not None and not df_fb.empty:
                    close_col = "Close" if "Close" in df_fb.columns else df_fb.columns[-1]
                    price = float(df_fb.iloc[-1][close_col])
            except Exception:
                price = 0.0

        return price

    def get_history(self, symbols: Iterable[str], lookback_days: int = 60) -> pd.DataFrame:
        symbols_norm = [s.strip().upper() for s in symbols if s]
        if not symbols_norm:
            return pd.DataFrame()
        if self.offline or self.data is None:
            return pd.DataFrame()
        req = StockBarsRequest(
            symbol_or_symbols=symbols_norm,
            timeframe=TimeFrame.Day,
            limit=max(lookback_days, 2),
        )
        try:
            bars = self.data.get_stock_bars(req)
        except Exception:
            return pd.DataFrame()
        try:
            df = getattr(bars, "df", None)
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "time"})
            if "close" not in df.columns and "Close" in df.columns:
                df = df.rename(columns={"Close": "close"})
            df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
            df = df.dropna(subset=["time"])
            # Inject symbol if missing and single ticker requested
            if "symbol" not in df.columns and len(symbols_norm) == 1:
                df["symbol"] = symbols_norm[0]
            return df
        except Exception:
            return pd.DataFrame()


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


def get_current_portfolio(client: AlpacaPortfolioClient) -> Dict[str, Any]:
    account = client.get_account()
    positions = client.get_positions()
    equity = float(account.get("equity", 0.0) or 0.0)
    cash = float(account.get("cash", 0.0) or 0.0)

    symbols: List[str] = []
    market_values: List[float] = []
    for pos in positions:
        symbols.append(pos.get("symbol"))
        market_values.append(float(pos.get("market_value", 0.0) or 0.0))

    total_mv = sum(abs(v) for v in market_values) or 1.0
    weights = [v / total_mv for v in market_values]

    return {
        "symbols": symbols,
        "market_values": market_values,
        "weights": weights,
        "equity": equity,
        "cash": cash,
    }


def compute_returns_matrix(
    client: AlpacaPortfolioClient, symbols: Iterable[str], lookback_days: int = 60
) -> Dict[str, Any]:
    df = client.get_history(symbols, lookback_days=lookback_days)
    symbols_list = [s for s in symbols if s]

    if df.empty or not {"time", "symbol", "close"}.issubset(set(df.columns)):
        # Fallback to cached/Stooq/Yahoo OHLC if Alpaca is offline or returns bad payloads
        from app.model.market_data.market_data import fetch_ohlc_history

        frames: list[pd.DataFrame] = []
        for sym in symbols_list:
            try:
                fb = fetch_ohlc_history(sym, period=f"{lookback_days}d", interval="1d")
            except Exception:
                fb = pd.DataFrame()
            if fb is None or fb.empty or "Date" not in fb.columns or "Close" not in fb.columns:
                continue
            f = fb[["Date", "Close"]].rename(columns={"Date": "time", "Close": "close"})
            f["symbol"] = sym.strip().upper()
            frames.append(f)
        if frames:
            df = pd.concat(frames, axis=0, ignore_index=True)
        else:
            return {"symbols": symbols_list, "returns": []}

    df = df[["time", "symbol", "close"]]
    pivot = df.pivot_table(index="time", columns="symbol", values="close").ffill().dropna(how="any")
    returns = pivot.pct_change().dropna(how="any")
    return {"symbols": list(pivot.columns), "returns": returns.values}


def _ensure_numpy(returns: Any) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _project_simplex(w: np.ndarray) -> np.ndarray:
    # Ensure weights are non-negative and sum to 1
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / total
    return w


def markowitz_optimize(returns: Any, mode: str = "min_var") -> List[float]:
    R = _ensure_numpy(returns)
    if R.shape[0] < 2:
        return [1.0 / R.shape[1]] * R.shape[1]
    mu = np.mean(R, axis=0)
    Sigma = np.cov(R, rowvar=False)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6  # regularize
    inv_Sigma = np.linalg.pinv(Sigma)

    if mode == "max_sharpe":
        w = inv_Sigma @ mu
    else:  # min variance
        ones = np.ones_like(mu)
        w = inv_Sigma @ ones

    w = _project_simplex(w)
    return w.tolist()


def risk_parity_optimize(returns: Any, max_iter: int = 500, lr: float = 0.01) -> List[float]:
    R = _ensure_numpy(returns)
    n = R.shape[1]
    if R.shape[0] < 2:
        return [1.0 / n] * n
    Sigma = np.cov(R, rowvar=False)
    Sigma += np.eye(n) * 1e-6

    w = np.ones(n) / n
    target_contrib = 1.0 / n
    for _ in range(max_iter):
        marginal = Sigma @ w
        port_var = w @ marginal
        if port_var <= 0:
            break
        risk_contrib = w * marginal / (np.sqrt(port_var) + 1e-12)
        gradient = risk_contrib - target_contrib
        w -= lr * gradient
        w = _project_simplex(w)
    return w.tolist()


def eigen_portfolio_optimize(returns: Any) -> List[float]:
    """
    Long-only EigenPortfolio using the first principal component of the
    correlation-adjusted returns matrix.

    - Standardises each asset return to avoid scale bias.
    - Uses the leading eigenvector (largest eigenvalue) of the correlation
      matrix as loadings.
    - Enforces a positive orientation and projects to the simplex to keep
      weights >= 0 and summing to 1.
    """
    R = _ensure_numpy(returns)
    n = R.shape[1]
    if R.shape[0] < 2:
        return [1.0 / n] * n

    # Drop rows with non-finite values to avoid NaN pollution
    mask = np.isfinite(R).all(axis=1)
    R = R[mask]
    if R.shape[0] < 2:
        return [1.0 / n] * n

    # Standardise to unit variance to mimic PCA on correlation matrix
    R = R - np.mean(R, axis=0, keepdims=True)
    std = np.std(R, axis=0, ddof=1, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    Z = R / std

    Sigma = np.cov(Z, rowvar=False)
    Sigma += np.eye(n) * 1e-6  # regularise to avoid singular covariance

    vals, vecs = np.linalg.eigh(Sigma)
    idx = np.argmax(vals)  # first principal component (largest eigenvalue)
    eig_vec = vecs[:, idx]

    # Fix arbitrary sign, then take absolute loadings for long-only
    if eig_vec.sum() < 0:
        eig_vec = -eig_vec
    w = np.abs(eig_vec)
    w = _project_simplex(w)
    return w.tolist()


def compute_rebalance_orders(
    current: Dict[str, Any], target: Dict[str, Any], client: AlpacaPortfolioClient
) -> List[Dict[str, Any]]:
    symbols = target.get("symbols") or current.get("symbols") or []
    target_weights = target.get("weights") or target.get("target_weights") or []
    current_symbols = current.get("symbols") or []
    current_values = current.get("market_values") or []
    current_map = {sym: float(val) for sym, val in zip(current_symbols, current_values)}

    equity = float(current.get("equity", 0.0) or 0.0)
    if equity <= 0:
        equity = sum(abs(v) for v in current_values)
    if equity <= 0:
        return []

    orders: List[Dict[str, Any]] = []
    for sym, tw in zip(symbols, target_weights):
        if tw is None:
            continue
        price = client.get_latest_price(sym)
        if price <= 0:
            continue
        target_value = float(tw) * equity
        current_value = float(current_map.get(sym, 0.0))
        delta_value = target_value - current_value
        delta_qty = delta_value / price
        if abs(delta_qty) < 1e-3:
            continue
        side = "buy" if delta_qty > 0 else "sell"
        orders.append(
            {
                "symbol": sym,
                "side": side,
                "qty": round(abs(delta_qty), 4),
                "current_value": current_value,
                "target_value": target_value,
                "price_used": price,
                "reason": f"rebalance {target.get('method', '')}".strip(),
            }
        )
    return orders


def execute_rebalance_orders(
    client: AlpacaPortfolioClient, orders: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    executions: List[Dict[str, Any]] = []
    for order in orders:
        symbol = order.get("symbol")
        qty = float(order.get("qty", 0.0) or 0.0)
        side = str(order.get("side", "buy")).lower()
        if not symbol or qty <= 0:
            continue
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,  # fractional orders require DAY TIF
        )
        alpaca_order = client.trading.submit_order(req)
        alpaca_dict = _to_dict(alpaca_order)
        executions.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "status": alpaca_dict.get("status"),
                "id": alpaca_dict.get("id"),
            }
        )
    return executions


__all__ = [
    "AlpacaPortfolioClient",
    "get_current_portfolio",
    "compute_returns_matrix",
    "markowitz_optimize",
    "risk_parity_optimize",
    "eigen_portfolio_optimize",
    "compute_rebalance_orders",
    "execute_rebalance_orders",
]
