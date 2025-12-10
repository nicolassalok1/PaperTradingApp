"""
Dashboard business services (no UI).
Provides pricing helpers, PnL aggregation, and refresh pipelines.
"""

from __future__ import annotations

import datetime
import time
from typing import Dict, Iterable, Tuple

from app.model.ai.chatgpt import chatgpt_response  # re-export convenience
from app.model.backtesting.engine import auto_execute_trading_levels
from app.model.dashboard.cache import (
    DASHBOARD_VARS_FILE,
    dashboard_cache_last_refresh,
    load_dashboard_cache,
    refresh_dashboard_cache,
    reset_dashboard,
    save_dashboard_cache,
)
from app.model.market_data.realtime import get_data
from app.model.market_data.market_data import (
    fetch_spot_price,
    clear_closing_history_cache,
    load_or_fetch_closing_history,
)
from app.model.market_data.cache_refresh import (
    collect_tickers,
    fetch_prices,
    update_dashboard_prices,
    update_portfolio_files,
)
from app.model.options.core.book import load_options_book, save_options_book
from app.model.portfolio.forwards import load_forwards, save_forwards
from app.model.portfolio.positions import load_portfolio_default, save_portfolio_default
from app.model.portfolio.settlement import process_matured_forwards
from app.model.portfolio.valuation import recompute_portfolio_value
from app.model.trading.buy_sell import buy_asset as core_buy_asset, sell_asset as core_sell_asset
from app.model.trading.execution import compute_spot_totals
from app.model.trading.logs import load_trades_log, log_trade
from app.model.trading.systems import (
    load_equities,
    load_ts_exec_log,
    save_equities,
    save_ts_exec_log,
    load_equities as load_sell_systems,
)
from app.model.dashboard.utils import collect_dashboard_tickers
from app.utils.io import load_json_file, save_json_file
from app.utils.math_utils import floor_4


EXPIRED_OPTIONS_FILE = DASHBOARD_VARS_FILE.parent / "options_expired.json"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def dashboard_price(sym: str, fallback: float = 0.0) -> float:
    """
    Fetch price from dashboard cache or live data; persist cache on success.
    """
    sym_clean = (sym or "").strip().upper()
    if not sym_clean:
        return float(fallback)
    cache = load_dashboard_cache()
    prices = cache.get("prices") or {}
    try:
        cached_val = prices.get(sym_clean)
        if cached_val is not None:
            return float(cached_val)
    except Exception:
        pass
    live_price = float(get_data(sym_clean).get("price", fallback) or fallback)
    if live_price > 0:
        prices[sym_clean] = live_price
        cache["prices"] = prices
        save_dashboard_cache(cache)
    return live_price


def compute_spot_pnl(portfolio: dict, price_fn) -> Tuple[float, float, float | None]:
    spot_total_pnl = 0.0
    spot_total_notional = 0.0
    for symbol, data in (portfolio or {}).items():
        avg_price = float(data.get("avg_price") or data.get("entry_price") or 0.0)
        quantity = float(data.get("quantity") or data.get("qty") or 0.0)
        side = (data.get("side") or data.get("direction") or "long").lower()
        current_price = price_fn(symbol, avg_price)
        if side == "long":
            pnl = (current_price - avg_price) * quantity
        else:
            pnl = (avg_price - current_price) * quantity
        spot_total_pnl += pnl
        notional = avg_price * quantity
        spot_total_notional += notional
    delta_spot = None
    if spot_total_notional > 0:
        delta_spot = (spot_total_pnl / spot_total_notional) * 100.0
    return spot_total_pnl, spot_total_notional, delta_spot


def compute_forward_pnl(forwards: dict, price_fn) -> Tuple[float, float, float | None]:
    total_forward_pnl = 0.0
    total_forward_notional = 0.0
    for f in forwards.values():
        symbol = f.get("symbol", "").upper()
        qty = float(f.get("quantity", 0.0) or 0.0)
        fwd_price = float(f.get("forward_price", 0.0) or 0.0)
        side = (f.get("side", "long") or "long").lower()
        spot_now = price_fn(symbol, 0.0)
        sign = +1.0 if side == "long" else -1.0
        pnl_fwd = sign * (spot_now - fwd_price) * qty
        total_forward_pnl += pnl_fwd
        total_forward_notional += abs(fwd_price * qty)
    delta_fwd = None
    if total_forward_notional > 0:
        delta_fwd = (total_forward_pnl / total_forward_notional) * 100.0
    return total_forward_pnl, total_forward_notional, delta_fwd


def load_expired_options() -> dict:
    return load_json_file(EXPIRED_OPTIONS_FILE, {})


def save_expired_options(data: dict) -> None:
    save_json_file(EXPIRED_OPTIONS_FILE, data if isinstance(data, dict) else {})


# ---------------------------------------------------------------------------
# Pipelines and trading systems refresh
# ---------------------------------------------------------------------------


def refresh_spot_prices_with_systems() -> None:
    """
    Refresh trading systems and apply auto-exec levels, updating dashboard cache balances.
    """
    cache = load_dashboard_cache()
    balance = float(cache.get("balance", 0.0) or 0.0)

    def _update_cache(portfolio: dict, realized_pnl: float, cash_delta: float) -> None:
        nonlocal balance
        balance = balance + cash_delta
        cache_local = load_dashboard_cache()
        cache_local["balance"] = balance
        save_dashboard_cache(cache_local)

    def _buy_fn(sym: str, qty: float, price: float):
        return core_buy_asset(
            sym,
            qty,
            price,
            load_portfolio=load_portfolio_default,
            save_portfolio=save_portfolio_default,
            update_cache_fn=_update_cache,
            floor_fn=floor_4,
            time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
            source="trading_system_refresh",
        )

    def _sell_fn(sym: str, qty: float, price: float):
        return core_sell_asset(
            sym,
            qty,
            price,
            load_portfolio=load_portfolio_default,
            save_portfolio=save_portfolio_default,
            update_cache_fn=_update_cache,
            floor_fn=floor_4,
            time_str_fn=lambda: time.strftime("%Y-%m-%d %H:%M:%S"),
            source="trading_system_refresh",
        )

    equities = load_equities()
    equities, _ = auto_execute_trading_levels(
        equities,
        load_ts_exec_log=load_ts_exec_log,
        save_ts_exec_log=save_ts_exec_log,
        floor_fn=floor_4,
        buy_fn=_buy_fn,
        sell_fn=_sell_fn,
    )
    save_equities(equities)


def refresh_all_spots_pipeline() -> None:
    """
    Run full refresh:
      1) update spot prices in cache/portfolio files
      2) auto-execute trading systems
      3) settle matured forwards
      4) recompute portfolio value
    """
    try:
        tickers = collect_tickers()
        prices = fetch_prices(tickers)
        update_dashboard_prices(prices)
        update_portfolio_files(prices)
    except Exception:
        pass

    try:
        refresh_spot_prices_with_systems()
    except Exception:
        pass

    try:
        process_matured_forwards()
    except Exception:
        pass

    try:
        recompute_portfolio_value()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Re-exports / simple wrappers for controllers
# ---------------------------------------------------------------------------


def load_portfolio() -> dict:
    return load_portfolio_default()


def save_portfolio(portfolio: dict) -> None:
    save_portfolio_default(portfolio)


def load_forwards_data() -> dict:
    return load_forwards()


def save_forwards_data(data: dict) -> None:
    save_forwards(data)


def load_options() -> dict:
    return load_options_book()


def save_options(data: dict) -> None:
    save_options_book(data)


__all__ = [
    "DASHBOARD_VARS_FILE",
    "auto_execute_trading_levels",
    "chatgpt_response",
    "collect_dashboard_tickers",
    "compute_forward_pnl",
    "compute_spot_pnl",
    "dashboard_cache_last_refresh",
    "dashboard_price",
    "fetch_spot_price",
    "load_dashboard_cache",
    "load_equities",
    "load_expired_options",
    "load_forwards_data",
    "load_options",
    "load_portfolio",
    "load_sell_systems",
    "load_trades_log",
    "load_ts_exec_log",
    "log_trade",
    "refresh_all_spots_pipeline",
    "refresh_dashboard_cache",
    "refresh_spot_prices_with_systems",
    "reset_dashboard",
    "save_dashboard_cache",
    "save_equities",
    "save_expired_options",
    "save_forwards_data",
    "save_options",
    "save_portfolio",
    "save_ts_exec_log",
    "load_or_fetch_closing_history",
    "compute_spot_totals",
    "clear_closing_history_cache",
]
