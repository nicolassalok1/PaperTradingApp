"""
Cache refresh helpers for dashboard and portfolio price data.
Moved from app.utils.scripts.update_spots to the model layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Set

from app.model.market_data import market_data
from app.model.market_data.realtime import get_data
from app.utils.io import load_json_file, save_json_file
from app.utils.math_utils import floor_4
from app.utils.paths import JSON_DIR

DASHBOARD_VARS_FILE = JSON_DIR / "dashboard_vars.json"
SPOT_FILE = JSON_DIR / "spot_portfolio.json"
FILES_TO_SCAN = [
    JSON_DIR / "options_portfolio.json",
    SPOT_FILE,
    JSON_DIR / "forwards_portfolio.json",
    JSON_DIR / "trading_systems.json",
]


def collect_tickers() -> Set[str]:
    tickers: Set[str] = set()
    for path in FILES_TO_SCAN:
        if not path.exists():
            continue
        try:
            data = load_json_file(path, {})
        except Exception:
            continue

        if isinstance(data, dict):
            for key, val in data.items():
                _maybe_add_ticker(tickers, key)
                _extract_tickers_from_entry(tickers, val)
        elif isinstance(data, list):
            for item in data:
                _extract_tickers_from_entry(tickers, item)
    return tickers


def fetch_prices(tickers: Iterable[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym in sorted(set(tickers)):
        price = 0.0
        try:
            spot = market_data.fetch_spot_price(sym)
            if spot is not None:
                price = floor_4(spot)
        except Exception:
            pass
        if price <= 0:
            try:
                price = floor_4(get_data(sym).get("price", 0.0) or 0.0)
            except Exception:
                price = 0.0
        if price > 0:
            prices[sym] = price
    return prices


def update_dashboard_prices(prices: Dict[str, float]) -> None:
    cache = load_json_file(DASHBOARD_VARS_FILE, {})
    cache["prices"] = prices
    save_json_file(DASHBOARD_VARS_FILE, cache)


def update_portfolio_files(prices: Dict[str, float]) -> None:
    for path in FILES_TO_SCAN:
        if not path.exists():
            continue
        try:
            data = load_json_file(path, {})
        except Exception:
            continue

        if path == SPOT_FILE:
            new_data = _sanitize_spot_portfolio(data)
        else:
            allow_current_price = path != SPOT_FILE
            new_data = _update_current_prices_in_dict(
                data, prices, allow_current_price=allow_current_price
            )
            if isinstance(data, list):
                new_list = []
                for item in data:
                    if isinstance(item, dict):
                        sym = item.get("symbol") or item.get("ticker")
                        sym_clean = str(sym).strip().upper() if sym else ""
                        if allow_current_price and sym_clean in prices:
                            item = dict(item)
                            item["current_price"] = prices[sym_clean]
                        elif not allow_current_price and "current_price" in item:
                            item = {k: v for k, v in item.items() if k != "current_price"}
                    new_list.append(item)
                new_data = new_list

        save_json_file(path, new_data)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _maybe_add_ticker(tickers: Set[str], value: Any) -> None:
    if isinstance(value, str):
        sym = value.strip().upper()
        if sym:
            tickers.add(sym)


def _extract_tickers_from_entry(tickers: Set[str], entry: Any) -> None:
    if not isinstance(entry, dict):
        return
    for key in ("symbol", "ticker"):
        if key in entry:
            _maybe_add_ticker(tickers, entry.get(key))


def _sanitize_spot_portfolio(data: Any) -> Dict[str, dict]:
    sanitized: Dict[str, dict] = {}
    if isinstance(data, dict):
        for sym, entry in data.items():
            e = entry if isinstance(entry, dict) else {}
            sanitized[sym] = {
                "quantity": float(e.get("quantity", 0.0) or 0.0),
                "side": str(e.get("side", "long")).lower(),
                "last_updated": e.get("last_updated"),
            }
    return sanitized


def _update_current_prices_in_dict(
    data: Any, prices: Dict[str, float], *, allow_current_price: bool
) -> Any:
    if isinstance(data, dict):
        updated = {}
        for key, val in data.items():
            sym_key = key.strip().upper() if isinstance(key, str) else ""
            if isinstance(val, dict):
                sym = val.get("symbol") or val.get("ticker") or sym_key
                sym_clean = str(sym).strip().upper()
                if allow_current_price and sym_clean in prices:
                    val = dict(val)
                    val["current_price"] = prices[sym_clean]
                elif not allow_current_price and isinstance(val, dict) and "current_price" in val:
                    val = {k: v for k, v in val.items() if k != "current_price"}
                updated[key] = val
            else:
                updated[key] = val
        return updated
    return data


__all__ = [
    "collect_tickers",
    "fetch_prices",
    "update_dashboard_prices",
    "update_portfolio_files",
]
