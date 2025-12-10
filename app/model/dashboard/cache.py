"""
Dashboard cache helpers (business logic only, no UI).
Moved from app.utils.dashboard.cache and reset to keep MVC boundaries.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

from app.model.market_data.realtime import get_data
from app.utils.paths import JSON_DIR, CACHE_CSV_DIR

DASHBOARD_VARS_FILE = JSON_DIR / "dashboard_vars.json"


def load_dashboard_cache() -> Dict:
    return _load_json(DASHBOARD_VARS_FILE, {"prices": {}, "last_refresh": None})


def save_dashboard_cache(cache: Dict) -> None:
    _save_json(DASHBOARD_VARS_FILE, cache)


def dashboard_cache_last_refresh(cache: Dict) -> datetime.date | None:
    try:
        ts = cache.get("last_refresh")
        if ts:
            return datetime.date.fromisoformat(str(ts))
    except Exception:
        return None
    return None


def refresh_dashboard_cache(tickers: Iterable[str], cache: Dict | None = None) -> Dict:
    cache = dict(cache or {"prices": {}})
    prices = dict(cache.get("prices") or {})
    for sym in tickers:
        data = get_data(sym)
        try:
            price = float(data.get("price", -1))
        except Exception:
            price = -1
        if price > 0:
            prices[sym] = price
    cache["prices"] = prices
    cache["last_refresh"] = datetime.date.today().isoformat()
    save_dashboard_cache(cache)
    return cache


def reset_dashboard() -> None:
    """
    Clear dashboard-related JSON files and cached CSVs.
    """
    files_defaults = {
        JSON_DIR
        / "dashboard_vars.json": {
            "prices": {},
            "last_refresh": None,
            "balance": 0.0,
            "portfolio_value": 0.0,
        },
        JSON_DIR / "forwards_portfolio.json": {},
        JSON_DIR / "options_expired.json": {},
        JSON_DIR / "options_portfolio.json": {},
        JSON_DIR / "spot_portfolio.json": {},
        JSON_DIR / "trades_log.json": [],
        JSON_DIR / "trading_systems_execs.json": {},
        JSON_DIR / "trading_systems.json": {},
        JSON_DIR / "options_h_params_calibrated.json": {},
    }

    for path, default_content in files_defaults.items():
        _clear_file(path, default_content)

    try:
        if CACHE_CSV_DIR.exists():
            for item in CACHE_CSV_DIR.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
    except Exception:
        pass

    try:
        save_dashboard_cache(
            {"prices": {}, "last_refresh": None, "balance": 0.0, "portfolio_value": 0.0}
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers (pure I/O)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clear_file(path: Path, default_content):
    """Best effort: overwrite the file with default_content (or unlink)."""
    try:
        if default_content is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_content, f, indent=2)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
