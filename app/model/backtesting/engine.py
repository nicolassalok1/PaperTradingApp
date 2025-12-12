import datetime
from typing import Any, Callable, Dict, Tuple

import pandas as pd

from app.model.market_data.service import fetch_historical_prices

from app.model.backtesting.signals import compute_level_prices


def auto_execute_trading_levels(
    equities_cfg: Dict,
    *,
    load_ts_exec_log: Callable[[], Dict],
    save_ts_exec_log: Callable[[Dict], None],
    floor_fn: Callable[[float], float],
    buy_fn: Callable[[str, float, float], None],
    sell_fn: Callable[[str, float, float], None],
    lookback_days: int = 30,
) -> Tuple[Dict, Dict]:
    """
    Check historical prices for configured trading systems and auto-fire one level when crossed.

    Args:
        equities_cfg: Trading systems config (mutable).
        load_ts_exec_log: Loader for execution log dict.
        save_ts_exec_log: Saver for execution log dict.
        floor_fn: Price rounding helper.
        buy_fn / sell_fn: Execution callbacks (symbol, qty, price).
        lookback_days: History window if no creation date is set.
    Returns:
        (updated_equities_cfg, exec_log)
    """
    exec_log = load_ts_exec_log()
    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(days=14)
    systems_changed = False
    # purge old executions
    for sym, rec in list(exec_log.items()):
        execs = rec.get("executions", [])
        filtered = []
        for ex in execs:
            try:
                ts = datetime.datetime.fromisoformat(ex.get("timestamp"))
            except Exception:
                ts = None
            if ts and ts < cutoff:
                continue
            filtered.append(ex)
        rec["executions"] = filtered
        rec["last_check"] = now.isoformat()
        exec_log[sym] = rec

    for symbol, cfg in (equities_cfg or {}).items():
        if str(cfg.get("status", "Off")).lower() != "on":
            continue
        levels = cfg.get("levels", {})
        if not levels:
            continue
        qty_cfg = cfg.get("quantity", 1)
        try:
            qty_default = max(0.0, float(qty_cfg))
        except Exception:
            qty_default = 1.0
        if qty_default <= 0:
            qty_default = 1.0
        direction = str(cfg.get("direction", "long")).lower()
        executed_for_symbol = False
        created_at_raw = cfg.get("created_at")
        try:
            created_at_dt = (
                datetime.datetime.fromisoformat(created_at_raw) if created_at_raw else None
            )
        except Exception:
            created_at_dt = None
        start_date = (
            created_at_dt.date()
            if created_at_dt
            else (datetime.date.today() - datetime.timedelta(days=lookback_days))
        )
        try:
            hist = fetch_historical_prices(symbol, start=start_date.isoformat(), end=None, freq="D")
        except Exception:
            hist = pd.DataFrame()
        if hist.empty or "close" not in hist.columns:
            continue
        hist = hist[hist["date"] >= pd.Timestamp(start_date)]
        prices = hist["close"].astype(float)
        p_min, p_max = prices.min(), prices.max()
        lvl_items = sorted(levels.items(), key=lambda kv: kv[0])
        for lvl_key, lvl_price in lvl_items:
            try:
                lvl_val = float(lvl_price)
            except Exception:
                continue
            if not (p_min <= lvl_val <= p_max):
                continue
            rec = exec_log.get(symbol, {"executions": []})
            already = False
            for ex in rec.get("executions", []):
                if ex.get("level_key") == str(lvl_key) and ex.get("level_price") == lvl_val:
                    already = True
                    break
            if already:
                continue
            try:
                qty = qty_default
            except Exception:
                qty = 1
            exec_price = floor_fn(lvl_val)
            if direction == "long":
                buy_fn(symbol, qty, exec_price)
                side_exec = "long"
            else:
                sell_fn(symbol, qty, exec_price)
                side_exec = "short"
            rec.setdefault("executions", []).append(
                {
                    "level_key": str(lvl_key),
                    "level_price": exec_price,
                    "side": side_exec,
                    "qty": qty,
                    "drawdown": cfg.get("drawdown"),
                    "timestamp": now.isoformat(),
                }
            )
            rec["last_check"] = now.isoformat()
            exec_log[symbol] = rec
            cfg["status"] = "Off"
            equities_cfg[symbol] = cfg
            systems_changed = True
            executed_for_symbol = True
            break
        if executed_for_symbol:
            continue
    if systems_changed:
        save_ts_exec_log(exec_log)
    return equities_cfg, exec_log


def add_trading_system(
    equities_cfg: Dict,
    *,
    symbol: str,
    direction: str,
    qty_per_trade: float,
    drawdown_pct: float,
    levels: int,
    get_data_fn: Callable[[str], Dict[str, Any]],
    save_equities_fn: Callable[[Dict], None],
    clear_cache_fn: Callable[[str], None],
) -> Tuple[bool, Dict, Dict]:
    """Create and persist a trading system configuration without Streamlit dependencies."""
    equities_cfg = dict(equities_cfg or {})
    sym = (symbol or "").upper().strip()
    if not sym:
        return False, {"error": "Please enter a symbol"}, equities_cfg
    if len(equities_cfg) >= 10:
        return False, {"error": "Maximum limit of 10 equities reached!"}, equities_cfg
    if sym in equities_cfg:
        return False, {"error": f"{sym} already exists!"}, equities_cfg

    price_data = get_data_fn(sym) or {}
    try:
        entry_price = float(price_data.get("price", 0.0))
    except Exception:
        entry_price = 0.0
    if entry_price <= 0:
        return False, {"error": f"Could not fetch price for {sym}"}, equities_cfg

    drawdown_decimal = drawdown_pct / 100.0
    level_prices = compute_level_prices(entry_price, drawdown_decimal, levels=levels)
    direction_clean = str(direction or "long").lower()
    try:
        qty_val = int(qty_per_trade)
    except Exception:
        qty_val = 1

    equities_cfg[sym] = {
        "position": 0,
        "entry_price": entry_price,
        "levels": level_prices,
        "drawdown": drawdown_decimal,
        "quantity": qty_val,
        "direction": direction_clean,
        "status": "Off",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    save_equities_fn(equities_cfg)
    clear_cache_fn(sym)
    return (
        True,
        {
            "symbol": sym,
            "entry_price": entry_price,
            "direction": direction_clean,
            "levels": level_prices,
        },
        equities_cfg,
    )


def set_trading_system_status(
    equities_cfg: Dict, symbol: str, active: bool, *, save_equities_fn: Callable[[Dict], None]
) -> Dict:
    equities_cfg = dict(equities_cfg or {})
    sym = (symbol or "").upper().strip()
    if sym not in equities_cfg:
        return equities_cfg
    equities_cfg[sym]["status"] = "On" if active else "Off"
    save_equities_fn(equities_cfg)
    return equities_cfg


def remove_trading_system(
    equities_cfg: Dict, symbol: str, *, save_equities_fn: Callable[[Dict], None]
) -> Dict:
    equities_cfg = dict(equities_cfg or {})
    sym = (symbol or "").upper().strip()
    if sym in equities_cfg:
        equities_cfg.pop(sym)
        save_equities_fn(equities_cfg)
    return equities_cfg
