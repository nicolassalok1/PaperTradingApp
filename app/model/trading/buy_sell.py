# placeholder
# Buy/Sell portfolio updates with injected dependencies.
import threading
import time
from typing import Callable, Dict, Optional

from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache
from app.utils.math_utils import floor_4
from app.model.trading.logs import log_trade


def _run_update_spots_async() -> None:
    """Fire-and-forget refresh of spots to keep cached prices in sync."""

    def _worker():
        try:
            from app.model.market_data.scripts import update_spots

            update_spots.main()
        except Exception:
            pass

    try:
        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        pass


def append_trade_log(
    symbol: str, side: str, qty: float, price: float, *, source: str, meta: Optional[Dict] = None
) -> None:
    """Single pipeline to append trades to trades_log.json."""
    try:
        log_trade(symbol, side, qty, price, source=source, meta=meta)
    except Exception:
        pass


def buy_asset(
    symbol: str,
    quantity: float,
    price: float,
    *,
    load_portfolio: Callable[[], Dict],
    save_portfolio: Callable[[Dict], None],
    update_cache_fn: Callable[[Dict, float, float], None],
    floor_fn: Callable[[float], float],
    time_str_fn: Callable[[], str],
    source: str = "manual",
    meta: Optional[Dict] = None,
) -> Optional[Dict]:
    portfolio = load_portfolio()
    now = time_str_fn()
    sym_clean = (symbol or "").strip().upper()

    position = portfolio.get(symbol)
    side = "long"
    realized_pnl = 0.0
    price_used = floor_fn(price) if floor_fn else floor_4(price)
    cash_delta = -abs(price_used * quantity)  # use provided price for balance adjustments
    action_label = "buy"

    def _adjust_balance(delta: float) -> None:
        cache = load_dashboard_cache()
        bal = float(cache.get("balance", 0.0) or 0.0) + float(delta)
        cache["balance"] = bal
        if sym_clean:
            prices = dict(cache.get("prices") or {})
            prices[sym_clean] = price_used
            cache["prices"] = prices
        save_dashboard_cache(cache)

    if position:
        old_qty = float(position.get("quantity", 0.0) or 0.0)
        old_entry = float(position.get("entry_price", 0.0) or price_used)
        s0_price = old_entry
        side = str(position.get("side", "long")).lower()

        if side == "long":
            new_qty = old_qty + quantity
            portfolio[symbol] = {
                "quantity": new_qty,
                "side": "long",
                "last_updated": now,
            }
        else:
            action_label = "buy_to_cover"
            if quantity < old_qty:
                realized_pnl += (s0_price - price) * quantity
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    "quantity": new_qty,
                    "side": "short",
                    "last_updated": now,
                }
            elif quantity == old_qty:
                realized_pnl += (s0_price - price) * quantity
                del portfolio[symbol]
            else:
                realized_pnl += (s0_price - price) * old_qty
                new_long_qty = quantity - old_qty
                portfolio[symbol] = {
                    "quantity": new_long_qty,
                    "side": "long",
                    "last_updated": now,
                }
    else:
        portfolio[symbol] = {
            "quantity": quantity,
            "side": "long",
            "last_updated": now,
        }

    _adjust_balance(cash_delta)
    update_cache_fn(portfolio, realized_pnl, cash_delta=cash_delta)
    save_portfolio(portfolio)
    log_side = side if action_label == "buy" else ("short" if "short" in action_label else "sell")
    append_trade_log(symbol, log_side, quantity, price_used, source=source, meta=meta)
    return portfolio.get(symbol)


def sell_asset(
    symbol: str,
    quantity: float,
    price: float,
    *,
    load_portfolio: Callable[[], Dict],
    save_portfolio: Callable[[Dict], None],
    update_cache_fn: Callable[[Dict, float, float], None],
    floor_fn: Callable[[float], float],
    time_str_fn: Callable[[], str],
    source: str = "manual",
    meta: Optional[Dict] = None,
) -> bool:
    portfolio = load_portfolio()
    now = time_str_fn()
    sym_clean = (symbol or "").strip().upper()

    position = portfolio.get(symbol)
    realized_pnl = 0.0
    price_used = floor_fn(price) if floor_fn else floor_4(price)
    cash_delta = abs(price_used * quantity)  # use provided price for balance adjustments
    action_label = "sell"

    def _adjust_balance(delta: float) -> None:
        cache = load_dashboard_cache()
        bal = float(cache.get("balance", 0.0) or 0.0) + float(delta)
        cache["balance"] = bal
        if sym_clean:
            prices = dict(cache.get("prices") or {})
            prices[sym_clean] = price_used
            cache["prices"] = prices
        save_dashboard_cache(cache)

    if position:
        old_qty = float(position.get("quantity", 0.0) or 0.0)
        old_entry = float(position.get("entry_price", 0.0) or price_used)
        s0_price = old_entry
        side = str(position.get("side", "long")).lower()

        if side == "long":
            if quantity < old_qty:
                realized_pnl += (price_used - s0_price) * quantity
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    "quantity": new_qty,
                    "entry_price": s0_price,
                    "side": "long",
                    "last_updated": now,
                }
            elif quantity == old_qty:
                realized_pnl += (price_used - s0_price) * quantity
                del portfolio[symbol]
            else:
                realized_pnl += (price_used - s0_price) * old_qty
                new_short_qty = quantity - old_qty
                portfolio[symbol] = {
                    "quantity": new_short_qty,
                    "entry_price": price_used,
                    "side": "short",
                    "last_updated": now,
                }
                action_label = "short_open"
        else:
            # Already short -> add more short
            new_qty = old_qty + quantity
            portfolio[symbol] = {
                "quantity": new_qty,
                "entry_price": s0_price if s0_price else price_used,
                "side": "short",
                "last_updated": now,
            }
            action_label = "short_add"
    else:
        portfolio[symbol] = {
            "quantity": quantity,
            "entry_price": price_used,
            "side": "short",
            "last_updated": now,
        }
        action_label = "short_open"

    _adjust_balance(cash_delta)
    update_cache_fn(portfolio, realized_pnl, cash_delta=cash_delta)
    save_portfolio(portfolio)
    side_logged = "buy" if action_label in ("buy", "buy_to_cover") else "sell"
    append_trade_log(symbol, side_logged, quantity, price_used, source=source, meta=meta)
    return True


__all__ = ["buy_asset", "sell_asset", "append_trade_log"]
