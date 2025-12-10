# placeholder
# Execution helpers for trading actions.
from typing import Any, Callable, Dict, Optional


def trade_spot(
    symbol: str,
    fallback: float,
    *,
    dashboard_price_fn: Optional[Callable[[str, float], float]] = None,
    get_data_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    floor_fn: Optional[Callable[[float], float]] = None,
) -> float:
    """Fetch spot using dashboard price when available, else fallback to provided data fetcher."""
    sym_clean = (symbol or "").strip().upper()
    if not sym_clean:
        return 0.0
    if dashboard_price_fn:
        try:
            return float(dashboard_price_fn(sym_clean, fallback))
        except Exception:
            pass
    if get_data_fn:
        try:
            val = float(get_data_fn(sym_clean).get("price", fallback) or fallback)
            return floor_fn(val) if floor_fn else val
        except Exception:
            return float(fallback)
    return float(fallback)


def compute_spot_totals(
    portfolio: dict,
    price_fn: Callable[[str, float], float],
) -> tuple[float, float, float]:
    """Retourne (pnl_total, notional_total, valeur_signée) pour le spot."""
    pnl_tot = 0.0
    notional_tot = 0.0
    long_val = 0.0
    short_val = 0.0
    for sym, data in (portfolio or {}).items():
        entry_price = float(data.get("current_price", data.get("avg_price", 0.0)) or 0.0)
        qty = float(data.get("quantity", 0.0) or 0.0)
        side = (data.get("side") or "long").lower()
        if qty <= 0:
            continue
        current_price = price_fn(sym, float(data.get("current_price", entry_price) or entry_price))
        if side == "long":
            long_val += qty * current_price
        else:
            short_val += qty * current_price
        if side == "long":
            pnl = (current_price - entry_price) * qty
        else:
            pnl = -(current_price - entry_price) * qty
        pnl_tot += pnl
        notional_tot += abs(entry_price * qty)
    value_tot = long_val - short_val
    return pnl_tot, notional_tot, value_tot


def execute_orders_with_trade_functions(
    orders_df,
    latest_prices,
    *,
    buy_fn: Callable[[str, float, float], Any],
    sell_fn: Callable[[str, float, float], Any],
) -> tuple[int, list[str]]:
    """Apply orders produced by eigen-portfolio calculation."""
    executed = 0
    skipped: list[str] = []
    if orders_df is None or getattr(orders_df, "empty", True):
        return executed, skipped
    for _, row in orders_df.iterrows():
        ticker = row.get("ticker")
        qty = float(row.get("order_qty", 0.0) or 0.0)
        price = float(getattr(latest_prices, "get", lambda k, d=0.0: d)(ticker) if ticker else 0.0)
        if not ticker or qty == 0 or price <= 0:
            skipped.append(str(ticker))
            continue
        if qty > 0:
            buy_fn(ticker, qty, price)
        else:
            sell_fn(ticker, abs(qty), price)
        executed += 1
    return executed, skipped
