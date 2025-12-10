from typing import Any, Callable, Dict, Optional

from app.utils.math_utils import floor_4


def dashboard_price(
    sym: str,
    fallback: float,
    dashboard_prices: Dict[str, Any],
    get_data: Callable[[str], Dict[str, Any]],
    floor_func: Callable[[float], float],
) -> float:
    """Return cached dashboard price when available, otherwise fetch live and cache it."""
    sym_clean = (sym or "").strip().upper()
    if sym_clean and dashboard_prices:
        cached_val = dashboard_prices.get(sym_clean)
        try:
            if cached_val is not None:
                return float(cached_val)
        except Exception:
            pass
    live_price = float(get_data(sym_clean).get("price", fallback) or fallback)
    live_price = floor_func(live_price)
    if live_price > 0 and sym_clean:
        dashboard_prices[sym_clean] = live_price
    return live_price


def spot_live(
    sym: str,
    dashboard_prices: Dict[str, Any],
    price_fn: Callable[[str, float], float],
    cache: Optional[Dict[str, float | None]] = None,
) -> float | None:
    """Best-effort spot lookup using cached dashboard prices and provided price function."""
    if not sym:
        return None
    sym_clean = (sym or "").strip().upper()
    if cache is not None and sym_clean in cache:
        return cache[sym_clean]
    if dashboard_prices.get(sym_clean) is not None:
        try:
            cached_val = float(dashboard_prices[sym_clean])
            if cache is not None:
                cache[sym_clean] = cached_val
            return cached_val
        except Exception:
            pass
    spot_val = price_fn(sym_clean, 0.0)
    if cache is not None:
        cache[sym_clean] = spot_val
    return spot_val


def spot_live_exp(
    sym: str,
    dashboard_prices: Dict[str, Any],
    price_fn: Callable[[str, float], float],
    cache: Optional[Dict[str, float | None]] = None,
) -> float | None:
    """Spot helper for expired options plots."""
    return spot_live(sym, dashboard_prices, price_fn, cache)


def get_data(symbol: str) -> Dict[str, Any]:
    """Fetch a simple price dictionary for a ticker (yfinance last close)."""
    try:
        import yfinance as yf
    except Exception:
        return {"price": -1}
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"price": -1}
    try:
        hist = yf.Ticker(sym).history(period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return {"price": float(hist["Close"].iloc[-1])}
    except Exception:
        pass
    return {"price": -1}
