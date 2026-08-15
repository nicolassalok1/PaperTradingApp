# placeholder
# Portfolio valuation helpers.
from typing import Callable, Dict, Tuple

import pandas as pd

from app.utils.io import load_json_file, save_json_file
from app.utils.paths import JSON_DIR


def portfolio_dict_to_df(portfolio: dict) -> pd.DataFrame:
    """Normalize the persisted portfolio dict into a clean DataFrame."""
    rows = []
    for ticker, entry in (portfolio or {}).items():
        qty = float(entry.get("quantity", 0.0) or 0.0)
        if qty == 0:
            continue
        rows.append(
            {
                "ticker": (ticker or "").strip().upper(),
                "quantity": qty,
                "avg_price": float(entry.get("avg_price", 0.0) or 0.0),
                "side": str(entry.get("side", "long")).lower(),
            }
        )
    return pd.DataFrame(rows)


def load_portfolio_price_panel(
    tickers: list[str],
    loader: Callable[[str], Tuple[pd.DataFrame | None, object, bool]],
    *,
    period: str = "1y",
    interval: str = "1d",
) -> tuple[pd.DataFrame | None, Dict[str, dict]]:
    """
    Load close history for portfolio tickers using provided loader (cache-aware).
    The loader should accept (ticker, period=..., interval=...) and return (df, path, from_cache).
    """
    merged = None
    sources: Dict[str, dict] = {}
    for tk in sorted({(t or "").strip().upper() for t in tickers if t}):
        df_hist, cache_path, from_cache = loader(tk, period=period, interval=interval)
        if df_hist is None or df_hist.empty:
            continue
        date_candidates = [
            c for c in df_hist.columns if str(c).lower() in {"date", "datetime", "index"}
        ]
        date_col = date_candidates[0] if date_candidates else df_hist.columns[0]
        price_cols = [c for c in df_hist.columns if c != date_col]
        price_col = tk if tk in price_cols else (price_cols[0] if price_cols else None)
        if not price_col:
            continue
        df_tmp = df_hist[[date_col, price_col]].copy()
        df_tmp[date_col] = pd.to_datetime(df_tmp[date_col], errors="coerce")
        df_tmp = df_tmp.dropna(subset=[date_col, price_col])
        if df_tmp.empty:
            continue
        df_tmp = df_tmp.rename(columns={price_col: tk})
        merged = df_tmp if merged is None else merged.merge(df_tmp, on=date_col, how="outer")
        sources[tk] = {"path": cache_path, "from_cache": from_cache}

    if merged is None or merged.empty:
        return None, sources

    merged = merged.sort_values(by=merged.columns[0])
    merged.set_index(merged.columns[0], inplace=True)
    merged.index.name = "Date"
    merged = merged.sort_index().ffill().dropna(how="all")
    return merged, sources


def compute_portfolio_value(positions, prices: dict | None = None) -> float:
    """
    Minimal portfolio valuation used by tests.
    Accepts an iterable of position dicts and an optional price map.
    """
    total = 0.0
    if not positions:
        return total
    price_map = {str(k).upper(): float(v or 0.0) for k, v in (prices or {}).items()}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        qty = float(pos.get("quantity", 0.0) or 0.0)
        if qty == 0:
            continue
        sym = (pos.get("symbol") or pos.get("ticker") or "").strip().upper()
        price = float(pos.get("price", 0.0) or 0.0)
        if sym and sym in price_map:
            price = price_map[sym]
        side = str(pos.get("side", "long")).lower()
        sign = 1.0 if side == "long" else -1.0
        total += sign * qty * price
    return float(total)


# ---------------------------------------------------------------------------
# Portfolio value refresh (moved from utils update_portfolio_value script)
# ---------------------------------------------------------------------------

DASHBOARD_VARS_FILE = JSON_DIR / "dashboard_vars.json"
SPOT_FILE = JSON_DIR / "spot_portfolio.json"
FORWARDS_FILE = JSON_DIR / "forwards_portfolio.json"


def _load_prices() -> dict:
    cache = load_json_file(DASHBOARD_VARS_FILE, {})
    prices = cache.get("prices") or {}
    return prices if isinstance(prices, dict) else {}


def _spot_value(prices: dict) -> float:
    portfolio = load_json_file(SPOT_FILE, {})
    total = 0.0
    if isinstance(portfolio, dict):
        for sym, entry in portfolio.items():
            if not isinstance(entry, dict):
                continue
            qty = float(entry.get("quantity", 0.0) or 0.0)
            side = str(entry.get("side", "long")).lower()
            sign = 1.0 if side == "long" else -1.0
            price = float(prices.get(sym.upper(), 0.0) or 0.0)
            total += sign * qty * price
    return total


def _forwards_value(prices: dict) -> float:
    """Value the open forward book at what settling it would actually pay.

    A forward is carried at (mark - forward_price) x quantity, signed by side —
    not at its notional. Carrying it at quantity x spot would count the underlying
    as owned outright while the strike has not been paid, and it would contradict
    `portfolio.settlement`, which credits only (spot - forward_price) x quantity to
    `balance` at maturity: settling would then drop `balance + portfolio_value` by
    forward_price x quantity out of nowhere.

    With no mark available the leg is worth zero, not its notional.
    """
    forwards = load_json_file(FORWARDS_FILE, {})
    total = 0.0
    if isinstance(forwards, dict):
        for _, entry in forwards.items():
            if not isinstance(entry, dict):
                continue
            qty = float(entry.get("quantity", 0.0) or 0.0)
            sym = (entry.get("symbol") or "").strip().upper()
            spot_price = float(prices.get(sym, 0.0) or 0.0)
            fwd_price = float(entry.get("forward_price", entry.get("price", 0.0)) or 0.0)
            price_ref = spot_price if spot_price > 0 else fwd_price
            side = str(entry.get("side", "long")).lower()
            sign = 1.0 if side == "long" else -1.0
            total += sign * qty * (price_ref - fwd_price)
    return total


def recompute_portfolio_value() -> float:
    prices = _load_prices()
    spot_val = _spot_value(prices)
    forwards_val = _forwards_value(prices)
    portfolio_value = spot_val + forwards_val

    cache = load_json_file(DASHBOARD_VARS_FILE, {})
    cache["portfolio_value"] = portfolio_value
    save_json_file(DASHBOARD_VARS_FILE, cache)
    return portfolio_value
