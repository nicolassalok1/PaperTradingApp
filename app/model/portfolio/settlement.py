"""
Forward settlement helpers moved from app.utils.scripts.update_balance.
"""

from __future__ import annotations

import datetime
from typing import Dict

import pandas as pd

from app.model.portfolio.forwards import load_forwards, save_forwards
from app.model.trading.logs import append_trade_log
from app.model.market_data.realtime import get_data
from app.model.market_data.history import load_or_fetch_closing_history
from app.model.dashboard.cache import load_dashboard_cache, save_dashboard_cache


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_and_settle_forward(fid: str, fwd: dict, today: datetime.date) -> bool:
    """
    If the forward is expired (maturity <= today), adjust balance by forward payoff.
    Returns True if a trade was executed (and the forward should be removed), else False.
    """
    try:
        maturity = datetime.date.fromisoformat(str(fwd.get("maturity")))
    except Exception:
        maturity = None

    if not maturity or maturity > today:
        return False

    qty = float(fwd.get("quantity", 0.0) or 0.0)
    fwd_price = float(fwd.get("forward_price", 0.0) or 0.0)
    side = (fwd.get("side", "long") or "long").lower()
    symbol = (fwd.get("symbol") or "").strip().upper()
    if not symbol or qty <= 0 or fwd_price <= 0:
        return False

    current_price = fwd_price
    try:
        df_hist, _, _ = load_or_fetch_closing_history(symbol, period="2y", interval="1d")
        if df_hist is not None and not df_hist.empty:
            date_col = "Date" if "Date" in df_hist.columns else df_hist.columns[0]
            price_col = "Close" if "Close" in df_hist.columns else df_hist.columns[-1]
            df_hist[date_col] = pd.to_datetime(df_hist[date_col], errors="coerce")
            df_hist = df_hist.dropna(subset=[date_col, price_col])
            if not df_hist.empty:
                target = pd.to_datetime(maturity)
                mask = df_hist[date_col] <= target
                df_prior = df_hist[mask]
                if not df_prior.empty:
                    current_price = float(df_prior.iloc[-1][price_col])
                else:
                    current_price = float(df_hist.iloc[0][price_col])
    except Exception:
        pass

    if current_price == fwd_price:
        current_price = float(get_data(symbol).get("price", fwd_price) or fwd_price)
    delta = (
        (current_price - fwd_price) * qty if side == "long" else (fwd_price - current_price) * qty
    )

    cache = load_dashboard_cache()
    balance = float(cache.get("balance", 0.0) or 0.0) + delta
    cache["balance"] = balance
    save_dashboard_cache(cache)

    append_trade_log(symbol, side, qty, current_price, source="forwards")
    return True


def process_matured_forwards() -> None:
    forwards = load_forwards()
    today = datetime.date.today()
    remaining = {}
    processed = 0

    for fid, f in forwards.items():
        if check_and_settle_forward(fid, f, today):
            processed += 1
        else:
            remaining[fid] = f

    if processed:
        save_forwards(remaining)
    print(f"Processed {processed} matured forwards. Remaining: {len(remaining)}")


__all__ = ["check_and_settle_forward", "process_matured_forwards"]
