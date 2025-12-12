"""
Payoff helpers covering all option types in the dashboard schema.

Each function accepts the JSON schema used by the dashboard (option dict).
If a payoff needs price paths (Asian, barrier…), recent closes are fetched and cached
as CSV files under cache/ for reuse.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from app.model.market_data.service import fetch_historical_prices
from app.utils.paths import CACHE_CSV_DIR

BASE_DIR = Path(__file__).resolve().parent
PRICE_CACHE_DIR = CACHE_CSV_DIR
PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------- Price utilities ----------------------- #
def download_prices_to_csv(
    tickers: Iterable[str],
    period: str = "6mo",
    interval: str = "1d",
) -> Dict[str, Path]:
    """Download close prices per ticker into CSV; return mapping ticker->path. Skip download if file already exists."""
    paths: Dict[str, Path] = {}
    for tkr in set(tickers):
        if not tkr:
            continue
        out = PRICE_CACHE_DIR / f"{tkr}_close.csv"
        if out.exists():
            paths[tkr] = out
            continue
        try:
            df = fetch_historical_prices(tkr, freq="D")
            if df is None or df.empty or "close" not in df.columns:
                continue
            date_col = pd.to_datetime(df["date"]) if "date" in df.columns else None
            close = df["close"] if "close" in df.columns else df.iloc[:, 0]
            if date_col is not None:
                close = pd.Series(close.values, index=date_col, name="Close")
            close.to_csv(out, index_label="Date")
            paths[tkr] = out
        except Exception:
            continue
    return paths


def _series_from_cache(ticker: str) -> pd.Series | None:
    path = PRICE_CACHE_DIR / f"{ticker}_close.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=["Date"], index_col="Date").iloc[:, 0]
    except Exception:
        try:
            return pd.read_csv(path, index_col=0).iloc[:, 0]
        except Exception:
            return None


# ----------------------- Payoff primitives ----------------------- #
def payoff_vanilla(option: dict, spot: float) -> float:
    opt = str(option.get("option_type") or option.get("type") or "call").lower()
    side_sign = 1.0 if str(option.get("side", "long")).lower() == "long" else -1.0
    strike = float(option.get("strike", 0.0) or 0.0)
    if opt == "put":
        return side_sign * max(strike - spot, 0.0)
    return side_sign * max(spot - strike, 0.0)


def call_payoff(spot: float | np.ndarray, strike: float) -> np.ndarray:
    """Simple call payoff helper used in pricing tests."""
    s = np.asarray(spot, dtype=float)
    return np.maximum(s - strike, 0.0)


def payoff_digital(option: dict, spot: float) -> float:
    payout = float(option.get("payout", 1.0) or 1.0)
    opt = str(option.get("option_type") or option.get("type") or "call").lower()
    side_sign = 1.0 if str(option.get("side", "long")).lower() == "long" else -1.0
    strike = float(option.get("strike", 0.0) or 0.0)
    hit = spot > strike if opt == "call" else spot < strike
    return side_sign * (payout if hit else 0.0)


def payoff_spread(option: dict, spot: float) -> float:
    opt = str(option.get("option_type") or option.get("type") or "call").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", strike)) or strike)
    if opt == "call":
        return payoff_vanilla(
            {"option_type": "call", "strike": strike, "side": "long"}, spot
        ) + payoff_vanilla({"option_type": "call", "strike": strike2, "side": "short"}, spot)
    return payoff_vanilla(
        {"option_type": "put", "strike": strike, "side": "long"}, spot
    ) + payoff_vanilla({"option_type": "put", "strike": strike2, "side": "short"}, spot)


def payoff_straddle(option: dict, spot: float) -> float:
    strike = float(option.get("strike", 0.0) or 0.0)
    return payoff_vanilla(
        {"option_type": "call", "strike": strike, "side": option.get("side", "long")}, spot
    ) + payoff_vanilla(
        {"option_type": "put", "strike": strike, "side": option.get("side", "long")}, spot
    )


def payoff_strangle(option: dict, spot: float) -> float:
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", strike) or strike)
    k_put = min(strike, strike2)
    k_call = max(strike, strike2)
    return payoff_vanilla(
        {"option_type": "put", "strike": k_put, "side": option.get("side", "long")}, spot
    ) + payoff_vanilla(
        {"option_type": "call", "strike": k_call, "side": option.get("side", "long")}, spot
    )


def payoff_asian(option: dict, prices: Sequence[float]) -> float:
    opt = str(option.get("option_type") or option.get("type") or "call").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    side_sign = 1.0 if str(option.get("side", "long")).lower() == "long" else -1.0
    if not prices:
        return 0.0
    if "geom" in str(option.get("product", "")).lower():
        avg = float(np.exp(np.mean(np.log(np.clip(prices, 1e-12, None)))))
    else:
        avg = float(np.mean(prices))
    payoff = max(avg - strike, 0.0) if opt == "call" else max(strike - avg, 0.0)
    return side_sign * payoff


def payoff_barrier(option: dict, prices: Sequence[float], final_spot: float) -> float:
    opt = str(option.get("option_type") or option.get("type") or "call").lower()
    side_sign = 1.0 if str(option.get("side", "long")).lower() == "long" else -1.0
    strike = float(option.get("strike", 0.0) or 0.0)
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    barrier = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
    barrier_type = str(misc.get("barrier_type", "up")).lower()
    knock = str(misc.get("knock", misc.get("direction", "out"))).lower()
    hit = False
    for p in prices:
        if barrier_type == "up" and p >= barrier:
            hit = True
            break
        if barrier_type == "down" and p <= barrier:
            hit = True
            break
    if knock == "out" and hit:
        return 0.0
    if knock == "in" and not hit:
        return 0.0
    payoff = max(final_spot - strike, 0.0) if opt == "call" else max(strike - final_spot, 0.0)
    return side_sign * payoff


def payoff_multi_leg(option: dict, spot: float) -> float:
    legs = option.get("legs") or []
    total = 0.0
    for leg in legs:
        product = str(leg.get("product_type") or leg.get("product") or "").lower()
        if "digital" in product:
            total += payoff_digital(leg, spot)
        else:
            total += payoff_vanilla(leg, spot)
    return total


# ----------------------- Dispatcher ----------------------- #
def compute_option_payoff(option: dict, spot: float | None = None) -> float:
    """
    Compute payoff for any option record from the dashboard schema.
    If a path is required and not supplied, recent closes are fetched to CSV and reused.
    """
    if not isinstance(option, dict):
        return 0.0
    product = str(
        option.get("product_type")
        or option.get("product")
        or option.get("structure")
        or option.get("type")
        or ""
    ).lower()
    opt_type = str(
        option.get("option_type") or option.get("cpflag") or option.get("type") or ""
    ).lower()
    underlying = option.get("underlying")
    prices = None

    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    if isinstance(misc, dict) and "closing_prices" in misc:
        prices = misc.get("closing_prices")

    if prices is None and underlying and any(k in product for k in ["asian", "barrier"]):
        download_prices_to_csv([underlying], period="6mo", interval="1d")
        series = _series_from_cache(underlying)
        if series is not None and not series.empty:
            prices = series.tolist()
            spot = spot if spot is not None else float(series.iloc[-1])

    if spot is None:
        if prices:
            spot = float(prices[-1])
        else:
            spot = float(
                option.get("S0", option.get("underlying_close", option.get("strike", 0.0))) or 0.0
            )

    if option.get("legs"):
        return payoff_multi_leg(option, spot)

    if "asian" in product:
        return payoff_asian(option, prices or [])
    if "digital" in product:
        return payoff_digital(option, spot)
    if "barrier" in product:
        return payoff_barrier(option, prices or [spot], spot)
    if product in {"straddle"}:
        return payoff_straddle(option, spot)
    if product in {"strangle"}:
        return payoff_strangle(option, spot)
    if product in {
        "call_spread",
        "bull_call_spread",
        "callspread",
        "put_spread",
        "bear_put_spread",
        "putspread",
    }:
        return payoff_spread(option, spot)
    if not opt_type and "put" in product:
        opt_type = "put"
    return payoff_vanilla(
        {
            "option_type": opt_type or "call",
            "strike": option.get("strike"),
            "side": option.get("side", "long"),
        },
        spot,
    )


__all__ = [
    "download_prices_to_csv",
    "_series_from_cache",
    "payoff_vanilla",
    "call_payoff",
    "payoff_digital",
    "payoff_spread",
    "payoff_straddle",
    "payoff_strangle",
    "payoff_asian",
    "payoff_barrier",
    "payoff_multi_leg",
    "compute_option_payoff",
]
