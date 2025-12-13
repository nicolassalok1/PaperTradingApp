from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.model.market_data.market_data import fetch_ohlc_history
from app.model.options.core.greeks import compute_option_greeks
from app.model.options.engines.black_scholes import black_scholes_price


def _regime_label(percentile: float | None) -> str:
    if percentile is None:
        return "N/A"
    p = float(percentile)
    if p >= 0.8:
        return "HIGH"
    if p >= 0.6:
        return "ABOVE AVG"
    if p >= 0.4:
        return "NORMAL"
    if p >= 0.2:
        return "BELOW AVG"
    return "LOW"


def compute_realized_vol_regime(
    symbol: str,
    *,
    period: str = "2y",
    window: int = 20,
    annualization: int = 252,
) -> dict[str, Any]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": "", "error": "Missing symbol"}

    df = fetch_ohlc_history(sym, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return {"symbol": sym, "error": "No OHLC history available"}

    close = pd.to_numeric(df["Close"], errors="coerce")
    dates = pd.to_datetime(df["Date"], errors="coerce")
    series = pd.Series(close.values, index=dates).dropna()
    if len(series) < max(5, window + 2):
        return {"symbol": sym, "error": "Insufficient history"}

    rets = np.log(series).diff()
    vol = rets.rolling(window=window).std() * np.sqrt(float(annualization))
    vol = vol.dropna()
    if vol.empty:
        return {"symbol": sym, "error": "Volatility series empty"}

    current_vol = float(vol.iloc[-1])
    percentile = float(vol.rank(pct=True).iloc[-1])
    return {
        "symbol": sym,
        "window": int(window),
        "annualization": int(annualization),
        "current_vol": current_vol,
        "percentile": percentile,
        "regime": _regime_label(percentile),
        "series": [{"date": d.date(), "vol": float(v)} for d, v in vol.items()],
    }


@dataclass(frozen=True)
class StraddleSnapshot:
    spot: float
    strike: float
    days_to_expiry: int
    iv: float
    r: float
    q: float
    call_price: float
    put_price: float
    straddle_price: float
    greeks: dict[str, float]


def compute_straddle_snapshot(
    *,
    spot: float,
    strike: float,
    days_to_expiry: int,
    iv: float,
    r: float = 0.0,
    q: float = 0.0,
) -> StraddleSnapshot:
    S0 = float(spot)
    K = float(strike)
    T = max(int(days_to_expiry), 0) / 365.0
    sigma = float(iv)
    rr = float(r)
    qq = float(q)

    call = float(black_scholes_price(S0, K, T, rr, sigma, "call", q=qq))
    put = float(black_scholes_price(S0, K, T, rr, sigma, "put", q=qq))

    call_g = compute_option_greeks(
        {"option_type": "call", "strike": K, "sigma": sigma, "r": rr, "q": qq, "T": T},
        spot=S0,
    )
    put_g = compute_option_greeks(
        {"option_type": "put", "strike": K, "sigma": sigma, "r": rr, "q": qq, "T": T},
        spot=S0,
    )

    greeks = {}
    for k in {"delta", "gamma", "vega", "theta", "rho"}:
        greeks[k] = float((call_g.get(k, 0.0) or 0.0) + (put_g.get(k, 0.0) or 0.0))

    return StraddleSnapshot(
        spot=S0,
        strike=K,
        days_to_expiry=int(days_to_expiry),
        iv=sigma,
        r=rr,
        q=qq,
        call_price=call,
        put_price=put,
        straddle_price=float(call + put),
        greeks=greeks,
    )


def compute_straddle_iv_crush(
    *,
    pre_spot: float,
    post_spot: float,
    strike: float,
    days_to_expiry: int,
    pre_iv: float,
    post_iv: float,
    r: float = 0.0,
    q: float = 0.0,
    qty: float = 1.0,
) -> dict[str, Any]:
    pre = compute_straddle_snapshot(
        spot=float(pre_spot),
        strike=float(strike),
        days_to_expiry=int(days_to_expiry),
        iv=float(pre_iv),
        r=float(r),
        q=float(q),
    )
    post = compute_straddle_snapshot(
        spot=float(post_spot),
        strike=float(strike),
        days_to_expiry=int(days_to_expiry),
        iv=float(post_iv),
        r=float(r),
        q=float(q),
    )
    qn = float(qty or 0.0)
    pnl_long = (post.straddle_price - pre.straddle_price) * qn
    pnl_short = -pnl_long
    return {
        "pre": pre,
        "post": post,
        "qty": qn,
        "pnl_long": float(pnl_long),
        "pnl_short": float(pnl_short),
        "iv_crush_pct": float((pre.iv - post.iv) / pre.iv) if pre.iv else None,
        "spot_move_pct": float((post.spot - pre.spot) / pre.spot) if pre.spot else None,
    }


__all__ = [
    "compute_realized_vol_regime",
    "StraddleSnapshot",
    "compute_straddle_snapshot",
    "compute_straddle_iv_crush",
]

