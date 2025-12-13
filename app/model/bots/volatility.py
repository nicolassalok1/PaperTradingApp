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


def _linregress_np(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Minimal linear regression helper to avoid adding new deps.

    Returns slope/intercept/r2 (best-effort; r2=nan if undefined).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def compute_realized_vol_mean_reversion(
    symbol: str,
    *,
    period: str = "2y",
    vol_window: int = 20,
    forward_window: int = 30,
    annualization: int = 252,
    min_points: int = 60,
) -> dict[str, Any]:
    """
    Mean-reversion style diagnostics inspired by QG/vol_dashboard, on realized volatility.

    - Computes realized vol as rolling std(log returns) * sqrt(annualization)
    - Computes forward realized vol as a forward rolling mean (shifted)
    - Fits regressions to estimate whether vol tends to revert
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": "", "error": "Missing symbol"}

    win = max(5, int(vol_window or 0))
    fwd = max(5, int(forward_window or 0))

    df = fetch_ohlc_history(sym, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return {"symbol": sym, "error": "No OHLC history available"}

    close = pd.to_numeric(df["Close"], errors="coerce")
    dates = pd.to_datetime(df["Date"], errors="coerce")
    series = pd.Series(close.values, index=dates).dropna()
    series = series[~series.index.isna()]
    if len(series) < max(min_points, win + fwd + 5):
        return {"symbol": sym, "error": "Insufficient history"}

    rets = np.log(series).diff()
    vol = rets.rolling(window=win).std() * np.sqrt(float(annualization))
    vol = vol.dropna()
    if vol.empty or len(vol) < min_points:
        return {"symbol": sym, "error": "Volatility series empty"}

    forward_vol = vol.rolling(window=fwd).mean().shift(-fwd)
    percentile = vol.rank(pct=True)

    df_an = pd.DataFrame(
        {
            "date": vol.index,
            "current_vol": vol.values,
            "forward_vol": forward_vol.values,
            "vol_diff": (forward_vol - vol).values,
            "percentile": percentile.values,
        }
    ).dropna()

    if df_an.empty or len(df_an) < min_points:
        return {"symbol": sym, "error": "Insufficient aligned samples for forward window"}

    reg_fwd = _linregress_np(df_an["current_vol"].to_numpy(), df_an["forward_vol"].to_numpy())

    slope = float(reg_fwd["slope"])
    intercept = float(reg_fwd["intercept"])
    if np.isfinite(slope) and abs(1.0 - slope) > 1e-6:
        split_x = float(intercept / (1.0 - slope))
    else:
        split_x = float(np.nanmedian(df_an["current_vol"].to_numpy()))

    hi_mask = df_an["current_vol"] > split_x
    lo_mask = ~hi_mask

    reg_diff_all = _linregress_np(df_an["current_vol"].to_numpy(), df_an["vol_diff"].to_numpy())
    reg_diff_hi = (
        _linregress_np(df_an.loc[hi_mask, "current_vol"].to_numpy(), df_an.loc[hi_mask, "vol_diff"].to_numpy())
        if int(hi_mask.sum()) >= 10
        else {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    )
    reg_diff_lo = (
        _linregress_np(df_an.loc[lo_mask, "current_vol"].to_numpy(), df_an.loc[lo_mask, "vol_diff"].to_numpy())
        if int(lo_mask.sum()) >= 10
        else {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    )

    current_vol = float(df_an["current_vol"].iloc[-1])
    current_pct = float(df_an["percentile"].iloc[-1])
    mean_rev_hint = "NEUTRAL"
    if current_pct >= 0.8:
        mean_rev_hint = "EXPECT MEAN REVERSION DOWN"
    elif current_pct <= 0.2:
        mean_rev_hint = "EXPECT MEAN REVERSION UP"

    return {
        "symbol": sym,
        "period": str(period),
        "vol_window": int(win),
        "forward_window": int(fwd),
        "annualization": int(annualization),
        "current_vol": current_vol,
        "percentile": current_pct,
        "regime": _regime_label(current_pct),
        "mean_reversion_hint": mean_rev_hint,
        "split_x": float(split_x),
        "reg_forward": reg_fwd,
        "reg_vol_diff_all": reg_diff_all,
        "reg_vol_diff_high": reg_diff_hi,
        "reg_vol_diff_low": reg_diff_lo,
        "series": df_an.to_dict(orient="records"),
    }


def compute_markov_vol_transition(
    symbol: str,
    *,
    period: str = "2y",
    window: int = 20,
    annualization: int = 252,
    n_states: int = 3,
) -> dict[str, Any]:
    """
    Discrete Markov transition matrix over realized volatility regimes.

    Regimes are defined by quantiles of the realized vol series (default: 3 states).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": "", "error": "Missing symbol"}

    n = int(n_states or 0)
    if n < 2 or n > 6:
        n = 3

    df = fetch_ohlc_history(sym, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return {"symbol": sym, "error": "No OHLC history available"}

    close = pd.to_numeric(df["Close"], errors="coerce")
    dates = pd.to_datetime(df["Date"], errors="coerce")
    series = pd.Series(close.values, index=dates).dropna()
    series = series[~series.index.isna()]
    if len(series) < max(10, int(window or 0) + 5):
        return {"symbol": sym, "error": "Insufficient history"}

    rets = np.log(series).diff()
    vol = rets.rolling(window=max(5, int(window or 0))).std() * np.sqrt(float(annualization))
    vol = vol.dropna()
    if vol.empty or len(vol) < 20:
        return {"symbol": sym, "error": "Volatility series empty"}

    # Quantile-based regimes
    qs = [i / n for i in range(1, n)]
    cuts = [float(vol.quantile(q)) for q in qs]

    def _state(v: float) -> int:
        for i, c in enumerate(cuts):
            if v <= c:
                return i
        return n - 1

    states = np.array([_state(float(v)) for v in vol.to_numpy()], dtype=int)

    counts = np.zeros((n, n), dtype=int)
    for a, b in zip(states[:-1], states[1:]):
        counts[int(a), int(b)] += 1

    probs = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_sum = float(counts[i].sum())
        if row_sum > 0:
            probs[i] = counts[i] / row_sum

    labels = [f"S{i+1}" for i in range(n)]
    if n == 3:
        labels = ["LOW", "MID", "HIGH"]

    current_state = int(states[-1])
    next_probs = {labels[j]: float(probs[current_state, j]) for j in range(n)}

    series_out = [
        {"date": d.date(), "vol": float(v), "state": labels[int(s)]}
        for d, v, s in zip(vol.index, vol.to_numpy(), states)
    ]

    return {
        "symbol": sym,
        "period": str(period),
        "window": int(window),
        "annualization": int(annualization),
        "n_states": int(n),
        "cuts": cuts,
        "labels": labels,
        "current_state": labels[current_state],
        "next_state_probs": next_probs,
        "transition_counts": counts.tolist(),
        "transition_matrix": probs.tolist(),
        "series": series_out,
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
    "compute_realized_vol_mean_reversion",
    "compute_markov_vol_transition",
    "StraddleSnapshot",
    "compute_straddle_snapshot",
    "compute_straddle_iv_crush",
]
