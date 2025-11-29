import math
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch

DEFAULT_R = 0.02
DEFAULT_Q = 0.0
DEFAULT_SIGMA = 0.2
DEFAULT_T = 1.0

# Anchor cache to the notebooks/ directory to avoid scattering files when running from subfolders
_CACHE_SPY_CLOSE = Path(__file__).resolve().parent.parent / "GPT" / "closing_cache.csv"
_LEGACY_CACHE_SPY_CLOSE = Path(__file__).resolve().parent.parent / "GPT" / "_cache_spy_close.csv"
_CLOSING_CACHE_AGE_HOURS = 0.0
_HES_DIR = Path(__file__).resolve().parents[2] / "scripts" / "scriptsGPT" / "pricing_scripts"

# Ensure Heston torch pricer is importable
if str(_HES_DIR) not in sys.path:
    sys.path.insert(0, str(_HES_DIR))

from Heston.heston_torch import HestonParams, carr_madan_call_torch  # noqa: E402


def _migrate_legacy_spy_cache() -> None:
    """
    If an old cache file name is present, migrate it to the new closing_cache.csv.
    Keeps the latest cache available under the unified name.
    """
    try:
        if _LEGACY_CACHE_SPY_CLOSE.exists():
            _CACHE_SPY_CLOSE.parent.mkdir(parents=True, exist_ok=True)
            _LEGACY_CACHE_SPY_CLOSE.replace(_CACHE_SPY_CLOSE)
    except Exception:
        pass


def _save_closing_cache(series: pd.Series) -> None:
    """Persist SPY close series and reset age indicator."""
    global _CLOSING_CACHE_AGE_HOURS
    try:
        _CACHE_SPY_CLOSE.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(_CACHE_SPY_CLOSE, index_label="date")
        _CLOSING_CACHE_AGE_HOURS = 0.0
    except Exception:
        pass


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_call(S: float, K: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_price_put(S: float, K: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def fetch_spy_history(period: str = "1y", interval: str = "1d", cache_path: Path = _CACHE_SPY_CLOSE) -> pd.Series:
    """Fetch SPY close prices with a simple CSV cache under notebooks/GPT/."""
    _migrate_legacy_spy_cache()
    try:
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty and "Close" in cached.columns:
                return cached["Close"]
    except Exception:
        pass
    import yfinance as yf

    data = yf.download("SPY", period=period, interval=interval, progress=False)
    if data.empty or "Close" not in data:
        raise RuntimeError("Impossible de récupérer les prix SPY")
    close = data["Close"]
    _save_closing_cache(close)
    return close


def last_spy_close(period: str = "1y", interval: str = "1d") -> float:
    _migrate_legacy_spy_cache()
    close = fetch_spy_history(period=period, interval=interval)
    return float(close.iloc[-1])


def price_heston_carr_madan(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    option_type: str = "call",
) -> float:
    """
    Price a European option under Heston using Carr–Madan FFT pricer.

    Args:
        S0: Spot price.
        K: Strike.
        T: Maturity (years).
        r: Risk-free rate.
        q: Continuous dividend (or repo).
        kappa, theta, sigma, rho, v0: Heston parameters.
        option_type: "call"/"c" or "put"/"p".

    Returns:
        Option price (float).
    """
    params = HestonParams(
        torch.tensor(float(kappa)),
        torch.tensor(float(theta)),
        torch.tensor(float(sigma)),
        torch.tensor(float(rho)),
        torch.tensor(float(v0)),
    )
    call_price = float(carr_madan_call_torch(float(S0), float(r), float(q), float(T), params, float(K)))
    if option_type.lower().startswith("c"):
        return call_price
    return float(call_price - float(S0) * math.exp(-float(q) * float(T)) + float(K) * math.exp(-float(r) * float(T)))


def payoff_call(spot, strike: float):
    s = np.asarray(spot, dtype=float)
    return np.maximum(s - strike, 0.0)


def payoff_put(spot, strike: float):
    s = np.asarray(spot, dtype=float)
    return np.maximum(strike - s, 0.0)


def payoff_straddle(spot, strike: float):
    return payoff_call(spot, strike) + payoff_put(spot, strike)


def payoff_strangle(spot, k_put: float, k_call: float):
    return payoff_put(spot, k_put) + payoff_call(spot, k_call)


def payoff_call_spread(spot, k_long: float, k_short: float):
    return payoff_call(spot, k_long) - payoff_call(spot, k_short)


def payoff_put_spread(spot, k_long: float, k_short: float):
    return payoff_put(spot, k_long) - payoff_put(spot, k_short)


def payoff_butterfly(spot, k1: float, k2: float, k3: float):
    return payoff_call(spot, k1) - 2.0 * payoff_call(spot, k2) + payoff_call(spot, k3)


def payoff_condor(spot, k1: float, k2: float, k3: float, k4: float):
    return payoff_call(spot, k1) - payoff_call(spot, k2) - payoff_call(spot, k3) + payoff_call(spot, k4)


def payoff_iron_butterfly(spot, k_put_long: float, k_center: float, k_call_long: float):
    return payoff_put(spot, k_put_long) - payoff_put(spot, k_center) - payoff_call(spot, k_center) + payoff_call(spot, k_call_long)


def payoff_iron_condor(spot, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float):
    return payoff_put(spot, k_put_long) - payoff_put(spot, k_put_short) - payoff_call(spot, k_call_short) + payoff_call(spot, k_call_long)


def price_straddle_bs(
    S: float,
    strike: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_call: float | None = None,
    sigma_put: float | None = None,
) -> float:
    """Straddle BS pricing with optional distinct call/put vols."""
    sigma_c = DEFAULT_SIGMA if sigma_call is None else sigma_call
    sigma_p = DEFAULT_SIGMA if sigma_put is None else sigma_put
    # Fallback: if specific vols absent, use the common sigma argument.
    if sigma_call is None:
        sigma_c = sigma
    if sigma_put is None:
        sigma_p = sigma
    return bs_price_put(S, strike, r=r, q=q, sigma=sigma_p, T=T) + bs_price_call(S, strike, r=r, q=q, sigma=sigma_c, T=T)


def price_strangle_bs(
    S: float,
    k_put: float,
    k_call: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_call: float | None = None,
    sigma_put: float | None = None,
) -> float:
    sigma_c = sigma if sigma_call is None else sigma_call
    sigma_p = sigma if sigma_put is None else sigma_put
    return bs_price_put(S, k_put, r=r, q=q, sigma=sigma_p, T=T) + bs_price_call(S, k_call, r=r, q=q, sigma=sigma_c, T=T)


def pricing_strangle_bs(
    S: float,
    k_put: float,
    k_call: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_call: float | None = None,
    sigma_put: float | None = None,
) -> float:
    """Alias explicite pour le pricing d'un strangle via Black-Scholes (somme put+call)."""
    return price_strangle_bs(S, k_put, k_call, r=r, q=q, sigma=sigma, T=T, sigma_call=sigma_call, sigma_put=sigma_put)


def price_call_spread_bs(
    S: float,
    k_long: float,
    k_short: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_long: float | None = None,
    sigma_short: float | None = None,
) -> float:
    sigma_l = sigma if sigma_long is None else sigma_long
    sigma_s = sigma if sigma_short is None else sigma_short
    return bs_price_call(S, k_long, r=r, q=q, sigma=sigma_l, T=T) - bs_price_call(S, k_short, r=r, q=q, sigma=sigma_s, T=T)


def price_put_spread_bs(
    S: float,
    k_long: float,
    k_short: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_long: float | None = None,
    sigma_short: float | None = None,
) -> float:
    sigma_l = sigma if sigma_long is None else sigma_long
    sigma_s = sigma if sigma_short is None else sigma_short
    return bs_price_put(S, k_long, r=r, q=q, sigma=sigma_l, T=T) - bs_price_put(S, k_short, r=r, q=q, sigma=sigma_s, T=T)


def price_butterfly_bs(
    S: float,
    k1: float,
    k2: float,
    k3: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_k1: float | None = None,
    sigma_k2: float | None = None,
    sigma_k3: float | None = None,
) -> float:
    sig1 = sigma if sigma_k1 is None else sigma_k1
    sig2 = sigma if sigma_k2 is None else sigma_k2
    sig3 = sigma if sigma_k3 is None else sigma_k3
    return bs_price_call(S, k1, r=r, q=q, sigma=sig1, T=T) - 2.0 * bs_price_call(S, k2, r=r, q=q, sigma=sig2, T=T) + bs_price_call(S, k3, r=r, q=q, sigma=sig3, T=T)


def price_condor_bs(
    S: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_k1: float | None = None,
    sigma_k2: float | None = None,
    sigma_k3: float | None = None,
    sigma_k4: float | None = None,
) -> float:
    sig1 = sigma if sigma_k1 is None else sigma_k1
    sig2 = sigma if sigma_k2 is None else sigma_k2
    sig3 = sigma if sigma_k3 is None else sigma_k3
    sig4 = sigma if sigma_k4 is None else sigma_k4
    return (
        bs_price_call(S, k1, r=r, q=q, sigma=sig1, T=T)
        - bs_price_call(S, k2, r=r, q=q, sigma=sig2, T=T)
        - bs_price_call(S, k3, r=r, q=q, sigma=sig3, T=T)
        + bs_price_call(S, k4, r=r, q=q, sigma=sig4, T=T)
    )


def price_iron_butterfly_bs(
    S: float,
    k_put_long: float,
    k_center: float,
    k_call_long: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_put_long: float | None = None,
    sigma_put_center: float | None = None,
    sigma_call_center: float | None = None,
    sigma_call_long: float | None = None,
) -> float:
    sig_pl = sigma if sigma_put_long is None else sigma_put_long
    sig_pc = sigma if sigma_put_center is None else sigma_put_center
    sig_cc = sigma if sigma_call_center is None else sigma_call_center
    sig_cl = sigma if sigma_call_long is None else sigma_call_long
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sig_pl, T=T)
        - bs_price_put(S, k_center, r=r, q=q, sigma=sig_pc, T=T)
        - bs_price_call(S, k_center, r=r, q=q, sigma=sig_cc, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sig_cl, T=T)
    )


def price_iron_condor_bs(
    S: float,
    k_put_long: float,
    k_put_short: float,
    k_call_short: float,
    k_call_long: float,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    sigma_put_long: float | None = None,
    sigma_put_short: float | None = None,
    sigma_call_short: float | None = None,
    sigma_call_long: float | None = None,
) -> float:
    sig_pl = sigma if sigma_put_long is None else sigma_put_long
    sig_ps = sigma if sigma_put_short is None else sigma_put_short
    sig_cs = sigma if sigma_call_short is None else sigma_call_short
    sig_cl = sigma if sigma_call_long is None else sigma_call_long
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sig_pl, T=T)
        - bs_price_put(S, k_put_short, r=r, q=q, sigma=sig_ps, T=T)
        - bs_price_call(S, k_call_short, r=r, q=q, sigma=sig_cs, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sig_cl, T=T)
    )


def _build_view(payoff_fn, premium: float, s0: float, args: Iterable, breakevens: Tuple[float, ...], span: float = 0.5, n: int = 300):
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = payoff_fn(s_grid, *args)
    pnl_grid = payoff_grid - premium
    return {
        "s_grid": s_grid,
        "payoff": payoff_grid,
        "pnl": pnl_grid,
        "premium": premium,
        "breakevens": tuple(breakevens),
    }


def view_vanilla_call(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = bs_price_call(s0, strike, **kwargs)
    be = strike + premium
    return _build_view(payoff_call, premium, s0, (strike,), (be,), span, n)


def view_vanilla_put(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = bs_price_put(s0, strike, **kwargs)
    be = strike - premium
    return _build_view(payoff_put, premium, s0, (strike,), (be,), span, n)


def view_straddle(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_straddle_bs(s0, strike, **kwargs)
    be_low, be_high = strike - premium, strike + premium
    return _build_view(payoff_straddle, premium, s0, (strike,), (be_low, be_high), span, n)


def view_strangle(s0: float, k_put: float, k_call: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_strangle_bs(s0, k_put, k_call, **kwargs)
    be_low, be_high = k_put - premium, k_call + premium
    return _build_view(payoff_strangle, premium, s0, (k_put, k_call), (be_low, be_high), span, n)


def view_call_spread(s0: float, k_long: float, k_short: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_call_spread_bs(s0, k_long, k_short, **kwargs)
    be = k_long + premium
    return _build_view(payoff_call_spread, premium, s0, (k_long, k_short), (be,), span, n)


def view_put_spread(s0: float, k_long: float, k_short: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_put_spread_bs(s0, k_long, k_short, **kwargs)
    be = k_long - premium
    return _build_view(payoff_put_spread, premium, s0, (k_long, k_short), (be,), span, n)


def view_butterfly(s0: float, k1: float, k2: float, k3: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_butterfly_bs(s0, k1, k2, k3, **kwargs)
    be_low, be_high = k1 + premium, k3 - premium
    return _build_view(payoff_butterfly, premium, s0, (k1, k2, k3), (be_low, be_high), span, n)


def view_condor(s0: float, k1: float, k2: float, k3: float, k4: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_condor_bs(s0, k1, k2, k3, k4, **kwargs)
    be_low, be_high = k1 + premium, k4 - premium
    return _build_view(payoff_condor, premium, s0, (k1, k2, k3, k4), (be_low, be_high), span, n)


def view_iron_butterfly(s0: float, k_put_long: float, k_center: float, k_call_long: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_iron_butterfly_bs(s0, k_put_long, k_center, k_call_long, **kwargs)
    credit = -premium
    be_low, be_high = k_center - credit, k_center + credit
    return _build_view(payoff_iron_butterfly, premium, s0, (k_put_long, k_center, k_call_long), (be_low, be_high), span, n)


def view_iron_condor(s0: float, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_iron_condor_bs(s0, k_put_long, k_put_short, k_call_short, k_call_long, **kwargs)
    credit = -premium
    be_low, be_high = k_put_short - credit, k_call_short + credit
    return _build_view(payoff_iron_condor, premium, s0, (k_put_long, k_put_short, k_call_short, k_call_long), (be_low, be_high), span, n)


# --- Exotic / path-dependent (simplified pedagogical views) ---

def _find_breakevens_from_grid(s_grid: np.ndarray, pnl_grid: np.ndarray) -> Tuple[float, ...]:
    roots = []
    signs = np.sign(pnl_grid)
    for i in range(len(s_grid) - 1):
        if signs[i] == 0:
            roots.append(float(s_grid[i]))
        if signs[i] * signs[i + 1] < 0:
            s1, s2 = s_grid[i], s_grid[i + 1]
            p1, p2 = pnl_grid[i], pnl_grid[i + 1]
            if p2 != p1:
                s_root = s1 - p1 * (s2 - s1) / (p2 - p1)
            else:
                s_root = s1
            roots.append(float(s_root))
    return tuple(sorted(set(round(r, 6) for r in roots)))


def payoff_digital(spot, strike: float, option_type: str = "call", payout: float = 1.0):
    s = np.asarray(spot, dtype=float)
    if option_type == "put":
        return payout * (s < strike)
    return payout * (s > strike)


def price_digital_bs(S: float, K: float, T: float = DEFAULT_T, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, option_type: str = "call", payout: float = 1.0) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df_r = math.exp(-r * T)
    if option_type == "put":
        return payout * df_r * _norm_cdf(-d2)
    return payout * df_r * _norm_cdf(d2)


def view_digital(s0: float, strike: float, T: float = DEFAULT_T, **kwargs):
    premium = price_digital_bs(s0, strike, T=T, **kwargs)
    s_grid = np.linspace(s0 * 0.5, s0 * 1.5, 300)
    payoff_grid = payoff_digital(s_grid, strike, option_type=kwargs.get("option_type", "call"), payout=kwargs.get("payout", 1.0))
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_asset_or_nothing(spot, strike: float, option_type: str = "call"):
    s = np.asarray(spot, dtype=float)
    return s * (s > strike) if option_type == "call" else s * (s < strike)


def price_asset_or_nothing_bs(S: float, K: float, T: float = DEFAULT_T, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, option_type: str = "call") -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if option_type == "put":
        return S * math.exp(-q * T) * _norm_cdf(-d1)
    return S * math.exp(-q * T) * _norm_cdf(d1)


def view_asset_or_nothing(s0: float, strike: float, T: float = DEFAULT_T, **kwargs):
    premium = price_asset_or_nothing_bs(s0, strike, T=T, **kwargs)
    s_grid = np.linspace(s0 * 0.5, s0 * 1.5, 300)
    payoff_grid = payoff_asset_or_nothing(s_grid, strike, option_type=kwargs.get("option_type", "call"))
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_chooser(spot, strike: float):
    s = np.asarray(spot, dtype=float)
    return np.abs(s - strike)


def price_chooser_bs(S: float, K: float, **kwargs) -> float:
    return price_straddle_bs(S, K, **kwargs)


def view_chooser(s0: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_chooser_bs(s0, strike, **kwargs)
    return _build_view(payoff_chooser, premium, s0, (strike,), (), span, n)


def payoff_quanto(spot, strike: float, option_type: str = "call", fx_rate: float = 1.0):
    s = np.asarray(spot, dtype=float)
    base = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)
    return fx_rate * base


def price_quanto_bs(S: float, K: float, fx_rate: float = 1.0, **kwargs) -> float:
    # Simplified: price vanilla then convert with fixed FX rate
    if kwargs.get("option_type", "call") == "put":
        return fx_rate * bs_price_put(S, K, **{k: v for k, v in kwargs.items() if k in {"r", "q", "sigma", "T"}})
    return fx_rate * bs_price_call(S, K, **{k: v for k, v in kwargs.items() if k in {"r", "q", "sigma", "T"}})


def view_quanto(s0: float, strike: float, span: float = 0.5, n: int = 300, fx_rate: float = 1.0, **kwargs):
    premium = price_quanto_bs(s0, strike, fx_rate=fx_rate, **kwargs)
    payoff_fn = lambda s, k: payoff_quanto(s, k, option_type=kwargs.get("option_type", "call"), fx_rate=fx_rate)
    return _build_view(payoff_fn, premium, s0, (strike,), (), span, n)


def payoff_rainbow(spot1, spot2, strike: float, option_type: str = "call"):
    s1 = np.asarray(spot1, dtype=float)
    s2 = np.asarray(spot2, dtype=float)
    best = np.maximum(s1, s2)
    if option_type == "put":
        return np.maximum(strike - best, 0.0)
    return np.maximum(best - strike, 0.0)


def price_rainbow_mc(
    S1: float,
    S2: float,
    K: float,
    T: float = DEFAULT_T,
    sigma1: float = DEFAULT_SIGMA,
    sigma2: float = DEFAULT_SIGMA,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    rho: float = 0.0,
    n_paths: int = 20_000,
    option_type: str = "call",
) -> float:
    """
    Monte Carlo pricing for a 2-asset rainbow (max(S1,S2) - K)^+ or (K - max)^+.
    Correlation rho is applied between the two asset Brownian motions.
    """
    dt = T
    sqrt_dt = math.sqrt(dt)
    z1 = np.random.randn(n_paths)
    z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho**2)) * np.random.randn(n_paths)
    s1_T = S1 * np.exp((r - q - 0.5 * sigma1**2) * dt + sigma1 * sqrt_dt * z1)
    s2_T = S2 * np.exp((r - q - 0.5 * sigma2**2) * dt + sigma2 * sqrt_dt * z2)
    best = np.maximum(s1_T, s2_T)
    if option_type == "put":
        payoff = np.maximum(K - best, 0.0)
    else:
        payoff = np.maximum(best - K, 0.0)
    return float(math.exp(-r * T) * np.mean(payoff))


def view_rainbow(s0: float, s0b: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    s_grid_b = np.linspace(s0b * (1.0 - span), s0b * (1.0 + span), n)
    payoff_grid = payoff_rainbow(s_grid, s_grid_b, strike, option_type=kwargs.get("option_type", "call"))
    opt_type = kwargs.get("option_type", "call")
    sigma_a = kwargs.get("sigma", DEFAULT_SIGMA)
    sigma_b = kwargs.get("sigma_b", sigma_a)
    premium = price_rainbow_mc(
        s0,
        s0b,
        strike,
        T=kwargs.get("T", DEFAULT_T),
        sigma1=sigma_a,
        sigma2=sigma_b,
        r=kwargs.get("r", DEFAULT_R),
        q=kwargs.get("q", DEFAULT_Q),
        rho=kwargs.get("rho", 0.0),
        n_paths=int(kwargs.get("n_paths", 20_000)),
        option_type=opt_type,
    )
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_asian_arith(avg_price: float, strike: float, option_type: str = "call"):
    if option_type == "put":
        return max(strike - avg_price, 0.0)
    return max(avg_price - strike, 0.0)


def price_asian_arith_approx(S: float, K: float, T: float = DEFAULT_T, sigma: float = DEFAULT_SIGMA, r: float = DEFAULT_R, q: float = DEFAULT_Q, option_type: str = "call") -> float:
    # Turnbull-Wakeman-esque approximation using adjusted sigma and strike
    sigma_adj = sigma / math.sqrt(3.0)
    if option_type == "put":
        return bs_price_put(S, K, r=r, q=q, sigma=sigma_adj, T=T)
    return bs_price_call(S, K, r=r, q=q, sigma=sigma_adj, T=T)


def view_asian_arith(s0: float, strike: float, avg_price: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_asian_arith_approx(s0, strike, **kwargs)
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = np.array([payoff_asian_arith(avg_price * (s / s0), strike, option_type=kwargs.get("option_type", "call")) for s in s_grid])
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_asian_geom(avg_price: float, strike: float, option_type: str = "call"):
    if option_type == "put":
        return max(strike - avg_price, 0.0)
    return max(avg_price - strike, 0.0)


def price_asian_geom(S: float, K: float, T: float = DEFAULT_T, sigma: float = DEFAULT_SIGMA, r: float = DEFAULT_R, q: float = DEFAULT_Q, option_type: str = "call") -> float:
    sigma_g = sigma / math.sqrt(3.0)
    if option_type == "put":
        return bs_price_put(S, K, r=r, q=q, sigma=sigma_g, T=T)
    return bs_price_call(S, K, r=r, q=q, sigma=sigma_g, T=T)


def view_asian_geom(s0: float, strike: float, avg_price: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_asian_geom(s0, strike, **kwargs)
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = np.array([payoff_asian_geom(avg_price * (s / s0), strike, option_type=kwargs.get("option_type", "call")) for s in s_grid])
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_lookback_floating(spot, min_path: float, max_path: float, option_type: str = "call"):
    s = np.asarray(spot, dtype=float)
    if option_type == "put":
        return np.maximum(max_path - s, 0.0)
    return np.maximum(s - min_path, 0.0)


def view_lookback(spot_ref: float, min_path: float, max_path: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = float(payoff_lookback_floating(spot_ref, min_path, max_path, option_type=kwargs.get("option_type", "call")))
    s_grid = np.linspace(spot_ref * (1.0 - span), spot_ref * (1.0 + span), n)
    payoff_grid = payoff_lookback_floating(s_grid, min_path, max_path, option_type=kwargs.get("option_type", "call"))
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_lookback_fixed(min_path: float, max_path: float, strike: float, option_type: str = "call"):
    if option_type == "put":
        return max(strike - min_path, 0.0)
    return max(max_path - strike, 0.0)


def view_lookback_fixed(spot_ref: float, min_path: float, max_path: float, strike: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = float(payoff_lookback_fixed(min_path, max_path, strike, option_type=kwargs.get("option_type", "call")))
    s_grid = np.linspace(spot_ref * (1.0 - span), spot_ref * (1.0 + span), n)
    payoff_grid = np.full_like(s_grid, premium)
    pnl_grid = payoff_grid - premium
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": ()}


def payoff_forward_start(spot_T, spot_start: float, m: float = 1.0, option_type: str = "call"):
    k_eff = m * spot_start
    if option_type == "put":
        return max(k_eff - spot_T, 0.0)
    return max(spot_T - k_eff, 0.0)


def view_forward_start(s0: float, spot_start: float, m: float = 1.0, span: float = 0.5, n: int = 300, **kwargs):
    k_eff = m * spot_start
    opt_type = kwargs.get("option_type", "call")
    premium = bs_price_call(s0, k_eff, r=kwargs.get("r", DEFAULT_R), q=kwargs.get("q", DEFAULT_Q), sigma=kwargs.get("sigma", DEFAULT_SIGMA), T=kwargs.get("T", DEFAULT_T)) if opt_type != "put" else bs_price_put(s0, k_eff, r=kwargs.get("r", DEFAULT_R), q=kwargs.get("q", DEFAULT_Q), sigma=kwargs.get("sigma", DEFAULT_SIGMA), T=kwargs.get("T", DEFAULT_T))
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = np.array([payoff_forward_start(s, spot_start, m=m, option_type=kwargs.get("option_type", "call")) for s in s_grid])
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_cliquet(spot_T, spot_0: float, floor: float = 0.0, cap: float = 0.1):
    ret = (spot_T - spot_0) / spot_0 if spot_0 else 0.0
    capped = min(max(ret, floor), cap)
    return max(capped, 0.0)


def view_cliquet(
    s0: float,
    floor: float = 0.0,
    cap: float = 0.1,
    span: float = 0.5,
    n: int = 300,
    T: float = DEFAULT_T,
    r: float = DEFAULT_R,
    q: float = DEFAULT_Q,
    sigma: float = DEFAULT_SIGMA,
    n_periods: int = 12,
    n_paths: int = 4000,
    seed: int = 42,
    k_ref: float | None = None,
):
    def _cliquet_mc(
        S0: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        n_periods: int,
        cap: float,
        floor: float,
        n_paths: int,
        seed: int,
    ) -> float:
        if n_periods <= 0 or n_paths <= 0 or T <= 0 or sigma <= 0 or S0 <= 0:
            return 0.0
        rng = np.random.default_rng(seed)
        dt = T / n_periods
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * math.sqrt(dt)
        disc = math.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            s = S0
            coupons = []
            for _ in range(n_periods):
                z = rng.normal()
                s_next = s * math.exp(drift + diff * z)
                ret = (s_next / s) - 1.0
                coupons.append(np.clip(ret, floor, cap))
                s = s_next
            payoffs.append(sum(coupons))
        return float(disc * np.mean(payoffs))

    base = float(k_ref) if k_ref is not None else float(s0)
    premium = _cliquet_mc(base, r, q, sigma, T, n_periods, cap, floor, n_paths, seed)
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = np.array([payoff_cliquet(s, base, floor=floor, cap=cap) for s in s_grid])
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def payoff_barrier(spot, strike: float, barrier: float, option_type: str = "call", direction: str = "up", knock: str = "out", payout: float = 1.0, binary: bool = False):
    s = np.asarray(spot, dtype=float)
    hit = (s >= barrier) if direction == "up" else (s <= barrier)

    # Determine where payoff is active depending on knock in/out
    active_mask = hit if knock == "in" else ~hit

    if binary:
        return payout * np.where(active_mask, 1.0, 0.0)

    base = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)
    return np.where(active_mask, base, 0.0)


def view_barrier(
    s0: float,
    strike: float,
    barrier: float,
    direction: str = "up",
    knock: str = "out",
    option_type: str = "call",
    payout: float = 1.0,
    binary: bool = False,
    span: float = 0.5,
    n: int = 300,
    n_paths: int = 8000,
    n_steps: int = 120,
    seed: int = 42,
    **kwargs,
):
    """
    Monte Carlo pricing for a simple barrier (in/out, up/down, vanilla or binary).
    Displays deterministic payoff curves while using MC for the premium to avoid 0 pricing.
    """
    def price_barrier_mc(
        S0: float,
        K: float,
        B: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str,
        direction: str,
        knock: str,
        payout: float,
        binary: bool,
        n_paths: int,
        n_steps: int,
        seed: int,
    ) -> float:
        if S0 <= 0 or K <= 0 or B <= 0 or T <= 0 or sigma <= 0 or n_paths <= 0 or n_steps <= 0:
            # Fallback: intrinsic masked by barrier condition at S0
            return float(
                np.asarray(
                    payoff_barrier(S0, K, B, option_type=option_type, direction=direction, knock=knock, payout=payout, binary=binary)
                ).item()
            )
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * math.sqrt(dt)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(n_paths, n_steps))
        log_paths = math.log(S0) + np.cumsum(drift + diff * Z, axis=1)
        paths = np.exp(log_paths)
        # include S0 for barrier monitoring
        paths = np.concatenate([np.full((n_paths, 1), S0), paths], axis=1)
        hit = np.max(paths, axis=1) >= B if direction == "up" else np.min(paths, axis=1) <= B
        ST = paths[:, -1]
        if binary:
            payoff = payout * np.where(hit if knock == "in" else ~hit, 1.0, 0.0)
        else:
            base = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
            payoff = base * np.where(hit if knock == "in" else ~hit, 1.0, 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = payoff_barrier(s_grid, strike, barrier, option_type=option_type, direction=direction, knock=knock, payout=payout, binary=binary)
    premium = price_barrier_mc(
        S0=s0,
        K=strike,
        B=barrier,
        T=float(kwargs.get("T", 1.0) or 1.0),
        r=float(kwargs.get("r", DEFAULT_R) or DEFAULT_R),
        q=float(kwargs.get("q", DEFAULT_Q) or DEFAULT_Q),
        sigma=float(kwargs.get("sigma", DEFAULT_SIGMA) or DEFAULT_SIGMA),
        option_type=option_type,
        direction=direction,
        knock=knock,
        payout=payout,
        binary=binary,
        n_paths=int(n_paths),
        n_steps=int(n_steps),
        seed=int(seed),
    )
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def price_calendar_spread_bs(S: float, K: float, T_short: float, T_long: float, option_type: str = "call", r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA) -> float:
    if option_type == "put":
        return bs_price_put(S, K, r=r, q=q, sigma=sigma, T=T_long) - bs_price_put(S, K, r=r, q=q, sigma=sigma, T=T_short)
    return bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T_long) - bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T_short)


def view_calendar_spread(s0: float, strike: float, T_short: float, T_long: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_calendar_spread_bs(s0, strike, T_short, T_long, **kwargs)
    def _payoff(s, k):
        base = payoff_call if kwargs.get("option_type", "call") != "put" else payoff_put
        return base(s, k) - base(s, k)
    # payoff at maturity is driven by long leg only (illustration), short leg assumed expired
    payoff_grid = (payoff_call if kwargs.get("option_type", "call") != "put" else payoff_put)(np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n), strike)
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n), pnl_grid)
    return {"s_grid": np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n), "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}


def price_diagonal_spread_bs(S: float, k_near: float, k_far: float, T_near: float, T_far: float, option_type: str = "call", r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA) -> float:
    if option_type == "put":
        return bs_price_put(S, k_far, r=r, q=q, sigma=sigma, T=T_far) - bs_price_put(S, k_near, r=r, q=q, sigma=sigma, T=T_near)
    return bs_price_call(S, k_far, r=r, q=q, sigma=sigma, T=T_far) - bs_price_call(S, k_near, r=r, q=q, sigma=sigma, T=T_near)


def view_diagonal_spread(s0: float, k_near: float, k_far: float, T_near: float, T_far: float, span: float = 0.5, n: int = 300, **kwargs):
    premium = price_diagonal_spread_bs(s0, k_near, k_far, T_near, T_far, **kwargs)
    base = payoff_call if kwargs.get("option_type", "call") != "put" else payoff_put
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = base(s_grid, k_far)  # near leg expired, keep long far leg payoff
    pnl_grid = payoff_grid - premium
    bes = _find_breakevens_from_grid(s_grid, pnl_grid)
    return {"s_grid": s_grid, "payoff": payoff_grid, "pnl": pnl_grid, "premium": premium, "breakevens": bes}
