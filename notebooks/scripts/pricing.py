import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

DEFAULT_R = 0.02
DEFAULT_Q = 0.0
DEFAULT_SIGMA = 0.2
DEFAULT_T = 1.0

_CACHE_SPY_CLOSE = Path("notebooks/GPT/_cache_spy_close.csv")


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
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(cache_path, index_label="date")
    except Exception:
        pass
    return close


def last_spy_close(period: str = "1y", interval: str = "1d") -> float:
    close = fetch_spy_history(period=period, interval=interval)
    return float(close.iloc[-1])


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


def price_straddle_bs(S: float, strike: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, strike, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, strike, r=r, q=q, sigma=sigma, T=T)


def price_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, k_put, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, k_call, r=r, q=q, sigma=sigma, T=T)


def pricing_strangle_bs(S: float, k_put: float, k_call: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    """Alias explicite pour le pricing d'un strangle via Black-Scholes (somme put+call)."""
    return price_strangle_bs(S, k_put, k_call, r=r, q=q, sigma=sigma, T=T)


def price_call_spread_bs(S: float, k_long: float, k_short: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_call(S, k_long, r=r, q=q, sigma=sigma, T=T) - bs_price_call(S, k_short, r=r, q=q, sigma=sigma, T=T)


def price_put_spread_bs(S: float, k_long: float, k_short: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_put(S, k_long, r=r, q=q, sigma=sigma, T=T) - bs_price_put(S, k_short, r=r, q=q, sigma=sigma, T=T)


def price_butterfly_bs(S: float, k1: float, k2: float, k3: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return bs_price_call(S, k1, r=r, q=q, sigma=sigma, T=T) - 2.0 * bs_price_call(S, k2, r=r, q=q, sigma=sigma, T=T) + bs_price_call(S, k3, r=r, q=q, sigma=sigma, T=T)


def price_condor_bs(S: float, k1: float, k2: float, k3: float, k4: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_call(S, k1, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k2, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k3, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k4, r=r, q=q, sigma=sigma, T=T)
    )


def price_iron_butterfly_bs(S: float, k_put_long: float, k_center: float, k_call_long: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sigma, T=T)
        - bs_price_put(S, k_center, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k_center, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sigma, T=T)
    )


def price_iron_condor_bs(S: float, k_put_long: float, k_put_short: float, k_call_short: float, k_call_long: float, r: float = DEFAULT_R, q: float = DEFAULT_Q, sigma: float = DEFAULT_SIGMA, T: float = DEFAULT_T) -> float:
    return (
        bs_price_put(S, k_put_long, r=r, q=q, sigma=sigma, T=T)
        - bs_price_put(S, k_put_short, r=r, q=q, sigma=sigma, T=T)
        - bs_price_call(S, k_call_short, r=r, q=q, sigma=sigma, T=T)
        + bs_price_call(S, k_call_long, r=r, q=q, sigma=sigma, T=T)
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


# --- Asian option functions ---

DEFAULT_N_OBS = 252
DEFAULT_N_PATHS = 100000


def _asian_geometric_closed_form(spot: float, strike: float, r: float, sigma: float, T: float, n_obs: int, option_type: str) -> float:
    """Closed-form price for Asian geometric option."""
    if n_obs < 1:
        return 0.0
    nu = r - 0.5 * sigma**2
    sigma_g_sq = (sigma**2) * (n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2)
    sigma_g = math.sqrt(sigma_g_sq)
    mu_g = (nu * (n_obs + 1) / (2 * n_obs) + 0.5 * sigma_g_sq) * T
    d1 = (math.log(spot / strike) + mu_g + 0.5 * sigma_g_sq * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)
    df = math.exp(-r * T)
    if option_type == "call":
        return df * (spot * math.exp(mu_g) * _norm_cdf(d1) - strike * _norm_cdf(d2))
    else:
        return df * (strike * _norm_cdf(-d2) - spot * math.exp(mu_g) * _norm_cdf(-d1))


def price_asian_arith_mc(
    S: float,
    K: float,
    r: float = DEFAULT_R,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    n_obs: int = DEFAULT_N_OBS,
    n_paths: int = DEFAULT_N_PATHS,
    option_type: str = "call",
    seed: int | None = None,
) -> float:
    """Monte Carlo price for Asian arithmetic option with control variate."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_obs
    drift = (r - 0.5 * sigma**2) * dt
    vol_step = sigma * math.sqrt(dt)

    n_base = max(1, n_paths // 2)
    z_base = np.random.randn(n_obs, n_base)
    z = np.concatenate([z_base, -z_base], axis=1)
    n_eff = z.shape[1]

    log_s = math.log(S) + np.cumsum(drift + vol_step * z, axis=0)
    s_paths = np.exp(log_s)

    arith_mean = s_paths.mean(axis=0)
    geom_mean = np.exp(np.log(s_paths).mean(axis=0))
    if option_type == "call":
        arith_payoff = np.maximum(arith_mean - K, 0.0)
        geom_payoff = np.maximum(geom_mean - K, 0.0)
    else:
        arith_payoff = np.maximum(K - arith_mean, 0.0)
        geom_payoff = np.maximum(K - geom_mean, 0.0)
    closed_geom = _asian_geometric_closed_form(S, K, r, sigma, T, n_obs, option_type)
    cov = np.cov(arith_payoff, geom_payoff)[0, 1]
    var_geom = np.var(geom_payoff)
    c = cov / var_geom if var_geom > 1e-10 else 0.0
    control_estimator = arith_payoff - c * (geom_payoff - closed_geom)
    disc = math.exp(-r * T)
    disc_payoff = disc * control_estimator
    return float(np.mean(disc_payoff))


def payoff_asian_arith(avg_price, strike: float, option_type: str = "call"):
    """
    Payoff of an Asian arithmetic option given the average price.

    Note: For Asian options, the payoff depends on the average of the underlying
    price over the observation period, not the final spot price. The avg_price
    parameter represents this average. For visualization purposes (view_asian_arith),
    we treat avg_price as a grid of hypothetical average prices to show the
    payoff profile.
    """
    avg = np.asarray(avg_price, dtype=float)
    if option_type == "call":
        return np.maximum(avg - strike, 0.0)
    else:
        return np.maximum(strike - avg, 0.0)


def view_asian_arith(
    s0: float,
    strike: float,
    option_type: str = "call",
    span: float = 0.5,
    n: int = 300,
    r: float = DEFAULT_R,
    sigma: float = DEFAULT_SIGMA,
    T: float = DEFAULT_T,
    n_obs: int = DEFAULT_N_OBS,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int | None = 42,
) -> dict:
    """
    Build view data for Asian arithmetic option payoff visualization.

    Parameters
    ----------
    s0 : float
        Current spot price.
    strike : float
        Strike price of the option.
    option_type : str
        'call' or 'put'.
    span : float
        Fraction of s0 to span for the grid (e.g., 0.5 means grid goes from s0*0.5 to s0*1.5).
    n : int
        Number of points in the grid.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.
    n_obs : int
        Number of averaging observations.
    n_paths : int
        Number of Monte Carlo paths for pricing.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: s_grid, payoff, pnl, premium, breakevens
    """
    premium = price_asian_arith_mc(
        S=s0,
        K=strike,
        r=r,
        sigma=sigma,
        T=T,
        n_obs=n_obs,
        n_paths=n_paths,
        option_type=option_type,
        seed=seed,
    )

    # s_grid represents hypothetical average prices for visualization.
    # For Asian options, the payoff depends on the average price over the
    # observation period, not the final spot price.
    s_grid = np.linspace(s0 * (1.0 - span), s0 * (1.0 + span), n)
    payoff_grid = payoff_asian_arith(s_grid, strike, option_type)
    pnl_grid = payoff_grid - premium

    if option_type == "call":
        be = strike + premium
    else:
        be = strike - premium

    return {
        "s_grid": s_grid,
        "payoff": payoff_grid,
        "pnl": pnl_grid,
        "premium": premium,
        "breakevens": (be,),
    }
