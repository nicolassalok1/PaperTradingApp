from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq


def bs_call_price(S0, K, t, r, q, vol):
    if t <= 0 or vol <= 0 or S0 <= 0 or K <= 0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    return S0 * math.exp(-q * t) * _norm_cdf(d1) - K * math.exp(-r * t) * _norm_cdf(d2)


def implied_vol_call(price, S0, K, t, r, q, vol_min=1e-4, vol_max=5.0):
    """
    Simple implied vol inverter for calls using Brent.
    Returns np.nan if price is outside arbitrage bounds.
    """
    if t <= 0 or S0 <= 0 or K <= 0:
        return np.nan
    forward = S0 * math.exp((r - q) * t)
    intrinsic = max(S0 * math.exp(-q * t) - K * math.exp(-r * t), 0.0)
    upper_bound = S0 * math.exp(-q * t)
    if price < intrinsic - 1e-8 or price > upper_bound + 1e-8:
        return np.nan

    def f(vol):
        return bs_call_price(S0, K, t, r, q, vol) - price

    try:
        return float(brentq(f, vol_min, vol_max, maxiter=100, xtol=1e-6))
    except Exception:
        return np.nan


def implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q):
    iv = np.zeros_like(price_grid, dtype=float)
    for i_t, t in enumerate(t_grid):
        for j_m, m in enumerate(m_grid):
            K = float(m * S0)
            price = price_grid[i_t, j_m]
            iv[i_t, j_m] = implied_vol_call(price, S0, K, t, r, q)
    return iv


__all__ = ["bs_call_price", "implied_vol_call", "implied_vol_grid"]
