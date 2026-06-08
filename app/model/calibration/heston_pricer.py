from __future__ import annotations

import cmath
import numpy as np


def heston_cf(u, S0, r, q, t, kappa, theta, sigma, rho, v0):
    """
    Heston characteristic function φ(u).
    """
    u = np.asarray(u, dtype=complex)
    rho = np.clip(rho, -0.999, 0.999)
    sigma = max(float(sigma), 1e-8)
    t = float(t)
    kappa = float(kappa)
    theta = float(theta)
    v0 = float(v0)

    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma * sigma * (iu + u * u))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d + 1e-16)
    exp_dt = np.exp(-d * t)
    one_minus_g_exp = 1 - g * exp_dt
    one_minus_g = 1 - g
    # Avoid division by zero
    one_minus_g_exp = np.where(one_minus_g_exp == 0, 1e-16, one_minus_g_exp)
    one_minus_g = np.where(one_minus_g == 0, 1e-16, one_minus_g)

    C = (r - q) * iu * t + (kappa * theta / (sigma * sigma)) * (
        (kappa - rho * sigma * iu - d) * t - 2.0 * np.log(one_minus_g_exp / one_minus_g)
    )
    D = ((kappa - rho * sigma * iu - d) / (sigma * sigma)) * (
        (1 - exp_dt) / one_minus_g_exp
    )
    return np.exp(C + D * v0 + iu * np.log(S0))


def _P(j, S0, K, t, r, q, params, u_max=50.0, N=2000):
    """
    Probability P1 (j=1) or P2 (j=2) using Heston integral.
    """
    kappa, theta, sigma, rho, v0 = params
    du = u_max / N
    u = np.arange(1, N + 1) * du
    iu = 1j * u
    if j == 1:
        phi = heston_cf(u - 1j, S0, r, q, t, kappa, theta, sigma, rho, v0)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        # P1 normaliser is phi(-i) = forward = S0 * exp((r-q)t), NOT S0*exp(-q t).
        # The missing exp(r t) factor biased P1 -> broke BS-limit convergence and
        # produced negative prices for short-maturity OTM. (PLAN.md step-4 bugs A/B)
        denom = 1j * u * S0 * np.exp((r - q) * t)
    else:
        phi = heston_cf(u, S0, r, q, t, kappa, theta, sigma, rho, v0)
        numerator = np.exp(-1j * u * np.log(K)) * phi
        denom = 1j * u
    integrand = (numerator / denom).real
    return 0.5 + (du / np.pi) * integrand.sum()


def call_price_cf(S0, K, t, r, q, params, u_max=50.0, N=2000):
    """
    Semi-closed form Heston call via Carr-Madan style integration.
    """
    if t <= 0 or S0 <= 0 or K <= 0:
        return 0.0
    kappa, theta, sigma, rho, v0 = params
    p1 = _P(1, S0, K, t, r, q, params, u_max=u_max, N=N)
    p2 = _P(2, S0, K, t, r, q, params, u_max=u_max, N=N)
    price = float(S0 * np.exp(-q * t) * p1 - K * np.exp(-r * t) * p2)
    # No-arbitrage clamp. A European call must lie in [max(F_S - F_K, 0), F_S],
    # F_S = S0 e^{-q t}, F_K = K e^{-r t}. The Heston P-integrand is hard for very
    # short-maturity deep-OTM cells, where finite integration can overshoot a few
    # bp negative (true value ~0); flooring at the no-arb bound guarantees a valid,
    # arbitrage-free price instead of a spurious negative.
    fwd_s = S0 * np.exp(-q * t)
    lower = max(fwd_s - K * np.exp(-r * t), 0.0)
    return float(min(max(price, lower), fwd_s))


def price_grid_from_params(S0, m_grid, t_grid, r, q, params):
    """
    Compute call prices on a fixed grid of moneyness x maturity.
    """
    prices = np.zeros((len(t_grid), len(m_grid)), dtype=float)
    for i_t, t in enumerate(t_grid):
        for j_m, m in enumerate(m_grid):
            K = float(m * S0)
            prices[i_t, j_m] = call_price_cf(S0, K, t, r, q, params)
    return prices


__all__ = ["heston_cf", "call_price_cf", "price_grid_from_params"]
