import numpy as np
from numpy import exp, log
from scipy.integrate import quad

"""
Module de pricing Heston (version compacte et propre).

Objectif :
  - Fournir un pricer européen de base sous Heston
  - Exposer une CF propre
  - Donner des hooks pour la calibration NN (params dict)

Paramètres Heston :
  kappa : vitesse de rappel
  theta : variance de long terme
  sigma : vol de la variance
  rho   : corrélation Brownien prix / variance
  v0    : variance initiale
"""


def heston_cf(u, T, kappa, theta, sigma, rho, v0, r=0.0, q=0.0):
    """
    Characteristic function de log(S_T) sous Heston.
    u : variable complexe
    T : maturité
    r : taux sans risque
    q : dividende continu
    """
    if T <= 0:
        return 1.0 + 0.0j

    alpha = -0.5 * (u * u + 1j * u)
    beta = kappa - rho * sigma * 1j * u
    gamma = 0.5 * sigma * sigma

    d = np.sqrt(beta * beta - 4.0 * alpha * gamma)
    g = (beta - d) / (beta + d)

    # C(T) et D(T) standard Heston
    exp_dt = np.exp(-d * T)
    one_minus_gexp = 1.0 - g * exp_dt
    one_minus_g = 1.0 - g

    C = 1j * u * (r - q) * T + kappa * theta / (sigma * sigma) * (
        (beta - d) * T - 2.0 * np.log(one_minus_gexp / one_minus_g)
    )
    D = (beta - d) / (sigma * sigma) * ((1.0 - exp_dt) / one_minus_gexp)

    return np.exp(C + D * v0)


def price_heston_call(S0, K, T, r, q, kappa, theta, sigma, rho, v0):
    """
    Pricing call européen sous Heston, via intégrale type Carr-Madan simplifiée.
    Approche compacte (pas optimisée production).
    """
    if T <= 0:
        return max(S0 - K, 0.0)

    # Intégrande inspiré de la formulation CF standard
    def integrand(u):
        u = u + 0.0j
        cf_val = heston_cf(u - 1j, T, kappa, theta, sigma, rho, v0, r, q)
        numer = exp(-1j * u * log(K)) * cf_val
        denom = 1j * u * S0 * exp(-q * T)
        return (numer / denom).real

    val, _ = quad(lambda uu: integrand(uu), 0.0, 200.0, limit=200)
    call_price = S0 * exp(-q * T) - (np.sqrt(K) / np.pi) * val
    return float(call_price)


def heston_delta(S0, K, T, r, q, params, eps=1e-4):
    """
    Delta via différentiation numérique.
    params : dict contenant kappa, theta, sigma, rho, v0
    """
    p_up = price_heston_call(S0 * (1 + eps), K, T, r, q, **params)
    p_dn = price_heston_call(S0 * (1 - eps), K, T, r, q, **params)
    return (p_up - p_dn) / (2 * S0 * eps)


def heston_vega(S0, K, T, r, q, params, eps=1e-4):
    """
    Vega numérique par perturbation de sigma.
    """
    params_up = params.copy()
    params_up["sigma"] += eps
    params_dn = params.copy()
    params_dn["sigma"] -= eps
    p_up = price_heston_call(S0, K, T, r, q, **params_up)
    p_dn = price_heston_call(S0, K, T, r, q, **params_dn)
    return (p_up - p_dn) / (2 * eps)


__all__ = [
    "heston_cf",
    "price_heston_call",
    "heston_delta",
    "heston_vega",
]
