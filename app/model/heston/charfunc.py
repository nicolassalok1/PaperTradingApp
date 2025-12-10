from __future__ import annotations

import cmath

from app.model.heston.params import HestonParams


def heston_charfunc(u: complex, T: float, S0: float, params: HestonParams) -> complex:
    """
    Characteristic function of log-price under the Heston model.

    Args:
        u: complex integration variable
        T: maturity
        params: HestonParams
    Returns:
        complex value of the characteristic function.
    """
    kappa, theta, sigma, rho, v0, r, q = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
        params.r,
        params.q,
    )
    iu = 1j * u
    d = cmath.sqrt((rho * sigma * iu - kappa) ** 2 + sigma * sigma * (iu + u * u))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_dt = cmath.exp(-d * T)

    C = iu * (cmath.log(S0) + (r - q) * T) + (kappa * theta) / (sigma * sigma) * (
        (kappa - rho * sigma * iu - d) * T - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((kappa - rho * sigma * iu - d) / (sigma * sigma)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return cmath.exp(C + D * v0)


__all__ = ["heston_charfunc"]
