from __future__ import annotations

import numpy as np


def heston_log_return_cf(
    u: np.ndarray,
    T: float,
    *,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    drift_adjust: float = 0.0,
) -> np.ndarray:
    """
    Characteristic function of log-return X_T = ln(S_T/S0) under Heston.

    drift_adjust lets higher-level models (e.g. Bates) shift (r-q) -> (r-q+drift_adjust).
    """
    u = np.asarray(u, dtype=complex)
    T = float(T)
    if T <= 0:
        return np.ones_like(u, dtype=complex)

    kappa = float(kappa)
    theta = float(theta)
    sigma = float(max(sigma, 1e-12))
    rho = float(np.clip(rho, -0.999, 0.999))
    v0 = float(max(v0, 1e-12))

    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma * sigma * (iu + u * u))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d + 1e-16)
    exp_dt = np.exp(-d * T)
    one_minus_g_exp = 1.0 - g * exp_dt
    one_minus_g = 1.0 - g
    one_minus_g_exp = np.where(np.abs(one_minus_g_exp) < 1e-16, 1e-16, one_minus_g_exp)
    one_minus_g = np.where(np.abs(one_minus_g) < 1e-16, 1e-16, one_minus_g)

    drift = (float(r) - float(q) + float(drift_adjust)) * T
    C = drift * iu + (kappa * theta / (sigma * sigma)) * (
        (kappa - rho * sigma * iu - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g)
    )
    D = ((kappa - rho * sigma * iu - d) / (sigma * sigma)) * ((1.0 - exp_dt) / one_minus_g_exp)
    return np.exp(C + D * v0)


__all__ = ["heston_log_return_cf"]

