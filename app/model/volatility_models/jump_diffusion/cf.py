from __future__ import annotations

import numpy as np


def merton_log_return_cf(
    u: np.ndarray,
    T: float,
    *,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    muj: float,
    sigj: float,
) -> np.ndarray:
    """
    Characteristic function of log-return X_T = ln(S_T/S0) under Merton (1976).
    """
    u = np.asarray(u, dtype=complex)
    T = float(T)
    sigma = float(max(sigma, 1e-12))
    lam = float(max(lam, 0.0))
    muj = float(muj)
    sigj = float(max(sigj, 1e-12))

    kappa = np.exp(muj + 0.5 * sigj * sigj) - 1.0
    drift = (float(r) - float(q) - 0.5 * sigma * sigma - lam * kappa) * T

    jump_cf = np.exp(1j * u * muj - 0.5 * sigj * sigj * u * u)
    exponent = 1j * u * drift - 0.5 * sigma * sigma * u * u * T + lam * T * (jump_cf - 1.0)
    return np.exp(exponent)


__all__ = ["merton_log_return_cf"]


def bates_log_return_cf(
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
    lam: float,
    muj: float,
    sigj: float,
) -> np.ndarray:
    """
    Bates (1996): Heston stochastic volatility + Merton lognormal jumps.
    """
    from app.model.volatility_models.heston.cf import heston_log_return_cf

    u = np.asarray(u, dtype=complex)
    T = float(T)
    lam = float(max(lam, 0.0))
    muj = float(muj)
    sigj = float(max(sigj, 1e-12))

    kappa_j = np.exp(muj + 0.5 * sigj * sigj) - 1.0
    drift_adjust = -lam * kappa_j

    jump_cf = np.exp(1j * u * muj - 0.5 * sigj * sigj * u * u)
    jump_part = np.exp(lam * T * (jump_cf - 1.0))
    heston_part = heston_log_return_cf(
        u,
        T,
        r=r,
        q=q,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        v0=v0,
        drift_adjust=drift_adjust,
    )
    return heston_part * jump_part


__all__ = ["merton_log_return_cf", "bates_log_return_cf"]
