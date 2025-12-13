from __future__ import annotations

import numpy as np


def hagan_black_iv(
    *,
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Hagan et al. (2002) SABR approximation for Black implied volatility.
    """
    F = float(F)
    K = float(K)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(np.clip(rho, -0.999, 0.999))
    nu = float(max(nu, 1e-12))
    alpha = float(max(alpha, 1e-12))

    if T <= 0 or F <= 0 or K <= 0:
        return float("nan")

    one_minus_beta = 1.0 - beta
    log_fk = float(np.log(F / K))

    # ATM handling
    if abs(log_fk) < 1e-10:
        f_beta = F ** one_minus_beta
        term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (f_beta * f_beta)
        term2 = (rho * beta * nu * alpha) / (4.0 * f_beta)
        term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0
        return float((alpha / f_beta) * (1.0 + (term1 + term2 + term3) * T))

    fk_beta = (F * K) ** (0.5 * one_minus_beta)
    z = (nu / alpha) * fk_beta * log_fk
    # x(z)
    sqrt_term = np.sqrt(max(1e-18, 1.0 - 2.0 * rho * z + z * z))
    xz_num = sqrt_term + z - rho
    xz_den = 1.0 - rho
    xz = np.log(max(1e-18, xz_num / max(1e-18, xz_den)))

    # Log strike correction
    log_corr = 1.0 + (one_minus_beta * one_minus_beta / 24.0) * (log_fk * log_fk) + (
        (one_minus_beta**4) / 1920.0
    ) * (log_fk**4)

    term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) / (fk_beta * fk_beta)
    term2 = (rho * beta * nu * alpha) / (4.0 * fk_beta)
    term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0

    z_over_xz = float(z / xz) if abs(xz) > 1e-14 else 1.0
    iv = (alpha / (fk_beta * log_corr)) * z_over_xz * (1.0 + (term1 + term2 + term3) * T)
    return float(iv)


def hagan_black_iv_vectorized(
    *,
    F: float,
    K: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    out = np.empty_like(K, dtype=float)
    for i, k in enumerate(K):
        out[i] = hagan_black_iv(F=F, K=float(k), T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
    return out


__all__ = ["hagan_black_iv", "hagan_black_iv_vectorized"]

