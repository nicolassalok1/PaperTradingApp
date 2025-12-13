from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


LogReturnCF = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class FFTConfig:
    alpha: float = 1.5
    n: int = 2048
    eta: float = 0.25


def _simpson_weights(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(n, dtype=float)
    w = np.ones(n, dtype=float)
    # Simpson weights: 1, 4, 2, 4, ..., 2, 4, 1 (scaled by 1/3)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w / 3.0


def carr_madan_fft_call_prices(
    *,
    S0: float,
    r: float,
    q: float,
    T: float,
    cf_log_return: LogReturnCF,
    cfg: FFTConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan (1999) FFT for European call prices across a strike grid.

    Computes call prices C(K) for strikes K_m = S0 * exp(k_m), where
    k_m is log-moneyness and the grid is centered around 0.

    Returns (K_grid, C_grid) with length N.
    """
    if T <= 0 or S0 <= 0:
        return np.array([]), np.array([])

    cfg = cfg or FFTConfig()
    alpha = float(cfg.alpha)
    n = int(cfg.n)
    eta = float(cfg.eta)
    if n <= 1 or eta <= 0:
        return np.array([]), np.array([])

    u = eta * np.arange(n, dtype=float)
    iu = 1j * u

    # Fourier transform of damped price (normalized underlying with S0=1)
    shifted = u - 1j * (alpha + 1.0)
    phi = cf_log_return(shifted, float(T))

    disc = np.exp(-float(r) * float(T))
    denom = (alpha * alpha + alpha - u * u) + 1j * (2.0 * alpha + 1.0) * u
    denom = np.where(np.abs(denom) < 1e-16, 1e-16, denom)
    psi = disc * phi / denom

    # FFT grid
    lam = 2.0 * np.pi / (n * eta)
    b = 0.5 * n * lam
    k = -b + lam * np.arange(n, dtype=float)  # log-moneyness grid

    w = _simpson_weights(n)
    x = np.exp(1j * u * b) * psi * w * eta
    y = np.fft.fft(x).real

    c_tilde = y / np.pi  # damped normalized call prices
    c_norm = np.exp(-alpha * k) * c_tilde
    K_grid = float(S0) * np.exp(k)
    C_grid = float(S0) * c_norm
    return K_grid.astype(float), C_grid.astype(float)


def interp_prices(
    *,
    K_grid: np.ndarray,
    C_grid: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    K_grid = np.asarray(K_grid, dtype=float)
    C_grid = np.asarray(C_grid, dtype=float)
    K = np.asarray(K, dtype=float)
    if K_grid.size == 0 or C_grid.size == 0 or K.size == 0:
        return np.full_like(K, np.nan, dtype=float)
    order = np.argsort(K_grid)
    Kg = K_grid[order]
    Cg = C_grid[order]
    return np.interp(K, Kg, Cg, left=np.nan, right=np.nan).astype(float)


__all__ = ["FFTConfig", "carr_madan_fft_call_prices", "interp_prices", "LogReturnCF"]
