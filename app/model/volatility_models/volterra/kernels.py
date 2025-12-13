from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from scipy.special import gamma
except Exception:  # pragma: no cover - optional dependency
    gamma = None


KernelFn = Callable[[np.ndarray], np.ndarray]


def exponential_kernel(lam: float) -> KernelFn:
    lam = float(max(lam, 0.0))

    def _k(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-lam * t)

    return _k


def fractional_kernel(H: float) -> KernelFn:
    H = float(H)
    if not (0.0 < H < 0.5):
        raise ValueError("H must be in (0, 0.5).")
    if gamma is None:
        raise RuntimeError("SciPy required (scipy.special.gamma) for fractional kernel.")

    denom = float(gamma(H + 0.5))

    def _k(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t, dtype=float)
        m = t > 0
        out[m] = (t[m] ** (H - 0.5)) / denom
        return out

    return _k


def sum_exponentials_kernel(rates: np.ndarray, weights: np.ndarray) -> KernelFn:
    rates = np.asarray(rates, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if rates.size != weights.size or rates.size == 0:
        raise ValueError("rates/weights must have same non-zero length.")

    def _k(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).reshape(-1)
        out = np.zeros_like(t, dtype=float)
        for x, w in zip(rates, weights):
            out += float(w) * np.exp(-float(x) * t)
        return out.reshape(t.shape)

    return _k


__all__ = ["KernelFn", "exponential_kernel", "fractional_kernel", "sum_exponentials_kernel"]

