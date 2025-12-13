from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.special import gamma
except Exception:  # pragma: no cover - optional dependency
    gamma = None


@dataclass(frozen=True)
class MarkovianKernel:
    rates: np.ndarray  # x_i
    weights: np.ndarray  # w_i


def fractional_kernel_markovian_approx(
    *,
    H: float,
    n_factors: int,
    x_min: float,
    x_max: float,
) -> MarkovianKernel:
    """
    Approximates the fractional kernel K(t)=t^{H-1/2}/Gamma(H+1/2) as a sum of exponentials:
      K(t) ≈ Σ w_i exp(-x_i t)
    via a geometric quadrature of the Laplace representation.
    """
    H = float(H)
    if not (0.0 < H < 0.5):
        raise ValueError("H must be in (0, 0.5) for rough volatility kernels.")
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1.")
    if x_min <= 0 or x_max <= 0 or x_min >= x_max:
        raise ValueError("Invalid x_min/x_max.")
    if gamma is None:
        raise RuntimeError("SciPy is required (scipy.special.gamma) for fractional kernel approximation.")

    power = 0.5 - H  # > 0
    c = 1.0 / (float(gamma(H + 0.5)) * float(gamma(0.5 - H)))

    edges = np.geomspace(float(x_min), float(x_max), int(n_factors) + 1)
    rates = np.sqrt(edges[:-1] * edges[1:]).astype(float)
    weights = (c * (edges[1:] ** power - edges[:-1] ** power) / power).astype(float)
    return MarkovianKernel(rates=rates, weights=weights)


__all__ = ["MarkovianKernel", "fractional_kernel_markovian_approx"]

