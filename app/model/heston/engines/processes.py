"""
Stochastic volatility process engines for the Heston model.
Extracted from app.model.options.american.* to keep Heston logic scoped to the Heston domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class StochasticProcess(ABC):
    """Abstract stochastic process interface; subclasses must implement simulate()."""

    @abstractmethod
    def simulate(self): ...


@dataclass
class HestonProcess(StochasticProcess):
    """
    Heston stochastic volatility process simulated via Milstein scheme.

    Attributes:
        mu: Drift term on price.
        kappa: Mean reversion speed of variance.
        theta: Long-term variance level.
        eta: Volatility of volatility.
        rho: Correlation between price and variance Brownian motions.
    """

    mu: float
    kappa: float
    theta: float
    eta: float
    rho: float

    def simulate(
        self, s0: float, v0: float, T: int, n: int, m: int
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points
        """
        Simulate Heston paths with Milstein correction for variance.

        Args:
            s0: Initial spot.
            v0: Initial variance.
            T: Horizon in years.
            n: Number of paths.
            m: Number of time steps.
        Returns:
            DataFrame of shape (m+1, n) with simulated spot levels.
        Notes:
            Variance is floored to stay non-negative; correlation is applied via
            Cholesky-equivalent construction.
        """
        dt = T / m
        z1 = np.random.randn(m, n)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn(m, n)

        s = np.zeros((m + 1, n))
        x = np.zeros((m + 1, n))
        v = np.zeros((m + 1, n))

        s[0] = s0
        v[0] = v0

        for i in range(m):

            v[i + 1] = (
                v[i]
                + self.kappa * (self.theta - v[i]) * dt
                + self.eta * np.sqrt(v[i] * dt) * z1[i]
                + self.eta**2 / 4 * (z1[i] ** 2 - 1) * dt
            )
            v = np.where(v > 0, v, -v)

            x[i + 1] = x[i] + (self.mu - v[i] / 2) * dt + np.sqrt(v[i] * dt) * z2[i]

            s[1:] = s[0] * np.exp(x[1:])

        return s


__all__ = ["HestonProcess", "StochasticProcess"]
