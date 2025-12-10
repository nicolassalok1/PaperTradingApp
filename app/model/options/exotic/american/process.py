"""
Stochastic process definitions and simulators used by Longstaff-Schwartz pricing.

Responsibilities:
- Provide base abstract process contract.
- Implement GBM processes with vectorized path generation.

External dependencies:
- NumPy for random sampling and vectorized math.
- Pandas for returning path matrices in DataFrame form.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class StochasticProcess(ABC):
    """Abstract stochastic process interface; subclasses must implement simulate()."""

    @abstractmethod
    def simulate(self): ...


@dataclass
class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion with closed-form path generation.

    Attributes:
        mu: Drift term (annualized).
        sigma: Volatility term.

    Notes:
        Paths are generated in a fully vectorized way for performance.
    """

    mu: float
    sigma: float

    def simulate(
        self, s0: float, T: int, n: int, m: int, v0: float = None
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points
        """
        Generate GBM paths using exact discretization.

        Args:
            s0: Initial spot.
            T: Horizon in years.
            n: Number of simulated paths.
            m: Number of time steps (discretization points).
            v0: Unused, kept for interface compatibility.
        Returns:
            DataFrame of shape (m+1, n) with simulated spot levels.
        """
        dt = T / m
        np.random.seed(0)
        W = np.cumsum(np.sqrt(dt) * np.random.randn(m + 1, n), axis=0)
        W[0] = 0

        T = np.ones(n).reshape(1, -1) * np.linspace(0, T, m + 1).reshape(-1, 1)

        s = s0 * np.exp((self.mu - 0.5 * self.sigma**2) * T + self.sigma * W)

        return s
