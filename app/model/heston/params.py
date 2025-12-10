from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HestonParams:
    """
    Container for Heston model parameters.

    Attributes:
        kappa: speed of mean reversion of variance
        theta: long-run variance
        sigma: volatility of variance (vol-of-vol)
        rho: correlation between asset and variance Brownian motions
        v0: initial variance
        r: risk-free rate
        q: dividend yield
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    r: float = 0.0
    q: float = 0.0
