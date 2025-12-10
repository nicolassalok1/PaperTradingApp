"""
Carr-Madan FFT pricing wrapper for the Heston model.
Migrated from app.model.options.pricing_lib to keep Heston logic scoped to the Heston domain.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HES_DIR = Path(__file__).resolve().parents[1] / "pricing_scripts"
if str(_HES_DIR) not in sys.path:
    sys.path.insert(0, str(_HES_DIR))

from app.model.heston.pricing_scripts.heston_torch import (
    HestonParams,
    carr_madan_call_torch,
)  # noqa: E402


def price_heston_carr_madan(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    option_type: str = "call",
) -> float:
    """
    Price a European option under Heston using Carr-Madan FFT pricer.

    Args:
        S0: Spot price.
        K: Strike.
        T: Maturity (years).
        r: Risk-free rate.
        q: Continuous dividend (or repo).
        kappa, theta, sigma, rho, v0: Heston parameters.
        option_type: "call"/"c" or "put"/"p".

    Returns:
        Option price (float).
    """
    params = HestonParams(
        torch.tensor(float(kappa)),
        torch.tensor(float(theta)),
        torch.tensor(float(sigma)),
        torch.tensor(float(rho)),
        torch.tensor(float(v0)),
    )
    call_price = float(
        carr_madan_call_torch(float(S0), float(r), float(q), float(T), params, float(K))
    )
    if option_type.lower().startswith("c"):
        return call_price
    return float(
        call_price
        - float(S0) * math.exp(-float(q) * float(T))
        + float(K) * math.exp(-float(r) * float(T))
    )


__all__ = ["price_heston_carr_madan", "HestonParams"]
