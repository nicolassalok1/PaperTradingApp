from __future__ import annotations

import math
from enum import Enum

import numpy as np

from app.model.options.logic import simulate_gbm_paths
from app.model.yieldcurve.rates_utils import get_q, get_r


class MCModel(Enum):
    BS = "bs"
    RHESTON = "rheston"
    RBERGOMI = "rbergomi"
    SABR = "sabr"
    VOLTERRA = "volterra"


def price_european_mc(
    S0,
    K,
    T,
    sigma,
    model: str = "bs",
    n_paths: int = 10000,
    n_steps: int = 252,
    ticker: str | None = None,
):
    """
    Monte Carlo pricing for a European call under GBM.
    """
    model_key = (model or "").lower()
    if model_key not in {MCModel.BS.value}:
        raise NotImplementedError("Model not implemented yet")

    if S0 is None or K is None:
        return float("nan")

    try:
        r = float(get_r(T))
    except Exception:
        r = 0.01
    try:
        q = float(get_q(ticker)) if ticker else 0.0
    except Exception:
        q = 0.0
    paths, _ = simulate_gbm_paths(S0, r, q, sigma, T, n_steps, n_paths)
    terminal = paths[-1]
    payoff = np.maximum(terminal - K, 0.0)
    price = math.exp(-r * T) * float(np.mean(payoff))
    return price


__all__ = ["MCModel", "price_european_mc"]
