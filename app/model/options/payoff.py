from __future__ import annotations

import numpy as np


def call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S - K, 0.0)


def put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)


__all__ = ["call_payoff", "put_payoff"]
