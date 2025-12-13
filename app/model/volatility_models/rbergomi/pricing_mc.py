from __future__ import annotations

import numpy as np


def price_call_mc(
    *,
    ST: np.ndarray,
    K: float,
    r: float,
    T: float,
) -> float:
    ST = np.asarray(ST, dtype=float)
    if ST.size == 0 or K <= 0 or T <= 0:
        return float("nan")
    payoff = np.maximum(ST - float(K), 0.0)
    return float(np.exp(-float(r) * float(T)) * np.mean(payoff))


def price_call_grid_mc(
    *,
    ST_by_T: dict[float, np.ndarray],
    strikes: np.ndarray,
    t_grid: np.ndarray,
    r: float,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    out = np.full((len(t_grid), len(strikes)), np.nan, dtype=float)
    for i_t, T in enumerate(t_grid):
        if float(T) <= 0:
            continue
        ST = ST_by_T.get(float(T))
        if ST is None:
            continue
        for j_k, K in enumerate(strikes):
            out[i_t, j_k] = price_call_mc(ST=ST, K=float(K), r=r, T=float(T))
    return out


__all__ = ["price_call_mc", "price_call_grid_mc"]

