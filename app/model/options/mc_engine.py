from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from app.model.options.logic import simulate_gbm_paths
from app.model.options.payoff import call_payoff, put_payoff
from app.model.yieldcurve.rates_utils import get_q, get_r


class MCModel(Enum):
    BS = "bs"
    RHESTON = "rheston"
    RBERGOMI = "rbergomi"
    SABR = "sabr"
    VOLTERRA = "volterra"


def simulate_paths(S0: float, r: float, q: float, vol: float, T: float, n_steps: int, n_paths: int, seed: int = 42):
    """Wrapper over simulate_gbm_paths to keep API stable."""
    return simulate_gbm_paths(S0, r, q, vol, T, n_steps, n_paths, seed=seed)


def price_mc_european(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 10000,
    n_steps: int = 252,
    ticker: str | None = None,
):
    """
    Monte Carlo pricing for a European option under GBM.
    """
    opt = option_type.lower()
    if opt not in {"call", "put"}:
        opt = "call"

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

    paths, _ = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths)
    terminal = paths[-1]
    payoff = call_payoff(terminal, K) if opt.startswith("c") else put_payoff(terminal, K)
    price = math.exp(-r * T) * float(np.mean(payoff))
    stderr = float(np.std(payoff, ddof=1) / math.sqrt(len(payoff)))
    return {"price": price, "stderr": stderr}


def _regress_continuation(values: np.ndarray, states: np.ndarray) -> np.ndarray:
    """
    Fit continuation value using polynomial basis [1, S, S^2].
    """
    X = np.vstack([np.ones_like(states), states, states**2]).T
    coeffs, *_ = np.linalg.lstsq(X, values, rcond=None)
    return X @ coeffs


def price_mc_lsmc(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    exercise_dates: Sequence[float],
    r: float,
    q: float = 0.0,
    n_steps: int = 252,
    n_paths: int = 20000,
    seed: int = 42,
) -> dict:
    """
    Longstaff-Schwartz Monte Carlo pricer for American/Bermudan/European options.
    exercise_dates: iterable of exercise times in years (ascending).
    """
    opt = option_type.lower()
    is_call = opt.startswith("c")
    times = sorted([t for t in exercise_dates if t > 0])
    if not times:
        times = [T]

    paths, dt = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, seed=seed)
    discount = math.exp(-r * dt)
    intrinsic = call_payoff if is_call else put_payoff

    # Map exercise dates to step indices
    step_indices: List[int] = []
    for t in times:
        idx = min(n_steps, max(1, int(round(t / T * n_steps))))
        if idx not in step_indices:
            step_indices.append(idx)
    step_indices = sorted(step_indices)

    cashflows = intrinsic(paths[-1], K)
    exercise_flags = np.zeros_like(cashflows, dtype=bool)

    for idx in reversed(step_indices[:-1]):  # exclude maturity handled by init cashflows
        state = paths[idx]
        immediate = intrinsic(state, K)
        in_money = immediate > 0
        if not np.any(in_money):
            continue
        cont_est = _regress_continuation(cashflows * (discount ** (n_steps - idx)), state)[in_money]
        exercise_mask = immediate[in_money] > cont_est
        # update cashflows where exercise happens
        update_indices = np.where(in_money)[0][exercise_mask]
        cashflows[update_indices] = immediate[update_indices]
        exercise_flags[update_indices] = True
        # discount remaining cashflows for non-exercised paths one step
        cashflows *= discount

    price = float(np.mean(cashflows) * (discount ** 0))  # already discounted per step
    stderr = float(np.std(cashflows, ddof=1) / math.sqrt(len(cashflows)))
    early_ex_ratio = float(np.mean(exercise_flags)) if len(exercise_flags) else 0.0

    return {"price": price, "stderr": stderr, "early_exercise_ratio": early_ex_ratio}


__all__ = [
    "MCModel",
    "simulate_paths",
    "price_mc_european",
    "price_mc_lsmc",
    "price_european_mc",
]


def price_european_mc(
    S0,
    K,
    T,
    sigma,
    model="bs",
    n_paths=10000,
    n_steps=252,
    ticker=None,
):
    """
    Backward-compatible alias for price_mc_european.
    Only Black-Scholes MC is implemented; other models not yet supported.
    """
    if str(model).lower() != "bs":
        raise NotImplementedError("Model not implemented yet")
    return price_mc_european(
        S0=S0,
        K=K,
        T=T,
        sigma=sigma,
        option_type="call",
        n_paths=n_paths,
        n_steps=n_steps,
        ticker=ticker,
    )["price"]
