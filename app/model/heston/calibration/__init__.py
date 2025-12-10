"""
Calibration helpers for the Heston model.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import numpy as np

from app.model.heston.params import HestonParams
from app.model.heston.pricing import heston_call_price_spot


def _bs_price_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S * math.exp(-q * T) * cdf(d1) - K * math.exp(-r * T) * cdf(d2)


def _prepare_targets(
    market_iv_grid: Iterable[dict] | dict | np.ndarray,
    strikes: Sequence[float] | None,
    maturities: Sequence[float] | None,
    S0: float,
    r: float,
    q: float,
):
    """
    Normalize inputs into arrays of (price, K, T).
    """
    targets = []

    def _append(k, t, iv):
        if k is None or t is None or iv is None:
            return
        price = _bs_price_call(S0, float(k), float(t), r, q, float(iv))
        targets.append((price, float(k), float(t)))

    if isinstance(market_iv_grid, dict) and {"K", "T", "iv"} <= set(market_iv_grid.keys()):
        for k, t, iv in zip(market_iv_grid["K"], market_iv_grid["T"], market_iv_grid["iv"]):
            _append(k, t, iv)
    elif (
        isinstance(market_iv_grid, np.ndarray)
        and market_iv_grid.ndim == 2
        and market_iv_grid.shape[1] >= 3
    ):
        for row in market_iv_grid:
            _append(row[0], row[1], row[2])
    else:
        for row in market_iv_grid or []:
            _append(row.get("K"), row.get("T"), row.get("iv"))

    # If explicit strikes/maturities provided separately, combine with surface ivs
    if strikes is not None and maturities is not None and targets:
        pass  # already handled

    return targets


def _regularization_penalty(params: HestonParams) -> float:
    return 0.01 * (
        params.kappa * params.kappa
        + params.theta * params.theta
        + params.sigma * params.sigma
        + params.rho * params.rho
        + params.v0 * params.v0
    )


def calibrate_heston(
    market_iv_grid: Iterable[dict] | dict | np.ndarray,
    strikes: Sequence[float] | None,
    maturities: Sequence[float] | None,
    S0: float,
    r: float,
    q: float,
) -> HestonParams:
    """
    Calibrate Heston parameters to a grid of market IVs.
    Uses scipy.optimize.minimize when available, else a simple random search.
    """
    targets = _prepare_targets(market_iv_grid, strikes, maturities, S0, r, q)
    if not targets:
        return HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04, r=r, q=q)

    def loss(vec) -> float:
        params = HestonParams(
            kappa=vec[0], theta=vec[1], sigma=vec[2], rho=vec[3], v0=vec[4], r=r, q=q
        )
        errs = []
        for price_target, k, t in targets:
            model_price = heston_call_price_spot(S0, k, t, params)
            errs.append((model_price - price_target) ** 2)
        return float(np.mean(errs) + _regularization_penalty(params))

    bounds = np.array(
        [
            (0.01, 5.0),  # kappa
            (1e-4, 1.0),  # theta
            (1e-3, 3.0),  # sigma
            (-0.999, 0.999),  # rho
            (1e-4, 2.0),  # v0
        ]
    )

    try:
        from scipy.optimize import minimize  # type: ignore

        x0 = np.array([1.0, 0.04, 0.5, -0.5, 0.04], dtype=float)
        res = minimize(
            loss,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )
        vec = res.x
    except Exception:
        # Fallback: coarse random search
        best = (float("inf"), np.array([1.0, 0.04, 0.5, -0.5, 0.04]))
        for _ in range(200):
            candidate = np.array(
                [
                    random.uniform(*bounds[0]),
                    random.uniform(*bounds[1]),
                    random.uniform(*bounds[2]),
                    random.uniform(*bounds[3]),
                    random.uniform(*bounds[4]),
                ]
            )
            l = loss(candidate)
            if l < best[0]:
                best = (l, candidate)
        vec = best[1]

    return HestonParams(
        kappa=float(vec[0]),
        theta=float(vec[1]),
        sigma=float(vec[2]),
        rho=float(vec[3]),
        v0=float(vec[4]),
        r=r,
        q=q,
    )


def calibrate_heston_to_market(market_options: Iterable[dict], market: dict):
    """Wrapper to calibrate against a list of market options with spot/r/q context."""
    mkt = market or {}
    S0 = float(mkt.get("S") or mkt.get("spot") or 0.0)
    r = float(mkt.get("r") or 0.0)
    q = float(mkt.get("q") or 0.0)
    return calibrate_heston(market_options, strikes=None, maturities=None, S0=S0, r=r, q=q)


__all__ = ["calibrate_heston", "calibrate_heston_to_market"]
