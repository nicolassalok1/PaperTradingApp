"""
Pricing routines for the Heston model (Fourier/FFT).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from app.model.heston.charfunc import heston_charfunc
from app.model.heston.params import HestonParams
from app.model.heston.pricing.carr_madan import price_heston_carr_madan


def _p_j(j: int, S0: float, K: float, T: float, params: HestonParams) -> float:
    """
    Compute the Heston probability P1 or P2 using the Lewis integral (trapezoidal rule).
    """
    u_max = 150.0
    n = 400
    u_vals = np.linspace(1e-5, u_max, n)
    du = u_vals[1] - u_vals[0]

    if j == 1:

        def integrand(u):
            cf = heston_charfunc(u - 1j, T, S0, params)
            return np.real(
                np.exp(-1j * u * np.log(K)) * cf / (1j * u * S0 * math.exp(-params.q * T))
            )

    else:

        def integrand(u):
            cf = heston_charfunc(u, T, S0, params)
            return np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))

    vals = integrand(u_vals)
    return 0.5 + (1.0 / math.pi) * np.trapz(vals, dx=du)


def heston_call_price_spot(S0: float, K: float, T: float, params: HestonParams) -> float:
    """
    Price a European call under Heston via Fourier integration.
    """
    if S0 <= 0 or K <= 0 or T <= 0:
        return 0.0
    p1 = _p_j(1, S0, K, T, params)
    p2 = _p_j(2, S0, K, T, params)
    return float(S0 * math.exp(-params.q * T) * p1 - K * math.exp(-params.r * T) * p2)


def heston_call_price_vectorized(
    S0: float,
    strikes: Iterable[float],
    maturities: Iterable[float],
    params: HestonParams,
) -> np.ndarray:
    """
    Vectorized pricing across a grid of strikes/maturities.
    Returns array shaped (len(maturities), len(strikes)).
    """
    strikes_arr = np.array(list(strikes), dtype=float)
    mats_arr = np.array(list(maturities), dtype=float)
    out = np.zeros((len(mats_arr), len(strikes_arr)), dtype=float)
    for i, T in enumerate(mats_arr):
        for j, K in enumerate(strikes_arr):
            out[i, j] = heston_call_price_spot(S0, K, T, params)
    return out


def compute_heston_price(option_dict: dict, market: dict, params: dict | HestonParams):
    """
    Thin wrapper to Heston pricing from option/market payloads.
    """
    opt = option_dict or {}
    mkt = market or {}
    par = params or {}
    S0 = float(opt.get("S0") or mkt.get("S") or mkt.get("spot") or opt.get("spot") or 0.0)
    K = float(opt.get("strike") or 0.0)
    T = float(opt.get("T") or opt.get("maturity") or opt.get("maturity_years") or 0.0)
    r = float(mkt.get("r") or 0.0)
    q = float(mkt.get("q") or 0.0)
    params_obj = (
        par
        if isinstance(par, HestonParams)
        else HestonParams(
            kappa=float(par.get("kappa", par.get("kappaH", 1.0))),
            theta=float(par.get("theta", 0.04)),
            sigma=float(par.get("sigma", par.get("sigma_v", 0.5))),
            rho=float(par.get("rho", -0.5)),
            v0=float(par.get("v0", par.get("v_init", 0.04))),
            r=r,
            q=q,
        )
    )
    return heston_call_price_spot(S0, K, T, params_obj)


__all__ = [
    "heston_call_price_spot",
    "heston_call_price_vectorized",
    "price_heston_carr_madan",
    "compute_heston_price",
]
