"""
Greeks calculator for dashboard options using Black-Scholes (vanilla proxy).
Accepts the option JSON schema used in options_book.json/custom options.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

_GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1_d2(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> Tuple[float, float]:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0, 0.0
    num = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    den = sigma * math.sqrt(T)
    d1 = num / den
    d2 = d1 - den
    return d1, d2


def compute_vanilla_greeks(option: Dict, spot: float) -> Dict[str, float]:
    opt_type = str(option.get("option_type") or option.get("type") or "call").lower()
    K = float(option.get("strike", 0.0) or 0.0)
    sigma = float(option.get("sigma", option.get("iv", option.get("vol", 0.2))) or 0.2)
    r = float(option.get("r", 0.0) or 0.0)
    q = float(option.get("q", 0.0) or 0.0)
    T = float(option.get("T", 0.0) or 0.0)
    if T <= 0 and option.get("expiration"):
        try:
            import datetime

            expiry = datetime.date.fromisoformat(str(option.get("expiration")))
            days = (expiry - datetime.date.today()).days
            T = max(days, 0) / 365.0
        except Exception:
            logging.warning(
                "[greeks] unreadable expiration %r — reporting zero greeks",
                option.get("expiration"),
            )
            T = 0.0
    if T <= 0:
        # An expired contract has no optionality left. Never fabricate a maturity:
        # a made-up T would report vega/gamma on a position that no longer exists.
        return dict.fromkeys(_GREEK_NAMES, 0.0)

    d1, d2 = _bs_d1_d2(spot, K, T, r, q, sigma)
    nd1 = _norm_pdf(d1)

    # delta_call = e^{-qT} N(d1) ; delta_put = -e^{-qT} N(-d1).
    # N(-d1) - 1 == -N(d1) only at the money, hence the historical put bug.
    delta = (
        math.exp(-q * T) * _norm_cdf(d1)
        if opt_type == "call"
        else -math.exp(-q * T) * _norm_cdf(-d1)
    )
    gamma = (
        math.exp(-q * T) * nd1 / (spot * sigma * math.sqrt(T))
        if spot > 0 and sigma > 0 and T > 0
        else 0.0
    )
    vega = spot * math.exp(-q * T) * nd1 * math.sqrt(T)
    theta_call = (
        -spot * nd1 * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * _norm_cdf(d2)
        + q * spot * math.exp(-q * T) * _norm_cdf(d1)
    )
    theta_put = (
        -spot * nd1 * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        - q * spot * math.exp(-q * T) * _norm_cdf(-d1)
    )
    theta = theta_call if opt_type == "call" else theta_put
    rho_call = K * T * math.exp(-r * T) * _norm_cdf(d2)
    rho_put = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    rho = rho_call if opt_type == "call" else rho_put

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),  # vega per 1% vol change
        "theta": float(theta / 365.0),  # per day
        "rho": float(rho / 100.0),  # per 1% rate change
    }


def compute_option_greeks(option: Dict, spot: float | None = None) -> Dict[str, float]:
    """
    Compute greeks for an option dict from options_book.json/custom options.
    For exotic structures, returns greeks of the first vanilla leg as proxy.
    """
    if not isinstance(option, dict):
        return {}
    if spot is None:
        try:
            spot = float(
                option.get("S0", option.get("underlying_close", option.get("strike", 0.0))) or 0.0
            )
        except Exception:
            spot = 0.0
    legs = option.get("legs") or []
    if legs:
        # use first leg as proxy
        proxy = dict(legs[0])
        proxy.setdefault("r", option.get("r", 0.0))
        proxy.setdefault("q", option.get("q", 0.0))
        proxy.setdefault("T", option.get("T", option.get("maturity_years", 0.0)))
        return compute_vanilla_greeks(proxy, spot)
    return compute_vanilla_greeks(option, spot)


__all__ = ["compute_option_greeks", "compute_vanilla_greeks"]
