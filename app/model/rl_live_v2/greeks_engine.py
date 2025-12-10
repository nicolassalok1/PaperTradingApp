"""
Lightweight Black-Scholes Greeks computation and aggregation.
"""

from __future__ import annotations

import math
from typing import Dict, List

from scipy.stats import norm


def compute_bs_greeks(spot: float, strike: float, vol: float, r: float, t: float, call_put: str) -> Dict[str, float]:
    # Basic BS greeks; assumes non-dividend-paying asset.
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if call_put.lower() == "call":
        delta = norm.cdf(d1)
        theta = -(spot * norm.pdf(d1) * vol) / (2 * math.sqrt(t)) - r * strike * math.exp(-r * t) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(spot * norm.pdf(d1) * vol) / (2 * math.sqrt(t)) + r * strike * math.exp(-r * t) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (spot * vol * math.sqrt(t))
    vega = spot * norm.pdf(d1) * math.sqrt(t)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


def aggregate_portfolio_greeks(option_positions: List[Dict[str, float]], underlying_price: float) -> Dict[str, float]:
    net_delta = net_gamma = net_vega = net_theta = 0.0
    for opt in option_positions:
        qty = float(opt.get("qty", 0.0) or 0.0)
        strike = float(opt.get("strike_price", opt.get("strike", 0.0)) or 0.0)
        t = float(opt.get("days_to_expiration", 30)) / 365.0
        call_put = str(opt.get("option_type", opt.get("type", "call"))).lower()
        vol = 0.20  # placeholder vol
        greeks = compute_bs_greeks(underlying_price, strike, vol, 0.0, max(t, 1e-6), call_put)
        net_delta += greeks["delta"] * qty
        net_gamma += greeks["gamma"] * qty
        net_vega += greeks["vega"] * qty
        net_theta += greeks["theta"] * qty
    return {
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_vega": net_vega,
        "net_theta": net_theta,
    }


__all__ = ["compute_bs_greeks", "aggregate_portfolio_greeks"]
