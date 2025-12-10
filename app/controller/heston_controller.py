"""
Controller + service for Heston pricing/calibration (merged).
"""

from __future__ import annotations

from typing import Dict

from app.model.heston.calibration import calibrate_heston
from app.model.heston.params import HestonParams
from app.model.heston.pricing import heston_call_price_spot


def price_heston_call(S0: float, K: float, T: float, params: HestonParams) -> float:
    return heston_call_price_spot(S0, K, T, params)


def calibrate_heston_surface(
    market_iv_surface: Dict, S0: float, r: float, q: float
) -> HestonParams:
    """Calibrate parameters to a market IV surface (expects keys K, T, iv)."""
    return calibrate_heston(market_iv_surface, strikes=None, maturities=None, S0=S0, r=r, q=q)


def compute_heston_price(payload: Dict) -> Dict:
    """
    Compute a Heston call price from a payload:
    {
        "S0": float,
        "K": float,
        "T": float,
        "params": {kappa, theta, sigma, rho, v0, r, q}
    }
    """
    params_dict = payload.get("params") or {}
    params = (
        params_dict
        if isinstance(params_dict, HestonParams)
        else HestonParams(
            kappa=float(params_dict.get("kappa", 1.0)),
            theta=float(params_dict.get("theta", 0.04)),
            sigma=float(params_dict.get("sigma", 0.5)),
            rho=float(params_dict.get("rho", -0.5)),
            v0=float(params_dict.get("v0", 0.04)),
            r=float(params_dict.get("r", payload.get("r", 0.0))),
            q=float(params_dict.get("q", payload.get("q", 0.0))),
        )
    )
    price = price_heston_call(
        float(payload.get("S0", 0.0)),
        float(payload.get("K", 0.0)),
        float(payload.get("T", 0.0)),
        params,
    )
    return {"price": price, "params": params.__dict__}


def calibrate_heston_from_market(payload: Dict) -> Dict:
    """
    Calibrate Heston parameters from a market IV surface payload containing:
        - market_iv_surface: dict with K, T, iv
        - S0, r, q
    """
    surface = payload.get("market_iv_surface") or payload
    params = calibrate_heston_surface(
        market_iv_surface=surface,
        S0=float(payload.get("S0", payload.get("spot", 0.0))),
        r=float(payload.get("r", 0.0)),
        q=float(payload.get("q", 0.0)),
    )
    return {"params": params.__dict__}


__all__ = [
    "compute_heston_price",
    "calibrate_heston_from_market",
    "price_heston_call",
    "calibrate_heston_surface",
]
