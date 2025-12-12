"""
Utilities to fetch risk-free rates and dividend yield.

- get_r(maturity_years): returns r(T) from the active cached yield curve, falling back to DEFAULT_RF.
- get_q(ticker): returns 0.0 (placeholder).
"""

from __future__ import annotations

import math
import os

DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))


def get_r(maturity_years: float, currency: str | None = None) -> float:
    """
    Risk-free rate r(T) in decimal using the shared yield curve model (cache-only).
    Falls back to DEFAULT_RF if unavailable.
    """
    try:
        from app.model.yieldcurve.service import get_active_curve

        curve, _, _, _, _ = get_active_curve(currency=currency, ensure_cache=True, allow_api=False)
        rate = curve.zero_rate(float(maturity_years))
        if math.isfinite(rate):
            return float(rate)
    except Exception:
        pass
    return DEFAULT_RF


def get_q(ticker: str) -> float:
    """
    Return dividend yield (continuous approx) for the given ticker.
    Fallback to 0.0.
    """
    return 0.0


if __name__ == "__main__":
    r = get_r(0.5)
    q = get_q("AAPL")
    print(r, q)
