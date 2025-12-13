from __future__ import annotations

import math
from typing import Iterable, List


def log_discount_interp(maturities: Iterable[float], discount_factors: Iterable[float], t: float) -> float:
    """
    Log-linear interpolation on discount factors, robust to sparse nodes.
    """
    pairs = [
        (float(T), float(df))
        for T, df in zip(maturities, discount_factors)
        if T is not None and df is not None and math.isfinite(T) and math.isfinite(df) and df > 0
    ]
    if not pairs:
        return math.nan
    pairs.sort(key=lambda x: x[0])
    Ts: List[float] = [p[0] for p in pairs]
    dfs: List[float] = [p[1] for p in pairs]
    t = float(t)
    if t <= Ts[0]:
        # Enforce DF(0)=1 by extrapolating with a flat *zero rate* implied by the first node.
        T0 = Ts[0]
        if T0 <= 0:
            return 1.0
        return math.exp(math.log(dfs[0]) * (t / T0))
    if t >= Ts[-1]:
        # Flat zero-rate extrapolation implied by the last node (consistent with NodeYieldCurve.zero_rate).
        Tn = Ts[-1]
        if Tn <= 0:
            return 1.0
        return math.exp(math.log(dfs[-1]) * (t / Tn))
    for i in range(len(Ts) - 1):
        t1, t2 = Ts[i], Ts[i + 1]
        if t1 <= t <= t2:
            w = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
            log_df = math.log(dfs[i]) * (1 - w) + math.log(dfs[i + 1]) * w
            return math.exp(log_df)
    # Should not happen due to explicit bounds above; keep safe fallback.
    return dfs[-1]


def linear_zero_interp(maturities: Iterable[float], zero_rates: Iterable[float], t: float) -> float:
    """
    Simple linear interpolation on zero rates as a fallback.
    """
    pairs = [
        (float(T), float(r))
        for T, r in zip(maturities, zero_rates)
        if T is not None and r is not None and math.isfinite(T) and math.isfinite(r)
    ]
    if not pairs:
        return math.nan
    pairs.sort(key=lambda x: x[0])
    Ts: List[float] = [p[0] for p in pairs]
    rates: List[float] = [p[1] for p in pairs]
    t = float(t)
    if t <= Ts[0]:
        return rates[0]
    if t >= Ts[-1]:
        return rates[-1]
    for i in range(len(Ts) - 1):
        t1, t2 = Ts[i], Ts[i + 1]
        if t1 <= t <= t2:
            w = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
            return rates[i] * (1 - w) + rates[i + 1] * w
    return rates[-1]
