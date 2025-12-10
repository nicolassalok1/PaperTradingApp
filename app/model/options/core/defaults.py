"""
Default parameter helpers for Options panels.
"""

from __future__ import annotations

import numpy as np


def pick_default_T(k_ref: float, maturities=None, target: float = 1.0):
    """
    Choose a default maturity near the target (in years).
    Mirrors ancienne behavior from the GPT Options tab.
    """
    if maturities is None or len(maturities) == 0:
        return float(target)
    mats = np.array([float(t) for t in maturities if t and t > 0], dtype=float)
    if mats.size == 0:
        return float(target)
    idx = np.argmin(np.abs(mats - target))
    return float(mats[idx])


# Default ranges (kept minimal for compatibility; extend as needed)
DEFAULT_SPAN = 25.0
DEFAULT_SIGMA = 0.2
DEFAULT_RATE = 0.02
DEFAULT_DIVIDEND = 0.0
