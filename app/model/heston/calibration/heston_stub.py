"""
TensorFlow-free Heston calibration stub.
Uses simple IV surface statistics to return plausible parameters
without any heavy ML dependencies.
"""

import numpy as np


def calibrate_heston_stub(strikes, maturities, iv_matrix, epochs=None, **kwargs):
    try:
        iv_flat = np.asarray(iv_matrix, dtype=float)
        iv_flat = iv_flat[np.isfinite(iv_flat)]
        iv_med = float(np.nanmedian(iv_flat)) if iv_flat.size else 0.5
        if not np.isfinite(iv_med) or iv_med <= 0:
            iv_med = 0.5
    except Exception:
        iv_med = 0.5

    sigma = max(0.01, iv_med)
    v0 = sigma**2
    return {
        "kappa": 1.5,
        "theta": v0,
        "sigma": sigma,
        "rho": -0.4,
        "v0": v0,
    }


__all__ = ["calibrate_heston_stub"]
