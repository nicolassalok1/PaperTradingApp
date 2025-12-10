"""
FFT-based Heston pricing (Carr-Madan).
Version compacte, non optimisée.
"""

import numpy as np
from numpy import exp, log
from scipy.fft import fft


# Skeleton only (for extension)
def price_heston_fft(S0, K, T, r, q, params):
    # TODO: implémentation FFT complète
    return 0.0


__all__ = ["price_heston_fft"]
