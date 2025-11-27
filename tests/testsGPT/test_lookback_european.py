import sys
import types
from unittest import TestCase

from math import exp, log, sqrt
from scipy.stats import norm


class LookbackEuropeanCallTests(TestCase):
    def test_price_exact_matches_bsm_formula(self):
        # Ensure Lookback package is importable by adjusting sys.path
        from pathlib import Path

        sys.path.insert(0, str(Path("scripts/scriptsGPT/pricing_scripts")))
        lookback_pkg = types.ModuleType("Lookback")
        lookback_pkg.__path__ = [str(Path("scripts/scriptsGPT/pricing_scripts/Lookback").resolve())]
        sys.modules["Lookback"] = lookback_pkg
        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba not installed")
        from scripts.scriptsGPT.pricing_scripts.Lookback.european_call import european_call_option

        S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
        opt = european_call_option(T=T, t=0.0, S0=S0, K=K, r=r, sigma=sigma)
        price = opt.price_exact()

        tau = T
        d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
        d2 = d1 - sigma * sqrt(tau)
        expected = S0 * norm.cdf(d1) - K * exp(-r * tau) * norm.cdf(d2)
        self.assertAlmostEqual(price, expected, places=6)
