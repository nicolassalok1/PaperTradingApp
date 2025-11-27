import sys
import types
from unittest import TestCase


class LookbackFloatingCallTests(TestCase):
    def setUp(self):
        from pathlib import Path

        sys.path.insert(0, str(Path("scripts/scriptsGPT/pricing_scripts")))
        pkg = types.ModuleType("Lookback")
        pkg.__path__ = [str(Path("scripts/scriptsGPT/pricing_scripts/Lookback").resolve())]
        sys.modules["Lookback"] = pkg

    def test_price_exact_positive(self):
        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba not installed")
        from scripts.scriptsGPT.pricing_scripts.Lookback.lookback_call import lookback_call_option

        opt = lookback_call_option(T=1.0, t=0.0, S0=100.0, r=0.01, sigma=0.2)
        price = opt.price_exact()
        self.assertGreaterEqual(price, 0.0)

    def test_price_monte_carlo_runs(self):
        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba not installed")
        from scripts.scriptsGPT.pricing_scripts.Lookback.lookback_call import lookback_call_option

        opt = lookback_call_option(T=0.5, t=0.0, S0=100.0, r=0.02, sigma=0.25)
        price = opt.price_monte_carlo(n_iters=50)
        self.assertIsInstance(price, float)
        self.assertGreaterEqual(price, 0.0)
