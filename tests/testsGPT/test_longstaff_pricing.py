from unittest import TestCase

import numpy as np


class LongstaffPricingTests(TestCase):
    def setUp(self):
        from scripts.scriptsGPT.pricing_scripts.Longstaff.pricing import GeometricBrownianMotion, Option

        self.Option = Option
        self.GBM = GeometricBrownianMotion

    def test_gbm_simulation_shape(self):
        gbm = self.GBM(mu=0.05, sigma=0.2)
        paths = gbm.simulate(s0=100, T=1, n=5, m=4, v0=None)
        self.assertEqual(paths.shape, (5, 5))
        self.assertTrue(np.all(paths > 0))

    def test_monte_carlo_pricing_positive(self):
        from scripts.scriptsGPT.pricing_scripts.Longstaff.pricing import monte_carlo_simulation

        option = self.Option(s0=100, T=1, K=100, call=True)
        gbm = self.GBM(mu=0.05, sigma=0.2)
        price = monte_carlo_simulation(option, gbm, n=200, m=10, alpha=0.1)
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0.0)

    def test_monte_carlo_LS_runs(self):
        from scripts.scriptsGPT.pricing_scripts.Longstaff.pricing import monte_carlo_simulation_LS

        option = self.Option(s0=100, T=1, K=100, call=True)
        gbm = self.GBM(mu=0.05, sigma=0.2)
        # Function prints price but returns None; verify it executes without raising.
        self.assertIsNone(monte_carlo_simulation_LS(option, gbm, n=200, m=10, alpha=0.1))
