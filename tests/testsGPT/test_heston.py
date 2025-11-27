from unittest import TestCase, skipIf

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without torch
    _TORCH_AVAILABLE = False


class HestonTests(TestCase):
    def test_params_from_unconstrained_produces_valid_ranges(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        from scripts.scriptsGPT.pricing_scripts.Heston.heston_torch import HestonParams

        params = HestonParams.from_unconstrained(0.1, 0.2, 0.3, 0.0, 0.1)
        self.assertGreater(params.kappa.item(), 0.0)
        self.assertGreater(params.theta.item(), 0.0)
        self.assertGreater(params.sigma.item(), 0.0)
        self.assertGreater(params.v0.item(), 0.0)
        self.assertTrue(-1.0 < params.rho.item() < 1.0)

    def test_heston_cf_returns_one_at_zero(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        from scripts.scriptsGPT.pricing_scripts.Heston.heston_torch import HestonParams, heston_cf

        params = HestonParams.from_unconstrained(0.5, 0.04, 0.3, 0.0, 0.04)
        val = heston_cf(u=0.0, T=1.0, S0=100.0, r=0.01, q=0.0, params=params)
        self.assertAlmostEqual(val.real.item(), 1.0, places=6)
        self.assertAlmostEqual(val.imag.item(), 0.0, places=6)

    def test_implied_vol_inverts_bs_price(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        import sys
        import types

        # Stub plotly to avoid heavy dependency for this unit-level check.
        plotly_stub = types.ModuleType("plotly")
        plotly_stub.graph_objects = types.SimpleNamespace()
        sys.modules.setdefault("plotly", plotly_stub)
        sys.modules.setdefault("plotly.graph_objects", plotly_stub.graph_objects)

        from scripts.scriptsGPT.pricing_scripts.Heston.heston_mc_heatmap_to_iv import bs_price, implied_vol_from_price

        target_vol = 0.2
        price = bs_price(S0=100, K=100, T=1.0, vol=target_vol, r=0.01, option_type="call")
        iv = implied_vol_from_price(price, S0=100, K=100, T=1.0, r=0.01, option_type="call")
        self.assertAlmostEqual(iv, target_vol, places=3)
