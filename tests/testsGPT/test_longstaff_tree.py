from unittest import TestCase


class LongstaffTreeTests(TestCase):
    def test_crr_pricing_put_and_call(self):
        from scripts.scriptsGPT.pricing_scripts.Longstaff.pricing import crr_pricing, Option

        put = Option(s0=100, T=1, K=100, call=False)
        call = Option(s0=100, T=1, K=100, call=True)

        put_price = crr_pricing(r=0.05, sigma=0.2, option=put, n=50)
        call_price = crr_pricing(r=0.05, sigma=0.2, option=call, n=50)

        self.assertGreater(put_price, 0.0)
        self.assertGreater(call_price, 0.0)
        self.assertNotEqual(put_price, call_price)
