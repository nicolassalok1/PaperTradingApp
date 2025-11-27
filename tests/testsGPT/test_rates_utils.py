import os
import sys
import types
from unittest import TestCase, mock


class RatesUtilsTests(TestCase):
    def test_get_r_respects_use_static_flag(self):
        os.environ["USE_STATIC_RF_RATE"] = "1"
        os.environ["DEFAULT_RF_RATE"] = "0.05"
        sys.modules.pop("scripts.scriptsGPT.rates_utils", None)
        sys.modules["yfinance"] = types.SimpleNamespace(Ticker=mock.Mock())
        sys.modules["requests"] = types.SimpleNamespace(get=mock.Mock())
        from scripts.scriptsGPT import rates_utils

        rates_utils.DEFAULT_RF = 0.05  # ensure constant aligns with env for this test
        self.assertAlmostEqual(rates_utils.get_r(1.0), 0.05)
        os.environ.pop("USE_STATIC_RF_RATE", None)
        os.environ.pop("DEFAULT_RF_RATE", None)

    def test_get_q_returns_dividend_yield_or_zero(self):
        sys.modules["yfinance"] = types.SimpleNamespace(Ticker=mock.Mock())
        sys.modules["requests"] = types.SimpleNamespace(get=mock.Mock())
        sys.modules.pop("scripts.scriptsGPT.rates_utils", None)
        from scripts.scriptsGPT import rates_utils

        class FakeTicker:
            def __init__(self, has_yield=True):
                self.info = {"dividendYield": 0.0123} if has_yield else {}

        with mock.patch.object(rates_utils.yf, "Ticker", return_value=FakeTicker(True)):
            self.assertAlmostEqual(rates_utils.get_q("AAPL"), 0.0123)

        with mock.patch.object(rates_utils.yf, "Ticker", return_value=FakeTicker(False)):
            self.assertEqual(rates_utils.get_q("MSFT"), 0.0)
