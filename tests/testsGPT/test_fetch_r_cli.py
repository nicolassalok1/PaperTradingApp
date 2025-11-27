import io
import sys
import types
from unittest import TestCase, mock


class FetchRCliTests(TestCase):
    def test_compute_r_interpolates_with_mock_data(self):
        # Stub external deps before import
        sys.modules.setdefault("yfinance", types.SimpleNamespace(Ticker=mock.Mock()))
        sys.modules.setdefault("requests", types.SimpleNamespace(get=mock.Mock()))
        from scripts.scriptsGPT import fetch_r_cli

        with mock.patch.object(fetch_r_cli, "_fetch_last_close", return_value=2.0):
            with mock.patch.object(fetch_r_cli, "_fetch_from_fred", return_value=1.5):
                val = fetch_r_cli.compute_r(2.0)
        # should interpolate between mocked points -> around 0.02
        self.assertGreater(val, 0.0)
        self.assertLess(val, 0.05)

    def test_main_uses_compute_r_and_prints_value(self):
        sys.modules.setdefault("yfinance", types.SimpleNamespace(Ticker=mock.Mock()))
        sys.modules.setdefault("requests", types.SimpleNamespace(get=mock.Mock()))
        from scripts.scriptsGPT import fetch_r_cli

        with mock.patch.object(fetch_r_cli, "compute_r", return_value=0.123456):
            buf = io.StringIO()
            with mock.patch.object(sys, "stdout", buf), mock.patch.object(sys, "argv", ["fetch_r_cli.py", "1.0"]):
                fetch_r_cli.main()
        self.assertIn("0.123456", buf.getvalue())
