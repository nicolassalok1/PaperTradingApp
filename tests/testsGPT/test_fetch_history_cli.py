import sys
import io
import types
from unittest import TestCase, mock

import pandas as pd


class FetchHistoryCliTests(TestCase):
    def test_main_success_writes_csv(self):
        # Stub yfinance before import to avoid external dependency.
        yf_stub = types.SimpleNamespace(download=mock.Mock())
        sys.modules["yfinance"] = yf_stub
        from scripts.scriptsGPT import fetch_history_cli

        fake_df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
        with mock.patch.object(fetch_history_cli.yf, "download", return_value=fake_df):
            with mock.patch.object(sys, "argv", ["fetch_history_cli.py", "--ticker", "AAPL"]):
                buf = io.StringIO()
                with mock.patch.object(sys, "stdout", buf):
                    code = fetch_history_cli.main()
        self.assertEqual(code, 0)
        self.assertIn("Close", buf.getvalue())

    def test_main_empty_returns_code_2(self):
        yf_stub = types.SimpleNamespace(download=mock.Mock())
        sys.modules["yfinance"] = yf_stub
        from scripts.scriptsGPT import fetch_history_cli

        with mock.patch.object(fetch_history_cli.yf, "download", return_value=pd.DataFrame()):
            with mock.patch.object(sys, "argv", ["fetch_history_cli.py", "--ticker", "AAPL"]):
                buf = io.StringIO()
                with mock.patch.object(sys, "stdout", buf):
                    code = fetch_history_cli.main()
        self.assertEqual(code, 2)
        self.assertEqual(buf.getvalue(), "")

    def test_main_exception_returns_code_1(self):
        yf_stub = types.SimpleNamespace(download=mock.Mock())
        sys.modules["yfinance"] = yf_stub
        from scripts.scriptsGPT import fetch_history_cli

        with mock.patch.object(fetch_history_cli.yf, "download", side_effect=RuntimeError("boom")):
            with mock.patch.object(sys, "argv", ["fetch_history_cli.py", "--ticker", "AAPL"]):
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with mock.patch.object(sys, "stdout", buf_out), mock.patch.object(sys, "stderr", buf_err):
                    code = fetch_history_cli.main()
        self.assertEqual(code, 1)
        self.assertIn("boom", buf_err.getvalue())
