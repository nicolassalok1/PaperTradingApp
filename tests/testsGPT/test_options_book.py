import ast
import json
import tempfile
from pathlib import Path
from unittest import TestCase


def _extract_functions(source_text, func_names):
    """Extract top-level function definitions by name."""
    tree = ast.parse(source_text)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments)


class OptionsBookTests(TestCase):
    def setUp(self):
        src = Path("streamlit_appGPT.py").read_text()
        code = "\n".join(
            [
                "import json, time, math",
                "from pathlib import Path",
                _extract_functions(
                    src,
                    {"load_options_book", "save_options_book", "add_option_to_dashboard"},
                ),
            ]
        )
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)
        ns = {}
        exec(code, ns)  # nosec: executed from local source

        # Override storage paths to the temp directory
        ns["OPTIONS_BOOK_FILE"] = tmp_path / "options_portfolio.json"
        ns["OPTIONS_BOOK_FILE_LEGACY"] = tmp_path / "options_book.json"
        ns["JSON_DIR"] = tmp_path
        self.api = ns
        self.book_path = ns["OPTIONS_BOOK_FILE"]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_add_option_variants(self):
        add_option = self.api["add_option_to_dashboard"]
        load_book = self.api["load_options_book"]

        payloads = {
            "vanilla_call": {
                "product_type": "vanilla",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "avg_price": 2.0,
                "expiration": "2025-12-31",
            },
            "vanilla_put": {
                "product_type": "vanilla",
                "option_type": "put",
                "strike": 95,
                "quantity": 1,
                "avg_price": 1.5,
                "expiration": "2025-12-31",
            },
            "digital": {
                "product_type": "Digital (cash-or-nothing)",
                "option_type": "call",
                "strike": 105,
                "quantity": 2,
                "avg_price": 0.9,
                "misc": {"payout": 10.0, "style": "cash_or_nothing"},
            },
            "asian_arith": {
                "product_type": "Asian arithmétique",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "avg_price": 1.2,
                "misc": {"method": "MC control variate", "n_obs": 20, "n_paths": 5000},
            },
            "asian_geom": {
                "product_type": "Asian géométrique",
                "option_type": "put",
                "strike": 90,
                "quantity": 1,
                "avg_price": 1.1,
                "misc": {"method": "closed_form_geom", "n_obs": 12},
            },
            "barrier": {
                "product_type": "Barrier up-and-out",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "avg_price": 1.8,
                "misc": {"barrier": 120.0, "direction": "out", "barrier_type": "up"},
            },
            "lookback": {
                "product_type": "Lookback call",
                "option_type": "call",
                "strike": 0,  # floating strike
                "quantity": 1,
                "avg_price": 2.5,
                "misc": {"method": "floating"},
            },
            "straddle": {
                "product_type": "Straddle",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "misc": {},
                "legs": [
                    {"option_type": "call", "strike": 100},
                    {"option_type": "put", "strike": 100},
                ],
            },
            "strangle": {
                "product_type": "Strangle",
                "option_type": "call",
                "strike": 95,
                "strike2": 105,
                "quantity": 1,
                "misc": {"strike_put": 95, "strike_call": 105, "wing": 10},
            },
            "call_spread": {
                "product_type": "Call spread",
                "option_type": "call",
                "strike": 100,
                "strike2": 105,
                "quantity": 1,
                "misc": {"width": 5},
            },
            "chooser": {
                "product_type": "Chooser",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "misc": {"t_choice": 0.5},
            },
            "forward_start": {
                "product_type": "Forward-start",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "misc": {"T_start": 0.25, "k_factor": 1.1, "n_paths": 5000, "n_steps": 200},
            },
            "basket_nn": {
                "product_type": "Basket NN",
                "option_type": "call",
                "strike": 100,
                "quantity": 1,
                "misc": {"components": ["AAPL", "MSFT"], "weights": [0.5, 0.5]},
            },
        }

        added = {}
        for name, payload in payloads.items():
            opt_id = add_option(payload)
            added[opt_id] = payload

        # Validate book persisted and entries normalized
        book = load_book()
        self.assertEqual(len(book), len(payloads))

        for opt_id, payload in added.items():
            self.assertIn(opt_id, book)
            entry = book[opt_id]
            # Default fields
            self.assertIn("product_type", entry)
            self.assertIn("option_type", entry)
            self.assertEqual(entry["product_type"], payload["product_type"])
            self.assertEqual(entry["option_type"], payload["option_type"])
            # Numeric normalization
            if payload.get("strike") is not None:
                self.assertIsInstance(entry["strike"], float)
            if payload.get("strike2") is not None:
                self.assertIsInstance(entry["strike2"], float)
            # Misc preserved when provided
            if payload.get("misc") is not None:
                for mk, mv in payload["misc"].items():
                    self.assertIn(mk, entry["misc"])

        # Ensure book file exists
        self.assertTrue(self.book_path.exists())
        saved = json.loads(self.book_path.read_text())
        self.assertEqual(len(saved), len(payloads))
