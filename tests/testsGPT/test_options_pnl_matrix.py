import ast
import json
import tempfile
from copy import deepcopy
from collections import defaultdict
import random
from pathlib import Path
from unittest import TestCase


def _extract_functions(source_text, func_names):
    tree = ast.parse(source_text)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments)


class OptionsPnLMatrixTests(TestCase):
    """
    Valide l’ajout et le PnL (mark-to-market et expiration ATM/ITM/OTM) pour les principaux types d’options.
    On utilise les fonctions de l’app GPT (add_option_to_dashboard, compute_option_payoff/pnl) sur un book temporaire.
    """

    def setUp(self):
        src = Path("streamlit_appGPT.py").read_text(encoding="utf-8")
        tests_json_dir = Path("tests/jsons")
        tests_json_dir.mkdir(parents=True, exist_ok=True)
        code = "\n".join(
            [
                "import json, time, math, numpy as np",
                "from pathlib import Path",
                _extract_functions(
                    src,
                    {
                        "load_options_book",
                        "save_options_book",
                        "load_options_portfolio",
                        "save_options_portfolio",
                        "_compute_leg_payoff",
                        "add_option_to_dashboard",
                        "_split_options_book",
                        "load_expired_options",
                        "save_expired_options",
                        "migrate_legacy_expired_options",
                        "describe_expired_option_payoff",
                        "compute_option_payoff",
                        "compute_option_pnl",
                    },
                ),
            ]
        )
        self.tmpdir = tempfile.TemporaryDirectory(dir=tests_json_dir)
        tmp_path = Path(self.tmpdir.name)
        ns = {}
        exec(code, ns)  # nosec: executed from local source

        ns["OPTIONS_BOOK_FILE"] = tmp_path / "options_portfolio.json"
        ns["OPTIONS_BOOK_FILE_LEGACY"] = tmp_path / "options_book.json"
        ns["JSON_DIR"] = tmp_path
        ns["LEGACY_EXPIRED_FILE"] = tmp_path / "expired_options.json"
        self.api = ns

    def tearDown(self):
        self.tmpdir.cleanup()

    # Helpers -------------------------------------------------
    def _add(self, payload):
        return self.api["add_option_to_dashboard"](payload)

    def _book(self):
        return self.api["load_options_book"]()

    def _payoff(self, opt, spot):
        return self.api["compute_option_payoff"](opt, spot)

    def _pnl(self, opt, spot, mark=None):
        return self.api["compute_option_pnl"](opt, spot_at_event=spot, mark_price=mark)

    def _expired(self):
        return self.api["load_expired_options"]()

    def _save_expired(self, payload):
        return self.api["save_expired_options"](payload)

    def _expected_pnl(self, opt, spot, mark=None):
        qty = float(opt.get("quantity", 1) or 0.0)
        premium = float(opt.get("avg_price", 0.0) or 0.0) * qty
        payoff = mark if mark is not None else self._payoff(opt, spot)
        payoff_total = payoff * qty
        side = (opt.get("side") or "long").lower()
        pnl_total = payoff_total - premium if side == "long" else premium - payoff_total
        pnl_per_unit = pnl_total / qty if qty else 0.0
        return {
            "payoff_per_unit": payoff if mark is not None else payoff / qty if qty else payoff,
            "pnl_per_unit": pnl_per_unit,
            "pnl_total": pnl_total,
        }

    def _spots(self, opt):
        strike = float(opt.get("strike", 100.0) or 100.0)
        strike2_val = opt.get("strike2", strike)
        strike2 = float(strike2_val) if strike2_val is not None else float(strike)
        product = str(opt.get("product_type") or opt.get("product") or "").lower()
        opt_type = str(opt.get("option_type") or opt.get("type") or "").lower()

        if "strangle" in product:
            k_put = min(strike, strike2)
            k_call = max(strike, strike2)
            return {
                "atm": (k_put + k_call) / 2,
                "itm": k_call + 10,  # call leg ITM
                "otm": k_put - 10,   # put leg ITM (call OTM)
            }
        if "straddle" in product:
            return {"atm": strike, "itm": strike + 10, "otm": strike}  # straddle nul seulement à K
        if "spread" in product and "put" in product:
            return {"atm": (strike + strike2) / 2, "itm": strike - 5, "otm": strike2 + 5}
        if "spread" in product and "call" in product:
            return {"atm": (strike + strike2) / 2, "itm": strike2 + 5, "otm": strike - 5}
        if opt_type == "put" or "put" in product:
            return {"atm": strike, "itm": strike - 10, "otm": strike + 10}
        # call/digital/barrier/etc.
        return {"atm": strike, "itm": strike + 10, "otm": strike - 10}

    # Tests ---------------------------------------------------
    def test_full_matrix_pnl(self):
        base_cases = [
            {"name": "vanilla_call", "product_type": "vanilla", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "vanilla_put", "product_type": "vanilla", "option_type": "put", "strike": 100, "avg_price": 2.0},
            {"name": "american_call", "product_type": "American", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "american_put", "product_type": "American", "option_type": "put", "strike": 100, "avg_price": 2.0},
            {"name": "bermudan_call", "product_type": "Bermudan", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "bermudan_put", "product_type": "Bermudan", "option_type": "put", "strike": 100, "avg_price": 2.0},
            {"name": "digital_call", "product_type": "Digital (cash-or-nothing)", "option_type": "call", "strike": 100, "avg_price": 1.0, "misc": {"payout": 10}},
            {"name": "digital_put", "product_type": "Digital (cash-or-nothing)", "option_type": "put", "strike": 100, "avg_price": 1.0, "misc": {"payout": 10}},
            {"name": "digital_asset", "product_type": "Digital asset-or-nothing", "option_type": "call", "strike": 100, "avg_price": 1.0, "misc": {"payout": 100}},
            {"name": "asian_arith", "product_type": "Asian arithmétique", "option_type": "call", "strike": 100, "avg_price": 1.0, "misc": {"closing_prices": [100, 102, 104]}},
            {"name": "asian_arith_put", "product_type": "Asian arithmétique", "option_type": "put", "strike": 100, "avg_price": 1.0, "misc": {"closing_prices": [100, 102, 104]}},
            {"name": "asian_geom", "product_type": "Asian géométrique", "option_type": "call", "strike": 100, "avg_price": 1.0, "misc": {"closing_prices": [100, 101, 99]}},
            {"name": "barrier_call", "product_type": "Barrier up-and-out", "option_type": "call", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 200, "direction": "out", "barrier_type": "up"}},
            {"name": "barrier_put", "product_type": "Barrier up-and-out", "option_type": "put", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 200, "direction": "out", "barrier_type": "up"}},
            {"name": "barrier_up_in_call", "product_type": "Barrier up-and-in", "option_type": "call", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 200, "direction": "in", "barrier_type": "up"}},
            {"name": "barrier_up_in_put", "product_type": "Barrier up-and-in", "option_type": "put", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 200, "direction": "in", "barrier_type": "up"}},
            {"name": "barrier_down_out_call", "product_type": "Barrier down-and-out", "option_type": "call", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 50, "direction": "out", "barrier_type": "down"}},
            {"name": "barrier_down_out_put", "product_type": "Barrier down-and-out", "option_type": "put", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 50, "direction": "out", "barrier_type": "down"}},
            {"name": "barrier_down_in_call", "product_type": "Barrier down-and-in", "option_type": "call", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 50, "direction": "in", "barrier_type": "down"}},
            {"name": "barrier_down_in_put", "product_type": "Barrier down-and-in", "option_type": "put", "strike": 100, "avg_price": 1.5, "misc": {"barrier": 50, "direction": "in", "barrier_type": "down"}},
            {"name": "lookback_floating", "product_type": "Lookback floating", "option_type": "call", "strike": 100, "avg_price": 2.0, "misc": {"closing_prices": [100, 110, 95]}},
            {"name": "lookback_fixed", "product_type": "Lookback fixed", "option_type": "call", "strike": 100, "avg_price": 2.0, "misc": {"closing_prices": [100, 110, 95]}},
            {"name": "straddle", "product_type": "Straddle", "option_type": "call", "strike": 100, "avg_price": 4.0},
            {"name": "straddle_put", "product_type": "Straddle", "option_type": "put", "strike": 100, "avg_price": 4.0},
            {"name": "strangle", "product_type": "Strangle", "option_type": "call", "strike": 95, "strike2": 105, "avg_price": 3.0},
            {"name": "strangle_put", "product_type": "Strangle", "option_type": "put", "strike": 95, "strike2": 105, "avg_price": 3.0},
            {"name": "call_spread", "product_type": "Call spread", "option_type": "call", "strike": 100, "strike2": 110, "avg_price": 2.0},
            {"name": "put_spread", "product_type": "Put spread", "option_type": "put", "strike": 110, "strike2": 100, "avg_price": 2.5},
            {"name": "butterfly", "product_type": "Butterfly", "option_type": "call", "strike": 100, "avg_price": 1.0, "legs": [
                {"option_type": "call", "strike": 90, "qty": 1, "side": "long"},
                {"option_type": "call", "strike": 100, "qty": 2, "side": "short"},
                {"option_type": "call", "strike": 110, "qty": 1, "side": "long"},
            ]},
            {"name": "condor", "product_type": "Condor", "option_type": "call", "strike": 100, "avg_price": 1.0, "legs": [
                {"option_type": "call", "strike": 90, "qty": 1, "side": "long"},
                {"option_type": "call", "strike": 100, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 110, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 120, "qty": 1, "side": "long"},
            ]},
            {"name": "iron_condor", "product_type": "Iron Condor", "option_type": "call", "strike": 100, "avg_price": 1.0, "legs": [
                {"option_type": "put", "strike": 90, "qty": 1, "side": "long"},
                {"option_type": "put", "strike": 95, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 105, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 110, "qty": 1, "side": "long"},
            ]},
            {"name": "iron_butterfly", "product_type": "Iron Butterfly", "option_type": "call", "strike": 100, "avg_price": 1.0, "legs": [
                {"option_type": "put", "strike": 95, "qty": 1, "side": "long"},
                {"option_type": "put", "strike": 100, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 100, "qty": 1, "side": "short"},
                {"option_type": "call", "strike": 105, "qty": 1, "side": "long"},
            ]},
            {"name": "diagonal_spread", "product_type": "Diagonal spread", "option_type": "call", "strike": 100, "avg_price": 1.5, "legs": [
                {"option_type": "call", "strike": 100, "qty": 1, "side": "long"},
                {"option_type": "call", "strike": 110, "qty": 1, "side": "short"},
            ]},
            {"name": "chooser", "product_type": "Chooser", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "chooser_put", "product_type": "Chooser", "option_type": "put", "strike": 100, "avg_price": 2.0},
            {"name": "forward_start", "product_type": "Forward-start", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "forward_start_put", "product_type": "Forward-start", "option_type": "put", "strike": 100, "avg_price": 2.0},
            {"name": "cliquet", "product_type": "Cliquet / Ratchet", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "quanto", "product_type": "Quanto", "option_type": "call", "strike": 100, "avg_price": 2.0},
            {"name": "rainbow", "product_type": "Rainbow", "option_type": "call", "strike": 100, "avg_price": 2.0},
        ]

        for case in base_cases:
            for side in ["long", "short"]:
                payload = deepcopy(case)
                payload["id"] = f"{case['name']}_{side}"
                payload["quantity"] = 1
                payload["side"] = side
                self._add(payload)

        book = self._book()
        self.assertEqual(len(book), len(base_cases) * 2)

        scenarios = ["atm", "itm", "otm"]
        # Trace détaillée pour supervision
        seen_types = set()
        seen_sections = defaultdict(set)
        expired_payloads = {}
        for opt_id, opt in sorted(book.items(), key=lambda kv: kv[0]):
            rng = random.Random(opt_id)
            chosen_scen = rng.choice(scenarios)
            base_name = str(opt.get("product_type") or opt.get("product") or opt.get("structure") or opt_id)
            opt_type = str(opt.get("option_type") or opt.get("type") or "").lower()
            side = (opt.get("side") or "long").lower()

            if base_name not in seen_types:
                if seen_types:
                    print("\n\n\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                seen_types.add(base_name)
                print("\n==============================")
                print(f"========== {base_name.upper()} ==========")
                print("==============================")
            # Affichage regroupé : bloc côté/option type avec sous-lignes OPEN/EXPIRATION
            label_type = opt_type.upper() if opt_type else "N/A"
            block_title = f"{side.upper()} / {label_type}"
            print(f"\n-- {block_title} --")
            spots = self._spots(opt)
            strike_base = float(opt.get("strike") or 1.0)
            strike2_val = opt.get("strike2")
            if strike2_val not in (None, ""):
                try:
                    strike_base = (strike_base + float(strike2_val)) / 2.0
                except Exception:
                    pass

            def moneyness(val: float) -> float:
                return val / strike_base if strike_base else 0.0

            # Opening snapshot (T%=100, mark = avg_price pour PnL nul à l'entrée) sur ATM/ITM/OTM
            print("---- OPENING ----")
            opening_mark = opt.get("avg_price")
            scen = chosen_scen
            open_spot = spots[scen]
            with self.subTest(option=opt_id, phase="opening", scenario=scen):
                open_res = self._pnl(opt, spot=open_spot, mark=opening_mark)
                self.assertAlmostEqual(open_res["pnl_total"], 0.0, places=6)
                print(f"   OPEN   [{scen.upper()}] price={opening_mark}")

            print("---- EXPIRATION ----")

            scen = chosen_scen
            with self.subTest(option=opt_id, phase="expiration", scenario=scen):
                spot = spots[scen]
                res = self._pnl(opt, spot=spot)
                expected = self._expected_pnl(opt, spot=spot)
                self.assertAlmostEqual(res["pnl_total"], expected["pnl_total"], places=6)
                self.assertAlmostEqual(res["pnl_per_unit"], expected["pnl_per_unit"], places=6)
                print(f"   EXPIRE [{scen.upper()}] payoff={res['payoff_per_unit']:.4f} (expired)")

                exp_entry = dict(opt)
                exp_entry.update(res)
                exp_entry.update(
                    {
                        "status": "expired",
                        "underlying_close": spot,
                        "closed_at": "T=0",
                        "event": "expiration",
                        "scenario": scen,
                    }
                )
                expired_payloads[opt_id] = exp_entry

        self._save_expired(expired_payloads)
        loaded_expired = self._expired()
        self.assertEqual(len(loaded_expired), len(expired_payloads))
