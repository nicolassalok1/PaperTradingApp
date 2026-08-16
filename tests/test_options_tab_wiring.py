"""Wiring contracts of the Options tab — the seams a review found silently broken.

Each test names the user-visible symptom it guards. No network: the close history
loader is patched, the IV surface is an in-memory frame, the tree is tiny.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

# Importing the bridge triggers _bootstrap_fake_streamlit() -> st is stubbed for bare runs.
import app.vue.components.options.controller_bridge as cb
from app.model.options import context as opt_context
from app.model.options.core import pricing_lib as pl
from app.model.options.engines import tree as crr_tree
from app.vue.state import options_context

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PANELS_DIR = REPO_ROOT / "app" / "vue" / "components" / "options" / "panels"


@pytest.fixture(autouse=True)
def _clean_session():
    for k in ("tkr_common", "common_underlying", "common_spot_value", "opt_iv_surface_df"):
        cb.st.session_state.pop(k, None)
    memo_key = getattr(options_context, "_CTX_MEMO_KEY", "_opt_ctx_memo")
    cb.st.session_state.pop(memo_key, None)
    yield
    cb.st.session_state.pop(memo_key, None)


def _patch_history(monkeypatch, closes):
    df = pd.DataFrame(
        {"Date": pd.date_range("2025-01-01", periods=len(closes), freq="D"), "Close": closes}
    )
    calls = {"n": 0}

    def _hist(ticker, period="2y", interval="1d"):
        calls["n"] += 1
        return df, None, None

    monkeypatch.setattr(opt_context, "load_or_fetch_closing_history", _hist)
    monkeypatch.setattr(opt_context, "fetch_spot_price", lambda tk: None)
    return calls


# --- 1. the spot follows the ticker, not a stale shared key -------------------


def test_option_context_spot_is_the_last_close_not_the_shared_default(monkeypatch):
    """Symptom: 'Spot actuel (AAPL) : 257.53' above panels priced at S0 = 100."""
    _patch_history(monkeypatch, [250.0, 257.53])
    cb.st.session_state["tkr_common"] = "AAPL"
    cb.st.session_state["common_spot_value"] = 100.0  # the router's own default

    ctx = options_context.get_option_context()

    assert ctx["ticker"] == "AAPL"
    assert ctx["close_available"] is True
    assert ctx["S0"] == pytest.approx(257.53)
    assert cb.current_spot(ctx) == pytest.approx(257.53)


def test_option_context_is_memoised_per_ticker_within_a_session(monkeypatch):
    """Symptom: 32 history loads (and, offline, 35 socket attempts) per rerun."""
    calls = _patch_history(monkeypatch, [10.0, 11.0])
    cb.st.session_state["tkr_common"] = "MSFT"

    first = options_context.get_option_context()
    second = options_context.get_option_context()

    assert calls["n"] == 1
    assert second["S0"] == first["S0"] == pytest.approx(11.0)

    cb.st.session_state["tkr_common"] = "AAPL"
    options_context.get_option_context()
    assert calls["n"] == 2  # a new ticker is a new load


def test_no_leaf_panel_reads_the_spot_from_session_state():
    """Every panel prices off the context spot; none off the mutable shared key."""
    offenders = []
    for path in sorted(PANELS_DIR.rglob("tab_*.py")):
        if path.parent == PANELS_DIR:
            continue  # family routers and shims
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "common_spot_value"
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


# --- 2. the cached IV actually reaches the panels ------------------------------


def test_bridge_iv_lookup_uses_the_loaded_surface():
    """Symptom: 'IV non trouvée dans le cache' even after 'Refresh IV surface'."""
    cb.st.session_state["opt_iv_surface_df"] = pd.DataFrame(
        {
            "K": [250.0, 260.0],
            "T": [1.0, 1.0],
            "iv": [0.31, 0.29],
            "type": ["call", "call"],
        }
    )
    assert cb._get_cached_iv_for(257.53, 1.0, "call") == pytest.approx(0.29)
    # A put has no row -> nothing to use, and no ticker to fall back on.
    assert cb._get_cached_iv_for(257.53, 1.0, "put") is None


# --- 3. deterministic Monte Carlo ---------------------------------------------


def test_rainbow_price_is_stable_across_reruns():
    """Symptom: the rainbow premium flickers on every slider move."""
    kw = dict(T=1.0, sigma1=0.2, sigma2=0.25, r=0.02, q=0.0, rho=0.3, n_paths=5_000)
    a = pl.price_rainbow_mc(100.0, 100.0, 100.0, **kw)
    b = pl.price_rainbow_mc(100.0, 100.0, 100.0, **kw)
    assert a == b
    assert pl.price_rainbow_mc(100.0, 100.0, 100.0, seed=7, **kw) != a


# --- 4. the drawn CRR tree carries the dividend yield --------------------------


class _Call:
    def __init__(self, s0, k, T):
        self.s0, self.K, self.T = s0, k, T

    def payoff(self, s):
        import numpy as np

        return np.maximum(np.asarray(s, dtype=float) - self.K, 0.0)


def test_build_crr_tree_honours_dividend_yield():
    """Symptom: tree root value != displayed premium as soon as q != 0."""
    opt = _Call(100.0, 100.0, 1.0)
    _, v0 = crr_tree.build_crr_tree(opt, r=0.05, sigma=0.2, n_steps=30)
    _, vq = crr_tree.build_crr_tree(opt, r=0.05, q=0.05, sigma=0.2, n_steps=30)
    assert vq[0, 0] < v0[0, 0]  # a dividend lowers a call
