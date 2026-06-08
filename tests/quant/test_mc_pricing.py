"""
Tests for the Monte-Carlo option pricing engine.

Target: app/model/options/mc_engine.py
  - simulate_paths (wrapper over simulate_gbm_paths)
  - price_mc_european  -> {"price", "stderr"}
  - price_european_mc  -> float (backward-compat alias, BS only)
and the service-level entry point:
  - app.model.options.service.price_option_mc_unified (european branch)

ORACLES (all independent of the MC code under test):
  - Closed-form Black-Scholes (analytic, implemented inline with scipy.norm)
    used as the statistical reference for the discounted MC mean. The MC
    estimate must lie within a few standard errors (the engine reports its
    own stderr) of the analytic value.
  - Structural invariants that hold pathwise / sample-wise and therefore are
    exact under a fixed seed:
      * reproducibility (same seed -> bit-identical price),
      * put-call parity on the *same* simulated sample,
      * no-arbitrage bounds (0 <= call <= S0, max(0, S0-K e^{-rT}) <= call),
      * monotonicity of a call price in the strike,
      * deep-OTM option -> price ~ 0.
  - Convergence: stderr decreases ~ 1/sqrt(n_paths) (variance reduction by
    sample size). Marked slow.

Environment note: in the test env get_r(T) returns the constant DEFAULT_RF
(0.02, no curve cache, network disabled) and get_q(ticker) returns 0.0. The
oracle reads get_r() directly so it always uses the *same* r the engine uses,
without ever pinning a magic number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from app.model.options.mc_engine import (
    price_european_mc,
    price_mc_european,
    simulate_paths,
)
from app.model.options.service import price_option_mc_unified
from app.model.yieldcurve.rates_utils import get_q, get_r

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Independent analytic oracle: closed-form Black-Scholes (no dividend handling
# needed here since get_q == 0 in this engine path for ticker=None).
# ---------------------------------------------------------------------------
def bs_price(S0: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Analytic Black-Scholes price with continuous rate r, no dividend."""
    sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sqrtT
    d2 = d1 - sqrtT
    if kind.startswith("c"):
        return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# r/q as the engine itself resolves them (deterministic in the test env).
R = float(get_r(1.0))
Q = float(get_q("AAPL"))


def test_engine_env_assumptions():
    """Guard the deterministic-env assumptions the oracles rely on."""
    assert math.isfinite(R)
    assert Q == 0.0
    # r is maturity-flat in the no-curve fallback, which the oracle relies on.
    assert get_r(0.25) == pytest.approx(R)
    assert get_r(2.0) == pytest.approx(R)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(
    "S0, K, T, sigma",
    [
        (100.0, 100.0, 1.0, 0.2),  # ATM
        (100.0, 90.0, 1.0, 0.2),   # ITM call / OTM put
        (100.0, 110.0, 0.5, 0.3),  # OTM call / ITM put
    ],
)
def test_mc_european_matches_bsm_within_tolerance(S0, K, T, sigma, kind):
    """MC discounted mean ~ analytic BSM within a statistical band.

    The band is built from the engine's own reported stderr (4 sigma) plus a
    small absolute cushion. With a fixed seed this is deterministic.
    """
    res = price_mc_european(S0, K, T, sigma, option_type=kind, n_paths=60000, n_steps=64)
    analytic = bs_price(S0, K, T, R, sigma, kind)
    tol = 4.0 * res["stderr"] + 1e-3
    assert res["price"] == pytest.approx(analytic, abs=tol)
    assert res["stderr"] > 0.0


def test_reproducibility_same_seed():
    """Same inputs -> bit-identical price (fixed default seed=42)."""
    a = price_mc_european(100.0, 100.0, 1.0, 0.2, option_type="call", n_paths=20000, n_steps=50)
    b = price_mc_european(100.0, 100.0, 1.0, 0.2, option_type="call", n_paths=20000, n_steps=50)
    assert a["price"] == b["price"]
    assert a["stderr"] == b["stderr"]


def test_put_call_parity_on_same_sample():
    """Call - Put MC estimates satisfy parity exactly on the shared sample.

    price_mc_european reuses seed=42, so call and put are priced on identical
    terminal prices. Hence (C - P) must equal the discounted sample forward
    e^{-rT} (mean(S_T) - K) up to float round-off, independent of statistics.
    """
    S0, K, T, sigma, n_steps, n_paths = 100.0, 100.0, 1.0, 0.2, 64, 50000
    c = price_mc_european(S0, K, T, sigma, option_type="call", n_paths=n_paths, n_steps=n_steps)
    p = price_mc_european(S0, K, T, sigma, option_type="put", n_paths=n_paths, n_steps=n_steps)
    paths, _ = simulate_paths(S0, R, Q, sigma, T, n_steps, n_paths)
    sample_rhs = math.exp(-R * T) * (float(np.mean(paths[-1])) - K)
    assert (c["price"] - p["price"]) == pytest.approx(sample_rhs, abs=1e-8)


def test_no_arbitrage_bounds_call():
    """0 <= C <= S0 and C >= max(0, S0 - K e^{-rT}) within MC error."""
    S0, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    res = price_mc_european(S0, K, T, sigma, option_type="call", n_paths=60000, n_steps=64)
    lower = max(0.0, S0 - K * math.exp(-R * T))
    assert res["price"] <= S0 + 3.0 * res["stderr"]
    assert res["price"] >= -3.0 * res["stderr"]
    assert res["price"] >= lower - 3.0 * res["stderr"]


def test_call_decreasing_in_strike():
    """A European call price is monotonically decreasing in the strike."""
    S0, T, sigma = 100.0, 1.0, 0.2
    strikes = [80.0, 100.0, 120.0]
    prices = [
        price_mc_european(S0, K, T, sigma, option_type="call", n_paths=60000, n_steps=64)["price"]
        for K in strikes
    ]
    # Strict decrease with margin well beyond combined MC noise.
    assert prices[0] > prices[1] > prices[2]


def test_deep_otm_call_near_zero():
    """Very deep OTM call (K >> S0) prices close to zero."""
    res = price_mc_european(100.0, 400.0, 0.5, 0.2, option_type="call", n_paths=60000, n_steps=64)
    assert 0.0 <= res["price"] < 0.05


def test_unknown_option_type_defaults_to_call():
    """Engine coerces an unknown option_type to 'call' (documented behavior)."""
    weird = price_mc_european(100.0, 100.0, 1.0, 0.2, option_type="banana", n_paths=20000, n_steps=50)
    ref = price_mc_european(100.0, 100.0, 1.0, 0.2, option_type="call", n_paths=20000, n_steps=50)
    assert weird["price"] == ref["price"]


def test_missing_inputs_return_nan():
    """None spot or strike returns NaN (no crash)."""
    assert math.isnan(price_mc_european(None, 100.0, 1.0, 0.2, option_type="call"))
    assert math.isnan(price_mc_european(100.0, None, 1.0, 0.2, option_type="call"))


def test_price_european_mc_alias_matches_dict_price():
    """price_european_mc (float) == price_mc_european(...)['price'] for BS."""
    f = price_european_mc(100.0, 95.0, 1.0, 0.25, model="bs", n_paths=30000, n_steps=50)
    d = price_mc_european(100.0, 95.0, 1.0, 0.25, option_type="call", n_paths=30000, n_steps=50)
    assert f == d["price"]


def test_price_european_mc_unknown_model_raises():
    """Non-BS models are not implemented and must raise."""
    with pytest.raises(NotImplementedError):
        price_european_mc(100.0, 100.0, 1.0, 0.2, model="rheston")


@pytest.mark.slow
def test_unified_european_branch_matches_bsm():
    """service.price_option_mc_unified (european) ~ analytic BSM.

    Fixed: service.py now imports price_mc_european, so the european branch runs.
    """
    res = price_option_mc_unified(
        ticker=None,
        S0=100.0,
        K=100.0,
        T=1.0,
        sigma=0.2,
        option_type="call",
        style="european",
        n_paths=60000,
        n_steps=64,
    )
    analytic = bs_price(100.0, 100.0, 1.0, R, 0.2, "call")
    tol = 4.0 * res["stderr"] + 1e-3
    assert res["price"] == pytest.approx(analytic, abs=tol)


# ---------------------------------------------------------------------------
# Convergence: stderr shrinks roughly like 1/sqrt(n_paths). Heavier MC.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stderr_decreases_with_n_paths():
    S0, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    small = price_mc_european(S0, K, T, sigma, option_type="call", n_paths=10000, n_steps=64)
    large = price_mc_european(S0, K, T, sigma, option_type="call", n_paths=160000, n_steps=64)
    assert large["stderr"] < small["stderr"]
    # 16x paths -> ~4x smaller stderr; allow generous slack.
    ratio = small["stderr"] / large["stderr"]
    assert ratio > 2.5


@pytest.mark.slow
def test_convergence_to_bsm_large_n():
    """With many paths the MC price is very close to analytic BSM."""
    S0, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    res = price_mc_european(S0, K, T, sigma, option_type="call", n_paths=400000, n_steps=100)
    analytic = bs_price(S0, K, T, R, sigma, "call")
    assert res["price"] == pytest.approx(analytic, abs=4.0 * res["stderr"] + 1e-3)
