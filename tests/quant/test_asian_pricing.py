"""D2.a / D2.b — Asian option pricers (app/model/options/core/pricing_lib.py).

Two distinct defects:

* `price_asian_geom` adjusts the volatility (sigma/sqrt(3)) but leaves r and q
  untouched. The Kemna-Vorst closed form also requires an adjusted drift
  b_G = (r - q - sigma^2/6)/2, so that the forward of the geometric average is
  S*exp(b_G*T) instead of S*exp((r-q)*T). Missing it overprices a call at every
  parameter set, INCLUDING r = q = 0 (there b_G = -sigma^2/12, not 0).

* `price_asian_arith_approx` claims a "Turnbull-Wakeman-esque" approximation but
  its body is character-for-character the same as `price_asian_geom`: same
  sigma/sqrt(3), no moment matching. So the arithmetic and geometric Asians
  return exactly the same price.

ORACLES — independent of the pricing algebra under test:

1. MODEL-FREE, EXACT: the arithmetic mean of a positive path is >= its geometric
   mean (AM-GM), strictly unless the path is constant. Hence
       asian_arith_call > asian_geom_call
   for any non-degenerate parameter set. This is the assertion that fails today,
   by exact equality.
2. MODEL-FREE, EXACT: averaging can only reduce dispersion, so both Asians are
   worth strictly less than the vanilla European call of the same strike.
3. QUANTITATIVE: a Monte-Carlo on the *actual* payoff — the average is
   accumulated along the simulated path (trapezoid rule over [0, T]) and the
   discounted payoff is averaged. It never touches the closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.options.core.pricing_lib import (
    bs_price_call,
    price_asian_arith_approx,
    price_asian_geom,
)

pytestmark = pytest.mark.unit


def _mc_asian_averages(S, r, q, sigma, T, n_steps=400, n_paths=100_000, seed=7):
    """Simulate GBM and return (arithmetic, geometric) continuous averages per path.

    The average over [0, T] is approximated by the trapezoid rule, whose error is
    O(1/n_steps^2) — an order of magnitude below the Monte-Carlo noise here.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    s = np.full(n_paths, float(S))
    # trapezoid: endpoints carry half weight
    acc_lin = 0.5 * s.copy()
    acc_log = 0.5 * np.log(s)
    for step in range(1, n_steps + 1):
        s = s * np.exp(drift + vol * rng.standard_normal(n_paths))
        w = 0.5 if step == n_steps else 1.0
        acc_lin += w * s
        acc_log += w * np.log(s)
    return acc_lin / n_steps, np.exp(acc_log / n_steps)


def _mc_price(avg, K, r, T):
    payoff = np.maximum(avg - K, 0.0) * math.exp(-r * T)
    return float(np.mean(payoff)), float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))


# Parameter sets chosen so the drift term actually matters (r != q, and one with r = q = 0
# where the geometric drift adjustment is still non-zero: b_G = -sigma^2/12).
CASES = [
    # (S, K, T, sigma, r, q)
    (100.0, 100.0, 1.0, 0.30, 0.05, 0.02),
    (100.0, 90.0, 1.0, 0.30, 0.05, 0.02),
    (100.0, 110.0, 2.0, 0.25, 0.08, 0.00),
    (100.0, 100.0, 1.0, 0.20, 0.00, 0.00),
]
_IDS = [f"S{s:g}-K{k:g}-T{t:g}-sig{v:g}-r{r:g}-q{q:g}" for s, k, t, v, r, q in CASES]


# --------------------------------------------------------------------------- #
# Oracle 1 — AM-GM: the arithmetic Asian strictly dominates the geometric one.  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_arithmetic_asian_strictly_dominates_geometric(S, K, T, sigma, r, q):
    arith = price_asian_arith_approx(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    geom = price_asian_geom(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    assert arith > geom, f"arith={arith!r} geom={geom!r} — AM-GM violated"


# --------------------------------------------------------------------------- #
# Oracle 2 — averaging reduces dispersion: both Asians < vanilla European.      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_both_asians_are_cheaper_than_the_vanilla_call(S, K, T, sigma, r, q):
    vanilla = bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T)
    arith = price_asian_arith_approx(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    geom = price_asian_geom(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    assert geom < arith < vanilla, (geom, arith, vanilla)


# --------------------------------------------------------------------------- #
# Oracle 3 — Monte-Carlo on the actual payoff.                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_geometric_asian_matches_monte_carlo(S, K, T, sigma, r, q):
    _, geo_avg = _mc_asian_averages(S, r, q, sigma, T)
    mc, stderr = _mc_price(geo_avg, K, r, T)
    got = price_asian_geom(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    assert got == pytest.approx(mc, abs=3.0 * stderr), (
        f"closed form {got:.4f} vs MC {mc:.4f} +/- {stderr:.4f}"
    )


@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_arithmetic_asian_matches_monte_carlo(S, K, T, sigma, r, q):
    arith_avg, _ = _mc_asian_averages(S, r, q, sigma, T)
    mc, stderr = _mc_price(arith_avg, K, r, T)
    got = price_asian_arith_approx(S, K, T=T, sigma=sigma, r=r, q=q, option_type="call")
    # Turnbull-Wakeman is a two-moment approximation, not an identity: allow its
    # documented approximation error (~0.5% of the price) on top of the MC noise.
    tol = 3.0 * stderr + 0.005 * mc
    assert got == pytest.approx(mc, abs=tol), (
        f"closed form {got:.4f} vs MC {mc:.4f} +/- {stderr:.4f} (tol {tol:.4f})"
    )


# --------------------------------------------------------------------------- #
# Put side — parity-free sanity: a put must stay non-negative and below its      #
# discounted strike, and both Asians must again be ordered.                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_asian_puts_are_bounded_and_ordered(S, K, T, sigma, r, q):
    arith = price_asian_arith_approx(S, K, T=T, sigma=sigma, r=r, q=q, option_type="put")
    geom = price_asian_geom(S, K, T=T, sigma=sigma, r=r, q=q, option_type="put")
    assert 0.0 <= arith <= K * math.exp(-r * T)
    assert 0.0 <= geom <= K * math.exp(-r * T)
    # AM >= GM makes the arithmetic *put* the cheaper of the two.
    assert arith < geom, f"arith={arith!r} geom={geom!r}"
