"""
Edge-case / limiting-behaviour tests for exotic option pricers.

Target: app/model/options/core/pricing_lib.py and app/model/options/core/payoff.py
(the deterministic payoff/closed-form layer used by the exotic dashboard views).

ORACLES are independent of the implementation under test:
  * digital cash-or-nothing call price == payout * e^{-rT} * N(d2)   (Hull, closed form)
  * digital put-call parity:  C_digital + P_digital == payout * e^{-rT}  (q=0)
  * Asian payoff with a single fixing == vanilla European payoff at that level
  * barrier far away (knock-out, never breached) ~ vanilla payoff
  * butterfly / condor terminal payoff >= 0 and bounded by the wing width
  * digital no-arbitrage bounds: 0 <= price <= payout * e^{-rT}

Network is disabled (pyproject --disable-socket); none of these tests touch IO/network.
All deterministic -> marked unit; the single MC convergence check is marked slow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.options.core import pricing_lib as L
from app.model.options.core import payoff as P


# ---- independent analytic helpers (NOT imported from the code under test) ---- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d2(S, K, r, q, sigma, T):
    return (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_digital_cash_or_nothing_call_equals_discounted_nd2():
    """Cash-or-nothing call == payout * e^{-rT} * N(d2)  (literature closed form)."""
    S, K, r, q, sigma, T, payout = 100.0, 95.0, 0.03, 0.0, 0.25, 0.75, 1.0
    oracle = payout * math.exp(-r * T) * _norm_cdf(_d2(S, K, r, q, sigma, T))
    got = L.price_digital_bs(
        S, K, T=T, r=r, q=q, sigma=sigma, option_type="call", payout=payout
    )
    assert got == pytest.approx(oracle, rel=1e-12, abs=1e-12)


@pytest.mark.unit
def test_digital_cash_or_nothing_put_equals_discounted_nminus_d2():
    """Cash-or-nothing put == payout * e^{-rT} * N(-d2)."""
    S, K, r, q, sigma, T, payout = 100.0, 110.0, 0.03, 0.0, 0.25, 0.75, 2.5
    oracle = payout * math.exp(-r * T) * _norm_cdf(-_d2(S, K, r, q, sigma, T))
    got = L.price_digital_bs(
        S, K, T=T, r=r, q=q, sigma=sigma, option_type="put", payout=payout
    )
    assert got == pytest.approx(oracle, rel=1e-12, abs=1e-12)


@pytest.mark.unit
def test_digital_put_call_parity_sums_to_discounted_payout():
    """C_digital + P_digital == payout * e^{-rT} (the two indicators partition the space, q=0)."""
    S, K, r, q, sigma, T, payout = 100.0, 100.0, 0.04, 0.0, 0.3, 1.0, 3.0
    c = L.price_digital_bs(S, K, T=T, r=r, q=q, sigma=sigma, option_type="call", payout=payout)
    p = L.price_digital_bs(S, K, T=T, r=r, q=q, sigma=sigma, option_type="put", payout=payout)
    assert c + p == pytest.approx(payout * math.exp(-r * T), rel=1e-12, abs=1e-12)


@pytest.mark.unit
def test_digital_price_within_no_arbitrage_bounds():
    """0 <= digital price <= payout * e^{-rT} for any reasonable spot/strike."""
    r, q, sigma, T, payout = 0.03, 0.0, 0.2, 1.0, 1.0
    upper = payout * math.exp(-r * T)
    for S in (50.0, 90.0, 100.0, 130.0, 200.0):
        for opt in ("call", "put"):
            price = L.price_digital_bs(
                S, 100.0, T=T, r=r, q=q, sigma=sigma, option_type=opt, payout=payout
            )
            assert -1e-12 <= price <= upper + 1e-12


@pytest.mark.unit
def test_asian_single_fixing_equals_vanilla_european_payoff():
    """An arithmetic Asian with a single fixing reduces to a vanilla European payoff.

    Oracle: max(S - K, 0) for a call computed independently of payoff_asian.
    """
    K = 100.0
    for S in (80.0, 100.0, 120.0):
        opt_call = {"product": "asian", "option_type": "call", "strike": K, "side": "long"}
        opt_put = {"product": "asian", "option_type": "put", "strike": K, "side": "long"}
        # single fixing == average is just that one price
        assert P.payoff_asian(opt_call, [S]) == pytest.approx(max(S - K, 0.0), abs=1e-12)
        assert P.payoff_asian(opt_put, [S]) == pytest.approx(max(K - S, 0.0), abs=1e-12)


@pytest.mark.unit
def test_barrier_far_knockout_equals_vanilla_payoff_path():
    """Up-and-out call with an unreachable barrier (never breached) == vanilla payoff.

    payoff_barrier(option, prices, final_spot): if knock=out and the barrier is never
    hit, payoff must equal the vanilla terminal payoff. Independent oracle = max(ST-K,0).
    """
    K = 100.0
    barrier = 10_000.0  # so high it can never be touched
    final_spot = 130.0
    path = [95.0, 102.0, 118.0, final_spot]  # all well below the barrier
    option = {
        "product_type": "barrier",
        "option_type": "call",
        "strike": K,
        "side": "long",
        "misc": {"barrier": barrier, "barrier_type": "up", "knock": "out"},
    }
    got = P.payoff_barrier(option, path, final_spot)
    assert got == pytest.approx(max(final_spot - K, 0.0), abs=1e-12)


@pytest.mark.unit
def test_barrier_knockout_hit_pays_zero():
    """Up-and-out: if the path breaches the barrier, knock-out pays exactly 0."""
    option = {
        "product_type": "barrier",
        "option_type": "call",
        "strike": 100.0,
        "side": "long",
        "misc": {"barrier": 120.0, "barrier_type": "up", "knock": "out"},
    }
    path = [95.0, 125.0, 130.0]  # breaches 120 -> knocked out
    assert P.payoff_barrier(option, path, 130.0) == 0.0


@pytest.mark.unit
def test_barrier_in_out_parity_equals_vanilla():
    """Knock-in payoff + knock-out payoff == vanilla payoff (same path), an exact identity."""
    K, final_spot = 100.0, 135.0
    path = [95.0, 125.0, final_spot]
    base_misc = {"barrier": 120.0, "barrier_type": "up"}
    opt_in = {
        "product_type": "barrier", "option_type": "call", "strike": K, "side": "long",
        "misc": {**base_misc, "knock": "in"},
    }
    opt_out = {
        "product_type": "barrier", "option_type": "call", "strike": K, "side": "long",
        "misc": {**base_misc, "knock": "out"},
    }
    in_payoff = P.payoff_barrier(opt_in, path, final_spot)
    out_payoff = P.payoff_barrier(opt_out, path, final_spot)
    assert in_payoff + out_payoff == pytest.approx(max(final_spot - K, 0.0), abs=1e-12)


@pytest.mark.unit
def test_long_call_butterfly_payoff_nonnegative_and_bounded():
    """Symmetric long call butterfly payoff in [0, wing] for all spots.

    Oracle (independent): peak payoff of a symmetric butterfly K1<K2<K3 (K2 centred)
    is exactly (K2 - K1) = (K3 - K2), reached at S == K2; floor is 0 everywhere.
    """
    k1, k2, k3 = 90.0, 100.0, 110.0
    wing = k2 - k1
    s_grid = np.linspace(50.0, 160.0, 1101)
    pf = L.payoff_butterfly(s_grid, k1, k2, k3)
    assert np.all(pf >= -1e-9)
    assert np.max(pf) <= wing + 1e-9
    # peak at the centre strike equals the wing width
    assert L.payoff_butterfly(np.array([k2]), k1, k2, k3)[0] == pytest.approx(wing, abs=1e-9)
    # deep ITM / OTM tails are worthless
    assert L.payoff_butterfly(np.array([50.0]), k1, k2, k3)[0] == pytest.approx(0.0, abs=1e-9)
    assert L.payoff_butterfly(np.array([160.0]), k1, k2, k3)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_long_call_condor_payoff_nonnegative_and_bounded():
    """Long call condor K1<K2<K3<K4 payoff in [0, (K2-K1)] for all spots.

    Oracle: max payoff of a symmetric long condor is the lower-wing width (K2 - K1),
    held flat on the plateau [K2, K3]; never negative.
    """
    k1, k2, k3, k4 = 80.0, 90.0, 110.0, 120.0
    plateau = k2 - k1
    s_grid = np.linspace(40.0, 170.0, 1301)
    pf = L.payoff_condor(s_grid, k1, k2, k3, k4)
    assert np.all(pf >= -1e-9)
    assert np.max(pf) <= plateau + 1e-9
    # mid-plateau value equals the wing width
    mid = 0.5 * (k2 + k3)
    assert L.payoff_condor(np.array([mid]), k1, k2, k3, k4)[0] == pytest.approx(plateau, abs=1e-9)
    # tails worthless
    assert L.payoff_condor(np.array([40.0]), k1, k2, k3, k4)[0] == pytest.approx(0.0, abs=1e-9)
    assert L.payoff_condor(np.array([170.0]), k1, k2, k3, k4)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_digital_call_increasing_in_spot():
    """Cash-or-nothing call price is non-decreasing in the underlying spot (monotonicity)."""
    r, q, sigma, T = 0.02, 0.0, 0.2, 1.0
    spots = [70.0, 85.0, 100.0, 115.0, 130.0]
    prices = [
        L.price_digital_bs(S, 100.0, T=T, r=r, q=q, sigma=sigma, option_type="call")
        for S in spots
    ]
    assert all(b >= a - 1e-12 for a, b in zip(prices, prices[1:]))


@pytest.mark.slow
def test_barrier_mc_far_barrier_converges_to_vanilla_call():
    """Down-and-out call with an unreachable (very low) barrier ~ vanilla BS call.

    view_barrier prices the premium by Monte Carlo. With a barrier far below spot it
    is (almost surely) never knocked out, so the premium must match the closed-form
    vanilla call. Oracle = bs_price_call (independent closed form).
    """
    S, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.0, 0.2, 1.0
    view = L.view_barrier(
        s0=S,
        strike=K,
        barrier=1.0,            # far below spot -> down-and-out never triggers
        direction="down",
        knock="out",
        option_type="call",
        n_paths=60_000,
        n_steps=200,
        seed=7,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
    )
    mc_premium = view["premium"]
    vanilla = L.bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T)
    assert mc_premium == pytest.approx(vanilla, rel=0.03)
