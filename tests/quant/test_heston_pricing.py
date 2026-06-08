"""Independent-oracle tests for the Heston CF pricer and BS implied-vol inverter.

Oracles are INDEPENDENT of the Heston code under test:
  * Black-Scholes closed form (bs_call_price) is itself validated against
    known analytic values / put-call parity before being used as a reference.
  * Heston is validated against BS only in the degenerate limit sigma->0,
    v0 == theta (constant variance), where Heston *must* reduce to BS with
    sigma_bs = sqrt(v0). This is a textbook limit, not "Heston by Heston".
  * Structural invariants: positivity, no-arbitrage bounds, put-call parity
    on the CF call, monotonicity in strike.
  * The IV inverter is checked to round-trip sigma on a synthetic BS surface
    (BS price -> implied_vol_call -> sigma), independent of Heston entirely.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.calibration.heston_pricer import (
    call_price_cf,
    price_grid_from_params,
)
from app.model.calibration.implied_vol import bs_call_price, implied_vol_call

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Reference Black-Scholes implementation, fully independent of bs_call_price.  #
# Used to validate bs_call_price itself before it serves as an oracle.        #
# --------------------------------------------------------------------------- #
def _ref_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ref_bs_call(S0, K, t, r, q, vol):
    sqrt_t = math.sqrt(t)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return S0 * math.exp(-q * t) * _ref_norm_cdf(d1) - K * math.exp(-r * t) * _ref_norm_cdf(d2)


# --------------------------------------------------------------------------- #
# 1. Validate the BS oracle itself against known analytic facts.              #
# --------------------------------------------------------------------------- #
def test_bs_call_matches_independent_reference():
    # Independent recomputation of the BS formula.
    val = bs_call_price(100.0, 100.0, 1.0, 0.03, 0.01, 0.2)
    ref = _ref_bs_call(100.0, 100.0, 1.0, 0.03, 0.01, 0.2)
    assert val == pytest.approx(ref, rel=1e-12)


def test_bs_atm_zero_rate_known_value():
    # ATM forward, r=q=0: C = S0 * (2 N(sigma*sqrt(t)/2) - 1).
    S0, vol, t = 100.0, 0.2, 1.0
    expected = S0 * (2.0 * _ref_norm_cdf(vol * math.sqrt(t) / 2.0) - 1.0)
    val = bs_call_price(S0, S0, t, 0.0, 0.0, vol)
    assert val == pytest.approx(expected, rel=1e-10)


def test_bs_deep_itm_approaches_discounted_intrinsic():
    # Very low vol, deep ITM: call ~ S0 e^{-qt} - K e^{-rt}.
    S0, K, t, r, q, vol = 100.0, 10.0, 0.5, 0.02, 0.0, 1e-3
    intrinsic = S0 * math.exp(-q * t) - K * math.exp(-r * t)
    assert bs_call_price(S0, K, t, r, q, vol) == pytest.approx(intrinsic, abs=1e-4)


def test_bs_guard_clauses_return_zero():
    assert bs_call_price(100.0, 100.0, 0.0, 0.03, 0.0, 0.2) == 0.0
    assert bs_call_price(100.0, 100.0, 1.0, 0.03, 0.0, 0.0) == 0.0
    assert bs_call_price(0.0, 100.0, 1.0, 0.03, 0.0, 0.2) == 0.0


# --------------------------------------------------------------------------- #
# 2. Heston CF -> BS limit (sigma -> 0, v0 == theta).                          #
# --------------------------------------------------------------------------- #
# params order = (kappa, theta, sigma, rho, v0)
#
# XFAIL (real application bug, NOT a tolerance issue): in the degenerate limit
# sigma->0, v0==theta the Heston CF call MUST converge to BS(sqrt(v0)). It does
# not. Measured errors are large and DO NOT shrink with integration resolution:
#   (u_max,N)=(200,4000): K=80 +1.15, K=100 +0.21, K=120 -0.84 (v0=0.04)
#   (u_max,N)=(500,20000): K=80 +1.19, K=100 +0.22, K=120 -0.82  -> identical.
# A persistent, resolution-independent bias => bug in the pricer formulation
# (heston_cf / _P integrand), not numerical truncation. Code left untouched
# per scope. Oracle (bs_call_price) is independently validated above.
@pytest.mark.xfail(
    reason="heston_pricer bug: CF call does not converge to BS in sigma->0,v0=theta "
    "limit; bias is resolution-independent (see diagnostics in test header).",
    strict=True,
)
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("v0", [0.04, 0.09])
def test_heston_reduces_to_bs_in_zero_vol_of_vol_limit(K, v0):
    S0, t, r, q = 100.0, 1.0, 0.03, 0.01
    sigma_bs = math.sqrt(v0)
    # sigma (vol-of-vol) -> 0, v0 == theta => variance is (essentially) constant.
    params = (1.5, v0, 1e-6, -0.3, v0)
    heston = call_price_cf(S0, K, t, r, q, params, u_max=200.0, N=4000)
    bs = bs_call_price(S0, K, t, r, q, sigma_bs)
    # Independent oracle: BS price. Tolerance accounts for finite integration.
    assert heston == pytest.approx(bs, abs=2e-2, rel=1e-2)


@pytest.mark.xfail(
    reason="heston_pricer bug: even ATM in the sigma->0 limit the CF call misses "
    "BS by ~0.2 (resolution-independent). See test_heston_reduces_to_bs header.",
    strict=True,
)
def test_heston_bs_limit_atm_tight():
    # ATM is the most numerically forgiving; demand a tighter match.
    S0, K, t, r, q = 100.0, 100.0, 0.75, 0.0, 0.0
    v0 = 0.04
    params = (2.0, v0, 1e-7, 0.0, v0)
    heston = call_price_cf(S0, K, t, r, q, params, u_max=300.0, N=6000)
    bs = bs_call_price(S0, K, t, r, q, math.sqrt(v0))
    assert heston == pytest.approx(bs, abs=5e-3)


# --------------------------------------------------------------------------- #
# 3. Structural invariants on the Heston CF call.                              #
# --------------------------------------------------------------------------- #
def _heston_params():
    # A generic, well-behaved (Feller-satisfying) Heston parameter set.
    # 2*kappa*theta = 2*2*0.06 = 0.24 > sigma^2 = 0.09.
    return (2.0, 0.06, 0.3, -0.5, 0.05)


@pytest.mark.parametrize("K", [70.0, 90.0, 100.0, 110.0, 130.0])
def test_heston_call_positive(K):
    price = call_price_cf(100.0, K, 1.0, 0.03, 0.01, _heston_params())
    assert price > 0.0


@pytest.mark.parametrize("K", [70.0, 90.0, 100.0, 110.0, 130.0])
def test_heston_call_within_no_arbitrage_bounds(K):
    S0, t, r, q = 100.0, 1.0, 0.03, 0.01
    price = call_price_cf(S0, K, t, r, q, _heston_params())
    lower = max(S0 * math.exp(-q * t) - K * math.exp(-r * t), 0.0)
    upper = S0 * math.exp(-q * t)
    assert price >= lower - 1e-6
    assert price <= upper + 1e-6


def test_heston_call_monotone_decreasing_in_strike():
    S0, t, r, q = 100.0, 1.0, 0.03, 0.01
    params = _heston_params()
    strikes = np.linspace(70.0, 130.0, 13)
    prices = [call_price_cf(S0, float(K), t, r, q, params) for K in strikes]
    diffs = np.diff(prices)
    # Calls are non-increasing in strike (allow tiny numerical noise).
    assert np.all(diffs <= 1e-6)


def test_heston_call_put_parity():
    # Build the European put from the CF call and check parity:
    #   C - P = S0 e^{-qt} - K e^{-rt}.
    # The put is recovered as P = C - (S0 e^{-qt} - K e^{-rt}); the test is
    # that this implied put is itself arbitrage-consistent (non-negative and
    # >= discounted intrinsic put value), i.e. the call respects parity bounds.
    S0, K, t, r, q = 100.0, 110.0, 1.0, 0.02, 0.01
    C = call_price_cf(S0, K, t, r, q, _heston_params())
    forward_term = S0 * math.exp(-q * t) - K * math.exp(-r * t)
    P = C - forward_term
    put_intrinsic = max(K * math.exp(-r * t) - S0 * math.exp(-q * t), 0.0)
    assert P >= put_intrinsic - 1e-6
    assert P >= -1e-6


def test_heston_call_guard_clauses_return_zero():
    params = _heston_params()
    assert call_price_cf(100.0, 100.0, 0.0, 0.03, 0.0, params) == 0.0
    assert call_price_cf(0.0, 100.0, 1.0, 0.03, 0.0, params) == 0.0
    assert call_price_cf(100.0, 0.0, 1.0, 0.03, 0.0, params) == 0.0


# --------------------------------------------------------------------------- #
# 4. price_grid_from_params shape / content.                                   #
# --------------------------------------------------------------------------- #
def test_price_grid_shape_and_monotone_in_moneyness():
    S0, r, q = 100.0, 0.03, 0.01
    m_grid = [0.8, 1.0, 1.2]
    t_grid = [0.25, 1.0]
    grid = price_grid_from_params(S0, m_grid, t_grid, r, q, _heston_params())
    assert grid.shape == (len(t_grid), len(m_grid))
    # Within a maturity, price decreases with moneyness (strike up => call down).
    for i in range(grid.shape[0]):
        assert np.all(np.diff(grid[i]) <= 1e-6)


@pytest.mark.xfail(
    reason="heston_pricer bug: short-maturity OTM call price goes NEGATIVE, "
    "violating the no-arbitrage lower bound C>=0. With default integration "
    "(u_max=50,N=2000), params (2,0.06,0.3,-0.5,0.05), t=0.25, K=120 -> -0.166. "
    "Real pricing bug, code left untouched per scope.",
    strict=True,
)
def test_price_grid_short_maturity_otm_is_positive():
    S0, r, q = 100.0, 0.03, 0.01
    m_grid = [0.8, 1.0, 1.2]
    t_grid = [0.25, 1.0]
    grid = price_grid_from_params(S0, m_grid, t_grid, r, q, _heston_params())
    # No-arbitrage: every call price must be >= 0.
    assert np.all(grid > 0.0)


def test_price_grid_matches_pointwise_call_price():
    S0, r, q = 100.0, 0.03, 0.01
    m_grid = [0.9, 1.1]
    t_grid = [0.5]
    params = _heston_params()
    grid = price_grid_from_params(S0, m_grid, t_grid, r, q, params)
    for j, m in enumerate(m_grid):
        K = m * S0
        assert grid[0, j] == pytest.approx(
            call_price_cf(S0, K, t_grid[0], r, q, params), rel=1e-12
        )


# --------------------------------------------------------------------------- #
# 5. Implied-vol inverter round-trip on a synthetic BS surface.                #
#    Fully independent of Heston: prices come straight from bs_call_price.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60])
@pytest.mark.parametrize("m", [0.85, 1.0, 1.2])
@pytest.mark.parametrize("t", [0.25, 1.0])
def test_implied_vol_recovers_sigma_on_bs_surface(sigma, m, t):
    S0, r, q = 100.0, 0.03, 0.01
    K = m * S0
    price = bs_call_price(S0, K, t, r, q, sigma)
    iv = implied_vol_call(price, S0, K, t, r, q)
    assert not np.isnan(iv)
    assert iv == pytest.approx(sigma, abs=1e-3)


def test_implied_vol_returns_nan_outside_bounds():
    S0, K, t, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    upper = S0 * math.exp(-q * t)
    # Price above the discounted-spot upper bound: no valid vol.
    assert np.isnan(implied_vol_call(upper + 1.0, S0, K, t, r, q))
    # Price below intrinsic (here 0 for ATM) is also rejected.
    assert np.isnan(implied_vol_call(-1.0, S0, K, t, r, q))
    # Non-positive maturity guard.
    assert np.isnan(implied_vol_call(5.0, S0, K, 0.0, r, q))
