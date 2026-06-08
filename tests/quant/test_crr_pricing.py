"""Quant tests for the CRR binomial engine (app/model/options/engines/crr.py).

Independent oracles used (no self-validation):
- Closed-form Black-Scholes-Merton (app.model.options.engines.black_scholes.black_scholes_price),
  a completely separate code path. The CRR price of a European-style payoff must converge
  to BSM as the number of steps grows (Cox-Ross-Rubinstein convergence theorem).
- No-arbitrage invariants from the literature:
    * American call without dividends == European call (no early exercise is optimal;
      Merton 1973). With q=0 the CRR American call therefore equals its BSM European value.
    * American put >= European put (early exercise has positive value for puts).
    * Price >= intrinsic value (lower no-arbitrage bound for an American option).
    * Monotone, bounded convergence in the number of steps (price stays in a sensible band).
"""

import math

import pytest

from app.model.options.engines.black_scholes import black_scholes_price
from app.model.options.engines.crr import price_american_crr

pytestmark = pytest.mark.unit

# A representative, well-conditioned parameter set (no degeneracies).
S0 = 100.0
K = 100.0
R = 0.05
T = 1.0
SIGMA = 0.20


def _bsm(option_type: str, q: float = 0.0) -> float:
    # black_scholes_price signature: (S, K, T, r, sigma, option_type, q=0.0)
    return black_scholes_price(S0, K, T, R, SIGMA, option_type, q)


# ---------------------------------------------------------------------------
# Convergence CRR -> BSM for a European-equivalent payoff.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_crr_converges_to_bsm_no_dividend(option_type):
    """With q=0 the American CRR call equals BSM; the put is bounded below by it.

    For the call (no early exercise) convergence is direct. To get a clean
    European convergence check that holds for both, we lean on the call case and,
    for the put, verify the CRR price sits in the no-arbitrage band [european, intrinsic+...].
    """
    bsm = _bsm(option_type, q=0.0)
    if option_type == "call":
        # American call without dividends == European call -> direct convergence to BSM.
        crr = price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=2000, option_type="call")
        assert crr == pytest.approx(bsm, abs=2e-2)
    else:
        # American put >= European put; CRR must be at least the BSM European value
        # and remain finite/sensible.
        crr = price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=2000, option_type="put")
        assert crr >= bsm - 1e-9
        # American put on a non-dividend underlying is still tightly bounded above by K.
        assert crr <= K


def test_crr_call_convergence_rate_no_dividend():
    """Error to BSM call shrinks as steps grow (CRR is O(1/N))."""
    bsm = _bsm("call", q=0.0)
    err_coarse = abs(
        price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=50, option_type="call") - bsm
    )
    err_fine = abs(
        price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=800, option_type="call") - bsm
    )
    assert err_fine < err_coarse
    assert err_fine < 1e-2


# ---------------------------------------------------------------------------
# American call without dividends == European call (no early exercise).
# ---------------------------------------------------------------------------
def test_american_call_equals_european_call_no_dividend():
    """q=0: early exercise of a call is never optimal -> American == European (== BSM)."""
    bsm_call = _bsm("call", q=0.0)
    am_call = price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=1500, option_type="call")
    assert am_call == pytest.approx(bsm_call, abs=2e-2)


# ---------------------------------------------------------------------------
# American put >= European put.
# ---------------------------------------------------------------------------
def test_american_put_geq_european_put():
    """Early exercise carries strictly positive value for an ITM put (Merton)."""
    bsm_put = _bsm("put", q=0.0)
    am_put = price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=1500, option_type="put")
    # American >= European, with a real early-exercise premium for an ATM put at r>0.
    assert am_put >= bsm_put - 1e-6
    assert am_put > bsm_put  # premium should be strictly positive here


def test_american_put_premium_with_deep_itm():
    """A deep ITM American put approaches its intrinsic value K - S0 (early exercise)."""
    s0_deep = 60.0
    am_put = price_american_crr(s0_deep, K, R, 0.0, T, SIGMA, steps=1500, option_type="put")
    intrinsic = K - s0_deep
    assert am_put >= intrinsic - 1e-6


# ---------------------------------------------------------------------------
# Price >= intrinsic value (no-arbitrage lower bound for an American option).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "s0,option_type",
    [
        (130.0, "call"),  # ITM call
        (70.0, "put"),    # ITM put
        (100.0, "call"),  # ATM
        (100.0, "put"),
        (70.0, "call"),   # OTM call
        (130.0, "put"),   # OTM put
    ],
)
def test_price_geq_intrinsic(s0, option_type):
    price = price_american_crr(s0, K, R, 0.0, T, SIGMA, steps=500, option_type=option_type)
    intrinsic = max(s0 - K, 0.0) if option_type == "call" else max(K - s0, 0.0)
    assert price >= intrinsic - 1e-9
    assert price >= 0.0


# ---------------------------------------------------------------------------
# Monotonicity / stability in the number of steps (bounded convergence).
# ---------------------------------------------------------------------------
def test_convergence_stable_in_steps():
    """Successive refinements stay in a tight band and converge (Cauchy-like)."""
    prices = [
        price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=n, option_type="put")
        for n in (100, 200, 400, 800, 1600)
    ]
    # The sequence must be bounded and consecutive gaps shrink overall.
    gap_early = abs(prices[1] - prices[0])
    gap_late = abs(prices[-1] - prices[-2])
    assert gap_late <= gap_early + 1e-9
    # No blow-up: every estimate within a sane band around the finest one.
    finest = prices[-1]
    assert all(abs(p - finest) < 0.5 for p in prices)


def test_call_price_increasing_in_spot():
    """Monotonicity in S0: a call is non-decreasing in the underlying."""
    base = price_american_crr(90.0, K, R, 0.0, T, SIGMA, steps=400, option_type="call")
    higher = price_american_crr(110.0, K, R, 0.0, T, SIGMA, steps=400, option_type="call")
    assert higher > base


def test_put_price_decreasing_in_spot():
    """Monotonicity in S0: a put is non-increasing in the underlying."""
    base = price_american_crr(90.0, K, R, 0.0, T, SIGMA, steps=400, option_type="put")
    higher = price_american_crr(110.0, K, R, 0.0, T, SIGMA, steps=400, option_type="put")
    assert higher < base


# ---------------------------------------------------------------------------
# Slow / high-resolution convergence check (tight tolerance, many steps).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_crr_call_high_resolution_matches_bsm():
    """Very fine tree matches BSM call to a tight tolerance."""
    bsm = _bsm("call", q=0.0)
    crr = price_american_crr(S0, K, R, 0.0, T, SIGMA, steps=6000, option_type="call")
    assert crr == pytest.approx(bsm, abs=5e-3)
    assert not math.isnan(crr)
