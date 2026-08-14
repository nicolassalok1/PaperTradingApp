"""D2.e — forward-start option (app/model/options/engines/black_scholes.py).

The strike is set at T_start to the spot observed then (moneyness 1 here), and the
option runs to T_end. Two defects, and both cancel out at r = q = 0, which is why
they survived:

1. The first term was never discounted. The body read
       call = forward * N(d1) - forward * N(d2) * exp(-r*tau)
   with `forward = S0 * exp((r-q)*T_start)` — the second term carries a discount
   factor, the first carries none, and the prefactor GROWS the spot at (r-q)
   instead of discounting the dividend over the fixing period.
2. The drift was dropped from d1: `d1 = (0.5*sigma^2*tau)/(sigma*sqrt(tau))`, i.e.
   sigma*sqrt(tau)/2, losing the (r-q)*tau term of the numerator.

Correct (Rubinstein), for strike m*S_{T_start} and tau = T_end - T_start:
    call = S0*exp(-q*T_start) * [ exp(-q*tau)*N(d1) - m*exp(-r*tau)*N(d2) ]
    d1   = [ln(1/m) + (r - q + sigma^2/2)*tau] / (sigma*sqrt(tau)),  d2 = d1 - sigma*sqrt(tau)

ORACLES:
1. T_start = 0 collapses the product to a plain at-the-money vanilla of maturity
   T_end — an exact identity the shipped code fails as soon as r != 0.
2. r = q = 0 is the regime where both defects cancel: the price must be UNCHANGED
   there. This is the non-regression anchor.
3. Put-call parity for forward-start:
       call - put = S0*exp(-q*T_start) * [exp(-q*tau) - m*exp(-r*tau)]
4. Monte-Carlo: draw S_{T_start}, value the vanilla struck at that spot over the
   remaining maturity with Black-Scholes, discount. Never evaluates the
   forward-start formula.
5. Homogeneity: the price is linear in S0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.options.core.pricing_lib import bs_price_call, bs_price_put
from app.model.options.engines.black_scholes import price_forward_start

pytestmark = pytest.mark.unit

# (S0, r, q, T_start, T_end, sigma)
CASES = [
    (100.0, 0.05, 0.00, 0.5, 1.5, 0.25),
    (100.0, 0.03, 0.02, 1.0, 2.0, 0.20),
    (80.0, 0.08, 0.01, 0.25, 1.0, 0.35),
]
_IDS = [f"S{s:g}-r{r:g}-q{q:g}-ts{a:g}-te{b:g}-sig{v:g}" for s, r, q, a, b, v in CASES]


@pytest.mark.parametrize(("S0", "r", "q", "_ts", "T_end", "sigma"), CASES, ids=_IDS)
def test_zero_fixing_delay_is_a_plain_atm_vanilla(S0, r, q, _ts, T_end, sigma):
    """With T_start = 0 the strike is S0 today: an ordinary at-the-money option."""
    for kind, vanilla in (
        ("call", bs_price_call(S0, S0, r=r, q=q, sigma=sigma, T=T_end)),
        ("put", bs_price_put(S0, S0, r=r, q=q, sigma=sigma, T=T_end)),
    ):
        got = price_forward_start(S0, r, q, 0.0, T_end, sigma, option_type=kind)
        assert got == pytest.approx(vanilla, rel=1e-12), kind


@pytest.mark.parametrize(("S0", "_r", "_q", "T_start", "T_end", "sigma"), CASES, ids=_IDS)
def test_at_zero_rates_the_price_is_the_atm_vanilla_on_the_remaining_maturity(
    S0, _r, _q, T_start, T_end, sigma
):
    """Non-regression anchor: at r = q = 0 both defects cancel, value must not move."""
    tau = T_end - T_start
    for kind, vanilla in (
        ("call", bs_price_call(S0, S0, r=0.0, q=0.0, sigma=sigma, T=tau)),
        ("put", bs_price_put(S0, S0, r=0.0, q=0.0, sigma=sigma, T=tau)),
    ):
        got = price_forward_start(S0, 0.0, 0.0, T_start, T_end, sigma, option_type=kind)
        assert got == pytest.approx(vanilla, rel=1e-12), kind


@pytest.mark.parametrize(("S0", "r", "q", "T_start", "T_end", "sigma"), CASES, ids=_IDS)
def test_forward_start_put_call_parity(S0, r, q, T_start, T_end, sigma):
    tau = T_end - T_start
    call = price_forward_start(S0, r, q, T_start, T_end, sigma, option_type="call")
    put = price_forward_start(S0, r, q, T_start, T_end, sigma, option_type="put")
    expected = S0 * math.exp(-q * T_start) * (math.exp(-q * tau) - math.exp(-r * tau))
    assert call - put == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(("S0", "r", "q", "T_start", "T_end", "sigma"), CASES, ids=_IDS)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_forward_start_matches_monte_carlo(S0, r, q, T_start, T_end, sigma, kind):
    rng = np.random.default_rng(23)
    n = 40_000
    z = rng.standard_normal(n)
    s1 = S0 * np.exp((r - q - 0.5 * sigma * sigma) * T_start + sigma * math.sqrt(T_start) * z)
    tau = T_end - T_start
    leg = bs_price_call if kind == "call" else bs_price_put
    # Strike is set at T_start to the spot observed then.
    values = np.array([leg(float(x), float(x), r=r, q=q, sigma=sigma, T=tau) for x in s1])
    disc = values * math.exp(-r * T_start)
    mc = float(np.mean(disc))
    stderr = float(np.std(disc, ddof=1) / math.sqrt(n))
    got = price_forward_start(S0, r, q, T_start, T_end, sigma, option_type=kind)
    assert got == pytest.approx(mc, abs=3.0 * stderr), (
        f"{kind}: closed form {got:.4f} vs MC {mc:.4f} +/- {stderr:.4f}"
    )


@pytest.mark.parametrize(("S0", "r", "q", "T_start", "T_end", "sigma"), CASES, ids=_IDS)
def test_price_is_linear_in_spot(S0, r, q, T_start, T_end, sigma):
    base = price_forward_start(S0, r, q, T_start, T_end, sigma, option_type="call")
    scaled = price_forward_start(3.0 * S0, r, q, T_start, T_end, sigma, option_type="call")
    assert scaled == pytest.approx(3.0 * base, rel=1e-12)


def test_degenerate_window_is_worthless():
    assert price_forward_start(100.0, 0.05, 0.0, 1.0, 1.0, 0.2) == 0.0
    assert price_forward_start(100.0, 0.05, 0.0, 1.5, 1.0, 0.2) == 0.0
