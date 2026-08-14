"""D2.d — quanto option (app/model/options/core/pricing_lib.py).

`price_quanto_bs` was, in its own words, "Simplified: price vanilla then convert with
fixed FX rate". That is an FX-converted vanilla, not a quanto. What makes an option a
quanto is precisely the term that was missing: paying a foreign-denominated asset at a
FIXED exchange rate changes the asset's drift under the domestic risk-neutral measure,

    mu_S = r_foreign - q - rho * sigma_S * sigma_FX

so the price depends on the spot/FX correlation. With no `rho` and no `sigma_FX` in the
signature, the returned price could not move when the correlation moved — the one
sensitivity that defines the product.

ORACLES:

1. EXACT BACKWARD COMPATIBILITY: at rho = 0 and sigma_FX = 0 the adjustment vanishes and
   the price must equal, to the last bit, fx_rate * vanilla Black-Scholes. This pins the
   fix as strictly additive — no existing caller changes value.
2. DEGENERACY: correlation can only act through FX volatility. With sigma_FX = 0 the
   price must be identical for every rho; with sigma_FX > 0 it must not be.
   The pre-fix code satisfied the first half for the wrong reason and failed the second.
3. SIGN AND MONOTONICITY, model-free given the drift: a higher rho lowers mu_S, hence
   lowers a call and raises a put, strictly and monotonically.
4. CONSISTENCY: the quanto equals a vanilla whose dividend yield is shifted by exactly
   the quanto adjustment — computed independently here from bs_price_call.
"""

from __future__ import annotations

import pytest

from app.model.options.core.pricing_lib import bs_price_call, bs_price_put, price_quanto_bs

pytestmark = pytest.mark.unit

S, K, T, SIGMA, R, Q = 100.0, 100.0, 1.0, 0.25, 0.04, 0.01
FX = 1.3
RHOS = [-0.8, -0.4, 0.0, 0.4, 0.8]


def _quanto(option_type="call", **over):
    kw = dict(r=R, q=Q, sigma=SIGMA, T=T, option_type=option_type)
    kw.update(over)
    return price_quanto_bs(S, K, fx_rate=FX, **kw)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_zero_correlation_and_zero_fx_vol_reproduce_the_converted_vanilla(option_type):
    vanilla = (
        bs_price_call(S, K, r=R, q=Q, sigma=SIGMA, T=T)
        if option_type == "call"
        else bs_price_put(S, K, r=R, q=Q, sigma=SIGMA, T=T)
    )
    got = _quanto(option_type, rho=0.0, sigma_fx=0.0)
    assert got == pytest.approx(FX * vanilla, rel=1e-14)


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_correlation_is_inert_without_fx_volatility(option_type, rho):
    """rho only ever acts multiplied by sigma_FX — no FX vol, no adjustment."""
    base = _quanto(option_type, rho=0.0, sigma_fx=0.0)
    assert _quanto(option_type, rho=rho, sigma_fx=0.0) == pytest.approx(base, rel=1e-14)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_correlation_moves_the_price_once_fx_is_volatile(option_type):
    at_zero = _quanto(option_type, rho=0.0, sigma_fx=0.15)
    at_high = _quanto(option_type, rho=0.8, sigma_fx=0.15)
    assert at_high != pytest.approx(at_zero, rel=1e-9), (
        "the quanto adjustment is inert: this prices an FX-converted vanilla, not a quanto"
    )


def test_call_decreases_and_put_increases_with_correlation():
    calls = [_quanto("call", rho=r, sigma_fx=0.15) for r in RHOS]
    puts = [_quanto("put", rho=r, sigma_fx=0.15) for r in RHOS]
    assert all(b < a for a, b in zip(calls, calls[1:])), calls
    assert all(b > a for a, b in zip(puts, puts[1:])), puts


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("sigma_fx", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_quanto_equals_a_vanilla_with_the_shifted_dividend_yield(option_type, sigma_fx, rho):
    """mu_S = r_f - q - rho*sigma_S*sigma_FX, i.e. an effective yield q + rho*sigma_S*sigma_FX
    when the foreign rate defaults to the domestic one."""
    q_eff = Q + rho * SIGMA * sigma_fx
    expected = (
        bs_price_call(S, K, r=R, q=q_eff, sigma=SIGMA, T=T)
        if option_type == "call"
        else bs_price_put(S, K, r=R, q=q_eff, sigma=SIGMA, T=T)
    )
    got = _quanto(option_type, rho=rho, sigma_fx=sigma_fx)
    assert got == pytest.approx(FX * expected, rel=1e-12)


def test_foreign_rate_can_differ_from_the_domestic_one():
    """A quanto discounts domestically but grows the asset at the foreign rate."""
    same = _quanto("call", rho=0.0, sigma_fx=0.0, r_foreign=R)
    lower = _quanto("call", rho=0.0, sigma_fx=0.0, r_foreign=R - 0.03)
    assert lower < same
    expected = bs_price_call(S, K, r=R, q=Q + 0.03, sigma=SIGMA, T=T)
    assert lower == pytest.approx(FX * expected, rel=1e-12)
