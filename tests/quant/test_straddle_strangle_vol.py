"""E1 — the common `sigma` argument must reach both legs of a straddle/strangle.

NOTE ON PROVENANCE: an earlier pass reported price_straddle_bs as ignoring its own
`sigma` and pricing at DEFAULT_SIGMA. That report was WRONG — it read

    sigma_c = DEFAULT_SIGMA if sigma_call is None else sigma_call

and stopped there, missing the two lines below that immediately override it with
`sigma`. The behaviour has always been correct. The dead assignments are removed so
the next reader cannot make the same mistake, and these tests pin the contract so a
future edit cannot quietly break it.

ORACLE: Black-Scholes call + put evaluated independently at the same volatility.
"""

from __future__ import annotations

import pytest

from app.model.options.core.pricing_lib import (
    DEFAULT_SIGMA,
    bs_price_call,
    bs_price_put,
    price_straddle_bs,
    price_strangle_bs,
)

pytestmark = pytest.mark.unit

S, K, R, Q, T = 100.0, 100.0, 0.05, 0.01, 1.0
VOLS = [0.05, 0.15, DEFAULT_SIGMA, 0.35, 0.60]


@pytest.mark.parametrize("sigma", VOLS)
def test_straddle_uses_the_common_sigma_on_both_legs(sigma):
    expected = bs_price_call(S, K, r=R, q=Q, sigma=sigma, T=T) + bs_price_put(
        S, K, r=R, q=Q, sigma=sigma, T=T
    )
    assert price_straddle_bs(S, K, r=R, q=Q, sigma=sigma, T=T) == pytest.approx(
        expected, rel=1e-14
    )


def test_straddle_actually_moves_with_sigma():
    """Guards against a regression to DEFAULT_SIGMA: the price must not be flat in sigma."""
    prices = [price_straddle_bs(S, K, r=R, q=Q, sigma=v, T=T) for v in VOLS]
    assert all(b > a for a, b in zip(prices, prices[1:])), prices


@pytest.mark.parametrize("sigma_call", [0.10, 0.40])
@pytest.mark.parametrize("sigma_put", [0.12, 0.45])
def test_per_leg_vols_override_the_common_sigma(sigma_call, sigma_put):
    expected = bs_price_call(S, K, r=R, q=Q, sigma=sigma_call, T=T) + bs_price_put(
        S, K, r=R, q=Q, sigma=sigma_put, T=T
    )
    got = price_straddle_bs(
        S, K, r=R, q=Q, sigma=0.99, T=T, sigma_call=sigma_call, sigma_put=sigma_put
    )
    assert got == pytest.approx(expected, rel=1e-14)


def test_one_leg_overridden_leaves_the_other_on_the_common_sigma():
    got = price_straddle_bs(S, K, r=R, q=Q, sigma=0.30, T=T, sigma_call=0.10)
    expected = bs_price_call(S, K, r=R, q=Q, sigma=0.10, T=T) + bs_price_put(
        S, K, r=R, q=Q, sigma=0.30, T=T
    )
    assert got == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("sigma", VOLS)
def test_strangle_uses_the_common_sigma_on_both_legs(sigma):
    k_put, k_call = 90.0, 110.0
    expected = bs_price_call(S, k_call, r=R, q=Q, sigma=sigma, T=T) + bs_price_put(
        S, k_put, r=R, q=Q, sigma=sigma, T=T
    )
    assert price_strangle_bs(S, k_put, k_call, r=R, q=Q, sigma=sigma, T=T) == pytest.approx(
        expected, rel=1e-14
    )
