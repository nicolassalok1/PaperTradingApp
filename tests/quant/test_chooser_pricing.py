"""D2.c — simple chooser (app/model/options/core/pricing_lib.py).

`price_chooser_bs` returned `price_straddle_bs(S, K, **kwargs)`. A chooser is not a
straddle: the holder picks ONE of the two legs at a choice date t1, and only that leg
survives to maturity. The straddle keeps both. The choice date did not even appear in
the signature, so the pricer had no way to be right for t1 < T.

ORACLES — all independent of the closed form under test:

1. MODEL-FREE, EXACT bounds, for every 0 < t1 <= T:
       max(call_BS, put_BS)  <=  chooser  <=  call_BS + put_BS
   with equality on the right exactly at t1 = T (choosing at maturity is the same
   as owning both legs). The pre-fix code sat AT the upper bound for every t1.
2. MODEL-FREE: the value is non-decreasing in t1 — a later choice date is a longer
   option on which leg to keep.
3. QUANTITATIVE: Monte-Carlo. Simulate S_t1 exactly (one lognormal step), value both
   legs at t1 with Black-Scholes on the remaining maturity, keep the better one, and
   discount. This never evaluates the chooser formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.options.core.pricing_lib import bs_price_call, bs_price_put, price_chooser_bs

pytestmark = pytest.mark.unit

# (S, K, T, sigma, r, q)
CASES = [
    (100.0, 100.0, 1.0, 0.25, 0.05, 0.00),
    (100.0, 90.0, 1.0, 0.30, 0.05, 0.02),
    (100.0, 115.0, 2.0, 0.20, 0.03, 0.01),
]
_IDS = [f"S{s:g}-K{k:g}-T{t:g}-sig{v:g}-r{r:g}-q{q:g}" for s, k, t, v, r, q in CASES]
_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def _legs(S, K, T, sigma, r, q):
    return (
        bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T),
        bs_price_put(S, K, r=r, q=q, sigma=sigma, T=T),
    )


def _mc_chooser(S, K, T, sigma, r, q, t1, n_paths=200_000, seed=11):
    """Value at t1 = max(call, put) on the remaining maturity, discounted."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    s1 = S * np.exp((r - q - 0.5 * sigma * sigma) * t1 + sigma * math.sqrt(t1) * z)
    tau = T - t1
    if tau <= 0.0:
        best = np.abs(s1 - K)
    else:
        calls = np.array([bs_price_call(float(x), K, r=r, q=q, sigma=sigma, T=tau) for x in s1[:20_000]])
        puts = np.array([bs_price_put(float(x), K, r=r, q=q, sigma=sigma, T=tau) for x in s1[:20_000]])
        best = np.maximum(calls, puts)
    disc = best * math.exp(-r * t1)
    return float(np.mean(disc)), float(np.std(disc, ddof=1) / math.sqrt(disc.size))


@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
@pytest.mark.parametrize("frac", _FRACTIONS)
def test_chooser_sits_between_the_best_leg_and_the_straddle(S, K, T, sigma, r, q, frac):
    t1 = frac * T
    call, put = _legs(S, K, T, sigma, r, q)
    got = price_chooser_bs(S, K, t1, r=r, q=q, sigma=sigma, T=T)
    assert got >= max(call, put) - 1e-12, (got, call, put)
    assert got <= call + put + 1e-12, (got, call + put)


@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_chooser_equals_the_straddle_only_when_the_choice_is_at_maturity(S, K, T, sigma, r, q):
    call, put = _legs(S, K, T, sigma, r, q)
    at_maturity = price_chooser_bs(S, K, T, r=r, q=q, sigma=sigma, T=T)
    assert at_maturity == pytest.approx(call + put, rel=1e-12)
    # ...and strictly cheaper as soon as the choice comes earlier.
    earlier = price_chooser_bs(S, K, 0.5 * T, r=r, q=q, sigma=sigma, T=T)
    assert earlier < call + put


@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
def test_chooser_increases_with_the_choice_date(S, K, T, sigma, r, q):
    values = [
        price_chooser_bs(S, K, f * T, r=r, q=q, sigma=sigma, T=T) for f in _FRACTIONS
    ]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), values
    assert values[-1] > values[0]


@pytest.mark.parametrize(("S", "K", "T", "sigma", "r", "q"), CASES, ids=_IDS)
@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_chooser_matches_monte_carlo(S, K, T, sigma, r, q, frac):
    t1 = frac * T
    mc, stderr = _mc_chooser(S, K, T, sigma, r, q, t1)
    got = price_chooser_bs(S, K, t1, r=r, q=q, sigma=sigma, T=T)
    assert got == pytest.approx(mc, abs=3.0 * stderr), (
        f"closed form {got:.4f} vs MC {mc:.4f} +/- {stderr:.4f}"
    )


def test_choice_date_is_required():
    """A chooser without a choice date is under-specified — fail loudly, never guess."""
    with pytest.raises(TypeError):
        price_chooser_bs(100.0, 100.0)  # type: ignore[call-arg]
