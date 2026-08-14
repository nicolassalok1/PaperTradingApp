"""E2 — fixed-strike lookback (app/model/options/core/pricing_lib.py).

A fixed-strike lookback pays max(M - K, 0) for a call (M = running maximum) and
max(K - m, 0) for a put (m = running minimum). `view_lookback_fixed` had no model:
its premium was the intrinsic payoff, and it then filled the whole payoff grid with
that same number, so the P&L was identically zero at every spot.

The flat payoff line is NOT the bug and is not "fixed" here: a fixed-strike lookback
pays on the path extremum, not on the terminal spot, so payoff-vs-spot really is
constant. What was wrong is that the premium carried no time value, which collapsed
the P&L (payoff - premium) to zero everywhere.

ORACLES:
1. QUANTITATIVE: Monte-Carlo on the actual path extremum, with the
   Broadie-Glasserman-Kou continuity correction (beta = 0.5826) so a discretely
   monitored simulation is comparable with a continuously monitored closed form.
2. MODEL-FREE, EXACT: at T = 0 the price is the intrinsic; while time remains it is
   STRICTLY above it (that strict inequality is what the old code failed, by
   returning the intrinsic itself).
3. MODEL-FREE: a fixed-strike lookback call dominates the vanilla call of the same
   strike, since M >= S_T pathwise.
4. MONOTONICITY: worth more with a longer life; the call falls as the strike rises.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.options.core.pricing_lib import bs_price_call, price_lookback_fixed

pytestmark = pytest.mark.unit

BGK_BETA = 0.5826


def _mc_fixed(S, extremum, K, r, q, sigma, T, option_type,
              n_paths=60_000, n_steps=2000, seed=17):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shift = math.exp(BGK_BETA * sigma * math.sqrt(dt))
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    s = np.full(n_paths, float(S))
    running = np.full(n_paths, float(extremum))
    for _ in range(n_steps):
        s = s * np.exp(drift + vol * rng.standard_normal(n_paths))
        if option_type == "call":
            np.maximum(running, s, out=running)
        else:
            np.minimum(running, s, out=running)
    if option_type == "call":
        corrected = np.maximum(running * shift, float(extremum))
        payoff = np.maximum(corrected - K, 0.0)
    else:
        corrected = np.minimum(running / shift, float(extremum))
        payoff = np.maximum(K - corrected, 0.0)
    disc = payoff * math.exp(-r * T)
    return float(np.mean(disc)), float(np.std(disc, ddof=1) / math.sqrt(n_paths))


# (S, extremum, K, r, q, sigma, T) — extremum is the running max (call) / min (put)
CALL_CASES = [
    (100.0, 100.0, 100.0, 0.05, 0.00, 0.25, 1.0),  # fresh, at the money
    (100.0, 100.0, 120.0, 0.05, 0.00, 0.25, 1.0),  # strike above the running max
    (100.0, 115.0, 100.0, 0.04, 0.02, 0.30, 1.0),  # seasoned: max already above K
    (100.0, 108.0, 110.0, 0.03, 0.01, 0.20, 0.5),  # seasoned, still out of the money
]
PUT_CASES = [
    (100.0, 100.0, 100.0, 0.05, 0.00, 0.25, 1.0),
    (100.0, 100.0, 80.0, 0.05, 0.00, 0.25, 1.0),
    (100.0, 85.0, 100.0, 0.04, 0.02, 0.30, 1.0),
    (100.0, 92.0, 90.0, 0.03, 0.01, 0.20, 0.5),
]
_CIDS = [f"S{a:g}-M{b:g}-K{c:g}-r{d:g}-q{e:g}-sig{f:g}-T{g:g}" for a, b, c, d, e, f, g in CALL_CASES]
_PIDS = [f"S{a:g}-m{b:g}-K{c:g}-r{d:g}-q{e:g}-sig{f:g}-T{g:g}" for a, b, c, d, e, f, g in PUT_CASES]


@pytest.mark.parametrize(("S", "M", "K", "r", "q", "sigma", "T"), CALL_CASES, ids=_CIDS)
def test_fixed_lookback_call_matches_monte_carlo(S, M, K, r, q, sigma, T):
    mc, stderr = _mc_fixed(S, M, K, r, q, sigma, T, "call")
    got = price_lookback_fixed(S, M, K, r=r, q=q, sigma=sigma, T=T, option_type="call")
    tol = 3.0 * stderr + 0.005 * max(mc, 1e-9)
    assert got == pytest.approx(mc, abs=tol), f"{got:.4f} vs MC {mc:.4f} +/- {stderr:.4f}"


@pytest.mark.parametrize(("S", "m", "K", "r", "q", "sigma", "T"), PUT_CASES, ids=_PIDS)
def test_fixed_lookback_put_matches_monte_carlo(S, m, K, r, q, sigma, T):
    mc, stderr = _mc_fixed(S, m, K, r, q, sigma, T, "put")
    got = price_lookback_fixed(S, m, K, r=r, q=q, sigma=sigma, T=T, option_type="put")
    tol = 3.0 * stderr + 0.005 * max(mc, 1e-9)
    assert got == pytest.approx(mc, abs=tol), f"{got:.4f} vs MC {mc:.4f} +/- {stderr:.4f}"


@pytest.mark.parametrize(("S", "M", "K", "r", "q", "sigma", "_T"), CALL_CASES, ids=_CIDS)
def test_zero_maturity_is_the_intrinsic(S, M, K, r, q, sigma, _T):
    got = price_lookback_fixed(S, M, K, r=r, q=q, sigma=sigma, T=0.0, option_type="call")
    assert got == pytest.approx(max(M - K, 0.0), abs=1e-12)


@pytest.mark.parametrize(("S", "M", "K", "r", "q", "sigma", "T"), CALL_CASES, ids=_CIDS)
def test_strictly_above_intrinsic_while_alive(S, M, K, r, q, sigma, T):
    """The old code returned exactly the intrinsic, so the P&L was flat at zero."""
    got = price_lookback_fixed(S, M, K, r=r, q=q, sigma=sigma, T=T, option_type="call")
    assert got > max(M - K, 0.0) + 1e-6, f"{got:.6f} vs intrinsic {max(M - K, 0.0):.6f}"


@pytest.mark.parametrize(("S", "M", "K", "r", "q", "sigma", "T"), CALL_CASES, ids=_CIDS)
def test_dominates_the_vanilla_call(S, M, K, r, q, sigma, T):
    """M >= S_T pathwise, so the lookback call is never worth less than the vanilla."""
    vanilla = bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T)
    got = price_lookback_fixed(S, M, K, r=r, q=q, sigma=sigma, T=T, option_type="call")
    assert got >= vanilla - 1e-9, (got, vanilla)


def test_value_grows_with_maturity():
    kw = dict(r=0.05, q=0.0, sigma=0.25, option_type="call")
    values = [price_lookback_fixed(100.0, 100.0, 100.0, T=t, **kw) for t in (0.25, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(values, values[1:])), values


def test_call_falls_as_the_strike_rises():
    kw = dict(r=0.05, q=0.0, sigma=0.25, T=1.0, option_type="call")
    values = [price_lookback_fixed(100.0, 100.0, k, **kw) for k in (80.0, 100.0, 120.0, 150.0)]
    assert all(b < a for a, b in zip(values, values[1:])), values


def test_zero_drift_case_is_finite():
    got = price_lookback_fixed(100.0, 100.0, 100.0, r=0.03, q=0.03, sigma=0.25, T=1.0,
                               option_type="call")
    assert math.isfinite(got) and got > 0.0
    mc, stderr = _mc_fixed(100.0, 100.0, 100.0, 0.03, 0.03, 0.25, 1.0, "call")
    assert got == pytest.approx(mc, abs=3.0 * stderr + 0.005 * mc)
