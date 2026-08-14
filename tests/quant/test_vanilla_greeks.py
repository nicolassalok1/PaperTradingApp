"""C2 — Black-Scholes vanilla greeks (app/model/options/core/greeks.py).

ORACLES — all independent of the code under test:

1. Closed-form Black-Scholes, evaluated *outside* this repo. With
   S = 100, T = 1, r = q = 0, sigma = 0.2 we have
   d1 = (ln(S/K) + sigma^2 T / 2) / (sigma sqrt(T)) and
   delta_put = -e^{-qT} N(-d1):

       K =  50 -> d1 =  3.565736 -> delta_put = -0.00018
       K = 100 -> d1 =  0.100000 -> delta_put = -0.46017
       K = 150 -> d1 = -1.927326 -> delta_put = -0.97303

   The pre-fix implementation returned -e^{-qT} N(d1) instead, i.e.
   -0.99982 / -0.53983 / -0.02697. Both formulas agree only at N(d1) = 1/2,
   which is why the bug survived at-the-money smoke checks.

2. Put-call parity on delta, which holds for any (S, K, T, r, q, sigma):
       delta_call - delta_put = e^{-qT}
   Checked on a strike grid with a non-zero dividend yield, so the e^{-qT}
   factor is actually exercised.

3. An expired option has no optionality left: every greek is exactly zero.
   The pre-fix code substituted a fabricated T = 0.5y whenever T <= 0 or the
   expiration date was unreadable, reporting vega = 0.2814 and gamma = 0.02814
   on a contract that no longer exists.
"""

from __future__ import annotations

import math

import pytest

from app.model.options.core.greeks import compute_vanilla_greeks

pytestmark = pytest.mark.unit

SPOT = 100.0
GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


def _opt(option_type: str, K: float, *, T: float = 1.0, sigma: float = 0.2,
         r: float = 0.0, q: float = 0.0, **extra) -> dict:
    out = {"option_type": option_type, "strike": K, "sigma": sigma, "r": r, "q": q, "T": T}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# Oracle 1 — put delta against externally computed Black-Scholes values.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("K", "expected_delta_put"),
    [
        (50.0, -0.00018),
        (100.0, -0.46017),
        (150.0, -0.97303),
    ],
)
def test_put_delta_matches_closed_form(K, expected_delta_put):
    got = compute_vanilla_greeks(_opt("put", K), SPOT)["delta"]
    assert got == pytest.approx(expected_delta_put, abs=1e-5)


@pytest.mark.parametrize(
    ("K", "expected_delta_call"),
    [
        (50.0, 0.99982),
        (100.0, 0.53983),
        (150.0, 0.02697),
    ],
)
def test_call_delta_matches_closed_form(K, expected_delta_call):
    got = compute_vanilla_greeks(_opt("call", K), SPOT)["delta"]
    assert got == pytest.approx(expected_delta_call, abs=1e-5)


def test_put_delta_is_negative_and_bounded():
    """Structural: a long put delta lives in [-1, 0] for every strike."""
    for K in (10.0, 50.0, 90.0, 100.0, 110.0, 150.0, 400.0):
        delta = compute_vanilla_greeks(_opt("put", K), SPOT)["delta"]
        assert -1.0 <= delta <= 0.0, (K, delta)


def test_put_delta_decreases_with_strike():
    """A put gets more sensitive (more negative delta) as the strike rises."""
    deltas = [compute_vanilla_greeks(_opt("put", K), SPOT)["delta"] for K in range(50, 160, 10)]
    assert all(b < a for a, b in zip(deltas, deltas[1:])), deltas


# --------------------------------------------------------------------------- #
# Oracle 2 — delta parity: delta_call - delta_put = e^{-qT}.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [0.0, 0.03, 0.07])
@pytest.mark.parametrize("K", [60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0])
def test_delta_parity_on_strike_grid(K, q):
    T, r, sigma = 0.75, 0.04, 0.3
    call = compute_vanilla_greeks(_opt("call", K, T=T, r=r, q=q, sigma=sigma), SPOT)["delta"]
    put = compute_vanilla_greeks(_opt("put", K, T=T, r=r, q=q, sigma=sigma), SPOT)["delta"]
    assert call - put == pytest.approx(math.exp(-q * T), abs=1e-12)


# --------------------------------------------------------------------------- #
# Oracle 3 — an expired / undated option carries no risk at all.                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [0.0, -1.0, -0.25])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_expired_option_has_zero_greeks(option_type, T):
    got = compute_vanilla_greeks(_opt(option_type, 100.0, T=T), SPOT)
    assert {k: got[k] for k in GREEK_NAMES} == dict.fromkeys(GREEK_NAMES, 0.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_past_expiration_date_has_zero_greeks(option_type):
    got = compute_vanilla_greeks(
        _opt(option_type, 100.0, T=0.0, expiration="2000-01-01"), SPOT
    )
    assert {k: got[k] for k in GREEK_NAMES} == dict.fromkeys(GREEK_NAMES, 0.0)


@pytest.mark.parametrize("bad", ["not-a-date", "2026-13-45", ""])
def test_unreadable_expiration_has_zero_greeks(bad):
    got = compute_vanilla_greeks(_opt("call", 100.0, T=0.0, expiration=bad), SPOT)
    assert {k: got[k] for k in GREEK_NAMES} == dict.fromkeys(GREEK_NAMES, 0.0)


def test_unreadable_expiration_is_logged(caplog):
    """A maturity that cannot be read must be reported, never invented."""
    with caplog.at_level("WARNING"):
        compute_vanilla_greeks(_opt("call", 100.0, T=0.0, expiration="not-a-date"), SPOT)
    assert caplog.records, "an unreadable expiration must leave a log trace"


# --------------------------------------------------------------------------- #
# Guard rail — the untouched greeks must keep their pre-existing values.        #
# S=100, K=100, T=1, r=q=0, sigma=0.2 (scaled as the module reports them).      #
# --------------------------------------------------------------------------- #
def test_other_greeks_are_unchanged_atm():
    got = compute_vanilla_greeks(_opt("call", 100.0), SPOT)
    # d1 = 0.1, d2 = -0.1, n(d1) = 0.3969525474770118
    assert got["gamma"] == pytest.approx(0.019847627, abs=1e-8)  # n(d1)/(S sigma sqrt(T))
    assert got["vega"] == pytest.approx(0.396952547, abs=1e-8)  # S n(d1) sqrt(T) / 100
    assert got["theta"] == pytest.approx(-0.010875412, abs=1e-8)  # -S n(d1) sigma/(2 sqrt(T)) /365
    assert got["rho"] == pytest.approx(0.460172163, abs=1e-8)  # K T N(d2) / 100
