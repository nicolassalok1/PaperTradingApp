"""C1 — Longstaff-Schwartz MC pricer (app/model/options/mc_engine.price_mc_lsmc).

The shipped implementation applied ONE time-step discount factor per *loop
iteration*, i.e. one per exercise DATE, while two consecutive dates can be tens
of steps apart. With 5 dates over 252 steps the cumulated factor was ~1, so the
returned "price" was an undiscounted expectation. Two further defects rode along:
the continuation regression was fitted on ALL paths instead of the in-the-money
ones, and `step_indices[:-1]` assumed the last listed date was maturity, silently
dropping a genuine exercise date when it was not.

ORACLES — none of them reads a value produced by the code under test:

* Closed-form Black-Scholes (implemented inline below with scipy.norm) and its
  externally computed values, pinned by `test_closed_form_oracles_are_correct`.
* The no-arbitrage relations
      american_call(q=0) == european_call          (never exercise early)
      american_price      >= immediate intrinsic
      american_put        >= european_put
* The repo's independent binomial trees `price_american_crr` /
  `price_bermuda_crr` at 2000 steps — a different algorithm, not the pricer
  under test (the same cross-check the existing test_crr_pricing.py relies on).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from app.model.options.engines.crr import price_american_crr, price_bermuda_crr
from app.model.options.exercise import build_exercise_dates
from app.model.options.mc_engine import price_mc_lsmc

pytestmark = pytest.mark.unit


def _bs(S0: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Closed-form Black-Scholes-Merton — the analytic oracle, no repo code involved."""
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)


# Case A — American call, zero dividend: early exercise is never optimal.
A_S0, A_K, A_T, A_SIGMA, A_R = 100.0, 100.0, 2.0, 0.25, 0.15
A_EUROPEAN_CALL = 29.2422  # closed form, computed outside this repo

# Case C — American put, at the money.
C_S0, C_K, C_T, C_SIGMA, C_R = 100.0, 100.0, 1.0, 0.25, 0.05
C_CRR_2000 = 7.9740  # binomial tree, 2000 steps
C_EUROPEAN_PUT = 7.4589  # closed form


def test_closed_form_oracles_are_correct():
    """Pin the literals used below against an independent closed-form evaluation."""
    assert _bs(A_S0, A_K, A_T, A_R, 0.0, A_SIGMA, "call") == pytest.approx(
        A_EUROPEAN_CALL, abs=5e-4
    )
    assert _bs(C_S0, C_K, C_T, C_R, 0.0, C_SIGMA, "put") == pytest.approx(
        C_EUROPEAN_PUT, abs=5e-4
    )
    assert price_american_crr(C_S0, C_K, C_R, 0.0, C_T, C_SIGMA, 2000, "put") == pytest.approx(
        C_CRR_2000, abs=5e-4
    )


# --------------------------------------------------------------------------- #
# (a) Invariance to the number of exercise dates.                              #
#     An American call on a non-dividend payer is worth exactly the European   #
#     call, whatever the exercise schedule.                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_dates", [2, 3, 5, 13, 52, 252])
def test_american_call_equals_european_for_any_number_of_dates(n_dates):
    dates = [float(t) for t in np.linspace(A_T / n_dates, A_T, num=n_dates, endpoint=True)]
    out = price_mc_lsmc(
        S0=A_S0,
        K=A_K,
        T=A_T,
        sigma=A_SIGMA,
        option_type="call",
        exercise_dates=dates,
        r=A_R,
        q=0.0,
        n_steps=252,
        n_paths=20000,
        seed=42,
    )
    tol = 3.0 * out["stderr"]
    assert out["price"] == pytest.approx(A_EUROPEAN_CALL, abs=tol), (
        f"{n_dates} dates: {out['price']:.4f} vs {A_EUROPEAN_CALL} (3 sigma = {tol:.4f})"
    )


def test_american_call_price_is_stable_across_schedules():
    """The spread between the coarsest and the densest schedule must be MC noise only."""
    prices = []
    for n_dates in (2, 252):
        dates = [float(t) for t in np.linspace(A_T / n_dates, A_T, num=n_dates, endpoint=True)]
        out = price_mc_lsmc(
            S0=A_S0, K=A_K, T=A_T, sigma=A_SIGMA, option_type="call",
            exercise_dates=dates, r=A_R, q=0.0, n_steps=252, n_paths=20000, seed=42,
        )
        prices.append(out["price"])
    assert abs(prices[0] - prices[1]) <= 0.02 * A_EUROPEAN_CALL, prices


# --------------------------------------------------------------------------- #
# (b) No-arbitrage: an American option is never worth less than exercising now. #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("S0", "intrinsic"), [(60.0, 40.0), (70.0, 30.0), (50.0, 50.0)])
def test_american_put_is_never_below_intrinsic(S0, intrinsic):
    out = price_mc_lsmc(
        S0=S0,
        K=100.0,
        T=1.0,
        sigma=0.25,
        option_type="put",
        exercise_dates=build_exercise_dates("american", 1.0),
        r=0.05,
        q=0.0,
        n_steps=252,
        n_paths=20000,
        seed=42,
    )
    assert out["price"] >= intrinsic, f"{out['price']:.4f} < intrinsic {intrinsic}"


# --------------------------------------------------------------------------- #
# (c) Convergence towards the binomial reference.                              #
# --------------------------------------------------------------------------- #
def test_american_put_matches_crr_and_dominates_the_european():
    out = price_mc_lsmc(
        S0=C_S0,
        K=C_K,
        T=C_T,
        sigma=C_SIGMA,
        option_type="put",
        exercise_dates=build_exercise_dates("american", C_T),
        r=C_R,
        q=0.0,
        n_steps=252,
        n_paths=20000,
        seed=42,
    )
    tol = 3.0 * out["stderr"]
    assert out["price"] == pytest.approx(C_CRR_2000, abs=tol), (
        f"{out['price']:.4f} vs CRR {C_CRR_2000} (3 sigma = {tol:.4f})"
    )
    assert out["price"] >= C_EUROPEAN_PUT, (out["price"], C_EUROPEAN_PUT)


# --------------------------------------------------------------------------- #
# Defect 3 — a schedule that does not contain T must not lose its last date.    #
# --------------------------------------------------------------------------- #
def test_bermudan_schedule_without_maturity_keeps_every_exercise_date():
    T, S0, K, r, sigma = 1.0, 100.0, 100.0, 0.05, 0.25
    dates = [0.25, 0.5, 0.75]  # deliberately no T
    out = price_mc_lsmc(
        S0=S0, K=K, T=T, sigma=sigma, option_type="put",
        exercise_dates=dates, r=r, q=0.0, n_steps=252, n_paths=20000, seed=42,
    )
    # Binomial reference on the same schedule; maturity is exercisable in both.
    steps = 2000
    reference = price_bermuda_crr(
        S0, K, r, 0.0, T, sigma, steps,
        exercise_dates=[int(round(t / T * steps)) for t in dates],
        option_type="put",
    )
    tol = 3.0 * out["stderr"]
    assert out["price"] == pytest.approx(reference, abs=tol), (
        f"{out['price']:.4f} vs bermudan CRR {reference:.4f} (3 sigma = {tol:.4f})"
    )


# --------------------------------------------------------------------------- #
# The intrinsic floor belongs to AMERICAN contracts only. A Bermudan whose      #
# first exercise date merely happens to sit one step away is still worth its    #
# continuation value, which is legitimately below the intrinsic.                #
# --------------------------------------------------------------------------- #
def test_deep_itm_bermudan_is_not_floored_at_intrinsic():
    T, S0, K, r, sigma, steps = 1.0, 60.0, 100.0, 0.05, 0.25, 50
    dates = [float(i) / steps for i in np.linspace(1, steps - 1, 5, dtype=int)]
    assert dates[0] > 0.0  # first exercise is NOT today
    out = price_mc_lsmc(
        S0=S0, K=K, T=T, sigma=sigma, option_type="put",
        exercise_dates=dates, r=r, q=0.0, n_steps=steps, n_paths=100_000, seed=42,
    )
    reference = price_bermuda_crr(
        S0, K, r, 0.0, T, sigma, 2000,
        exercise_dates=[int(round(t * 2000)) for t in dates],
        option_type="put",
    )
    assert reference < K - S0, "test setup: this Bermudan really is worth less than intrinsic"
    assert out["price"] == pytest.approx(reference, abs=3.0 * out["stderr"])
    assert out["price"] < K - S0, f"{out['price']:.4f} was floored at the intrinsic {K - S0:.2f}"


# --------------------------------------------------------------------------- #
# Degenerate schedule — only maturity: the pricer must reproduce Black-Scholes. #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["call", "put"])
def test_single_maturity_date_reproduces_black_scholes(kind):
    T, S0, K, r, sigma = 1.0, 100.0, 100.0, 0.05, 0.25
    out = price_mc_lsmc(
        S0=S0, K=K, T=T, sigma=sigma, option_type=kind,
        exercise_dates=[T], r=r, q=0.0, n_steps=252, n_paths=20000, seed=42,
    )
    expected = _bs(S0, K, T, r, 0.0, sigma, kind)
    assert out["price"] == pytest.approx(expected, abs=3.0 * out["stderr"])
