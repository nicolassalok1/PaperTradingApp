"""E3 — `logic.price_bermudan_lsmc` must be `mc_engine.price_mc_lsmc`, not a twin.

Two Longstaff-Schwartz implementations coexisted since pass 1: `logic.py` returned
a bare `float` and regressed the continuation value on a degree-3 basis, while
`mc_engine.py` returned a dict, regressed on a hard-wired degree-2 basis, and
carried an immediate-exercise floor for continuously exercisable schedules. Both
already drew their paths from the same `simulate_gbm_paths`, so the duplication
bought nothing but a second place for the exercise logic to rot.

ORACLES — none of them reads a price produced by the code under test:

* `price_bermuda_crr` at 2000 steps on the SAME exercise schedule: an independent
  algorithm (binomial tree), the cross-check `test_lsmc_pricing.py` already uses.
* Closed-form Black-Scholes (inline, scipy.norm) for the degenerate one-step
  schedule, where no early exercise can happen.
* The no-arbitrage statement that a Bermudan is NOT floored at its intrinsic.

Tolerances only are scaled by the engine's own `stderr` — a precision estimate,
never an expected value — as `test_lsmc_pricing.py` does. `_SIGMAS = 4` because a
seed sweep put the worst |price - CRR| at 2.5 standard errors: the residual is
Longstaff-Schwartz regression bias, not a wrong price.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from app.model.options.engines.crr import price_bermuda_crr
from app.model.options.logic import price_bermudan_lsmc
from app.model.options.mc_engine import price_mc_lsmc

pytestmark = pytest.mark.unit

# Number of standard errors allowed between the MC price and the tree reference.
_SIGMAS = 4.0

# The schedule `price_bermudan_lsmc` is documented to price: `M` time steps,
# `n_ex_dates` exercise opportunities in total, maturity always being the last.
_M = 50
_N_EX = 6
_N_PATHS = 100_000
_DEGREE = 3
_SEED = 42


def _bs(S0: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Closed-form Black-Scholes-Merton — analytic oracle, no repo code involved."""
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)


def _early_exercise_steps(M: int, n_ex_dates: int) -> list[int]:
    """Step indices of the EARLY exercise dates, pinned from the documented contract.

    `n_ex_dates - 1` opportunities evenly spread over steps 1..M-1; maturity (step
    M) is the n-th one and is always exercisable. Re-stated here so that any change
    to the schedule the pricer offers fails this module instead of silently
    repricing a different contract.
    """
    return sorted({int(i) for i in np.linspace(1, M - 1, max(1, n_ex_dates - 1), dtype=int)})


def _exercise_times(T: float, M: int, n_ex_dates: int) -> list[float]:
    return [i * T / M for i in _early_exercise_steps(M, n_ex_dates)]


def _crr_reference(S0, K, T, r, q, sigma, kind, M=_M, n_ex_dates=_N_EX, steps=2000) -> float:
    """Binomial tree on the very same Bermudan schedule — the independent oracle."""
    return price_bermuda_crr(
        S0, K, r, q, T, sigma, steps,
        exercise_dates=[int(round(i / M * steps)) for i in _early_exercise_steps(M, n_ex_dates)],
        option_type=kind,
    )


def _stderr(S0, K, T, r, q, sigma, kind, M=_M, n_paths=_N_PATHS, n_ex_dates=_N_EX) -> float:
    """MC standard error of the estimator — sizes the tolerance, never the target."""
    return price_mc_lsmc(
        S0=S0, K=K, T=T, sigma=sigma, option_type=kind,
        exercise_dates=_exercise_times(T, M, n_ex_dates),
        r=r, q=q, n_steps=M, n_paths=n_paths, seed=_SEED,
    )["stderr"]


# --------------------------------------------------------------------------- #
# The delegation itself: same schedule, same seed, same basis => same number.   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("S0", "K", "T", "r", "q", "sigma", "cpflag", "kind"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.00, 0.25, "p", "put"),
        (100.0, 100.0, 1.0, 0.05, 0.03, 0.25, "c", "call"),
        (60.0, 100.0, 1.0, 0.05, 0.00, 0.25, "p", "put"),
    ],
)
def test_bermudan_lsmc_is_the_shared_engine_on_the_documented_schedule(
    S0, K, T, r, q, sigma, cpflag, kind
):
    """One implementation, not two: the wrapper must reproduce the engine exactly.

    Same `simulate_gbm_paths` seed on both sides, so the paths are identical; the
    only freedom left is the exercise decision, hence the tiny relative tolerance
    rather than an exact equality.
    """
    wrapper = price_bermudan_lsmc(
        S0, K, T, r, q, sigma, cpflag, _M, _N_PATHS, _DEGREE, _N_EX, _SEED
    )
    engine = price_mc_lsmc(
        S0=S0, K=K, T=T, sigma=sigma, option_type=kind,
        exercise_dates=_exercise_times(T, _M, _N_EX),
        r=r, q=q, n_steps=_M, n_paths=_N_PATHS, seed=_SEED, degree=_DEGREE,
    )["price"]
    assert wrapper == pytest.approx(engine, rel=1e-9), f"{wrapper!r} vs {engine!r}"


def test_regression_degree_is_configurable_and_still_defaults_to_two():
    """`_regress_continuation` gains a degree; every existing caller keeps degree 2."""
    kwargs = dict(
        S0=100.0, K=100.0, T=1.0, sigma=0.25, option_type="put",
        exercise_dates=_exercise_times(1.0, _M, _N_EX),
        r=0.05, q=0.0, n_steps=_M, n_paths=20_000, seed=_SEED,
    )
    assert price_mc_lsmc(**kwargs)["price"] == price_mc_lsmc(**kwargs, degree=2)["price"]

    cubic = price_mc_lsmc(**kwargs, degree=3)["price"]
    reference = _crr_reference(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, "put")
    tol = _SIGMAS * price_mc_lsmc(**kwargs, degree=3)["stderr"]
    assert cubic == pytest.approx(reference, abs=tol), f"{cubic:.4f} vs CRR {reference:.4f}"


# --------------------------------------------------------------------------- #
# The price itself, against an independent binomial tree.                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("S0", "K", "T", "r", "q", "sigma", "cpflag", "kind"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.00, 0.25, "p", "put"),
        (100.0, 100.0, 1.0, 0.05, 0.03, 0.25, "c", "call"),
        (110.0, 100.0, 1.0, 0.05, 0.00, 0.30, "p", "put"),
    ],
)
def test_bermudan_lsmc_matches_the_binomial_tree_on_the_same_schedule(
    S0, K, T, r, q, sigma, cpflag, kind
):
    price = price_bermudan_lsmc(
        S0, K, T, r, q, sigma, cpflag, _M, _N_PATHS, _DEGREE, _N_EX, _SEED
    )
    reference = _crr_reference(S0, K, T, r, q, sigma, kind)
    tol = _SIGMAS * _stderr(S0, K, T, r, q, sigma, kind)
    assert price == pytest.approx(reference, abs=tol), (
        f"{price:.4f} vs bermudan CRR {reference:.4f} ({_SIGMAS} sigma = {tol:.4f})"
    )


def test_single_step_schedule_reproduces_black_scholes():
    """`M=1` leaves no step before maturity: the Bermudan degenerates to European.

    Also the edge case of the step-index/time round-trip, where `np.linspace`
    truncation feeds a step 0 that must be dropped rather than exercised at t=0.
    """
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.25
    price = price_bermudan_lsmc(S0, K, T, r, q, sigma, "p", 1, _N_PATHS, _DEGREE, _N_EX, _SEED)
    expected = _bs(S0, K, T, r, q, sigma, "put")
    tol = _SIGMAS * _stderr(S0, K, T, r, q, sigma, "put", M=1)
    assert price == pytest.approx(expected, abs=tol), f"{price:.4f} vs BS {expected:.4f}"


# --------------------------------------------------------------------------- #
# The immediate-exercise floor lives in the engine and must stay dormant here.  #
# --------------------------------------------------------------------------- #
def test_deep_itm_bermudan_is_not_floored_at_intrinsic():
    """`n_ex_dates=6` over `M=50` is nowhere near a continuously exercisable
    contract, so the engine's American floor must not fire: the price is a pure
    continuation value and legitimately sits below `K - S0`."""
    S0, K, T, r, q, sigma = 60.0, 100.0, 1.0, 0.05, 0.0, 0.25
    assert _early_exercise_steps(_M, _N_EX)[0] >= 1, "first exercise is not today"

    price = price_bermudan_lsmc(S0, K, T, r, q, sigma, "p", _M, _N_PATHS, _DEGREE, _N_EX, _SEED)
    reference = _crr_reference(S0, K, T, r, q, sigma, "put")
    assert reference < K - S0, "test setup: this Bermudan really is worth less than intrinsic"
    assert price < K - S0, f"{price:.4f} was floored at the intrinsic {K - S0:.2f}"
    assert price == pytest.approx(reference, abs=_SIGMAS * _stderr(S0, K, T, r, q, sigma, "put"))


# --------------------------------------------------------------------------- #
# Signature and return contract — the one caller (engines/pricing.py) relies on #
# a bare float and on the ValueError for an unknown cpflag.                     #
# --------------------------------------------------------------------------- #
def test_returns_a_plain_float_not_a_dict():
    out = price_bermudan_lsmc(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, "p", 10, 2_000, 3, 4, _SEED)
    assert type(out) is float


def test_positional_signature_is_preserved():
    """`S0, K, T, r, q, sigma, cpflag, M, N_paths, degree, n_ex_dates, seed`."""
    positional = price_bermudan_lsmc(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, "p", 10, 2_000, 3, 4, 7)
    keyword = price_bermudan_lsmc(
        S0=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.25, cpflag="p",
        M=10, N_paths=2_000, degree=3, n_ex_dates=4, seed=7,
    )
    assert positional == keyword


@pytest.mark.parametrize("cpflag", ["x", "C", "call", ""])
def test_unknown_cpflag_still_raises(cpflag):
    with pytest.raises(ValueError):
        price_bermudan_lsmc(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, cpflag, 10, 1_000, 3, 4, _SEED)
