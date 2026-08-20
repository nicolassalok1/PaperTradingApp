"""
Tests for the rough-vol variance-swap strike curve.

Target: app/model/calibration/rough_vol/variance_swap.py (spec 4.3)

ORACLES (all independent of the code under test)
------------------------------------------------
* **Analytic oracle.** Under Black-Scholes with a *constant* volatility
  ``sigma``, the realised variance of the log-contract is deterministic, so the
  fair variance-swap strike is exactly ``sigma^2``. The synthetic chains below
  are therefore priced at one constant ``sigma`` with the repo's own Black-76
  pricer, and ``K_var`` must come back as ``sigma^2``.
* **Continuous-formula oracle.** ``test_continuous_replication_is_exactly_sigma_squared``
  restates the Demeterfi-Derman-Kamal-Zou integral inline, with Black-76 written
  out through ``scipy.stats.norm`` and evaluated by ``scipy.integrate.quad`` — no
  code from the module under test is involved. It confirms the *continuous*
  replication returns ``sigma^2`` to ~1e-12, which is the reference the discrete
  CBOE sum is then shown to converge to.
* Put prices are built by **exact put-call parity** off the call prices, so the
  chain is arbitrage-free by construction and spec 4.1's cleaning drops nothing
  (asserted).

TOLERANCE ACHIEVED, AND WHY IT IS NOT MACHINE PRECISION
-------------------------------------------------------
The *continuous* formula is exact. The **required discrete implementation** is
not, and cannot be: it carries two deliberate approximations, both of them part
of the CBOE specification and both measured here.

1. ``sum_i (dK_i / K_i^2) Q(K_i)`` is a mid-cell rectangle rule for
   ``int Q(K)/K^2 dK``. Its error is ``O(h^2)``;
   ``test_discrete_sum_converges_monotonically`` shows the error falling by a
   factor of ~4 for every halving of the strike spacing, which is exactly that
   order.
2. ``-(1/T)(F/K_0 - 1)^2`` is the second-order truncation of
   ``(2/T)[1 - F/K_0 + ln(F/K_0)]``; the neglected term is ``O((F/K_0 - 1)^3)``.

On the reference grid of ``test_flat_vol_surface_recovers_sigma_squared``
(641 strikes spanning +/- 4 standard deviations, spacing ~0.19 on a forward of
~101.5) the measured error is ``1.2e-6`` in variance — i.e. ``3e-5`` in relative
terms, or ``1.4e-5`` in volatility points. The assertion is made at ``5e-6``
absolute, comfortably above the measured value and far below any market
resolution: a one-cent quote change on a single ATM option moves ``K_var`` by
more than that.

Determinism: no RNG, no Monte-Carlo, no network. Discounting is always injected.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from app.model.calibration.rough_vol.chain_cleaning import (
    CALL,
    PUT,
    CleanChain,
    CleaningConfig,
    CleanQuote,
    ViabilityReport,
    clean_expiry_chain,
)
from app.model.calibration.rough_vol.forward_curve import (
    ParitySlopeDiag,
    ForwardPoint,
    black76_call_price,
    build_forward_curve,
    build_forward_point,
)
from app.model.calibration.rough_vol.variance_swap import (
    FLAG_COARSE_STRIKE_LADDER,
    FLAG_IRREGULAR_SPACING,
    FLAG_K0_SINGLE_SIDED,
    FLAG_NO_PUT_TAIL_ANCHOR,
    FLAG_STRIKES_SKIPPED,
    REASON_NEGATIVE_K_VAR,
    REASON_NO_FORWARD_POINT,
    REASON_NO_STRIKE_BELOW_FORWARD,
    REASON_TOO_FEW_STRIKES,
    REASON_VIABILITY_FAILED,
    TAIL_STOP_INTEGRAND,
    VarianceSwapConfig,
    VarianceSwapCurve,
    VarianceSwapFailure,
    VarianceSwapPoint,
    black76_put_price,
    build_variance_swap_curve,
    build_variance_swap_point,
    variance_swap_report,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Known market — one constant volatility, so K_var must equal VOL**2
# ---------------------------------------------------------------------------

S0 = 100.0
T = 0.5
R = 0.03
VOL = 0.20
D = math.exp(-R * T)
F = S0 * math.exp(R * T)
SD = VOL * math.sqrt(T)
EXPIRY_TS = 1_782_864_000


def call_price(K: float) -> float:
    return black76_call_price(F=F, K=float(K), T=T, D=D, vol=VOL)


def put_price(K: float) -> float:
    """Exact put-call parity: ``P = C - D (F - K)``."""
    return call_price(K) - D * (F - float(K))


def make_row(
    option_type: str,
    K: float,
    price: float,
    *,
    spread_rel: float = 0.02,
    maturity: float = T,
    expiry_ts: int = EXPIRY_TS,
    vendor_iv: float = 0.99,
) -> dict:
    half = 0.5 * float(spread_rel)
    return {
        "underlying": "TEST",
        "contractSymbol": f"TEST-{option_type}-{K:g}-{maturity:g}",
        "expiry": "2026-07-01",
        "expiry_ts": int(expiry_ts),
        "T": float(maturity),
        "strike": float(K),
        # Deliberately absurd vendor IV: K_var must never read it.
        "iv": float(vendor_iv),
        "bid": price * (1.0 - half),
        "ask": price * (1.0 + half),
        "lastPrice": price,
        "openInterest": 25,
        "volume": 7,
        "inTheMoney": False,
        "type": option_type,
        "S0": S0,
    }


def synthetic_chain(
    strikes,
    *,
    spread_rel: float = 0.02,
    vendor_iv: float = 0.99,
    drop_call_at: float | None = None,
    drop_put_at: float | None = None,
) -> CleanChain:
    """Cleaned chain of the known flat-vol market; asserts nothing was dropped."""
    def _keep(side: str, K: float) -> bool:
        target = drop_call_at if side == CALL else drop_put_at
        return target is None or abs(float(K) - float(target)) > 1e-12

    rows = [
        make_row(CALL, K, call_price(K), spread_rel=spread_rel, vendor_iv=vendor_iv)
        for K in strikes
        if _keep(CALL, K)
    ]
    rows += [
        make_row(PUT, K, put_price(K), spread_rel=spread_rel, vendor_iv=vendor_iv)
        for K in strikes
        if _keep(PUT, K)
    ]
    chain = clean_expiry_chain(rows, config=CleaningConfig())
    assert chain.removals == (), f"le chantier de test doit être propre: {chain.removals_by_reason()}"
    return chain


def forward_point_for(chain: CleanChain, *, expect_F: bool = True) -> ForwardPoint:
    point = build_forward_point(chain, D=D, S0=S0)
    assert point is not None
    if expect_F:
        assert point.F == pytest.approx(F, rel=1e-9)
    return point


def linear_strikes(n_std: float, n: int) -> np.ndarray:
    lo = F * math.exp(-float(n_std) * SD)
    hi = F * math.exp(+float(n_std) * SD)
    return np.linspace(lo, hi, int(n))


def k_var_of(
    strikes,
    *,
    drop_call_at: float | None = None,
    drop_put_at: float | None = None,
    forward_override: float | None = None,
):
    """
    ``K_var`` on the known flat-vol market.

    ``drop_call_at`` / ``drop_put_at`` remove one leg at a chosen strike, so the
    ``K_0`` single-sided path can be exercised. ``forward_override`` substitutes a
    deliberately wrong forward, standing in for a spec-4.2 parity failure.

    Returns a ``VarianceSwapPoint`` normally; the refusal paths return a
    ``VarianceSwapFailure``, so callers that expect success assert on the type.
    """
    chain = synthetic_chain(strikes, drop_call_at=drop_call_at, drop_put_at=drop_put_at)
    point = forward_point_for(chain, expect_F=forward_override is None)
    if forward_override is not None:
        point = dataclasses.replace(point, F=float(forward_override))
    result = build_variance_swap_point(chain, point)
    if drop_call_at is None and drop_put_at is None and forward_override is None:
        assert isinstance(result, VarianceSwapPoint), getattr(result, "message_fr", result)
    return result


# ---------------------------------------------------------------------------
# Oracle guards
# ---------------------------------------------------------------------------


def test_known_market_is_self_consistent():
    """The oracle itself: F = S0 e^{rT}, D = e^{-rT}, and parity holds."""
    assert F == pytest.approx(101.5113064615719, abs=1e-9)
    assert math.exp(-(-math.log(D) / T) * T) == pytest.approx(D, abs=1e-15)
    K = 95.0
    assert call_price(K) - put_price(K) == pytest.approx(D * (F - K), abs=1e-12)
    assert put_price(K) > 0.0


def test_continuous_replication_is_exactly_sigma_squared():
    """
    Independent oracle: the DDKZ integral, restated inline with scipy.stats.norm
    and integrated by scipy.integrate.quad, returns exactly ``VOL**2``.

    Nothing from ``variance_swap.py`` is used here. This pins the target that the
    discrete CBOE sum is required to converge to.
    """

    def black76_call(K: float) -> float:
        sqrt_t = VOL * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * VOL * VOL * T) / sqrt_t
        d2 = d1 - sqrt_t
        return float(D * (F * norm.cdf(d1) - K * norm.cdf(d2)))

    def black76_put(K: float) -> float:
        sqrt_t = VOL * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * VOL * VOL * T) / sqrt_t
        d2 = d1 - sqrt_t
        return float(D * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))

    put_leg, _ = quad(lambda K: black76_put(K) / (K * K), 1e-8, F, limit=400)
    call_leg, _ = quad(lambda K: black76_call(K) / (K * K), F, np.inf, limit=400)
    k_var_continuous = (2.0 / (T * D)) * (put_leg + call_leg)

    assert k_var_continuous == pytest.approx(VOL * VOL, abs=1e-12)


def test_black76_put_helper_matches_an_independent_pricer():
    """``black76_put_price`` is parity off the Phase-1 call; check it inline."""
    for K in (70.0, 95.0, 101.5113064615719, 130.0):
        sqrt_t = VOL * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * VOL * VOL * T) / sqrt_t
        d2 = d1 - sqrt_t
        expected = float(D * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))
        got = black76_put_price(F=F, K=K, T=T, D=D, vol=VOL)
        assert got == pytest.approx(expected, abs=1e-11, rel=1e-11)


# ---------------------------------------------------------------------------
# The analytic oracle: flat-vol surface => K_var == sigma**2
# ---------------------------------------------------------------------------


def test_flat_vol_surface_recovers_sigma_squared():
    """
    Reference grid: 641 strikes over +/- 4 standard deviations.

    Measured error ~1.2e-6 in variance; asserted at 5e-6 absolute (see the module
    docstring for why machine precision is unreachable with the *required*
    discrete formula).
    """
    point = k_var_of(linear_strikes(4.0, 641))

    assert point.k_var == pytest.approx(VOL * VOL, abs=5e-6)
    assert point.k_var_trunc == pytest.approx(VOL * VOL, abs=5e-6)
    assert math.sqrt(point.k_var) == pytest.approx(VOL, abs=2e-5)

    # The tail anchors must have recovered the flat vol they were built with.
    diag = point.diagnostics
    assert diag.put_tail.anchor_vol == pytest.approx(VOL, abs=1e-6)
    assert diag.call_tail.anchor_vol == pytest.approx(VOL, abs=1e-6)
    assert diag.put_tail.stop_reason == TAIL_STOP_INTEGRAND
    assert diag.call_tail.stop_reason == TAIL_STOP_INTEGRAND

    # Structure: every strike below K_0 is a put, every strike above is a call.
    assert point.n_puts + point.n_calls == diag.n_strikes + 1  # K_0 quoted both sides
    assert point.F == pytest.approx(F, rel=1e-13)
    assert point.D == pytest.approx(D, rel=1e-15)
    assert point.total_variance == pytest.approx(T * point.k_var, rel=0, abs=1e-15)


def test_vendor_implied_vols_are_never_read():
    """Two identical chains differing only by their (absurd) vendor IVs agree bit for bit."""
    strikes = linear_strikes(3.0, 161)
    sane = build_variance_swap_point(
        synthetic_chain(strikes, vendor_iv=0.20),
        forward_point_for(synthetic_chain(strikes, vendor_iv=0.20)),
    )
    absurd_chain = synthetic_chain(strikes, vendor_iv=4.75)
    absurd = build_variance_swap_point(absurd_chain, forward_point_for(absurd_chain))
    assert isinstance(sane, VarianceSwapPoint)
    assert isinstance(absurd, VarianceSwapPoint)
    assert absurd.k_var == sane.k_var
    assert absurd.k_var_trunc == sane.k_var_trunc


def test_discrete_sum_converges_monotonically():
    """
    Refine the strike grid on a FIXED range: the error must fall monotonically,
    at the ``O(h^2)`` rate of the mid-cell rectangle rule.
    """
    errors = []
    for n in (41, 81, 161, 321, 641):
        point = k_var_of(linear_strikes(4.0, n))
        errors.append(abs(point.k_var_trunc - VOL * VOL))

    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"la suite d'erreurs n'est pas monotone: {errors}"
        # Halving h must divide the error by ~4 (second order).
        assert 2.5 < coarse / fine < 6.0, f"ordre de convergence inattendu: {errors}"

    assert errors[-1] < 5e-6

    # The tail-completed value converges as well.
    tail_errors = [abs(k_var_of(linear_strikes(4.0, n)).k_var - VOL * VOL) for n in (81, 321)]
    assert tail_errors[1] < tail_errors[0]


# ---------------------------------------------------------------------------
# Missing strikes / irregular spacing are absorbed by dK_i
# ---------------------------------------------------------------------------


def test_missing_interior_strikes_are_absorbed_by_delta_k():
    """
    Drop one interior strike out of three: ``dK_i`` widens on the survivors and
    the result barely moves. No special-casing anywhere in the module.
    """
    dense = linear_strikes(3.0, 121)
    holed = np.array([K for i, K in enumerate(dense) if i % 3 != 1], dtype=float)
    assert holed.size < dense.size

    full = k_var_of(dense)
    partial = k_var_of(holed)

    assert partial.diagnostics.n_strikes < full.diagnostics.n_strikes
    assert partial.diagnostics.dk_max > full.diagnostics.dk_max
    # "Barely moves": less than 0.2 % of the variance, i.e. under one basis point
    # of volatility, for a third of the ladder removed.
    assert abs(partial.k_var / full.k_var - 1.0) < 2e-3
    assert partial.k_var == pytest.approx(VOL * VOL, abs=1.5e-4)


def test_irregular_spacing_is_handled_and_flagged():
    """A deliberately non-uniform ladder still lands on sigma^2 and is flagged."""
    tight = np.arange(90.0, 115.0 + 0.1, 1.0)
    wide = np.concatenate([np.arange(62.0, 90.0, 4.0), np.arange(120.0, 168.0, 4.0)])
    strikes = np.unique(np.concatenate([tight, wide]))

    point = k_var_of(strikes)
    assert FLAG_IRREGULAR_SPACING in point.flags
    assert point.diagnostics.dk_ratio > 3.0
    assert point.k_var == pytest.approx(VOL * VOL, abs=5e-4)


def test_one_sided_ladder_skips_the_missing_leg_and_flags_it():
    """
    A strike quoted only on the wrong side of ``K_0`` is dropped from the ladder
    (there is no OTM price there) and reported in ``skipped_strikes``.
    """
    strikes = np.arange(62.0, 168.0, 1.0)
    rows = [make_row(CALL, K, call_price(K)) for K in strikes]
    rows += [make_row(PUT, K, put_price(K)) for K in strikes if K != 80.0]
    chain = clean_expiry_chain(rows, config=CleaningConfig())
    assert chain.removals == ()

    point = build_variance_swap_point(chain, forward_point_for(chain))
    assert isinstance(point, VarianceSwapPoint)
    assert FLAG_STRIKES_SKIPPED in point.flags
    assert point.diagnostics.skipped_strikes == (80.0,)
    assert point.k_var == pytest.approx(VOL * VOL, abs=5e-4)


# ---------------------------------------------------------------------------
# The (F/K_0 - 1)^2 correction
# ---------------------------------------------------------------------------


def test_k0_is_the_largest_strike_at_or_below_the_forward_not_the_nearest():
    """
    ``F = 101.5113``. On an integer ladder the *nearest* strike is 102 (distance
    0.489) but ``K_0`` must be **101** (distance 0.511) — the largest strike at
    or below F. Getting this wrong silently flips the sign of ``F/K_0 - 1``.
    """
    strikes = np.arange(62.0, 168.0, 1.0)
    assert 101.0 in strikes and 102.0 in strikes
    assert abs(102.0 - F) < abs(101.0 - F)  # the nearest strike really is above F

    point = k_var_of(strikes)
    assert point.diagnostics.K_0 == 101.0
    assert point.diagnostics.forward_over_k0 > 1.0

    # The pipeline uses the EXACT Demeterfi-Derman-Kamal-Zou first bracket at
    # S* = K_0, not CBOE's second-order truncation. Oracle, written inline from
    # the continuous formula: (2/T)*[1 - F/K_0 + ln(F/K_0)], i.e. the subtracted
    # term is -(2/T)*[ln(F/K_0) - x] with x = F/K_0 - 1.
    x = F / 101.0 - 1.0
    exact = -(2.0 / T) * (math.log(F / 101.0) - x)
    assert point.diagnostics.correction_term == pytest.approx(exact, rel=1e-12)
    assert point.diagnostics.correction_term > 0.0

    # CBOE's truncated form is still reported for comparability. Expanding
    # x - ln(1+x) = x^2/2 - x^3/3 + x^4/4 - ... gives
    #     exact = (1/T)*x^2 - (2/(3T))*x^3 + ...  <  (1/T)*x^2 = cboe   (x > 0)
    # The correction is SUBTRACTED, so CBOE's larger term biases K_var
    # one-sidedly LOW, and the neglected O(x^3)/T grows without bound as T -> 0.
    cboe = (1.0 / T) * x * x
    assert point.diagnostics.correction_term_cboe == pytest.approx(cboe, rel=1e-12)
    assert point.diagnostics.correction_term < point.diagnostics.correction_term_cboe
    # ... and the gap is the third-order term to leading order. 1 % is loose
    # enough for the neglected x^4 (x ~ 5e-3 here, so the next term is ~0.4 %)
    # and still tight enough that a wrong power would miss by orders of magnitude.
    assert cboe - point.diagnostics.correction_term == pytest.approx(
        (2.0 / (3.0 * T)) * x**3, rel=1e-2
    )

    assert point.k_var == pytest.approx(VOL * VOL, abs=5e-4)


def test_correction_term_vanishes_when_k0_equals_the_forward():
    """Same ladder plus a strike exactly at F: the correction collapses to zero."""
    integer_ladder = np.arange(62.0, 168.0, 1.0)
    with_forward = np.unique(np.concatenate([integer_ladder, np.array([F])]))

    offset = k_var_of(integer_ladder)
    aligned = k_var_of(with_forward)

    assert aligned.diagnostics.K_0 == pytest.approx(F, rel=1e-15)
    assert aligned.diagnostics.correction_term == pytest.approx(0.0, abs=1e-18)
    assert offset.diagnostics.correction_term > 1e-6

    # Both must land on the same variance: the correction is doing its job.
    assert aligned.k_var == pytest.approx(VOL * VOL, abs=5e-4)
    assert offset.k_var == pytest.approx(aligned.k_var, abs=2e-4)

    # And dropping the correction from the offset ladder would be visibly worse.
    uncorrected = offset.k_var + offset.diagnostics.correction_term
    assert abs(uncorrected - VOL * VOL) > abs(offset.k_var - VOL * VOL)


# ---------------------------------------------------------------------------
# Truncation diagnostic
# ---------------------------------------------------------------------------


def test_truncation_gap_is_positive_and_shrinks_as_the_range_widens():
    """
    ``K_var^tail >= K_var^trunc`` (the tails only add positive mass), and the gap
    shrinks strictly as the quoted range widens. Both the sign and the direction
    are asserted.
    """
    gaps = []
    values = []
    for n_std in (1.5, 2.0, 2.5, 3.0, 3.5):
        lo = round(F * math.exp(-n_std * SD))
        hi = round(F * math.exp(+n_std * SD))
        point = k_var_of(np.arange(lo, hi + 0.5, 1.0))
        assert point.k_var >= point.k_var_trunc
        assert point.truncation_error >= 0.0
        assert point.diagnostics.truncation_error == pytest.approx(point.truncation_error)
        gaps.append(point.truncation_error)
        values.append(point.k_var)

    for wide_gap, wider_gap in zip(gaps, gaps[1:]):
        assert wider_gap < wide_gap, f"l'écart de troncature ne décroît pas: {gaps}"

    assert gaps[0] > 1e-4  # a narrow chain really is materially truncated
    assert gaps[-1] < 1e-5

    # Tail completion makes K_var essentially independent of the quoted range.
    assert max(values) - min(values) < 1e-5
    for value in values:
        assert value == pytest.approx(VOL * VOL, abs=1e-4)


def test_tail_is_left_empty_and_flagged_when_no_reliable_otm_quote_exists():
    """
    Every put is one-sided (zero bid), so no put carries a mid the flat-IV
    extrapolation could be anchored on. The put wing then contributes exactly
    nothing, the point is flagged and a French message says so. No number is
    invented for the missing wing.

    The CBOE zero-bid wall of spec 4.1 is switched off here (its documented
    ``zero_bid_stop_count=0`` escape) purely so the one-sided quotes survive far
    enough to reach this code path.
    """
    strikes = np.arange(62.0, 168.0, 1.0)
    rows = [make_row(CALL, K, call_price(K)) for K in strikes]
    for K in strikes:
        price = put_price(K)
        row = make_row(PUT, K, price)
        row["bid"] = 0.0
        row["ask"] = 2.0 * price  # keeps the mid on the true price
        rows.append(row)
    chain = clean_expiry_chain(rows, config=CleaningConfig(zero_bid_stop_count=0))
    assert chain.viability.usable_for_kvar
    assert all(q.one_sided for q in chain.puts)

    point = build_variance_swap_point(chain, fake_forward_point())
    assert isinstance(point, VarianceSwapPoint)
    assert FLAG_NO_PUT_TAIL_ANCHOR in point.flags
    assert point.diagnostics.put_tail.integral == 0.0
    assert not point.diagnostics.put_tail.available
    assert point.diagnostics.call_tail.available
    assert any("Aile put" in message for message in point.diagnostics.messages_fr)
    # The call wing is still completed, so the point is not simply the truncated
    # value either: only the unanchored side is dropped.
    assert point.k_var > point.k_var_trunc


# ---------------------------------------------------------------------------
# Refusals — flagged failures, never a number
# ---------------------------------------------------------------------------


def hand_made_quote(option_type: str, K: float, mid: float, *, one_sided: bool = False) -> CleanQuote:
    half = 0.01 * mid
    return CleanQuote(
        option_type=option_type,
        strike=float(K),
        T=T,
        bid=0.0 if one_sided else mid - half,
        ask=mid + half,
        mid=float(mid),
        spread_abs=2.0 * half,
        spread_rel=2.0 * half / mid,
        volume=10.0,
        open_interest=100.0,
        vendor_iv=0.2,
        one_sided=bool(one_sided),
        contract_symbol=f"HAND-{option_type}-{K:g}",
    )


def hand_made_chain(
    calls, puts, *, usable_for_kvar: bool, reasons: tuple[str, ...] = ()
) -> CleanChain:
    """A CleanChain assembled by hand so the viability verdict can be pinned."""
    return CleanChain(
        T=T,
        underlying="TEST",
        expiry="2026-07-01",
        S0=S0,
        forward_ref=S0,
        calls=tuple(sorted(calls, key=lambda q: q.strike)),
        puts=tuple(sorted(puts, key=lambda q: q.strike)),
        removals=(),
        viability=ViabilityReport(
            forward_ref=S0,
            n_otm_calls=len(calls),
            n_otm_puts=len(puts),
            n_strikes_near_atm=len(calls) + len(puts),
            usable_for_kvar=bool(usable_for_kvar),
            usable_for_skew=True,
            reasons=reasons,
        ),
    )


def fake_forward_point(*, forward: float = F, discount: float = D) -> ForwardPoint:
    diag = ParitySlopeDiag(
        method="test_fixture",
        n_pairs=0,
        slope=-discount,
        slope_expected=-discount,
        slope_abs_error=0.0,
        slope_rel_error=0.0,
        implied_discount=discount,
        intercept=forward * discount,
        forward_free_slope=forward,
        forward_minus_free=0.0,
        r_squared=1.0,
        residual_rmse=0.0,
    )
    return ForwardPoint(
        T=T,
        F=float(forward),
        D=float(discount),
        r=-math.log(discount) / T,
        q_implied=0.0,
        parity_slope_diag=diag,
        S0=S0,
        discount_source="explicit_D",
    )


def test_chain_failing_spec_41_viability_is_refused_in_french():
    chain = hand_made_chain(
        [hand_made_quote(CALL, 105.0, 2.0)],
        [hand_made_quote(PUT, 95.0, 1.5)],
        usable_for_kvar=False,
        reasons=("insufficient_otm_puts",),
    )
    outcome = build_variance_swap_point(chain, fake_forward_point())

    assert isinstance(outcome, VarianceSwapFailure)
    assert not isinstance(outcome, VarianceSwapPoint)
    assert outcome.reason == REASON_VIABILITY_FAILED
    assert "refusé" in outcome.message_fr
    assert "viabilité" in outcome.message_fr
    assert "insufficient_otm_puts" in outcome.message_fr
    assert outcome.to_dict()["reason_fr"].startswith("Échéance non exploitable")


def test_degenerate_chain_with_two_usable_strikes_is_a_flagged_failure():
    """
    Viability is forced to pass, yet only two strikes survive the ladder: the
    integral is degenerate and the module must refuse rather than return a
    number that no one could interpret.
    """
    chain = hand_made_chain(
        [hand_made_quote(CALL, 110.0, 1.10)],
        [hand_made_quote(PUT, 95.0, 1.90)],
        usable_for_kvar=True,
    )
    outcome = build_variance_swap_point(chain, fake_forward_point())

    assert isinstance(outcome, VarianceSwapFailure)
    assert outcome.reason == REASON_TOO_FEW_STRIKES
    assert "dégénérée" in outcome.message_fr
    assert "2 strike(s) exploitable(s)" in outcome.message_fr
    assert outcome.detail["n_strikes"] == 2.0
    assert not hasattr(outcome, "k_var")


def test_single_strike_chain_is_also_refused():
    chain = hand_made_chain(
        [], [hand_made_quote(PUT, 95.0, 1.90)], usable_for_kvar=True
    )
    outcome = build_variance_swap_point(chain, fake_forward_point())
    assert isinstance(outcome, VarianceSwapFailure)
    assert outcome.reason == REASON_TOO_FEW_STRIKES


def test_chain_entirely_above_the_forward_is_refused():
    chain = hand_made_chain(
        [hand_made_quote(CALL, K, 1.0) for K in (110.0, 115.0, 120.0, 125.0)],
        [],
        usable_for_kvar=True,
    )
    outcome = build_variance_swap_point(chain, fake_forward_point())
    assert isinstance(outcome, VarianceSwapFailure)
    assert outcome.reason == REASON_NO_STRIKE_BELOW_FORWARD
    assert "K_0" in outcome.message_fr


def test_invalid_forward_and_discount_are_refused():
    chain = synthetic_chain(np.arange(80.0, 125.0, 1.0))
    bad_forward = build_variance_swap_point(chain, fake_forward_point(forward=float("nan")))
    assert isinstance(bad_forward, VarianceSwapFailure)
    assert "Forward invalide" in bad_forward.to_dict()["reason_fr"]

    bad_discount = build_variance_swap_point(
        chain, fake_forward_point(), D=0.0
    )
    assert isinstance(bad_discount, VarianceSwapFailure)
    assert "actualisation" in bad_discount.message_fr


# ---------------------------------------------------------------------------
# Public curve contract (spec 5)
# ---------------------------------------------------------------------------


def multi_expiry_chains():
    """Three expiries of the same flat-vol market, distinct ``expiry_ts``."""
    chains = []
    for i, maturity in enumerate((0.25, 0.5, 1.0)):
        discount = math.exp(-R * maturity)
        forward = S0 * math.exp(R * maturity)
        strikes = np.arange(round(forward * 0.65), round(forward * 1.55), 1.0)
        rows = []
        for K in strikes:
            call = black76_call_price(F=forward, K=float(K), T=maturity, D=discount, vol=VOL)
            put = call - discount * (forward - float(K))
            rows.append(make_row(CALL, K, call, maturity=maturity, expiry_ts=EXPIRY_TS + i))
            rows.append(make_row(PUT, K, put, maturity=maturity, expiry_ts=EXPIRY_TS + i))
        chain = clean_expiry_chain(rows, config=CleaningConfig(), T=maturity)
        assert chain.removals == ()
        chains.append(chain)
    return chains


def test_build_variance_swap_curve_positional_contract():
    """``build_variance_swap_curve(option_surface, forward_curve, rates)``."""
    chains = multi_expiry_chains()
    discounts = {float(c.T): math.exp(-R * float(c.T)) for c in chains}
    forward_curve = build_forward_curve(chains, discounts=discounts, S0=S0)
    assert len(forward_curve) == 3

    curve = build_variance_swap_curve(chains, forward_curve, None)

    assert isinstance(curve, VarianceSwapCurve)
    assert len(curve) == 3
    assert curve.failures == ()
    assert list(curve.maturities) == sorted(curve.maturities)
    for point, maturity in zip(curve, (0.25, 0.5, 1.0)):
        assert point.T == pytest.approx(maturity)
        assert point.k_var == pytest.approx(VOL * VOL, abs=5e-4)
        assert point.total_variance == pytest.approx(maturity * VOL * VOL, abs=5e-4)

    # Serialisation must survive the controller's JSON pass.
    payload = json.dumps(curve.to_dict())
    assert "k_var" in payload
    report = variance_swap_report(curve)
    assert report["n_points"] == 3
    assert report["n_failures"] == 0
    assert len(report["total_variances"]) == 3


def test_curve_accepts_chain_forward_point_pairs():
    chains = multi_expiry_chains()
    discounts = {float(c.T): math.exp(-R * float(c.T)) for c in chains}
    forward_curve = build_forward_curve(chains, discounts=discounts, S0=S0)
    pairs = list(zip(chains, forward_curve))

    from_pairs = build_variance_swap_curve(pairs, None, None)
    from_chains = build_variance_swap_curve(chains, forward_curve, None)

    assert [p.k_var for p in from_pairs] == [p.k_var for p in from_chains]


def test_curve_records_a_failure_when_no_forward_point_matches():
    chains = multi_expiry_chains()
    discounts = {float(c.T): math.exp(-R * float(c.T)) for c in chains}
    forward_curve = build_forward_curve(chains[:2], discounts=discounts, S0=S0)

    curve = build_variance_swap_curve(chains, forward_curve, None)
    assert len(curve) == 2
    assert len(curve.failures) == 1
    assert curve.failures[0].reason == REASON_NO_FORWARD_POINT
    assert "courbe forward" in curve.failures[0].message_fr


def test_rates_override_replaces_the_discount_factor():
    """
    ``rates`` pins ``r(T)``; ``D`` is then ``exp(-rT)`` through spec 4.2's
    ``resolve_discount``, never a guessed flat rate applied to ``e^{rT}``.
    """
    chains = multi_expiry_chains()
    discounts = {float(c.T): math.exp(-R * float(c.T)) for c in chains}
    forward_curve = build_forward_curve(chains, discounts=discounts, S0=S0)

    override = {float(c.T): 0.12 for c in chains}
    curve = build_variance_swap_curve(chains, forward_curve, override)

    for point in curve:
        assert point.D == pytest.approx(math.exp(-0.12 * point.T), rel=1e-15)
    # A larger discount rate inflates 1/D, hence K_var: the override really bites.
    baseline = build_variance_swap_curve(chains, forward_curve, None)
    for shifted, base in zip(curve, baseline):
        assert shifted.k_var > base.k_var


def test_curve_uses_forward_point_discount_by_default():
    chains = multi_expiry_chains()
    discounts = {float(c.T): math.exp(-R * float(c.T)) for c in chains}
    forward_curve = build_forward_curve(chains, discounts=discounts, S0=S0)
    curve = build_variance_swap_curve(chains, forward_curve)
    for point, forward in zip(curve, forward_curve):
        assert point.D == forward.D


def test_curve_rejects_unknown_surface_entries():
    with pytest.raises(TypeError, match="CleanChain"):
        build_variance_swap_curve([object()], [])


def test_config_min_usable_strikes_is_honoured():
    """Raising the minimum turns a perfectly good expiry into an explicit refusal."""
    chain = synthetic_chain(np.arange(90.0, 115.0, 1.0))
    strict = VarianceSwapConfig(min_usable_strikes=999)
    outcome = build_variance_swap_point(chain, forward_point_for(chain), config=strict)
    assert isinstance(outcome, VarianceSwapFailure)
    assert outcome.reason == REASON_TOO_FEW_STRIKES


# ---------------------------------------------------------------------------
# Strike-grid discretisation bias, K_0 parity recovery, and the K_var refusal
#
# Oracle for the bias: the discrete replication portfolio interpolates the log
# payoff PIECEWISE LINEARLY between listed strikes. For a convex payoff the
# linear interpolant sits above the function, and integrating the interpolation
# error f''*(K-K_i)(K_{i+1}-K)/2 against the risk-neutral density with
# f'' = 2/K^2 gives, per cell, f''*h^2/12 -- i.e. a total-variance excess of
#     V_bias = h^2 / (6 F^2)
# independent of T and of the vol level. Verified here against the module.
# ---------------------------------------------------------------------------
def test_discretisation_bias_matches_the_closed_form_and_is_flagged():
    """The bias is h^2/(6F^2) in V, hence h^2/(6F^2T) in K_var: 1/T-decaying."""
    coarse = np.arange(60.0, 170.0, 10.0)
    point = k_var_of(coarse)

    h = point.diagnostics.h_atm
    expected = (h * h) / (6.0 * F * F) / T
    assert point.diagnostics.discretisation_bias == pytest.approx(expected, rel=1e-12)

    # It really is the error: subtracting it moves k_var towards the truth.
    assert abs(point.k_var - point.diagnostics.discretisation_bias - VOL * VOL) < abs(
        point.k_var - VOL * VOL
    )

    # A ladder this coarse must announce itself rather than pass silently.
    assert FLAG_COARSE_STRIKE_LADDER in point.flags
    assert any("discrétisation" in m for m in point.diagnostics.messages_fr)

    # The bias scales as h^2: halving the spacing must quarter it.
    finer = k_var_of(np.arange(60.0, 170.0, 5.0))
    ratio = point.diagnostics.discretisation_bias / finer.diagnostics.discretisation_bias
    assert ratio == pytest.approx(4.0, rel=1e-9)


def test_fine_ladder_does_not_raise_the_coarse_flag():
    point = k_var_of(np.arange(60.0, 165.0, 0.5))
    assert FLAG_COARSE_STRIKE_LADDER not in point.flags
    assert point.diagnostics.discretisation_bias_rel < 0.02


def test_k0_single_sided_is_recovered_by_parity_not_by_the_lone_mid():
    """
    Dropping one leg at K_0 must NOT move K_var: put-call parity determines the
    missing mid exactly, so the CBOE average is still recoverable.

    Using the lone mid instead offsets Q(K_0) by +/- D*(F-K_0)/2, which the
    2/(T*D) prefactor turns into a 1/T-decaying bias -- the same maturity-
    dependent class of defect as the discretisation bias.
    """
    strikes = np.arange(62.0, 168.0, 1.0)
    both = k_var_of(strikes)
    K0 = both.diagnostics.K_0
    assert K0 < F  # the pivot is genuinely off the forward, so the fix bites

    call_only = k_var_of(strikes, drop_put_at=K0)
    put_only = k_var_of(strikes, drop_call_at=K0)

    assert call_only.k_var == pytest.approx(both.k_var, rel=1e-12)
    assert put_only.k_var == pytest.approx(both.k_var, rel=1e-12)
    # Still reported as a data-quality signal even though it is now corrected.
    assert FLAG_K0_SINGLE_SIDED in call_only.flags
    assert FLAG_K0_SINGLE_SIDED in put_only.flags


def test_negative_k_var_is_refused_with_a_french_message():
    """
    A wrong spec-4.2 forward can drive the CBOE sum negative. K_var is a fair
    variance, so no-arbitrage forbids it: refuse rather than hand spec 4.4 a
    number it would turn into an absurd xi0 on the following interval.
    """
    strikes = np.arange(62.0, 168.0, 1.0)
    point = k_var_of(strikes, forward_override=3.0 * F)

    assert isinstance(point, VarianceSwapFailure)
    assert point.reason == REASON_NEGATIVE_K_VAR
    assert "négatif" in point.message_fr
