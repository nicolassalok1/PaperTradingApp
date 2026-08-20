"""
Tests for the rough-vol forward-variance curve.

Target: app/model/calibration/rough_vol/forward_variance.py (spec 4.4)

ORACLES (all independent of the code under test)
------------------------------------------------
* **Isotonic regression.** ``pool_adjacent_violators`` is checked against the
  closed-form min-max characterisation of the isotonic least-squares fit,
  ``yhat_i = max_{k<=i} min_{l>=i} mean(y[k..l])`` (Ayer et al. 1955), written
  out inline as a brute-force ``O(n^3)`` loop. Two independent algorithms for the
  same unique solution.
* **Reconstruction.** ``int_0^{T_j} xi0`` is recomputed by an inline quadrature
  that never calls ``ForwardVarianceCurve.integrated``: because the default
  ``xi0`` is piecewise constant, a midpoint rule with one cell per knot interval
  is *exact*, so the check has no discretisation error of its own and can be
  asserted at 1e-12.
* **Levels.** The piecewise-constant levels are recomputed inline straight from
  ``(T_j, V_j)`` and compared element by element.

Determinism: no RNG, no Monte-Carlo, no network, no market data.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from app.model.calibration.rough_vol.forward_variance import (
    EXTRAPOLATION_FLAT,
    FLAG_ISOTONIC_REPAIR,
    FLAG_PCHIP_FALLBACK,
    FLAG_SINGLE_MATURITY,
    FLAG_XI0_FLOORED,
    METHOD_PCHIP,
    METHOD_PIECEWISE_CONSTANT,
    REJECT_POSITIVITY,
    REJECT_TOO_FEW_MATURITIES,
    ForwardVarianceConfig,
    ForwardVarianceCurve,
    build_forward_variance_curve,
    forward_variance_report,
    material_turning_points,
    pool_adjacent_violators,
)
from app.model.calibration.rough_vol.variance_swap import VarianceSwapPoint

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures: variance-swap points with a chosen K_var term structure
# ---------------------------------------------------------------------------

MATURITIES = (0.08, 0.25, 0.5, 1.0, 1.5, 2.0)


def points_from_k_var(maturities, k_var_values):
    return [
        VarianceSwapPoint(
            T=float(T),
            k_var=float(k),
            k_var_trunc=float(k),
            n_puts=6,
            n_calls=6,
            F=100.0,
            D=math.exp(-0.03 * float(T)),
        )
        for T, k in zip(maturities, k_var_values)
    ]


def points_from_levels(maturities, levels):
    """Build points whose total variance is exactly ``cumsum(level * dT)``."""
    T = np.asarray(maturities, dtype=float)
    dT = np.diff(np.concatenate(([0.0], T)))
    V = np.cumsum(np.asarray(levels, dtype=float) * dT)
    return points_from_k_var(T, V / T), V


def rising_term_structure():
    """A smooth, strictly rising variance term structure (no repair needed)."""
    T = np.asarray(MATURITIES, dtype=float)
    # V(0,T) = 0.04 T + 0.01 T^2  =>  xi0(t) = 0.04 + 0.02 t  (strictly positive)
    V = 0.04 * T + 0.01 * T * T
    return points_from_k_var(T, V / T), V


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------


def isotonic_minmax_oracle(y):
    """``yhat_i = max_{k<=i} min_{l>=i} mean(y[k..l])`` — brute force, O(n^3)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        best = -np.inf
        for k in range(i + 1):
            worst = np.inf
            for l in range(i, n):
                worst = min(worst, float(np.mean(y[k : l + 1])))
            best = max(best, worst)
        out[i] = best
    return out


def integrate_xi0_independently(curve: ForwardVarianceCurve, T_end: float) -> float:
    """
    Exact quadrature of a piecewise-constant ``xi0`` on ``[0, T_end]``.

    One midpoint evaluation per knot cell: the integrand is constant inside each
    cell, so the rule is exact and this is a genuine oracle for
    ``curve.integrated`` rather than a restatement of it.
    """
    edges = [0.0] + [float(t) for t in curve.T_knots if t < T_end] + [float(T_end)]
    total = 0.0
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            continue
        total += float(curve.xi0(0.5 * (left + right))) * (right - left)
    return total


def expected_levels(T, V):
    """The spec-4.4 piecewise-constant levels, restated inline."""
    T = np.asarray(T, dtype=float)
    V = np.asarray(V, dtype=float)
    levels = [V[0] / T[0]]
    for j in range(1, T.size):
        levels.append((V[j] - V[j - 1]) / (T[j] - T[j - 1]))
    return np.asarray(levels, dtype=float)


# ---------------------------------------------------------------------------
# PAVA
# ---------------------------------------------------------------------------


def test_pool_adjacent_violators_matches_the_minmax_oracle():
    for values in (
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [1.0, 3.0, 2.0, 4.0],
        [0.05, 0.04, 0.06, 0.03, 0.09, 0.08],
        [2.0, 2.0, 2.0],
        [-1.0, -3.0, 0.5],
    ):
        got = pool_adjacent_violators(values)
        want = isotonic_minmax_oracle(values)
        assert np.allclose(got, want, atol=1e-14), (values, got, want)
        assert np.all(np.diff(got) >= -1e-15)
        # The fit preserves the total mass of the data (unit weights).
        assert float(np.sum(got)) == pytest.approx(float(np.sum(values)), abs=1e-12)


def test_pool_adjacent_violators_is_the_identity_on_monotone_data():
    values = [0.01, 0.02, 0.05, 0.05, 0.11]
    got = pool_adjacent_violators(values)
    assert np.array_equal(got, np.asarray(values, dtype=float))


def test_pool_adjacent_violators_rejects_bad_input():
    with pytest.raises(ValueError):
        pool_adjacent_violators([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError):
        pool_adjacent_violators([1.0, 2.0], weights=[1.0, 0.0])
    with pytest.raises(ValueError):
        pool_adjacent_violators([[1.0, 2.0]])
    assert pool_adjacent_violators([]).size == 0


def test_material_turning_points_ignores_sub_band_wiggles():
    assert material_turning_points([1.0, 2.0, 3.0, 4.0], 0.0) == 0
    assert material_turning_points([1.0, 3.0, 2.0, 4.0], 0.0) == 2
    # The same wiggle, now inside the dead band.
    assert material_turning_points([1.0, 3.0, 2.999, 4.0], 0.01) == 0
    assert material_turning_points([1.0], 0.0) == 0


# ---------------------------------------------------------------------------
# The core guarantee: exact reconstruction
# ---------------------------------------------------------------------------


def test_exact_reconstruction_at_every_market_maturity():
    """``int_0^{T_j} xi0 = V_j = T_j K_var(T_j)`` to 1e-12, at every maturity."""
    points, V = rising_term_structure()
    curve = build_forward_variance_curve(points)

    assert curve.method == METHOD_PIECEWISE_CONSTANT
    assert curve.metadata.flags == ()

    levels = expected_levels(MATURITIES, V)
    assert np.allclose(np.asarray(curve.levels), levels, rtol=0, atol=1e-15)

    for T_j, V_j in zip(MATURITIES, V):
        assert curve.integrated(T_j) == pytest.approx(V_j, rel=0, abs=1e-12)
        assert integrate_xi0_independently(curve, T_j) == pytest.approx(V_j, rel=0, abs=1e-12)

    assert max(abs(e) for e in curve.reconstruction_errors()) < 1e-12
    assert curve.metadata.max_reconstruction_error < 1e-12


def test_consistency_integral_equals_T_times_k_var():
    """The spec-4.4 consistency test, stated on ``K_var`` rather than on ``V``."""
    k_var = (0.062, 0.055, 0.049, 0.045, 0.0435, 0.043)
    points = points_from_k_var(MATURITIES, k_var)
    curve = build_forward_variance_curve(points)

    for T_j, k in zip(MATURITIES, k_var):
        assert curve.integrated(T_j) == pytest.approx(T_j * k, rel=0, abs=1e-12)
        assert integrate_xi0_independently(curve, T_j) == pytest.approx(
            T_j * k, rel=0, abs=1e-12
        )


def test_reconstruction_holds_on_a_non_uniform_maturity_grid():
    maturities = (0.019178, 0.076712, 0.170, 0.44, 0.9, 1.73)
    k_var = (0.081, 0.070, 0.063, 0.056, 0.052, 0.050)
    curve = build_forward_variance_curve(points_from_k_var(maturities, k_var))
    for T_j, k in zip(maturities, k_var):
        assert curve.integrated(T_j) == pytest.approx(T_j * k, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Positivity floor
# ---------------------------------------------------------------------------


def test_non_positive_forward_variance_increment_is_floored_and_flagged():
    """
    A *flat* segment of ``V(0,T)`` gives a forward-variance increment of exactly
    zero, hence ``xi0 = 0`` on that interval. The floor kicks in, the flag is
    raised, and the resulting reconstruction gap is reported rather than hidden.

    Note this is reached through a legitimately non-decreasing ``V``: a negative
    ``K_var`` is refused outright upstream (it is non-physical), so it cannot be
    used as a shortcut to this branch -- see
    ``test_negative_k_var_is_refused_not_floored``.
    """
    # V = T*k_var = (0.0032, 0.0032, 0.025, 0.0605): flat across the first two
    # maturities, so the increment on (T_1, T_2] is exactly 0.
    points = points_from_k_var((0.08, 0.25, 0.5, 1.0), (0.04, 0.0128, 0.05, 0.0605))
    curve = build_forward_variance_curve(points)

    assert FLAG_XI0_FLOORED in curve.metadata.flags
    assert curve.metadata.floored_indices == (1,)
    assert curve.levels[1] == pytest.approx(1e-6)
    assert all(level > 0.0 for level in curve.levels)
    assert curve.xi0(0.20) == pytest.approx(1e-6)

    # The exactness claim is explicitly withdrawn on the floored interval, and
    # by exactly the amount the floor injected: raising a zero level to eps_xi
    # over an interval of width dT adds eps_xi * dT to the integral.
    errors = curve.reconstruction_errors()
    expected_gap = 1e-6 * (0.25 - 0.08)
    assert abs(errors[1]) == pytest.approx(expected_gap, rel=1e-9)
    assert curve.metadata.max_reconstruction_error == pytest.approx(
        expected_gap, rel=1e-9
    )
    # ... and the curve stays internally consistent: integrated() is still the
    # true integral of the xi0 it exposes.
    for T_j in curve.T_knots:
        assert curve.integrated(T_j) == pytest.approx(
            integrate_xi0_independently(curve, T_j), rel=0, abs=1e-12
        )
    assert any("plancheré" in m for m in curve.metadata.messages_fr)


def test_flat_total_variance_segment_is_floored():
    """
    ``K_var`` decaying exactly as ``1/T`` gives a *constant* total variance:
    monotone, so the isotonic repair leaves it alone, but every increment past
    the first is zero and the floor must catch them.
    """
    points = points_from_k_var((0.25, 0.5, 1.0), (0.04, 0.02, 0.01))
    curve = build_forward_variance_curve(points)
    assert curve.V_market == curve.V_repaired
    assert curve.metadata.isotonic_adjustments == ()
    assert FLAG_XI0_FLOORED in curve.metadata.flags
    assert curve.metadata.floored_indices == (1, 2)
    assert min(curve.levels) == pytest.approx(1e-6)


def test_isotonic_repair_and_floor_can_fire_together():
    """A strictly decreasing V is pooled flat, which then trips the floor."""
    points = points_from_k_var((0.25, 0.5, 1.0), (0.06, 0.02, 0.005))
    curve = build_forward_variance_curve(points)
    assert FLAG_ISOTONIC_REPAIR in curve.metadata.flags
    assert FLAG_XI0_FLOORED in curve.metadata.flags
    assert min(curve.levels) == pytest.approx(1e-6)


def test_positivity_floor_level_is_configurable():
    # Flat V across the first two maturities => zero increment on (T_1, T_2].
    points = points_from_k_var((0.08, 0.25, 0.5), (0.04, 0.0128, 0.05))
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(eps_xi=1e-4)
    )
    assert curve.levels[1] == pytest.approx(1e-4)
    assert curve.metadata.eps_xi == pytest.approx(1e-4)


def test_negative_k_var_is_refused_not_floored():
    """
    A negative ``K_var`` is non-physical: ``V(0,T)`` is the integral of a
    positive instantaneous variance. It must be REFUSED, not quietly floored.

    The isotonic repair cannot catch it -- a negative-then-rising sequence is
    already non-decreasing, so PAVA is a no-op -- and the piecewise-constant
    level on the following interval would otherwise absorb the whole negative
    offset and come out enormous (a measured case produced xi0 = 12.58, i.e.
    355 % instantaneous vol, carrying no flag at all).
    """
    points = points_from_k_var((0.08, 0.25, 0.5), (-0.01, 0.04, 0.05))
    with pytest.raises(ValueError, match="variance totale négative"):
        build_forward_variance_curve(points)


# ---------------------------------------------------------------------------
# Isotonic repair
# ---------------------------------------------------------------------------


def test_isotonic_repair_on_decreasing_total_variance_is_applied_and_logged():
    """
    ``V(0,T)`` falling with ``T`` is a calendar arbitrage. It must be repaired,
    the repair must match the independent isotonic oracle, and every adjusted
    point must be logged with its size.
    """
    maturities = (0.25, 0.5, 1.0, 2.0)
    k_var = (0.05, 0.04, 0.045, 0.02)
    points = points_from_k_var(maturities, k_var)
    curve = build_forward_variance_curve(points)

    V_market = np.asarray([T * k for T, k in zip(maturities, k_var)])
    assert np.any(np.diff(V_market) < 0.0), "l'oracle doit vraiment être décroissant"

    assert np.allclose(np.asarray(curve.V_market), V_market, atol=1e-15)
    assert np.allclose(
        np.asarray(curve.V_repaired), isotonic_minmax_oracle(V_market), atol=1e-14
    )
    assert np.all(np.diff(np.asarray(curve.V_repaired)) >= -1e-15)

    assert FLAG_ISOTONIC_REPAIR in curve.metadata.flags
    adjustments = curve.metadata.isotonic_adjustments
    assert len(adjustments) >= 1
    for adjustment in adjustments:
        assert adjustment.delta != 0.0
        assert adjustment.v_fitted != adjustment.v_raw
        assert adjustment.T in maturities
        payload = adjustment.to_dict()
        assert payload["delta"] == pytest.approx(payload["v_fitted"] - payload["v_raw"])
        assert payload["k_var_raw"] == pytest.approx(payload["v_raw"] / payload["T"])
    assert curve.metadata.isotonic_max_abs_adjustment == pytest.approx(
        max(abs(a.delta) for a in adjustments)
    )
    assert any("isotone" in m for m in curve.metadata.messages_fr)
    assert any(f"{curve.metadata.isotonic_max_abs_adjustment:.6g}" in m
               for m in curve.metadata.messages_fr)


def test_no_repair_is_logged_when_the_data_is_already_monotone():
    points, _ = rising_term_structure()
    curve = build_forward_variance_curve(points)
    assert curve.metadata.isotonic_adjustments == ()
    assert FLAG_ISOTONIC_REPAIR not in curve.metadata.flags
    assert curve.metadata.isotonic_max_abs_adjustment == 0.0
    assert curve.V_market == curve.V_repaired


# ---------------------------------------------------------------------------
# Smooth (PCHIP) variant
# ---------------------------------------------------------------------------


def test_pchip_variant_is_accepted_on_smooth_data():
    points, V = rising_term_structure()
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(method=METHOD_PCHIP)
    )

    assert curve.method == METHOD_PCHIP
    assert curve.metadata.requested_method == METHOD_PCHIP
    assert curve.metadata.rejection_reasons == ()
    assert FLAG_PCHIP_FALLBACK not in curve.metadata.flags

    # The three documented validations.
    assert curve.metadata.min_xi0 > curve.metadata.eps_xi
    assert curve.metadata.n_turning_points_model <= curve.metadata.n_turning_points_data
    assert curve.metadata.quadrature_max_error < 1e-8

    for T_j, V_j in zip(MATURITIES, V):
        assert curve.integrated(T_j) == pytest.approx(V_j, rel=0, abs=1e-10)

    grid = np.linspace(0.0, MATURITIES[-1], 501)
    xi = curve.xi0(grid)
    assert np.all(xi > 0.0)
    # The true forward variance of this term structure is 0.04 + 0.02 t; PCHIP
    # recovers it closely because it is smooth by construction.
    assert np.max(np.abs(xi - (0.04 + 0.02 * grid))) < 5e-3


def test_pchip_falls_back_when_xi0_would_touch_zero():
    """A flat segment of ``V`` makes ``xi0 = 0``: the smooth variant is refused."""
    points = points_from_k_var((0.25, 0.5, 1.0, 2.0), (0.05, 0.04, 0.045, 0.02))
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(method=METHOD_PCHIP)
    )
    assert curve.method == METHOD_PIECEWISE_CONSTANT
    assert curve.metadata.requested_method == METHOD_PCHIP
    assert REJECT_POSITIVITY in curve.metadata.rejection_reasons
    assert FLAG_PCHIP_FALLBACK in curve.metadata.flags
    assert any("repli" in m for m in curve.metadata.messages_fr)


def test_pchip_falls_back_when_the_interpolant_dips_below_the_floor():
    """
    Every *level* stays above ``eps_xi``, but the monotone interpolant undershoots
    them between the knots. That is exactly the case the positivity validation on
    a fine grid exists for, and the builder must fall back.
    """
    maturities = (0.25, 0.5, 0.75, 1.0, 1.5)
    levels = (0.05, 0.05, 0.0011, 0.05, 0.05)
    points, _ = points_from_levels(maturities, levels)
    config = ForwardVarianceConfig(method=METHOD_PCHIP, eps_xi=1e-3)

    piecewise = build_forward_variance_curve(points)
    assert min(piecewise.levels) > config.eps_xi, "aucun niveau ne doit être plancheré"

    curve = build_forward_variance_curve(points, config=config)
    assert curve.method == METHOD_PIECEWISE_CONSTANT
    assert curve.metadata.rejection_reasons == (REJECT_POSITIVITY,)
    assert FLAG_PCHIP_FALLBACK in curve.metadata.flags
    # The fallback is still exact.
    for T_j, V_j in zip(curve.T_knots, curve.V_repaired):
        assert curve.integrated(T_j) == pytest.approx(V_j, rel=0, abs=1e-12)


def test_pchip_needs_more_than_one_maturity():
    curve = build_forward_variance_curve(
        points_from_k_var((0.5,), (0.04,)),
        config=ForwardVarianceConfig(method=METHOD_PCHIP),
    )
    assert curve.method == METHOD_PIECEWISE_CONSTANT
    assert curve.metadata.rejection_reasons == (REJECT_TOO_FEW_MATURITIES,)
    assert FLAG_SINGLE_MATURITY in curve.metadata.flags


def test_unknown_method_is_rejected():
    points, _ = rising_term_structure()
    with pytest.raises(ValueError, match="méthode inconnue"):
        build_forward_variance_curve(
            points, config=ForwardVarianceConfig(method="spline_cubique")
        )


# ---------------------------------------------------------------------------
# Extrapolation policy
# ---------------------------------------------------------------------------


def test_extrapolation_before_the_first_and_beyond_the_last_maturity():
    points, V = rising_term_structure()
    curve = build_forward_variance_curve(points)
    T_1 = MATURITIES[0]
    T_last = MATURITIES[-1]

    assert curve.extrapolation_policy == EXTRAPOLATION_FLAT
    assert curve.metadata.extrapolation_policy == EXTRAPOLATION_FLAT

    # [0, T_1]: the constant V_1 / T_1.
    front = V[0] / T_1
    for t in (0.0, 0.25 * T_1, T_1):
        assert curve.xi0(t) == pytest.approx(front, rel=0, abs=1e-15)
    assert curve.integrated(0.5 * T_1) == pytest.approx(0.5 * V[0], rel=0, abs=1e-15)

    # Just past T_1 the second level takes over (the intervals are (T_j, T_{j+1}]).
    assert curve.xi0(T_1 + 1e-9) == pytest.approx(curve.levels[1])

    # Beyond T_last: xi0 held flat, integral extended linearly.
    assert curve.xi0(T_last) == pytest.approx(curve.levels[-1])
    assert curve.xi0(3.0 * T_last) == pytest.approx(curve.levels[-1])
    assert curve.integrated(T_last + 0.75) == pytest.approx(
        V[-1] + curve.levels[-1] * 0.75, rel=0, abs=1e-12
    )

    # Negative time has no forward variance.
    assert math.isnan(curve.xi0(-0.1))
    assert math.isnan(curve.integrated(-0.1))


def test_pchip_extrapolation_beyond_the_last_maturity_is_also_flat():
    points, V = rising_term_structure()
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(method=METHOD_PCHIP)
    )
    T_last = MATURITIES[-1]
    tail_level = curve.xi0(T_last)
    assert curve.xi0(T_last + 5.0) == pytest.approx(tail_level)
    assert curve.integrated(T_last + 2.0) == pytest.approx(
        V[-1] + tail_level * 2.0, rel=0, abs=1e-10
    )


def test_single_maturity_curve_is_flat_everywhere():
    curve = build_forward_variance_curve(points_from_k_var((0.75,), (0.0361,)))
    assert FLAG_SINGLE_MATURITY in curve.metadata.flags
    assert curve.levels == (pytest.approx(0.0361),)
    assert curve.xi0(0.0) == pytest.approx(0.0361)
    assert curve.xi0(10.0) == pytest.approx(0.0361)
    assert curve.integrated(0.75) == pytest.approx(0.75 * 0.0361, rel=0, abs=1e-15)


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------


def test_xi0_and_integrated_are_vectorised():
    points, _ = rising_term_structure()
    curve = build_forward_variance_curve(points)

    scalar = curve.xi0(0.37)
    assert isinstance(scalar, float)

    grid = np.array([0.0, 0.05, 0.37, 1.0, 2.0, 7.5])
    vector = curve.xi0(grid)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == grid.shape
    for i, t in enumerate(grid):
        assert vector[i] == pytest.approx(curve.xi0(float(t)))

    two_d = curve.xi0(grid.reshape(2, 3))
    assert two_d.shape == (2, 3)
    assert np.allclose(two_d.reshape(-1), vector)

    integrals = curve.integrated(grid)
    assert integrals.shape == grid.shape
    for i, t in enumerate(grid):
        assert integrals[i] == pytest.approx(curve.integrated(float(t)))

    # A python list is accepted too, and NaN propagates for negative times.
    from_list = curve.xi0([0.37, -1.0])
    assert from_list[0] == pytest.approx(scalar)
    assert math.isnan(from_list[1])


def test_pchip_curve_is_vectorised_the_same_way():
    points, _ = rising_term_structure()
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(method=METHOD_PCHIP)
    )
    grid = np.array([[0.0, 0.4], [1.3, 4.0]])
    values = curve.xi0(grid)
    assert values.shape == (2, 2)
    assert values[0, 0] == pytest.approx(curve.xi0(0.0))
    assert values[1, 1] == pytest.approx(curve.xi0(4.0))
    assert isinstance(curve.xi0(0.4), float)


# ---------------------------------------------------------------------------
# Immutability / shareability
# ---------------------------------------------------------------------------


def test_curve_is_effectively_immutable_and_safe_to_share():
    points, _ = rising_term_structure()
    curve = build_forward_variance_curve(points)

    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.levels = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.metadata = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.metadata.method = "hack"

    # Every field is an immutable container.
    assert isinstance(curve.T_knots, tuple)
    assert isinstance(curve.levels, tuple)
    assert isinstance(curve.V_knots, tuple)
    assert hash(curve) == hash(curve)

    # The cached numpy views cannot be written through.
    for name in ("_T_arr", "_levels_arr", "_V_arr"):
        assert not getattr(curve, name).flags.writeable

    # Output arrays are fresh: mutating one must not disturb the curve.
    grid = np.array([0.1, 0.6, 1.7])
    before = curve.xi0(grid)
    out = curve.xi0(grid)
    out[:] = -999.0
    assert np.allclose(curve.xi0(grid), before)
    assert np.all(np.asarray(curve.levels) > 0.0)


def test_pchip_internals_are_frozen_too():
    points, _ = rising_term_structure()
    curve = build_forward_variance_curve(
        points, config=ForwardVarianceConfig(method=METHOD_PCHIP)
    )
    assert curve.pchip is not None
    with pytest.raises(ValueError):
        curve.pchip.c[0, 0] = 1.0
    with pytest.raises(ValueError):
        curve.pchip.x[0] = 1.0


# ---------------------------------------------------------------------------
# Input handling and reporting
# ---------------------------------------------------------------------------


def test_builder_accepts_a_variance_swap_curve_object():
    from app.model.calibration.rough_vol.variance_swap import VarianceSwapCurve

    points, V = rising_term_structure()
    container = VarianceSwapCurve(points=tuple(points))
    curve = build_forward_variance_curve(container)
    assert len(curve) == len(MATURITIES)
    assert curve.integrated(MATURITIES[-1]) == pytest.approx(V[-1], rel=0, abs=1e-12)


def test_builder_sorts_unsorted_input():
    points, V = rising_term_structure()
    shuffled = [points[3], points[0], points[5], points[1], points[4], points[2]]
    curve = build_forward_variance_curve(shuffled)
    assert list(curve.T_knots) == sorted(MATURITIES)
    for T_j, V_j in zip(MATURITIES, V):
        assert curve.integrated(T_j) == pytest.approx(V_j, rel=0, abs=1e-12)


def test_builder_refuses_degenerate_inputs():
    with pytest.raises(ValueError, match="aucune maturité"):
        build_forward_variance_curve([])
    with pytest.raises(ValueError, match="dupliquées"):
        build_forward_variance_curve(points_from_k_var((0.5, 0.5), (0.04, 0.05)))
    with pytest.raises(ValueError, match="maturités invalides"):
        build_forward_variance_curve(points_from_k_var((0.0, 0.5), (0.04, 0.05)))
    with pytest.raises(ValueError, match="K_var non fini"):
        build_forward_variance_curve(points_from_k_var((0.25, 0.5), (0.04, float("nan"))))
    with pytest.raises(TypeError, match="VarianceSwapPoint"):
        build_forward_variance_curve([object()])


def test_report_and_serialisation_are_json_safe():
    points = points_from_k_var((0.25, 0.5, 1.0, 2.0), (0.05, 0.04, 0.045, 0.02))
    curve = build_forward_variance_curve(points)

    payload = curve.to_dict()
    assert json.dumps(payload)
    assert payload["metadata"]["n_isotonic_adjustments"] >= 1
    assert payload["metadata"]["extrapolation_policy"] == EXTRAPOLATION_FLAT

    report = forward_variance_report(curve)
    assert json.dumps(report)
    assert report["method"] == METHOD_PIECEWISE_CONSTANT
    assert report["n_maturities"] == 4
    assert len(report["xi0_vol"]) == 4
    assert report["metadata"]["rejection_reasons_fr"] == []
