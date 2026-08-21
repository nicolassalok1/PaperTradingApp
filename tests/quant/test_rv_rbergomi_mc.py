"""
Tests for the xi0-curve rough Bergomi Monte-Carlo simulator and its pricing layer.

Target
  - app/model/volatility_models/rbergomi/simulator_xi_curve.py   (spec 4.8 simulation)
  - app/model/volatility_models/rbergomi/pricing.py              (spec 4.8 pricing)

HEAVY Monte-Carlo: this module is ``slow``-ONLY (module-level ``pytestmark``),
never ``unit``, so ``-m "unit or smoke"`` cannot drag it into CI.

ORACLES (none of them shares a code path with the code under test)
==================================================================
* **Closed-form Black-Scholes / Black-76**, written inline here with
  ``scipy.stats.norm`` (the module under test uses ``scipy.special.ndtr``), used
  for the ``eta -> 0`` degeneracy and for the deterministic-drift checks.
* **The exact moments of the model**, derived analytically and independent of
  any implementation:
    - ``E[V_t] = xi0(t)`` (this is what pins the ``-0.5 eta^2 t^{2H}``
      compensator: it holds *only* if the compensator matches
      ``Var(W~_t) = t^{2H}`` exactly);
    - ``E[(1/T) int_0^T V du] = (1/T) int_0^T xi0 = K_var(T)`` - the
      variance-swap input the pipeline fed in;
    - ``E[W^S_t W~_t] = rho sqrt(2H)/(H+1/2) t^{H+1/2}`` (MATH_ORACLE.md sec. 5)
      - the cross moment that can only hold if the spot noise and the Volterra
      driver share the SAME Brownian increments, and that a naive
      "correlate the fBm values" construction violates;
    - ``E[D(T) S_T e^{qT}] = S0`` - exact for the left-point (adapted) log-Euler
      scheme, so any deviation is pure Monte-Carlo noise.
* **Structural invariants that are exact on the shared sample** and therefore
  deterministic under a fixed seed: put-call parity strike by strike,
  monotonicity and convexity in the strike (both hold *pathwise*, hence exactly
  in the sample mean), bit-identical reproducibility, exact antithetic
  mirroring, exact float equality of the quoted maturities on the grid.
* **An independent recomputation of the deterministic term structure** from
  ``grid.dt`` and the per-step rate array, used to pin that every ``dt`` in the
  drift / compensator / integrals is the real, NON-UNIFORM step.

STATISTICAL TOLERANCES follow tests/quant/test_mc_pricing.py: every band is
``4 * (the estimator's own reported stderr) + <small absolute floor>``, with the
stderr computed from the independent antithetic **pair means**.  Seeds are
fixed, so every outcome is deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from app.model.calibration.rough_vol.forward_variance import (
    ForwardVarianceCurve,
    build_forward_variance_curve,
)
from app.model.calibration.rough_vol.variance_swap import VarianceSwapPoint
from app.model.volatility_models.rbergomi import pricing as rb_pricing
from app.model.volatility_models.rbergomi.pricing import (
    ESTIMATOR_CONDITIONAL,
    ESTIMATOR_PLAIN,
    forward_variance_expectation,
    implied_vol_surface,
    martingale_diagnostics,
    price_call,
    price_calls_and_puts,
    price_put,
    pricing_report,
    put_call_parity_residual,
    spot_volterra_cross_moment,
    variance_swap_estimate,
)
from app.model.volatility_models.rbergomi.simulator_xi_curve import (
    XI0_CELL_AVERAGE,
    XI0_NODE,
    RBergomiParams,
    RBergomiSimulationError,
    SimulationConfig,
    clear_xi0_cache,
    evaluate_xi0_on_grid,
    resolve_step_rates,
    simulate_rbergomi_xi_curve,
    xi0_cache_info,
)
from app.model.volatility_models.rbergomi.volterra_gaussian import (
    GridConfig,
    build_simulation_grid,
    clear_factor_cache,
    factor_cache_info,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Market-shaped inputs (synthetic, never a real quote)
# ---------------------------------------------------------------------------
S0 = 100.0
Q_DIV = 0.01
R_FLAT = 0.02

#: Listed-style maturities in the repo day count (calendar days / 365).
MATURITIES = (0.1, 0.25, 0.5, 1.0)
#: Variance-swap strikes: a rising term structure, ~20 % to ~23.5 % vol.
K_VARS = (0.04, 0.045, 0.05, 0.055)

STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
#: Evenly spaced ladder, for the butterfly (convexity) test.
LADDER = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

PARAMS = RBergomiParams(H=0.12, eta=1.2, rho=-0.7)

#: 16 384 paths on a 96-step grid: modest by design - the tolerances below are
#: built from the estimator's OWN reported standard error, so statistical power
#: is spent on honesty rather than on brute force.  Every draw stays around a
#: sixth of a second and the ATM standard error is ~0.07 price points.
STAT_PATHS = 16_384
SEED = 20_260_820


# ---------------------------------------------------------------------------
# Independent analytic oracles
# ---------------------------------------------------------------------------
def bs_call_forward(F: float, K: float, total_variance: float, D: float) -> float:
    """Discounted Black-76 call from a TOTAL variance, via ``scipy.stats.norm``."""
    if total_variance <= 0.0:
        return D * max(F - K, 0.0)
    sqrt_w = math.sqrt(total_variance)
    d1 = (math.log(F / K) + 0.5 * total_variance) / sqrt_w
    d2 = d1 - sqrt_w
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_put_forward(F: float, K: float, total_variance: float, D: float) -> float:
    """Discounted Black-76 put from a TOTAL variance."""
    if total_variance <= 0.0:
        return D * max(K - F, 0.0)
    sqrt_w = math.sqrt(total_variance)
    d1 = (math.log(F / K) + 0.5 * total_variance) / sqrt_w
    d2 = d1 - sqrt_w
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def cross_moment_oracle(T: float, H: float, rho: float) -> float:
    """``E[W^S_T W~_T] = rho sqrt(2H)/(H+1/2) T^{H+1/2}`` (MATH_ORACLE.md sec. 5)."""
    return rho * math.sqrt(2.0 * H) / (H + 0.5) * T ** (H + 0.5)


def make_curve(maturities, k_vars) -> ForwardVarianceCurve:
    """Build a spec-4.4 forward-variance curve from synthetic variance-swap points."""
    points = [
        VarianceSwapPoint(
            T=float(T),
            k_var=float(k),
            k_var_trunc=float(k),
            n_puts=8,
            n_calls=8,
            F=S0,
            D=1.0,
        )
        for T, k in zip(maturities, k_vars)
    ]
    return build_forward_variance_curve(points)


# ---------------------------------------------------------------------------
# Module-scoped fixtures: one grid, a couple of path sets, reused everywhere.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def curve() -> ForwardVarianceCurve:
    return make_curve(MATURITIES, K_VARS)


@pytest.fixture(scope="module")
def flat_curve() -> ForwardVarianceCurve:
    """A single-maturity curve: xi0 is constant, so node == cell average."""
    return make_curve((1.0,), (0.04,))


@pytest.fixture(scope="module")
def grid():
    return build_simulation_grid(
        maturities=MATURITIES, config=GridConfig(n_max=96, min_steps=12)
    )


def _simulate(curve, grid, *, params=PARAMS, config=None, r=R_FLAT, q=Q_DIV):
    return simulate_rbergomi_xi_curve(
        S0=S0,
        xi_curve=curve,
        params=params,
        grid=grid,
        r=r,
        q=q,
        config=config if config is not None else SimulationConfig(
            n_paths=STAT_PATHS, antithetic=True, seed=SEED
        ),
    )


@pytest.fixture(scope="module")
def paths(curve, grid):
    """Reference path set: spec-verbatim node evaluation of xi0."""
    return _simulate(curve, grid)


@pytest.fixture(scope="module")
def cell_paths(curve, grid):
    """Same draw, exact cell-averaged xi0 (zero quadrature error)."""
    return _simulate(
        curve,
        grid,
        config=SimulationConfig(
            n_paths=STAT_PATHS,
            antithetic=True,
            seed=SEED,
            xi0_evaluation=XI0_CELL_AVERAGE,
        ),
    )


# ---------------------------------------------------------------------------
# Grid / plumbing premises the statistical tests rely on
# ---------------------------------------------------------------------------
def test_grid_premises_quoted_maturities_exact_and_dt_non_uniform(grid):
    """Every quoted maturity is EXACTLY on the grid and dt is strongly graded."""
    for T in MATURITIES:
        assert grid.t[grid.index_of(T)] == T  # exact float equality, no snapping
    assert grid.dt.min() > 0.0
    # The premise of the non-uniform-dt tests: three orders of magnitude apart.
    assert grid.dt.max() / grid.dt.min() > 100.0
    assert grid.t[-1] == pytest.approx(max(MATURITIES), abs=0.0)
    assert float(np.sum(grid.dt)) == pytest.approx(max(MATURITIES), rel=1e-15)


def test_variance_strictly_positive_pathwise(paths):
    """xi0 positivity carries through the exponential: V > 0 on every node."""
    assert np.all(np.isfinite(paths.variance))
    assert np.all(paths.variance > 0.0)
    assert np.all(np.isfinite(paths.integrated_variance))
    assert np.all(paths.integrated_variance[:, 1:] > 0.0)
    assert np.all(np.isfinite(paths.log_spot))


# ---------------------------------------------------------------------------
# Model-level statistics
# ---------------------------------------------------------------------------
def test_expected_variance_matches_xi0_curve(paths, grid):
    """E[V_t] = xi0(t) at several t - this is what pins the compensator."""
    probe = [
        float(grid.t[3]),
        float(grid.t[15]),
        float(grid.t[grid.index_of(0.25)]),
        float(grid.t[grid.index_of(1.0)]),
    ]
    stats = forward_variance_expectation(paths, times=probe)
    assert np.all(stats["stderr"] > 0.0)
    for t, mean, stderr, xi0 in zip(
        stats["times"], stats["mean"], stats["stderr"], stats["xi0"]
    ):
        assert mean == pytest.approx(xi0, abs=4.0 * stderr + 1e-6), t
    assert np.nanmax(np.abs(stats["z_score"])) < 4.0


def test_expected_variance_would_fail_without_the_compensator(paths, grid):
    """
    The compensator is not decorative: dropping it biases E[V_t] by exp(eta^2 t^{2H}/2).

    Re-deriving V WITHOUT the ``-0.5 eta^2 t^{2H}`` term from the very same
    driver sample must land many standard errors away from xi0 - otherwise the
    test above would pass on a model that has no compensator at all.
    """
    k = grid.index_of(1.0)
    t = float(grid.t[k])
    eta, H = paths.params.eta, paths.params.H
    uncompensated = np.asarray(paths.xi0_on_grid.nodes)[k] * np.exp(
        eta * paths.volterra(t)
    )
    mean, stderr = paths.mean_stderr(uncompensated)
    xi0 = float(paths.xi0_on_grid.nodes[k])
    inflation = math.exp(0.5 * eta * eta * t ** (2.0 * H))
    assert inflation > 1.5
    assert abs(float(mean) - xi0) > 10.0 * float(stderr)
    assert float(mean) == pytest.approx(xi0 * inflation, abs=4.0 * float(stderr))


@pytest.mark.parametrize("rho", [-0.85, 0.35])
def test_spot_volterra_cross_moment(curve, grid, rho):
    """
    E[W^S_t W~_t] = rho sqrt(2H)/(H+1/2) t^{H+1/2} for two rho, one negative.

    Only the shared-driver construction produces this: correlating fBm VALUES
    with the spot noise would leave the cross moment at the wrong level and with
    the wrong power of t.
    """
    params = RBergomiParams(H=0.12, eta=1.2, rho=rho)
    path_set = _simulate(
        curve,
        grid,
        params=params,
        config=SimulationConfig(n_paths=STAT_PATHS, antithetic=True, seed=SEED + 1),
    )
    stats = spot_volterra_cross_moment(path_set)
    for T, mean, stderr, expected in zip(
        stats["maturities"], stats["mean"], stats["stderr"], stats["expected"]
    ):
        oracle = cross_moment_oracle(float(T), params.H, rho)
        # The module's own closed form and the inline oracle must agree exactly.
        assert float(expected) == pytest.approx(oracle, rel=1e-14)
        assert float(mean) == pytest.approx(oracle, abs=4.0 * float(stderr) + 1e-4), T
    # Sign of the correlation is carried, not just the magnitude.
    assert np.all(np.sign(stats["mean"]) == np.sign(rho))


def test_variance_swap_reproduction_exact_in_cell_average_mode(cell_paths, curve):
    """E[(1/T) int V dt] reproduces the K_var input when xi0 is integrated exactly."""
    estimate = variance_swap_estimate(cell_paths, xi_curve=curve)
    assert float(np.max(np.abs(estimate.quadrature_error))) == pytest.approx(
        0.0, abs=1e-15
    )
    for T, k_var, stderr, reference in zip(
        estimate.maturities,
        estimate.k_var,
        estimate.k_var_stderr,
        estimate.xi0_reference,
    ):
        market = float(K_VARS[MATURITIES.index(T)])
        assert float(reference) == pytest.approx(market, rel=1e-12)
        assert float(k_var) == pytest.approx(market, abs=4.0 * float(stderr) + 1e-6), T


def test_variance_swap_node_mode_matches_its_reported_quadrature_error(grid, curve):
    """
    In spec-verbatim node mode the left-point sum of a piecewise-constant xi0
    misses ``int xi0`` at every knot.  The gap is DETERMINISTIC and reported;
    the Monte-Carlo estimate must match ``reference + gap/T``, not ``reference``.

    Node mode is no longer the default (cell average is), so it is requested
    explicitly here.  The reported ``quadrature_error`` is checked against an
    INDEPENDENT recomputation of the left-point sum straight off the public
    curve API -- comparing it against the module's own ``steps`` array would be
    an identity, true for any xi0 evaluation whether right or wrong.
    """
    node_paths = _simulate(
        curve,
        grid,
        config=SimulationConfig(
            n_paths=STAT_PATHS,
            antithetic=True,
            seed=SEED,
            xi0_evaluation=XI0_NODE,
        ),
    )
    estimate = variance_swap_estimate(node_paths, xi_curve=curve)
    assert float(np.max(np.abs(estimate.quadrature_error))) > 0.0

    # Independent oracle: sum_i xi0(t_i) dt_i - int_0^{t_k} xi0, from the curve.
    t_nodes = np.asarray(grid.t, dtype=float)
    dt = np.diff(t_nodes)
    left_sum = np.concatenate(([0.0], np.cumsum(curve.xi0(t_nodes[:-1]) * dt)))
    for T, reported in zip(estimate.maturities, estimate.quadrature_error):
        k = int(np.argmin(np.abs(t_nodes - float(T))))
        oracle = float(left_sum[k] - curve.integrated(float(T)))
        assert float(reported) == pytest.approx(oracle, rel=1e-9, abs=1e-15), T
    for T, k_var, stderr, expected in zip(
        estimate.maturities, estimate.k_var, estimate.k_var_stderr, estimate.expected
    ):
        assert float(k_var) == pytest.approx(
            float(expected), abs=4.0 * float(stderr) + 1e-6
        ), T


def test_martingale_property_at_every_maturity(paths):
    """E[D(T) S_T e^{qT}] = S0 within Monte-Carlo error, all maturities at once."""
    diagnostics = martingale_diagnostics(paths)
    assert diagnostics.maturities == tuple(float(x) for x in MATURITIES)
    assert np.all(diagnostics.deflated_stderr > 0.0)
    for T, mean, stderr in zip(
        diagnostics.maturities,
        diagnostics.deflated_mean,
        diagnostics.deflated_stderr,
    ):
        assert float(mean) == pytest.approx(S0, abs=4.0 * float(stderr)), T
    assert diagnostics.max_abs_z < 4.0


def test_martingale_correction_is_off_by_default_and_exact_when_enabled(curve, grid):
    """The default-OFF flag; when ON, the sample martingale defect is removed."""
    default_paths = _simulate(
        curve,
        grid,
        config=SimulationConfig(n_paths=4096, antithetic=True, seed=SEED + 2),
    )
    assert default_paths.config.martingale_correction is False
    assert default_paths.martingale_ratio is None

    corrected = _simulate(
        curve,
        grid,
        config=SimulationConfig(
            n_paths=4096,
            antithetic=True,
            seed=SEED + 2,
            martingale_correction=True,
        ),
    )
    assert corrected.martingale_ratio is not None
    # The uncorrected sample is already within a hair of the martingale.
    assert float(np.max(np.abs(corrected.martingale_ratio - 1.0))) < 5e-3
    after = martingale_diagnostics(corrected)
    for mean in after.deflated_mean:
        assert float(mean) == pytest.approx(S0, rel=1e-12)


# ---------------------------------------------------------------------------
# Pricing: exact sample-wise invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("estimator", [ESTIMATOR_PLAIN, ESTIMATOR_CONDITIONAL])
def test_put_call_parity_exact_on_the_shared_sample(paths, estimator):
    """
    C - P = D(T) (mean(S_T) - K) to round-off - an identity, not a statistic.

    ``(x)^+ - (-x)^+ = x`` pathwise for the plain estimator, and
    ``Black_call - Black_put = F - K`` identically for the conditional one.
    """
    calls, puts = price_calls_and_puts(paths, strikes=STRIKES, estimator=estimator)
    residual = put_call_parity_residual(calls, puts)
    assert np.max(np.abs(residual)) < 1e-8
    # ... and, for the plain estimator, the sample forward IS mean(S_T).
    if estimator == ESTIMATOR_PLAIN:
        for row, T in enumerate(calls.maturities):
            assert float(calls.sample_forward[row]) == pytest.approx(
                float(np.mean(paths.spot(T))), rel=1e-14
            )
            for col, K in enumerate(calls.strikes[row]):
                lhs = float(calls.price[row, col] - puts.price[row, col])
                rhs = float(
                    calls.discount_factors[row]
                    * (float(np.mean(paths.spot(T))) - float(K))
                )
                assert lhs == pytest.approx(rhs, abs=1e-8)


@pytest.mark.parametrize("estimator", [ESTIMATOR_PLAIN, ESTIMATOR_CONDITIONAL])
def test_call_monotone_and_convex_in_strike(paths, estimator):
    """
    Decreasing and convex in K - both hold PATHWISE, hence exactly in the mean.

    Put prices are increasing and convex in K by the same argument.
    """
    calls = price_call(paths, strikes=LADDER, estimator=estimator)
    puts = price_put(paths, strikes=LADDER, estimator=estimator)
    tol = 1e-10
    for row in range(len(calls.maturities)):
        first_call = np.diff(calls.price[row])
        assert np.all(first_call <= tol)
        assert np.all(np.diff(first_call) >= -tol)  # butterfly >= 0
        first_put = np.diff(puts.price[row])
        assert np.all(first_put >= -tol)
        assert np.all(np.diff(first_put) >= -tol)
    # The ladder is evenly spaced, so the second differences ARE butterflies.
    assert np.all(np.diff(LADDER) == pytest.approx(LADDER[1] - LADDER[0], rel=1e-15))


@pytest.mark.parametrize("estimator", [ESTIMATOR_PLAIN, ESTIMATOR_CONDITIONAL])
def test_no_arbitrage_bounds(paths, estimator):
    """0 <= C, intrinsic <= C <= S0 e^{-qT}, and the mirror bounds for puts."""
    calls, puts = price_calls_and_puts(paths, strikes=STRIKES, estimator=estimator)
    for row, T in enumerate(calls.maturities):
        discount = float(calls.discount_factors[row])
        upper = S0 * math.exp(-Q_DIV * float(T))
        for col, K in enumerate(calls.strikes[row]):
            call = float(calls.price[row, col])
            put = float(puts.price[row, col])
            se_c = float(calls.stderr[row, col])
            se_p = float(puts.stderr[row, col])
            intrinsic_call = max(upper - float(K) * discount, 0.0)
            intrinsic_put = max(float(K) * discount - upper, 0.0)
            assert call >= -4.0 * se_c - 1e-12
            assert put >= -4.0 * se_p - 1e-12
            assert call >= intrinsic_call - 4.0 * se_c - 1e-9
            assert call <= upper + 4.0 * se_c
            assert put >= intrinsic_put - 4.0 * se_p - 1e-9
            assert put <= float(K) * discount + 4.0 * se_p
    assert np.all(calls.stderr >= 0.0)
    assert np.all(np.isfinite(calls.price))


def test_conditional_estimator_agrees_with_plain_and_reduces_variance(paths):
    """
    The mixed estimator must land inside the plain estimator's error band, and
    it must actually reduce the variance (that is its whole purpose).
    """
    plain = price_call(paths, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    conditional = price_call(paths, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    combined = np.sqrt(plain.stderr**2 + conditional.stderr**2)
    deviation = np.abs(plain.price - conditional.price)

    resolvable = plain.stderr > 0.0
    assert np.any(resolvable)
    assert np.all(deviation[resolvable] <= 4.0 * combined[resolvable] + 1e-6)

    # Where NOT A SINGLE path finished in the money the plain sample variance
    # collapses to exactly zero. That is a degenerate error bar, not a precise
    # one - and it is exactly the regime the conditional estimator exists for:
    # it still resolves a small, strictly positive, economically sane value.
    degenerate = ~resolvable
    if np.any(degenerate):
        assert np.all(plain.price[degenerate] == 0.0)
        assert np.all(conditional.price[degenerate] > 0.0)
        assert np.all(conditional.price[degenerate] < 1e-2)

    atm = int(np.argmin(np.abs(STRIKES - S0)))
    ratios = plain.stderr[:, atm] / conditional.stderr[:, atm]
    assert np.all(ratios > 1.2), f"conditional/plain stderr ratios: {ratios}"


def test_antithetic_variance_reduction_is_measured(curve, grid):
    """
    Antithetic sampling MUST reduce the standard error at equal path count.

    Both standard errors are computed from independent samples (pair means when
    antithetic, raw paths otherwise), so the ratio is a like-for-like
    measurement, not an artefact of the estimator.
    """
    common = dict(n_paths=8192, seed=SEED + 3)
    with_anti = _simulate(
        curve, grid, config=SimulationConfig(antithetic=True, **common)
    )
    without = _simulate(
        curve, grid, config=SimulationConfig(antithetic=False, **common)
    )
    priced_anti = price_call(with_anti, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    priced_plain = price_call(without, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    atm = int(np.argmin(np.abs(STRIKES - S0)))
    ratios = priced_plain.stderr[:, atm] / priced_anti.stderr[:, atm]
    assert np.all(np.isfinite(ratios))
    assert np.all(ratios > 1.1), f"antithetic stderr reduction ratios (ATM): {ratios}"
    # And the two price estimates still agree.
    combined = np.sqrt(
        priced_anti.stderr[:, atm] ** 2 + priced_plain.stderr[:, atm] ** 2
    )
    assert np.all(
        np.abs(priced_anti.price[:, atm] - priced_plain.price[:, atm])
        <= 4.0 * combined
    )


def test_mc_convergence_is_order_one_over_sqrt_n(curve, grid):
    """16x paths -> ~4x smaller stderr (generous slack, as in test_mc_pricing)."""
    small = _simulate(
        curve, grid, config=SimulationConfig(n_paths=2048, antithetic=True, seed=101)
    )
    large = _simulate(
        curve, grid, config=SimulationConfig(n_paths=32768, antithetic=True, seed=101)
    )
    atm = int(np.argmin(np.abs(STRIKES - S0)))
    small_price = price_call(small, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    large_price = price_call(large, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    ratio = small_price.stderr[:, atm] / large_price.stderr[:, atm]
    assert np.all(large_price.stderr[:, atm] < small_price.stderr[:, atm])
    assert np.all(ratio > 2.5), f"stderr ratios for a 16x path increase: {ratio}"
    combined = np.sqrt(
        small_price.stderr[:, atm] ** 2 + large_price.stderr[:, atm] ** 2
    )
    assert np.all(
        np.abs(small_price.price[:, atm] - large_price.price[:, atm])
        <= 4.0 * combined
    )


# ---------------------------------------------------------------------------
# eta -> 0: the model must degenerate to Black-Scholes
# ---------------------------------------------------------------------------
def test_eta_to_zero_degenerates_to_black_scholes_flat_iv(curve, grid):
    """
    With eta -> 0 the variance path is deterministic, so the model IS
    Black-Scholes with total variance ``int_0^T xi0`` and the implied-vol
    surface must be FLAT across strikes.

    With ``rho = 0`` the conditional estimator integrates out the only source of
    randomness, so the price is exact to machine precision - no statistical
    slack is needed and the test is a hard equality against the inline
    ``scipy.stats.norm`` oracle.
    """
    degenerate = _simulate(
        curve,
        grid,
        params=RBergomiParams(H=0.12, eta=1e-8, rho=0.0),
        config=SimulationConfig(
            n_paths=2048,
            antithetic=True,
            seed=SEED + 4,
            xi0_evaluation=XI0_CELL_AVERAGE,
        ),
    )
    calls, puts = price_calls_and_puts(
        degenerate, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL
    )
    ivs = implied_vol_surface(calls)
    assert np.all(np.isfinite(ivs))
    for row, T in enumerate(calls.maturities):
        total_variance = float(curve.integrated(float(T)))
        forward = float(degenerate.model_forward(float(T)))
        discount = float(degenerate.discount_factor(float(T)))
        for col, K in enumerate(calls.strikes[row]):
            oracle = bs_call_forward(forward, float(K), total_variance, discount)
            assert float(calls.price[row, col]) == pytest.approx(oracle, abs=1e-8)
            put_oracle = bs_put_forward(forward, float(K), total_variance, discount)
            assert float(puts.price[row, col]) == pytest.approx(put_oracle, abs=1e-8)
        flat_vol = math.sqrt(total_variance / float(T))
        assert np.nanmax(ivs[row]) - np.nanmin(ivs[row]) < 1e-5
        # Brent's own xtol is 1e-6, so that is the floor of this comparison.
        assert float(np.nanmean(ivs[row])) == pytest.approx(flat_vol, abs=2e-6)
    assert float(np.max(calls.stderr)) < 1e-10


def test_eta_to_zero_matches_black_scholes_with_correlation_and_plain_mc(curve, grid):
    """
    Same degeneracy with rho = -0.7 and the PLAIN estimator: the correlation
    cannot matter when the variance is deterministic, so the prices must still
    be Black-Scholes, now within Monte-Carlo error.
    """
    degenerate = _simulate(
        curve,
        grid,
        params=RBergomiParams(H=0.12, eta=1e-8, rho=-0.7),
        config=SimulationConfig(
            n_paths=STAT_PATHS,
            antithetic=True,
            seed=SEED + 5,
            xi0_evaluation=XI0_CELL_AVERAGE,
        ),
    )
    calls = price_call(degenerate, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    for row, T in enumerate(calls.maturities):
        total_variance = float(curve.integrated(float(T)))
        forward = float(degenerate.model_forward(float(T)))
        discount = float(degenerate.discount_factor(float(T)))
        for col, K in enumerate(calls.strikes[row]):
            oracle = bs_call_forward(forward, float(K), total_variance, discount)
            tol = 4.0 * float(calls.stderr[row, col]) + 1e-3
            assert float(calls.price[row, col]) == pytest.approx(oracle, abs=tol)


# ---------------------------------------------------------------------------
# NON-UNIFORM dt: the classic silent bug
# ---------------------------------------------------------------------------
def test_non_uniform_dt_is_used_in_drift_compensator_and_integrals(curve, grid):
    """
    A strongly graded grid + a per-step rate array + the eta -> 0 degeneracy.

    Everything is recomputed here from ``grid.dt`` and the rate array with no
    reference to the simulator's internals.  A constant ``T/n`` anywhere - in the
    drift, in the discount factor or in ``int V dt`` - moves the forward and the
    total variance by percent and this test fails loudly.
    """
    n = grid.n
    dt = np.asarray(grid.dt, dtype=float)
    # A genuinely non-flat term structure of rates, one value per CELL.
    r_steps = 0.005 + 0.05 * (np.asarray(grid.t[:-1]) / float(grid.t[-1]))
    assert r_steps.size == n
    assert r_steps.max() / r_steps.min() > 5.0

    degenerate = simulate_rbergomi_xi_curve(
        S0=S0,
        xi_curve=curve,
        params=RBergomiParams(H=0.12, eta=1e-8, rho=0.0),
        grid=grid,
        r=r_steps,
        q=Q_DIV,
        config=SimulationConfig(
            n_paths=2048,
            antithetic=True,
            seed=SEED + 6,
            xi0_evaluation=XI0_CELL_AVERAGE,
        ),
    )
    assert np.array_equal(degenerate.r_steps, r_steps)

    calls = price_call(degenerate, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    naive_dt = float(grid.t[-1]) / n
    naive_gaps = []
    for row, T in enumerate(calls.maturities):
        k = grid.index_of(float(T))
        # Independent recomputation with the REAL per-step dt.
        integral_r = float(np.sum(r_steps[:k] * dt[:k]))
        discount = math.exp(-integral_r)
        forward = S0 * math.exp(integral_r - Q_DIV * float(T))
        total_variance = float(curve.integrated(float(T)))
        assert float(degenerate.discount_factor(float(T))) == pytest.approx(
            discount, rel=1e-14
        )
        assert float(degenerate.model_forward(float(T))) == pytest.approx(
            forward, rel=1e-14
        )
        naive_gaps.append(float(np.sum(r_steps[:k])) * naive_dt / integral_r)
        for col, K in enumerate(calls.strikes[row]):
            oracle = bs_call_forward(forward, float(K), total_variance, discount)
            assert float(calls.price[row, col]) == pytest.approx(oracle, abs=1e-8)

    # Premise of the test: a constant `T_max / n` step misplaces every interior
    # node (it maps the 0.1y maturity onto k * T_max/n = 0.65y), so the drift
    # integral it produces is several times the true one at the short end. It
    # only happens to agree near the last node, which is why the check is on the
    # WORST maturity, not on each one.
    assert max(naive_gaps) > 3.0

    # int V dt with a deterministic V must be int xi0, which needs the real dt:
    # a constant step would inflate the shortest maturity's total variance by
    # more than a factor 6, so this equality pins every dt in the integral.
    swap = variance_swap_estimate(degenerate, xi_curve=curve)
    for T, k_var in zip(swap.maturities, swap.k_var):
        assert float(k_var) == pytest.approx(
            float(curve.integrated(float(T))) / float(T), rel=1e-10
        )
    naive_total_variance = float(
        np.sum(np.asarray(degenerate.xi0_on_grid.steps)[: grid.index_of(0.1)])
        * naive_dt
    )
    assert naive_total_variance > 5.0 * float(curve.integrated(0.1))


def test_step_rates_from_a_yield_curve_reproduce_its_discount_factors(curve, grid):
    """A curve object is converted to EXACT per-step forward rates."""

    class _Curve:
        """Minimal stand-in for the repo's yield curve (no network, no cache)."""

        @staticmethod
        def discount_factor(T: float) -> float:
            return math.exp(-0.03 * T - 0.01 * T * T)

        @staticmethod
        def zero_rate(T: float) -> float:
            return 0.03 + 0.01 * T

    r_steps = resolve_step_rates(_Curve(), grid)
    assert r_steps.shape == (grid.n,)
    dt = np.asarray(grid.dt, dtype=float)
    for k in range(grid.n + 1):
        rebuilt = math.exp(-float(np.sum(r_steps[:k] * dt[:k])))
        assert rebuilt == pytest.approx(
            _Curve.discount_factor(float(grid.t[k])), rel=1e-13
        )
    path_set = _simulate(
        curve,
        grid,
        r=_Curve(),
        config=SimulationConfig(n_paths=1024, antithetic=True, seed=SEED + 7),
    )
    for T in MATURITIES:
        assert path_set.discount_factor(T) == pytest.approx(
            _Curve.discount_factor(float(T)), rel=1e-13
        )


def test_scalar_and_curveless_rates_are_accepted(grid):
    """Scalar, ``None`` and per-step forms; an ``n+1`` array is rejected."""
    assert np.array_equal(resolve_step_rates(None, grid), np.zeros(grid.n))
    assert np.array_equal(
        resolve_step_rates(0.03, grid), np.full(grid.n, 0.03, dtype=float)
    )
    with pytest.raises(ValueError, match="per-step rates"):
        resolve_step_rates(np.zeros(grid.n + 1), grid)
    with pytest.raises(ValueError, match="finite"):
        resolve_step_rates(float("nan"), grid)


# ---------------------------------------------------------------------------
# Implied volatilities
# ---------------------------------------------------------------------------
def test_implied_vol_inversion_of_model_prices(paths):
    """
    Model call prices invert to finite implied vols with the right skew sign,
    and stay NaN-safe outside the no-arbitrage bounds.
    """
    calls = price_call(paths, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    ivs = implied_vol_surface(calls)
    assert ivs.shape == calls.price.shape
    assert np.all(np.isfinite(ivs))
    assert np.all(ivs > 0.0)
    for row in range(len(calls.maturities)):
        # rho < 0 -> negative skew: low strikes carry the higher implied vol.
        assert ivs[row, 0] > ivs[row, -1]

    # Puts invert through the SAMPLE parity, so wherever both sides resolve they
    # land on the SAME smile -- put-call parity in the volatility domain.
    puts = price_put(paths, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    put_ivs = implied_vol_surface(puts)

    both = np.isfinite(ivs) & np.isfinite(put_ivs)
    assert both.any()
    assert np.allclose(put_ivs[both], ivs[both], atol=1e-5)

    # The put side may legitimately resolve FEWER cells: converting an ITM put
    # (K > F) to its OTM call equivalent subtracts two nearly equal numbers, so
    # the surviving time value can fall below the round-off / Monte-Carlo floor
    # even where the call itself -- whose price IS its time value there -- is
    # perfectly well conditioned. That asymmetry is exactly why the pipeline
    # builds its market surface from the OTM side of each strike
    # (forward_curve.build_otm_surface). The reverse must never happen: a put
    # that resolves where the call does not would mean the guard is inverted.
    assert not np.any(np.isfinite(put_ivs) & ~np.isfinite(ivs))


def test_implied_vol_is_nan_safe_outside_no_arbitrage_bounds(paths):
    """
    Un-invertible prices come back as NaN, never as a fabricated volatility.

    A call-equivalent price of exactly zero carries NO volatility information:
    every path finished out of the money.  Handed to the repo inverter it would
    come back as the Brent floor ``vol_min = 1e-4`` -- ``bs_call_price``
    underflows to exactly 0.0 at that vol for a far strike, so the bracket has a
    root sitting on the endpoint -- i.e. a finite, plausible-looking, entirely
    fabricated volatility.  The number of such cells is monotone in maturity, so
    left alone it corrupts the short end of the surface specifically.  The
    contract is NaN.  Prices genuinely outside the band - above ``S0 e^{-qT}`` or
    below the intrinsic - and NaN inputs must all yield NaN too.
    """
    calls = price_call(paths, strikes=np.array([1.0e6]), estimator=ESTIMATOR_PLAIN)
    assert float(calls.price[0, 0]) == pytest.approx(0.0, abs=1e-12)
    ivs = implied_vol_surface(calls)
    assert ivs.shape == (len(calls.maturities), 1)
    assert np.all(np.isnan(ivs))  # never the inverter's fabricated floor

    above = dataclasses.replace(calls, price=np.full_like(calls.price, 5.0 * S0))
    assert np.all(np.isnan(implied_vol_surface(above)))
    below = dataclasses.replace(calls, price=np.full_like(calls.price, -1.0))
    assert np.all(np.isnan(implied_vol_surface(below)))
    missing = dataclasses.replace(calls, price=np.full_like(calls.price, np.nan))
    assert np.all(np.isnan(implied_vol_surface(missing)))


# ---------------------------------------------------------------------------
# QMC, caching, reproducibility
# ---------------------------------------------------------------------------
def test_scrambled_sobol_qmc_agrees_with_pseudo_random(curve, grid):
    """QMC is a variance-reduction switch, not a different model."""
    reference = _simulate(
        curve,
        grid,
        config=SimulationConfig(n_paths=8192, antithetic=True, seed=SEED + 8),
    )
    sobol = _simulate(
        curve,
        grid,
        config=SimulationConfig(
            n_paths=8192, antithetic=True, seed=SEED + 8, qmc=True, qmc_scramble=True
        ),
    )
    plain_prices = price_call(reference, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    qmc_prices = price_call(sobol, strikes=STRIKES, estimator=ESTIMATOR_PLAIN)
    assert plain_prices.stderr_is_conservative is False
    # The QMC stderr is a conservative pseudo-MC proxy - flagged, never hidden.
    assert qmc_prices.stderr_is_conservative is True
    assert np.all(
        np.abs(qmc_prices.price - plain_prices.price)
        <= 4.0 * plain_prices.stderr + 1e-3
    )
    martingale = martingale_diagnostics(sobol)
    assert martingale.max_abs_z < 6.0


def test_randomised_qmc_reduces_the_error_versus_pseudo_random(curve, grid):
    """
    Measure the QMC benefit the ONLY honest way: independent randomisations.

    A single scrambled Sobol' set has no usable internal error estimate, so the
    error is measured across ``N_REPLICATES`` independent Owen scramblings and
    compared with the same number of independent pseudo-random runs of identical
    size.  The reported reduction ratio is asserted, not assumed.
    """
    n_replicates = 12
    n_paths = 1024
    strikes = np.array([95.0, 100.0, 105.0, 110.0])

    def replicate_means(use_qmc: bool) -> np.ndarray:
        means = []
        for replicate in range(n_replicates):
            path_set = _simulate(
                curve,
                grid,
                config=SimulationConfig(
                    n_paths=n_paths,
                    antithetic=True,
                    seed=3000 + replicate,
                    qmc=use_qmc,
                ),
            )
            means.append(
                price_call(
                    path_set, strikes=strikes, estimator=ESTIMATOR_CONDITIONAL
                ).price
            )
        return np.asarray(means, dtype=float)

    qmc_means = replicate_means(True)
    mc_means = replicate_means(False)
    qmc_stderr = qmc_means.std(axis=0, ddof=1) / math.sqrt(n_replicates)
    mc_stderr = mc_means.std(axis=0, ddof=1) / math.sqrt(n_replicates)
    assert np.all(qmc_stderr > 0.0)

    # Same estimator of the same quantity: the two must agree.
    combined = np.sqrt(qmc_stderr**2 + mc_stderr**2)
    assert np.all(
        np.abs(qmc_means.mean(axis=0) - mc_means.mean(axis=0)) <= 4.0 * combined
    )
    ratio = mc_stderr / qmc_stderr
    assert np.all(ratio > 1.0), f"randomised-QMC error reduction ratios: {ratio}"
    assert float(np.mean(ratio)) > 1.2, f"mean reduction ratio {np.mean(ratio)}"


def test_prices_are_grid_converged_near_the_money(curve):
    """
    Refining the grid must not move the near-the-money prices.

    The log-Euler scheme approximates ``int_{t_i}^{t_{i+1}} V du`` by
    ``V_{t_i} dt_i``; the residual bias has to be invisible where the calibration
    actually looks.  A 96-step and a 224-step grid are compared on independent
    samples, so the band is the combined standard error.

    Recorded caveat, measured on this configuration: the far short-dated WING
    (a 10 % OTM call at 36 days) is *not* grid-converged at 96 steps - it moves
    by about 3 combined standard errors.  Deep out-of-the-money short maturities
    need a finer grid; that is a property of the scheme, not a defect of the
    implementation, and it is why ``GridConfig.n_max`` is a caller-facing knob.
    """
    fine = build_simulation_grid(
        maturities=MATURITIES, config=GridConfig(n_max=224, min_steps=16)
    )
    coarse = build_simulation_grid(
        maturities=MATURITIES, config=GridConfig(n_max=96, min_steps=12)
    )
    strikes = np.array([95.0, 100.0, 105.0])
    config = SimulationConfig(n_paths=STAT_PATHS, antithetic=True, seed=555)
    coarse_prices = price_call(
        _simulate(curve, coarse, config=config),
        strikes=strikes,
        estimator=ESTIMATOR_CONDITIONAL,
    )
    fine_prices = price_call(
        _simulate(curve, fine, config=config),
        strikes=strikes,
        estimator=ESTIMATOR_CONDITIONAL,
    )
    combined = np.sqrt(coarse_prices.stderr**2 + fine_prices.stderr**2)
    deviation = np.abs(coarse_prices.price - fine_prices.price) / combined
    assert np.all(deviation < 4.0), f"grid-refinement deviations (sigmas): {deviation}"


def test_cholesky_and_xi0_factors_are_cached_and_reused(curve, grid):
    """Phase 4 hammers this under common random numbers: the caches must hit."""
    clear_factor_cache()
    clear_xi0_cache()
    config = SimulationConfig(n_paths=1024, antithetic=True, seed=SEED + 9)
    _simulate(curve, grid, config=config)
    assert factor_cache_info()["misses"] == 1
    assert xi0_cache_info()["misses"] == 1
    _simulate(curve, grid, config=config)
    assert factor_cache_info()["hits"] >= 1
    assert xi0_cache_info()["hits"] >= 1
    # A different xi0 evaluation mode is a different cache entry, same Cholesky.
    # (cell average is the default, so NODE is the contrasting mode here)
    _simulate(
        curve,
        grid,
        config=SimulationConfig(
            n_paths=1024,
            antithetic=True,
            seed=SEED + 9,
            xi0_evaluation=XI0_NODE,
        ),
    )
    assert xi0_cache_info()["misses"] == 2
    assert factor_cache_info()["misses"] == 1


def test_reproducibility_is_bit_identical_under_a_fixed_seed(curve, grid):
    """Same seed -> bit-identical paths, prices and standard errors."""
    config = SimulationConfig(n_paths=2048, antithetic=True, seed=4242)
    first = _simulate(curve, grid, config=config)
    second = _simulate(curve, grid, config=config)
    assert np.array_equal(first.log_spot, second.log_spot)
    assert np.array_equal(first.variance, second.variance)
    assert np.array_equal(first.integrated_variance, second.integrated_variance)
    assert np.array_equal(first.draw.W_tilde, second.draw.W_tilde)

    a = price_call(first, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    b = price_call(second, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL)
    assert np.array_equal(a.price, b.price)
    assert np.array_equal(a.stderr, b.stderr)

    # Antithetic mirroring is exact negation, path by path.
    base = first.n_base_paths
    assert np.array_equal(first.draw.W_tilde[:base], -first.draw.W_tilde[base:])
    assert np.array_equal(first.draw.dB[:base], -first.draw.dB[base:])
    assert np.array_equal(first.draw.dB_perp[:base], -first.draw.dB_perp[base:])


def test_qmc_reproducibility_is_bit_identical(curve, grid):
    """The scrambled Sobol' stream is seeded too."""
    config = SimulationConfig(n_paths=1024, antithetic=True, seed=99, qmc=True)
    first = _simulate(curve, grid, config=config)
    second = _simulate(curve, grid, config=config)
    assert np.array_equal(first.log_spot, second.log_spot)


# ---------------------------------------------------------------------------
# xi0 plumbing, flat curve and reporting
# ---------------------------------------------------------------------------
def test_flat_curve_makes_node_and_cell_average_identical(flat_curve, grid):
    """A constant xi0 has no jumps, so the two evaluation modes coincide."""
    node = evaluate_xi0_on_grid(curve=flat_curve, grid=grid, mode=XI0_NODE)
    cell = evaluate_xi0_on_grid(curve=flat_curve, grid=grid, mode=XI0_CELL_AVERAGE)
    assert np.allclose(node.steps, cell.steps, rtol=0.0, atol=1e-15)
    assert float(np.max(np.abs(node.quadrature_error))) < 1e-15
    assert np.all(node.nodes == pytest.approx(float(flat_curve.levels[0]), rel=1e-15))


def test_pricing_report_is_json_safe(paths):
    """Everything the controller layer would serialise must survive json.dumps."""
    report = pricing_report(
        paths, strikes=STRIKES, estimator=ESTIMATOR_CONDITIONAL
    )
    payload = json.dumps(report)
    assert len(payload) > 0
    assert report["put_call_parity_max_abs_residual"] < 1e-8
    assert report["martingale"]["max_abs_z"] < 4.0
    assert report["simulation"]["antithetic"] is True
    assert report["simulation"]["params"]["H"] == pytest.approx(PARAMS.H)


# ---------------------------------------------------------------------------
# Validation: the guardrails refuse rather than silently approximate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "H, eta, rho",
    [
        (0.5, 1.0, -0.5),      # H above the spec bound
        (0.005, 1.0, -0.5),    # H below the spec bound
        (0.2, 0.0, -0.5),      # eta must be strictly positive
        (0.2, -1.0, -0.5),
        (0.2, 1.0, -1.5),      # |rho| above the spec bound
        (0.2, 1.0, float("nan")),
    ],
)
def test_parameter_bounds_are_enforced(H, eta, rho):
    with pytest.raises(ValueError):
        RBergomiParams(H=H, eta=eta, rho=rho)


def test_simulation_config_rejects_odd_path_counts_and_unknown_modes():
    with pytest.raises(ValueError, match="even n_paths"):
        SimulationConfig(n_paths=1001, antithetic=True)
    with pytest.raises(ValueError, match="xi0_evaluation"):
        SimulationConfig(n_paths=16, xi0_evaluation="linear")
    # Without antithetic sampling an odd count is fine.
    assert SimulationConfig(n_paths=1001, antithetic=False).n_base_paths == 1001


def test_maturity_not_on_the_grid_is_refused_never_snapped(paths, curve, grid):
    """No maturity snapping: an off-grid maturity raises instead of drifting."""
    off_grid = 0.37
    assert off_grid not in grid.quoted_maturities
    with pytest.raises(KeyError):
        paths.index_of(off_grid)
    with pytest.raises(KeyError):
        price_call(paths, strikes=STRIKES, maturities=[off_grid])
    with pytest.raises(KeyError):
        simulate_rbergomi_xi_curve(
            S0=S0,
            xi_curve=curve,
            params=PARAMS,
            grid=grid,
            maturities=[off_grid],
            r=R_FLAT,
            q=Q_DIV,
            config=SimulationConfig(n_paths=64, antithetic=True, seed=1),
        )


def test_invalid_inputs_are_refused(curve, grid):
    with pytest.raises(ValueError, match="S0"):
        simulate_rbergomi_xi_curve(
            S0=-1.0, xi_curve=curve, params=PARAMS, grid=grid, r=R_FLAT
        )
    with pytest.raises(TypeError, match="ForwardVarianceCurve"):
        simulate_rbergomi_xi_curve(
            S0=S0, xi_curve=object(), params=PARAMS, grid=grid, r=R_FLAT
        )
    with pytest.raises(TypeError, match="RBergomiParams"):
        simulate_rbergomi_xi_curve(
            S0=S0, xi_curve=curve, params=(0.1, 1.0, -0.5), grid=grid, r=R_FLAT
        )
    paths_small = _simulate(
        curve, grid, config=SimulationConfig(n_paths=64, antithetic=True, seed=2)
    )
    with pytest.raises(ValueError, match="estimator"):
        price_call(paths_small, strikes=STRIKES, estimator="antithetic")
    with pytest.raises(ValueError, match="strictly positive"):
        price_call(paths_small, strikes=np.array([-10.0]))
    with pytest.raises(TypeError, match="RBergomiPathSet"):
        price_call(object(), strikes=STRIKES)


def test_non_positive_forward_variance_is_refused(grid):
    """A hand-built curve with a non-positive level cannot produce sqrt(V)."""
    broken = ForwardVarianceCurve(
        T_knots=(0.5, 1.0),
        levels=(0.04, -0.01),
        V_knots=(0.02, 0.015),
        V_market=(0.02, 0.015),
        V_repaired=(0.02, 0.015),
    )
    with pytest.raises(RBergomiSimulationError):
        evaluate_xi0_on_grid(curve=broken, grid=grid, mode=XI0_NODE, use_cache=False)


def test_module_exports_are_declared():
    """Both modules end with an explicit, accurate ``__all__`` (repo convention)."""
    from app.model.volatility_models.rbergomi import simulator_xi_curve

    for module in (simulator_xi_curve, rb_pricing):
        assert isinstance(module.__all__, list)
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.{name}"
