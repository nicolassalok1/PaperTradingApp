"""
Joint ``(H, eta, rho)`` rough Bergomi calibration - spec 4.10 and 4.11.

Target: ``app/model/volatility_models/rbergomi/calibrator_joint_mc.py``

This module is **slow-only** (``pytestmark = pytest.mark.slow``): every test here
runs real Monte-Carlo pricing.  Nothing in it is ever selected by
``-m "unit or smoke"``.

ORACLES (all independent of the code under test)
------------------------------------------------
* **Common random numbers.** The claim "same seed + same grid => bit-identical
  ``Z`` whatever ``H``" is checked against ``numpy.random.default_rng(seed)``
  driven **directly** in the test, not against anything the calibrator computes:
  ``Z_oracle = default_rng(seed).standard_normal((n_base, 2n))`` and the joint
  block must equal ``Z_oracle @ L(H).T`` bit for bit, for two different ``H``.
* **Forward matching.** ``forward_step_rates`` is checked by re-accumulating
  ``S0 * exp(sum r_i dt_i)`` in the test and comparing against the market
  forwards - an independent recomputation of the identity the module claims.
* **Weights.** The weighting formula is re-derived inline from
  ``black76_vega`` and compared to the module's normalised weights.
* **Batch pooling.** Pooling ``B`` copies of one ``PriceResult`` must leave the
  mean untouched and divide the standard error by ``sqrt(B)`` - exact algebra,
  checked to round-off.
* **Synthetic recovery.** The market surface is generated **forward** by the
  Phase-3 simulator over a deliberately **non-flat** ``xi0`` curve, on a grid
  twice as fine as the one the calibration uses and with three times as many
  paths, so the calibration is not simply inverting its own discretisation.

ANTI-TAUTOLOGY (mandatory, spec 4.9 defaults collide with the canonical truth)
------------------------------------------------------------------------------
The spec-4.9 ``rho0 = -0.7`` prior and the spec-4.5 fallback ``H0 = 0.1``
coincide with ``RHO_TRUE`` and ``H_TRUE``, and the spec-4.9 initializer already
lands inside all three tolerances on that surface - so on the canonical market a
**no-op optimizer would pass every recovery assertion**.  That is a test-design
defect, not a code defect, and four guards close it:

* :func:`test_recovery_at_a_truth_that_collides_with_no_default` - the PRIMARY
  recovery test.  A second market is generated at
  ``(H, eta, rho) = (0.16, 1.1, -0.45)``, which coincides with no initializer
  default and no fallback: the pipeline start there is measured at
  ``(0.1444, 0.643, -0.700)``, i.e. ``eta`` off by 0.46 (> ``TOL_ETA``) and
  ``rho`` off by 0.25 (> ``TOL_RHO``), so the tolerances can only be met if the
  optimizer really moves.  The canonical ``(0.10, 1.5, -0.70)`` run is kept for
  comparability.
* :func:`test_recovery_fails_when_the_optimizer_is_a_no_op` - the guard that
  keeps the suite honest.  The local stage is monkeypatched to return ``x0`` and
  the recovery assertions must **fail**, and ``success`` must be ``False``.
* :func:`test_optimum_beats_the_initial_point_by_more_than_the_noise_floor` -
  the CRN objective must improve by strictly more than the **measured**
  CRN-difference noise floor, on the same common draw.
* :func:`test_recovery_from_a_deliberately_offset_start` - a run started at
  ``(0.30, 0.8, -0.30)``, nowhere near any default, must land inside the same
  tolerances.

WHAT THIS MODULE PINS AFTER THE PHASE-4 ADVERSARIAL PANEL
---------------------------------------------------------
Each of the eleven defects the panel measured has a test here that fails against
the pre-fix module: the truncating ``max_nfev`` default and the stationarity
invariant (D1), the hard-coded ``success`` and the false French label (D2), the
fabricated ``theta_shift = 0.0`` (D3), this very anti-tautology section (D4), the
silently truncated ``n_starts`` (D5), the clamped forward past the last quote
(D6), the ``w ~ vega^3`` weight collapse (D7), the non-neutral
``default_spread_iv`` (D8), the across-draw noise floor judged against
within-draw spans (D9), the span-based flatness test (D10), and the unfireable
fresh-seed flag with the wrong-scale bias threshold (D11).

Tolerances are Monte-Carlo loose and named once at the top of the module.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from app.model.calibration.base_calibrator import CalibratorSettings, SurfaceGrid
from app.model.calibration.rough_vol.forward_curve import SurfacePoint
from app.model.calibration.rough_vol.forward_variance import (
    ForwardVarianceCurve,
    ForwardVarianceMetadata,
)
from app.model.calibration.rough_vol.hurst_estimator import (
    HurstConfig,
    black76_vega,
    estimate_hurst_from_skew,
)
from app.model.volatility_models.rbergomi import calibrator_joint_mc as joint_mc
from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
    BLOCKING_FLAGS,
    DEFAULT_BOUNDS,
    FLAG_FRESH_SEED_LOSS_GAP,
    FLAG_GRID_BIAS_NOT_MEASURED,
    FLAG_H_OUTSIDE_H0_CI,
    FLAG_H_PROFILE_FLAT,
    FLAG_H_WEAKLY_IDENTIFIED,
    FLAG_NO_IMPROVEMENT,
    FLAG_PARAMETER_PINNED,
    FLAG_PROFILE_NOT_STATIONARY,
    FLAG_REPORT_BEYOND_QUOTES,
    FLAG_RESTARTS_TRUNCATED,
    JOINT_CALIBRATION_LABELS_FR,
    OBJECTIVE_PRICE_RELATIVE,
    PARAM_ORDER,
    PARAM_SCALE,
    REASON_K_TOO_FAR,
    REASON_NON_FINITE_IV,
    FrozenXi0,
    JointMCConfig,
    JointObjective,
    RBergomiCalibrationError,
    RBergomiJointHCalibrator,
    WeightConfig,
    _batch_seeds,
    _batch_sizes,
    _pool_price_results,
    build_calibration_quotes,
    calibrate_rbergomi,
    calibration_report,
    forward_step_rates,
    grid_refinement_bias,
    measure_noise_floor,
    profile_slice,
    resolve_bounds,
)
from app.model.volatility_models.rbergomi.initializer import initial_rbergomi_params
from app.model.volatility_models.rbergomi.pricing import (
    ESTIMATOR_CONDITIONAL,
    implied_vol_surface,
    price_call,
    price_put,
)
from app.model.volatility_models.rbergomi.simulator_xi_curve import (
    RBergomiParams,
    SimulationConfig,
    simulate_rbergomi_xi_curve,
)
from app.model.volatility_models.rbergomi.volterra_gaussian import (
    GridConfig,
    build_simulation_grid,
    cholesky_factor,
    draw_joint_gaussian,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# The synthetic world
# ---------------------------------------------------------------------------
H_TRUE = 0.10
ETA_TRUE = 1.5
RHO_TRUE = -0.70

#: The spec-4.9 ``rho`` prior and the spec-4.5 ``H0`` fallback. They are EQUAL to
#: ``RHO_TRUE`` / ``H_TRUE`` above, which is the whole reason the second truth
#: below exists: on the canonical market a dead optimizer scores as a success.
RHO0_PRIOR = -0.70
H0_FALLBACK = 0.10

#: A second truth colliding with NO default anywhere in the pipeline.
H_TRUE_B = 0.16
ETA_TRUE_B = 1.1
RHO_TRUE_B = -0.45

#: Monte-Carlo-loose recovery tolerances (suggested spec starting values).
#: The module mirrors them in ``PARAM_SCALE`` and expresses every "is this
#: material?" threshold against them - see the D11(b) test.
TOL_H = 0.05
TOL_ETA = 0.35
TOL_RHO = 0.12

S0 = 100.0
#: Continuous drift of the market forward curve and its discount rate.
DRIFT = 0.01
DISCOUNT_RATE = 0.02

MATURITIES = np.array([7.0, 14.0, 30.0, 60.0, 91.0, 182.0, 365.0, 730.0]) / 365.0
#: Deliberately NON-FLAT forward variance: it rises, dips and rises again, so a
#: flat-xi0 shortcut cannot reproduce the surface.
XI0_LEVELS = np.array([0.040, 0.046, 0.052, 0.048, 0.044, 0.050, 0.056, 0.060])

#: Strike ladder in standard deviations of the ATM total variance.
SD_LADDER = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

#: Grid and path budget of the *market*: strictly finer / larger than the
#: calibration's, so the fit is not inverting its own discretisation error.
MARKET_N_MAX = 768
MARKET_PATHS_PER_BATCH = 40_000
MARKET_BATCHES = 2
MARKET_SEED = 987_654
#: The second (offset-truth) market: same grid, one batch - it drives the
#: primary recovery test and its accuracy is already far inside the tolerances.
MARKET_SEED_B = 424_242
MARKET_BATCHES_B = 1


def _piecewise_curve(maturities: np.ndarray, levels: np.ndarray) -> ForwardVarianceCurve:
    """A spec-4.4 piecewise-constant curve built directly from its levels."""
    widths = np.diff(np.concatenate([[0.0], maturities]))
    totals = np.cumsum(levels * widths)
    return ForwardVarianceCurve(
        T_knots=tuple(float(x) for x in maturities),
        levels=tuple(float(x) for x in levels),
        V_knots=tuple(float(x) for x in totals),
        V_market=tuple(float(x) for x in totals),
        V_repaired=tuple(float(x) for x in totals),
        metadata=ForwardVarianceMetadata(n_maturities=int(maturities.size)),
    )


def _market_forwards() -> dict[float, float]:
    return {float(T): S0 * math.exp(DRIFT * float(T)) for T in MATURITIES}


def _market_discounts() -> dict[float, float]:
    return {float(T): math.exp(-DISCOUNT_RATE * float(T)) for T in MATURITIES}


def _strike_ladder(curve: ForwardVarianceCurve) -> np.ndarray:
    forwards = _market_forwards()
    rows = []
    for T in MATURITIES:
        sigma = math.sqrt(float(curve.integrated(float(T))) / float(T))
        rows.append(
            forwards[float(T)] * np.exp(SD_LADDER * sigma * math.sqrt(float(T)))
        )
    return np.asarray(rows, dtype=float)


def _drift_rates(grid, forwards: dict[float, float]) -> np.ndarray:
    """Log-linear market-forward drift, recomputed here rather than imported."""
    knots = np.concatenate([[0.0], MATURITIES])
    logs = np.concatenate([[math.log(S0)], [math.log(forwards[float(T)]) for T in MATURITIES]])
    nodes = np.interp(np.asarray(grid.t, dtype=float), knots, logs)
    return np.diff(nodes) / np.asarray(grid.dt, dtype=float)


def _simulate_market_iv(
    curve: ForwardVarianceCurve,
    strikes: np.ndarray,
    *,
    H: float,
    eta: float,
    rho: float,
    n_max: int,
    n_paths: int,
    batches: int,
    seed: int,
) -> np.ndarray:
    """Generate a market IV surface forward from the Phase-3 simulator."""
    config = GridConfig(n_max=int(n_max), min_steps=16)
    grid = build_simulation_grid(maturities=MATURITIES, config=config)
    rates = _drift_rates(grid, _market_forwards())
    maturities = [float(T) for T in MATURITIES]
    half = SD_LADDER.size // 2
    calls, puts = [], []
    for batch in range(int(batches)):
        simulation = SimulationConfig(
            n_paths=int(n_paths), antithetic=True, seed=int(seed) + batch, grid_config=config
        )
        paths = simulate_rbergomi_xi_curve(
            S0=S0,
            xi_curve=curve,
            params=RBergomiParams(H=H, eta=eta, rho=rho),
            maturities=MATURITIES,
            grid=grid,
            r=rates,
            q=0.0,
            config=simulation,
        )
        calls.append(
            price_call(
                paths,
                strikes=strikes[:, half:],
                maturities=maturities,
                estimator=ESTIMATOR_CONDITIONAL,
            )
        )
        puts.append(
            price_put(
                paths,
                strikes=strikes[:, : half + 1],
                maturities=maturities,
                estimator=ESTIMATOR_CONDITIONAL,
            )
        )
    iv_call = implied_vol_surface(_pool_price_results(calls))
    iv_put = implied_vol_surface(_pool_price_results(puts))
    return np.concatenate([iv_put[:, :half], iv_call], axis=1)


def _surface_points(strikes: np.ndarray, iv: np.ndarray) -> list[SurfacePoint]:
    forwards = _market_forwards()
    discounts = _market_discounts()
    points: list[SurfacePoint] = []
    for i, T in enumerate(MATURITIES):
        F = forwards[float(T)]
        for j, K in enumerate(strikes[i]):
            points.append(
                SurfacePoint(
                    T=float(T),
                    K=float(K),
                    k=math.log(float(K) / F),
                    F=float(F),
                    D=float(discounts[float(T)]),
                    iv=float(iv[i, j]),
                    option_type="call" if float(K) >= F else "put",
                    mid=float("nan"),
                    call_equivalent_price=float("nan"),
                    vendor_iv=float("nan"),
                    one_sided=False,
                )
            )
    return points


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: the Monte Carlo is paid once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def xi0_curve() -> ForwardVarianceCurve:
    return _piecewise_curve(MATURITIES, XI0_LEVELS)


@pytest.fixture(scope="module")
def strikes(xi0_curve: ForwardVarianceCurve) -> np.ndarray:
    return _strike_ladder(xi0_curve)


@pytest.fixture(scope="module")
def market_iv(xi0_curve: ForwardVarianceCurve, strikes: np.ndarray) -> np.ndarray:
    return _simulate_market_iv(
        xi0_curve,
        strikes,
        H=H_TRUE,
        eta=ETA_TRUE,
        rho=RHO_TRUE,
        n_max=MARKET_N_MAX,
        n_paths=MARKET_PATHS_PER_BATCH,
        batches=MARKET_BATCHES,
        seed=MARKET_SEED,
    )


@pytest.fixture(scope="module")
def market_points(strikes: np.ndarray, market_iv: np.ndarray) -> list[SurfacePoint]:
    return _surface_points(strikes, market_iv)


@pytest.fixture(scope="module")
def quote_set(market_points):
    return build_calibration_quotes(market_points, S0=S0)


@pytest.fixture(scope="module")
def objective(quote_set, xi0_curve):
    return JointObjective(
        quotes=quote_set,
        xi0=FrozenXi0.freeze(xi0_curve),
        config=JointMCConfig(grid_n_max=384, stage2_paths=8_000),
    )


def _recovery_config(*, refinement: bool) -> JointMCConfig:
    return JointMCConfig(
        grid_n_max=384,
        n_design=12,
        stage1_paths=6_000,
        top_k=2,
        stage2_paths=8_000,
        final_paths=40_000,
        batch_paths=20_000,
        profile_points=5,
        valley_points=5,
        noise_replicates=2,
        refinement_check=refinement,
        refinement_factor=2,
    )


@pytest.fixture(scope="module")
def pipeline_start(market_points, xi0_curve):
    """Run the real downstream pipeline: skew -> H0 -> (H0, eta0, rho0)."""
    hurst = estimate_hurst_from_skew(
        market_points,
        None,
        hurst_config=HurstConfig(short_maturity_window=(5.0 / 365.0, 0.30)),
    )
    params, diagnostics = initial_rbergomi_params(
        hurst, market_points, xi0_curve=xi0_curve
    )
    return hurst, params, diagnostics


@pytest.fixture(scope="module")
def recovered(market_points, xi0_curve, pipeline_start):
    """
    Canonical market, real pipeline start, and **no explicit ``max_nfev``**.

    Leaving ``max_nfev`` out is deliberate: it is what exercises the
    calibrator's own budget (D1). The repo default of 80 truncates Nelder-Mead
    on this problem.
    """
    _hurst, params, diagnostics = pipeline_start
    return calibrate_rbergomi(
        market_points,
        xi0_curve,
        (params, diagnostics),
        mc_cfg=_recovery_config(refinement=True),
        settings=CalibratorSettings(n_starts=1, seed=20_240_101),
        S0=S0,
    )


@pytest.fixture(scope="module")
def recovered_offset(market_points, xi0_curve):
    """Recovery from a start nowhere near any spec default."""
    return calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(H=0.30, eta=0.8, rho=-0.30),
        mc_cfg=_recovery_config(refinement=False),
        settings=CalibratorSettings(n_starts=1, seed=777_001),
        S0=S0,
    )


# ---------------------------------------------------------------------------
# The SECOND market: a truth that collides with no default (anti-tautology)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def market_iv_b(xi0_curve: ForwardVarianceCurve, strikes: np.ndarray) -> np.ndarray:
    return _simulate_market_iv(
        xi0_curve,
        strikes,
        H=H_TRUE_B,
        eta=ETA_TRUE_B,
        rho=RHO_TRUE_B,
        n_max=MARKET_N_MAX,
        n_paths=MARKET_PATHS_PER_BATCH,
        batches=MARKET_BATCHES_B,
        seed=MARKET_SEED_B,
    )


@pytest.fixture(scope="module")
def market_points_b(strikes: np.ndarray, market_iv_b: np.ndarray) -> list[SurfacePoint]:
    return _surface_points(strikes, market_iv_b)


@pytest.fixture(scope="module")
def pipeline_start_b(market_points_b, xi0_curve):
    hurst = estimate_hurst_from_skew(
        market_points_b,
        None,
        hurst_config=HurstConfig(short_maturity_window=(5.0 / 365.0, 0.30)),
    )
    params, diagnostics = initial_rbergomi_params(
        hurst, market_points_b, xi0_curve=xi0_curve
    )
    return hurst, params, diagnostics


@pytest.fixture(scope="module")
def recovered_b(market_points_b, xi0_curve, pipeline_start_b):
    _hurst, params, diagnostics = pipeline_start_b
    return calibrate_rbergomi(
        market_points_b,
        xi0_curve,
        (params, diagnostics),
        mc_cfg=_recovery_config(refinement=True),
        settings=CalibratorSettings(n_starts=1, seed=20_240_202),
        S0=S0,
    )


def _cheap_config(**overrides) -> JointMCConfig:
    """A deliberately small configuration for the bookkeeping / flag tests."""
    fields = dict(
        grid_n_max=192,
        n_design=0,
        stage1_paths=2_000,
        top_k=1,
        stage2_paths=2_000,
        profile_paths=2_000,
        final_paths=4_000,
        batch_paths=4_000,
        profile_points=5,
        valley_points=4,
        noise_replicates=2,
        refinement_check=False,
    )
    fields.update(overrides)
    return JointMCConfig(**fields)


# ---------------------------------------------------------------------------
# 1. Common random numbers
# ---------------------------------------------------------------------------
def test_common_random_numbers_are_shared_across_H():
    """
    Same seed + same grid => bit-identical ``Z``, whatever ``H``.

    Oracle: the normals are drawn **in the test** from
    ``numpy.random.default_rng(seed)`` and pushed through the Cholesky factor;
    the simulator's own draw must match bit for bit at two different ``H``.
    """
    grid = build_simulation_grid(
        maturities=MATURITIES, config=GridConfig(n_max=64, min_steps=8)
    )
    n = grid.n
    n_paths, seed = 64, 4_242
    n_base = n_paths // 2

    oracle = np.random.default_rng(seed)
    z_joint = oracle.standard_normal((n_base, 2 * n))
    z_perp = oracle.standard_normal((n_base, n))
    perp = z_perp * np.sqrt(np.asarray(grid.dt, dtype=float))

    for H in (0.10, 0.42):
        factor = cholesky_factor(H=H, grid=grid)
        draw = draw_joint_gaussian(
            factor=factor, n_paths=n_paths, seed=seed, antithetic=True
        )
        expected = z_joint @ np.asarray(factor.L).T
        assert np.array_equal(draw.W_tilde[:n_base], expected[:, :n])
        assert np.array_equal(draw.dB[:n_base], expected[:, n:])
        assert np.array_equal(draw.dB_perp[:n_base], perp)

    # And the recovered Z agrees between the two H, to the round-off of two
    # different triangular solves - the same normals really did drive both.
    recovered = []
    for H in (0.10, 0.42):
        factor = cholesky_factor(H=H, grid=grid)
        draw = draw_joint_gaussian(
            factor=factor, n_paths=n_paths, seed=seed, antithetic=True
        )
        joint = np.concatenate([draw.W_tilde, draw.dB], axis=1)[:n_base]
        recovered.append(solve_triangular(np.asarray(factor.L), joint.T, lower=True).T)
    assert np.max(np.abs(recovered[0] - recovered[1])) < 1e-9


def test_objective_is_bitwise_deterministic_under_crn(objective):
    """Two evaluations at the same theta and seed return the SAME number."""
    theta = [H_TRUE, ETA_TRUE, RHO_TRUE]
    first = objective.evaluate(theta, n_paths=6_000, seed=31, use_cache=False)
    second = objective.evaluate(theta, n_paths=6_000, seed=31, use_cache=False)
    assert first.loss == second.loss
    assert np.array_equal(first.iv_model, second.iv_model)
    assert np.array_equal(first.residuals, second.residuals)

    # A different theta on the same draw moves the loss (the draw is shared, the
    # model is not), and moving back returns exactly the first value.
    moved = objective.evaluate([0.25, 1.0, -0.4], n_paths=6_000, seed=31, use_cache=False)
    assert moved.loss != first.loss
    again = objective.evaluate(theta, n_paths=6_000, seed=31, use_cache=False)
    assert again.loss == first.loss


def test_objective_rejects_anything_but_three_parameters(objective):
    """``theta`` is structurally (H, eta, rho); a fourth slot does not exist."""
    with pytest.raises(ValueError, match="exactly three parameters"):
        objective.evaluate([H_TRUE, ETA_TRUE, RHO_TRUE, 0.04], n_paths=1_000, seed=1)
    with pytest.raises(ValueError):
        objective.evaluate([H_TRUE, ETA_TRUE], n_paths=1_000, seed=1)


def test_batch_seeds_depend_only_on_the_stage_seed():
    """Batching must not break common random numbers."""
    assert _batch_seeds(11, 1) == [11]
    assert _batch_seeds(11, 4) == _batch_seeds(11, 4)
    assert _batch_seeds(11, 4) != _batch_seeds(12, 4)
    assert _batch_seeds(None, 3) == [None, None, None]
    assert _batch_sizes(10_000, 20_000, antithetic=True) == [10_000]
    assert _batch_sizes(100_000, 20_000, antithetic=True) == [20_000] * 5
    assert all(size % 2 == 0 for size in _batch_sizes(9_999, 4_000, antithetic=True))


# ---------------------------------------------------------------------------
# 2. xi0 is frozen - structurally, not by inspection
# ---------------------------------------------------------------------------
def test_xi0_object_is_returned_by_identity_and_bitwise_unchanged(
    recovered, xi0_curve
):
    """The calibration hands back the caller's OWN curve, byte for byte."""
    before_levels = np.array(xi0_curve.levels, dtype=float)
    before_knots = np.array(xi0_curve.T_knots, dtype=float)
    before_V = np.array(xi0_curve.V_knots, dtype=float)

    assert recovered.xi0_curve is xi0_curve
    assert recovered.xi0.curve is xi0_curve
    assert np.array_equal(np.array(xi0_curve.levels, dtype=float), before_levels)
    assert np.array_equal(np.array(xi0_curve.T_knots, dtype=float), before_knots)
    assert np.array_equal(np.array(xi0_curve.V_knots, dtype=float), before_V)

    # The content hash recorded at freeze time still matches: nothing moved.
    recovered.xi0.verify()
    assert recovered.to_dict()["xi0"]["fingerprint"] == recovered.xi0.fingerprint


def test_xi0_is_structurally_outside_the_optimiser(recovered):
    """
    ``xi0`` is not a parameter: not in the order, not in the bounds, not pinnable.

    This is the demonstration, not an assertion by inspection - the optimiser's
    search space is exactly what ``PARAM_ORDER`` and ``bounds`` describe.
    """
    assert PARAM_ORDER == ("H", "eta", "rho")
    assert "xi0" not in DEFAULT_BOUNDS
    assert set(recovered.bounds) == {"H", "eta", "rho"}
    assert recovered.theta.size == 3
    assert set(recovered.params.to_dict()) == {"H", "eta", "rho"}


def test_constraining_xi0_raises_rather_than_being_ignored(market_points, xi0_curve):
    with pytest.raises(RBergomiCalibrationError, match="pas un param"):
        resolve_bounds({"xi0": 0.04})
    with pytest.raises(RBergomiCalibrationError):
        calibrate_rbergomi(
            market_points,
            xi0_curve,
            RBergomiParams(0.2, 1.0, -0.5),
            settings=CalibratorSettings(seed=1),
            S0=S0,
            constraints={"xi0": 0.04},
        )


def test_freeze_rejects_a_non_curve():
    with pytest.raises(RBergomiCalibrationError, match="ForwardVarianceCurve"):
        FrozenXi0.freeze({"levels": [0.04]})


# ---------------------------------------------------------------------------
# 3. Synthetic recovery + the anti-tautology guards
# ---------------------------------------------------------------------------
def test_downstream_pipeline_produces_a_usable_start(pipeline_start):
    """The skew -> H0 stage really ran on the synthetic surface."""
    hurst, params, diagnostics = pipeline_start
    assert hurst.unstable is False
    assert hurst.n_expiries >= 3
    assert 0.01 < params.H < 0.49
    assert diagnostics["H0_is_fallback"] is False
    # ``rho0`` is the spec-4.9 PRIOR, not a measurement, and it happens to equal
    # RHO_TRUE on this market. Asserting it against RHO_TRUE would look like a
    # recovery check while carrying zero discriminating power, so it is asserted
    # against the prior it actually is - and the offset-truth market below is
    # what makes the recovery of rho a real test.
    assert params.rho == pytest.approx(RHO0_PRIOR)
    assert RHO0_PRIOR == RHO_TRUE and H0_FALLBACK == H_TRUE


def _recovery_gaps(result, truth) -> tuple[float, float, float]:
    return (
        abs(result.params.H - truth[0]),
        abs(result.params.eta - truth[1]),
        abs(result.params.rho - truth[2]),
    )


def test_recovery_at_a_truth_that_collides_with_no_default(
    recovered_b, pipeline_start_b
):
    """
    ANTI-TAUTOLOGY, PRIMARY: recover ``(0.16, 1.1, -0.45)``.

    Nothing in the pipeline points at this triple.  The measured start on this
    market is ``(0.1444, 0.643, -0.700)``: ``eta`` is off by more than
    ``TOL_ETA`` and ``rho`` by more than ``TOL_RHO``, so the three assertions
    below can only pass if the optimizer moved.  The companion test
    :func:`test_recovery_fails_when_the_optimizer_is_a_no_op` proves that claim
    rather than asserting it.
    """
    _hurst, start, _diagnostics = pipeline_start_b
    # The start is genuinely OUTSIDE the tolerances - the premise of this test.
    assert abs(start.eta - ETA_TRUE_B) > TOL_ETA
    assert abs(start.rho - RHO_TRUE_B) > TOL_RHO

    dH, d_eta, d_rho = _recovery_gaps(recovered_b, (H_TRUE_B, ETA_TRUE_B, RHO_TRUE_B))
    assert dH <= TOL_H
    assert d_eta <= TOL_ETA
    assert d_rho <= TOL_RHO
    assert recovered_b.success is True
    assert recovered_b.rmse_fresh < 0.01
    # And it improved on the start by more than the measured noise.
    report = recovered_b.identifiability
    assert report.improvement > report.noise_floor.difference_value
    assert report.improvement_significant is True


def test_recovery_fails_when_the_optimizer_is_a_no_op(
    monkeypatch, market_points_b, xi0_curve, pipeline_start_b
):
    """
    THE GUARD THAT KEEPS THE SUITE HONEST.

    Stub the local stage into a no-op that returns its own ``x0`` and re-run the
    recovery.  At least one recovery assertion MUST fail, and ``success`` must be
    ``False``: an optimizer that does nothing cannot be reported as a success.
    Against the canonical ``(0.10, 1.5, -0.70)`` market this stub still passed
    every tolerance, which is exactly why the offset truth exists.
    """
    calls: list[int] = []

    def _no_op(objective, x0, *, bounds, config, n_paths, seed, max_nfev):
        calls.append(1)
        return np.asarray(x0, dtype=float).copy(), 0, True, "no-op stub"

    monkeypatch.setattr(joint_mc, "_run_local", _no_op)

    _hurst, params, diagnostics = pipeline_start_b
    result = calibrate_rbergomi(
        market_points_b,
        xi0_curve,
        (params, diagnostics),
        mc_cfg=_cheap_config(),
        settings=CalibratorSettings(n_starts=1, seed=13_579),
        S0=S0,
    )
    assert calls, "the stub was never reached - the monkeypatch missed its target"

    # The returned point IS the start: nothing moved.
    assert result.params.H == pytest.approx(params.H)
    assert result.params.eta == pytest.approx(params.eta)
    assert result.params.rho == pytest.approx(params.rho)

    gaps = _recovery_gaps(result, (H_TRUE_B, ETA_TRUE_B, RHO_TRUE_B))
    assert any(
        gap > tol for gap, tol in zip(gaps, (TOL_H, TOL_ETA, TOL_RHO))
    ), f"a dead optimizer passed every tolerance: gaps={gaps}"

    assert result.success is False
    assert FLAG_NO_IMPROVEMENT in result.flags
    assert any(flag in BLOCKING_FLAGS for flag in result.flags)


def test_synthetic_recovery_from_the_pipeline_start(recovered):
    """The canonical ``(0.10, 1.5, -0.70)`` run, kept for comparability."""
    assert recovered.success is True
    assert abs(recovered.params.H - H_TRUE) <= TOL_H
    assert abs(recovered.params.eta - ETA_TRUE) <= TOL_ETA
    assert abs(recovered.params.rho - RHO_TRUE) <= TOL_RHO
    # The out-of-sample (fresh seed) fit is a fraction of a volatility point.
    assert recovered.rmse_fresh < 0.01
    # It ran on the calibrator's OWN budget, not the repo default of 80.
    assert recovered.stage2.max_nfev_source == "config"
    assert recovered.stage2.max_nfev == recovered.config.local_max_nfev(3)
    assert recovered.stage2.max_nfev >= 150


def test_recovery_from_a_deliberately_offset_start(recovered_offset):
    """
    ANTI-TAUTOLOGY: start at (0.30, 0.8, -0.30) - nowhere near the spec-4.5
    fallback ``H0 = 0.1`` nor the spec-4.9 prior ``rho0 = -0.7`` - and still land
    on the truth.
    """
    start = recovered_offset.initial_params
    assert (start.H, start.eta, start.rho) == (0.30, 0.8, -0.30)
    assert abs(recovered_offset.params.H - H_TRUE) <= TOL_H
    assert abs(recovered_offset.params.eta - ETA_TRUE) <= TOL_ETA
    assert abs(recovered_offset.params.rho - RHO_TRUE) <= TOL_RHO


def test_optimum_beats_the_initial_point_by_more_than_the_noise_floor(recovered_offset):
    """
    ANTI-TAUTOLOGY: the optimizer must have DONE something.

    Both losses are evaluated on the same common-random-number draw, so their
    difference is a CRN difference and must be judged against the CRN-DIFFERENCE
    noise floor, not against the scatter of the loss level across independent
    seeds (spec 4.11 / D9).
    """
    report = recovered_offset.identifiability
    assert report is not None
    assert report.loss_optimum < report.loss_initial
    assert report.improvement > report.noise_floor.difference_value
    assert report.improvement_significant is True
    assert report.noise_floor.difference_value > 0.0
    assert report.noise_floor.value > 0.0


# ---------------------------------------------------------------------------
# 3bis. The evaluation budget and the stationarity invariant (D1 / D2)
# ---------------------------------------------------------------------------
def test_the_calibrator_carries_its_own_evaluation_budget():
    """
    The repo default of 80 truncates Nelder-Mead here; the module must not use it.

    Measured on the reference surface: 115-162 evaluations are needed for three
    free parameters, and stopping at 80 cost up to ``dH = -0.026``.
    ``CalibratorSettings`` is shared repo-wide and is NOT modified - the
    calibrator carries its own default and honours ``settings.max_nfev`` only
    when the caller changed it.
    """
    config = JointMCConfig()
    assert config.local_max_nfev(3) >= 150
    assert config.local_max_nfev(3) == config.local_nfev_per_param * 3
    assert config.local_max_nfev(1) == config.local_nfev_per_param
    # The shared dataclass is untouched.
    assert CalibratorSettings().max_nfev == 80
    assert config.local_max_nfev(3) > CalibratorSettings().max_nfev


def test_an_explicit_max_nfev_is_still_honoured(market_points, xi0_curve):
    """A caller who sets a budget gets that budget, and it is reported as such."""
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=_cheap_config(),
        settings=CalibratorSettings(max_nfev=9, n_starts=1, seed=4_141),
        S0=S0,
    )
    assert result.stage2.max_nfev == 9
    assert result.stage2.max_nfev_source == "settings"
    assert result.details["settings"]["max_nfev_effective"] == 9
    assert result.stage2.runs[0].nfev <= 9


def test_a_non_stationary_optimum_is_flagged_and_fails(market_points, xi0_curve):
    """
    THE MODULE ASSERTS ITS OWN INVARIANT (D1).

    ``theta*`` must be the cheapest point of its own profile, up to the measured
    noise floor - every one of those losses lives on the SAME draw.  A brutally
    truncated local stage started far from the optimum violates that, and the
    module must say so and refuse to call the run a success, instead of shipping
    a point that its own diagnostics contradict.
    """
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(H=0.35, eta=0.6, rho=-0.20),
        mc_cfg=_cheap_config(),
        settings=CalibratorSettings(max_nfev=4, n_starts=1, seed=8_642),
        S0=S0,
    )
    offenders = [p for p in result.identifiability.profiles if not p.stationary]
    assert offenders, "a 4-evaluation local stage was reported as stationary"
    for slice_ in offenders:
        assert slice_.optimum_loss > slice_.losses.min() + slice_.stationarity_floor
        assert "NON STATIONNAIRE" in slice_.message_fr
    assert FLAG_PROFILE_NOT_STATIONARY in result.flags
    assert result.success is False
    assert "NON CONCLUANTE" in result.message_fr
    assert FLAG_PROFILE_NOT_STATIONARY in JOINT_CALIBRATION_LABELS_FR


def test_success_is_a_verdict_not_a_constant():
    """``success`` is False exactly when a blocking flag is present."""
    assert set(BLOCKING_FLAGS) == {
        FLAG_H_PROFILE_FLAT,
        FLAG_NO_IMPROVEMENT,
        FLAG_PROFILE_NOT_STATIONARY,
    }
    for flag in BLOCKING_FLAGS:
        assert flag in JOINT_CALIBRATION_LABELS_FR


def test_the_flat_H_label_does_not_claim_the_initialisation_is_returned():
    """
    D2: the label used to say "la valeur rendue est l'initialisation".

    It is not: on the measured degenerate market the initial ``H`` was 0.25 and
    the returned one 0.3375.  A simplex moves even on a flat surface.
    """
    label = JOINT_CALIBRATION_LABELS_FR[FLAG_H_PROFILE_FLAT]
    assert "PLAT" in label
    assert "n'est PAS l'initialisation" in label
    assert "la valeur rendue est l'initialisation" not in label


def test_fresh_seed_loss_is_reported_and_comparable(recovered):
    """Spec 4.10: report BOTH the in-sample CRN loss and the fresh-seed loss."""
    assert recovered.loss_crn > 0.0
    assert recovered.loss_fresh > 0.0
    assert recovered.final_evaluation.seed != recovered.identifiability.profiles[0].seed
    assert recovered.final_evaluation.n_paths >= 40_000


def test_the_over_fitting_test_compares_matched_path_counts(recovered):
    """
    D11(a): ``loss_fresh`` lives at ``final_paths``, ``loss_crn`` at ``stage2_paths``.

    Comparing them compares two estimators, not two draws: measured
    ``E[L(12k)] = 6.647e-06`` against ``E[L(100k)] = 5.163e-06`` (ratio 0.777),
    so the fresh loss is structurally SMALLER and the flag needed a genuine 3.9x
    over-fit before it could fire.  The reported comparison must therefore be at
    matched path counts, on the draw the local stage actually fitted.
    """
    gap = recovered.to_dict()["fresh_seed_gap"]
    assert gap["n_paths_crn"] == gap["n_paths_fresh"] == recovered.matched_paths
    # The in-sample draw is the one the local stage fitted, not the report draw.
    assert recovered.matched_paths == recovered.config.stage2_paths
    assert recovered.loss_crn_matched == pytest.approx(
        recovered.stage2.runs[recovered.stage2.best_index].loss_own_draw
    )
    assert recovered.loss_fresh_matched > 0.0
    # The structural rigging is gone: the fresh loss is now the LARGER of the
    # two, which is what "out of sample" is supposed to mean.
    assert recovered.loss_fresh_matched > recovered.loss_crn_matched
    assert gap["ratio"] == pytest.approx(
        recovered.loss_fresh_matched / recovered.loss_crn_matched
    )
    assert gap["ratio"] < recovered.config.fresh_seed_gap_ratio
    assert FLAG_FRESH_SEED_LOSS_GAP not in recovered.flags


def test_the_fresh_seed_gap_flag_can_actually_fire(market_points, xi0_curve):
    """The flag is live: tighten the ratio to 1 and the measured gap trips it."""
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.12, 1.4, -0.65),
        mc_cfg=_cheap_config(fresh_seed_gap_ratio=1.0),
        settings=CalibratorSettings(max_nfev=30, n_starts=1, seed=515),
        S0=S0,
    )
    assert result.loss_fresh_matched > result.loss_crn_matched
    assert FLAG_FRESH_SEED_LOSS_GAP in result.flags


# ---------------------------------------------------------------------------
# 4. Identifiability (spec 4.11)
# ---------------------------------------------------------------------------
def test_profile_slices_are_produced_bound_to_bound(recovered):
    report = recovered.identifiability
    assert report is not None
    assert {slice_.parameter for slice_ in report.profiles} == {"H", "eta", "rho"}
    for slice_ in report.profiles:
        lower, upper = recovered.bounds[slice_.parameter]
        assert slice_.values.min() == pytest.approx(lower)
        assert slice_.values.max() == pytest.approx(upper)
        assert slice_.values.size >= 5
        assert np.all(np.diff(slice_.values) > 0.0)
        assert np.all(np.isfinite(slice_.losses))
        # The optimum is the cheapest point of its own slice, up to the noise -
        # and the module ASSERTS it rather than assuming it (D1).
        assert slice_.optimum_loss <= slice_.losses.min() + slice_.stationarity_floor
        assert slice_.stationary is True
        assert slice_.stationarity_gap >= 0.0
        assert slice_.flat is False
        assert slice_.curvature > 0.0
    assert FLAG_PROFILE_NOT_STATIONARY not in recovered.flags


def test_standard_errors_are_read_off_the_profile_curvature(recovered, pipeline_start):
    """
    D10: spec 4.11 asks how well the surface identifies H, and the SPAN does not
    answer it.

    ``SE(p) = sqrt(2 sigma_L / (d2L/dp2))`` is the half-width over which the loss
    rises by less than one Monte-Carlo sigma.  On the reference surface the
    module's own measurement is ``SE(H) ~ 0.017`` against the spec-4.5 ``H0``
    standard error of ~0.006, so the joint fit is the looser of the two by a
    factor of a few - and it is inside ``TOL_H``, which is what makes the
    recovery assertions meaningful.
    """
    hurst, _params, _diagnostics = pipeline_start
    report = recovered.identifiability
    errors = report.standard_errors
    assert set(errors) == {"H", "eta", "rho"}
    for name, value in errors.items():
        assert math.isfinite(value) and value > 0.0
        # An identified parameter's standard error is inside its own scale.
        assert value < PARAM_SCALE[name]

    se_H = report.H_standard_error
    assert se_H == pytest.approx(recovered.identifiability.profile("H").standard_error)
    assert se_H < TOL_H
    # Reconstruct it from the two published ingredients - the oracle is the
    # formula, recomputed here rather than read back.
    slice_H = report.profile("H")
    assert se_H == pytest.approx(
        math.sqrt(2.0 * slice_H.sigma_level / slice_H.curvature), rel=1e-12
    )
    assert slice_H.sigma_level == pytest.approx(
        max(report.noise_floor.replicated_std, report.noise_floor.delta_std)
    )
    # ... and it is comparable to the spec-4.5 standard error, as spec 4.11 asks.
    assert se_H < float(recovered.config.se_vs_h0_factor) * hurst.se
    assert FLAG_H_WEAKLY_IDENTIFIED not in recovered.flags
    assert "SE(H)" in report.identification_fr


def test_a_flat_span_does_not_prove_identification(recovered):
    """
    The span diagnostic is kept, but as the SECONDARY one.

    Measured (D10): on a quote set restricted to 1 y and 2 y, three seeds gave
    ``H = 0.1260 / 0.0447 / 0.1780`` - a spread of 0.133, i.e. 2.7x ``TOL_H`` -
    while ``span / noise_floor`` sat between 250x and 320x in all three, so the
    span could never have fired.  Both numbers are therefore reported side by
    side, and the flags are built on the standard error.
    """
    slice_H = recovered.identifiability.profile("H")
    payload = slice_H.to_dict()
    assert payload["span"] > payload["noise_floor"]  # the span says "informative"
    assert set(payload) >= {
        "span",
        "noise_floor",
        "standard_error",
        "se_threshold",
        "weakly_identified",
        "stationarity_gap",
        "stationary",
    }
    assert payload["se_threshold"] == pytest.approx(
        recovered.config.se_material_ratio * PARAM_SCALE["H"]
    )


def test_eta_rho_valley_at_constant_product_is_curved(recovered):
    """
    The spec-4.9 degeneracy (``MATH_ORACLE`` section 8: the short-dated skew
    identifies only ``rho * eta``) must be BROKEN by the joint fit.
    """
    valley = recovered.identifiability.valley
    assert valley is not None
    assert valley.n_points >= 3
    assert valley.product == pytest.approx(
        recovered.params.eta * recovered.params.rho, rel=1e-12
    )
    assert np.allclose(valley.eta_values * valley.rho_values, valley.product)
    assert valley.span > valley.noise_floor
    assert valley.flat is False
    assert "levée" in valley.message_fr


def test_H0_versus_H_calibrated_is_reported_with_the_confidence_interval(
    recovered, pipeline_start
):
    hurst, _params, _diagnostics = pipeline_start
    report = recovered.identifiability
    assert report.has_H0_ci is True
    assert report.H0 == pytest.approx(hurst.H0)
    assert report.H0_se == pytest.approx(hurst.se)
    low, high = report.H0_ci95
    assert low < report.H0 < high
    # The flag and the interval must agree - whichever way the comparison falls.
    inside = low <= report.H_calibrated <= high
    assert report.H_in_ci is inside
    assert (FLAG_H_OUTSIDE_H0_CI in recovered.flags) is (not inside)
    assert "IC95" in report.h_comparison_fr
    assert f"{report.H_calibrated:.4f}" in report.h_comparison_fr


def test_noise_floor_is_measured_not_assumed(recovered):
    floor = recovered.identifiability.noise_floor
    assert floor.n_replicates >= 2
    assert len(set(floor.seeds)) == floor.n_replicates
    assert floor.replicated_std >= 0.0
    assert floor.delta_std >= 0.0
    assert floor.value == pytest.approx(
        floor.multiplier * max(floor.replicated_std, floor.delta_std)
    )
    # The DIFFERENCE floor is the one the flatness / improvement tests use.
    assert floor.difference_is_fallback is False
    assert floor.difference_value == pytest.approx(
        floor.multiplier * floor.difference_std
    )
    assert set(floor.difference_parameters) == {"H", "eta", "rho"}
    for slice_ in recovered.identifiability.profiles:
        assert slice_.noise_floor == pytest.approx(
            floor.difference_for(slice_.parameter)
        )
        assert slice_.stationarity_floor == pytest.approx(
            max(floor.difference_for(slice_.parameter), floor.value)
        )
    payload = floor.to_dict()
    assert "covariance" in payload["delta_std_caveat_fr"]
    assert set(payload["difference_by_parameter"]) == {"H", "eta", "rho"}


def test_the_noise_floor_is_measured_on_CRN_DIFFERENCES_not_on_levels(objective):
    """
    D9: the yardstick has to match the quantity it judges.

    Every span, gradient and improvement this module reports is a difference of
    two losses computed on ONE shared draw, and CRN cancels the sampling noise in
    exactly those differences.  Measured here, independently of the calibration:
    the run-to-run scatter of a CRN one-step difference is far below the scatter
    of the loss LEVEL across the same seeds - the panel measured up to 25x at
    ``n_max = 128`` / 3 000 paths.
    """
    theta = [0.15, 1.1, -0.55]
    floor = measure_noise_floor(
        objective,
        theta,
        config=JointMCConfig(noise_replicates=4, noise_sigma_multiplier=2.0),
        seeds=[11, 22, 33, 44],
        n_paths=3_000,
        delta_std=0.0,
        bounds=DEFAULT_BOUNDS,
        free=[0, 1, 2],
    )
    assert floor.n_replicates == 4
    assert floor.replicated_std > 0.0
    assert floor.difference_is_fallback is False
    per_parameter = dict(
        zip(floor.difference_parameters, floor.difference_by_parameter)
    )
    assert set(per_parameter) == {"H", "eta", "rho"}
    for name, value in per_parameter.items():
        assert math.isfinite(value) and value > 0.0
    # The H difference is the one the H span is judged against, and it is far
    # below the level scatter - the whole point of common random numbers.
    assert per_parameter["H"] < floor.replicated_std / 3.0
    assert floor.difference_for("H") == pytest.approx(
        floor.multiplier * per_parameter["H"]
    )
    # A single seed cannot measure a standard deviation: fall back, and say so,
    # rather than return a zero floor that would make every test unfireable.
    lone = measure_noise_floor(
        objective,
        theta,
        config=JointMCConfig(noise_replicates=1),
        seeds=[11],
        n_paths=1_000,
        delta_std=1e-6,
        bounds=DEFAULT_BOUNDS,
        free=[0, 1, 2],
    )
    assert lone.difference_is_fallback is True
    assert lone.difference_value == pytest.approx(lone.value)
    assert lone.difference_value > 0.0


@pytest.fixture(scope="module")
def degenerate_run(xi0_curve, strikes):
    """
    A market with NO vol-of-vol: ``H = 0.25, eta = 0.05``, ``eta`` and ``rho``
    pinned and the initial ``H`` set to the truth.  The smile carries no ``H``
    signal at all, so the calibration cannot measure ``H`` - and must say so.
    """
    iv = _simulate_market_iv(
        xi0_curve,
        strikes,
        H=0.25,
        eta=0.05,
        rho=-0.70,
        n_max=256,
        n_paths=20_000,
        batches=1,
        seed=555,
    )
    points = _surface_points(strikes, iv)
    config = JointMCConfig(
        grid_n_max=256,
        n_design=0,
        stage2_paths=1_000,
        profile_paths=1_000,
        final_paths=4_000,
        batch_paths=4_000,
        profile_points=5,
        valley_points=4,
        noise_replicates=4,
        refinement_check=False,
    )
    return calibrate_rbergomi(
        points,
        xi0_curve,
        RBergomiParams(H=0.25, eta=0.05, rho=-0.70),
        mc_cfg=config,
        settings=CalibratorSettings(max_nfev=6, n_starts=1, seed=9),
        S0=S0,
        constraints={"eta": 0.05, "rho": -0.70},
    )


def test_a_surface_without_vol_of_vol_cannot_measure_H_and_says_so(degenerate_run):
    """
    D2 + D10: the run must FAIL, and for a reason it actually measured.

    Before the fix this exact configuration returned ``success = True`` with
    ``H = 0.3375`` from an initial ``H = 0.25``, carrying the flags
    ``('parameter_pinned', 'local_stage_not_converged', 'h_profile_flat',
    'no_improvement_over_initial')`` - and a French label claiming the returned
    value *was* the initialisation, which ``0.3375 != 0.25`` falsifies.
    """
    result = degenerate_run
    profile = result.identifiability.profile("H")
    assert profile is not None

    # The honest measurement: the H profile has essentially no curvature, so the
    # standard error explodes past the parameter's own scale.
    assert profile.standard_error > TOL_H
    assert profile.weakly_identified is True
    assert FLAG_H_WEAKLY_IDENTIFIED in result.flags

    # And the verdict follows the diagnostics instead of being hard-coded.
    assert result.success is False
    assert FLAG_NO_IMPROVEMENT in result.flags
    assert "NON CONCLUANTE" in result.message_fr
    assert any(flag in BLOCKING_FLAGS for flag in result.flags)

    # Pinning is honoured and reported.
    assert result.params.eta == pytest.approx(0.05)
    assert result.params.rho == pytest.approx(-0.70)
    assert FLAG_PARAMETER_PINNED in result.flags
    assert set(result.pinned) == {"eta", "rho"}


def test_the_returned_H_is_not_the_initialisation_on_a_flat_surface(degenerate_run):
    """
    The claim the old French label made, checked against what the code does.

    A simplex moves even where the surface is flat, so the returned ``H`` is
    where the search happened to stop - which is precisely why the run is
    reported as a failure rather than as "the initialisation".
    """
    result = degenerate_run
    assert result.initial_params.H == pytest.approx(0.25)
    assert result.params.H != pytest.approx(result.initial_params.H, abs=1e-6)


def test_flat_H_flag_fires_when_the_span_really_is_below_the_floor(
    market_points, xi0_curve
):
    """
    The (secondary) span diagnostic, and the whole chain it drives.

    ``noise_sigma_multiplier`` is the documented "how many sigmas before I
    believe a variation" knob; pushing it far above every span is the honest way
    to exercise the flat verdict, its French label and its effect on ``success``
    without fabricating a market.
    """
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.12, 1.4, -0.65),
        mc_cfg=_cheap_config(noise_sigma_multiplier=1e6),
        settings=CalibratorSettings(max_nfev=20, n_starts=1, seed=2_468),
        S0=S0,
    )
    profile = result.identifiability.profile("H")
    assert profile.span < profile.noise_floor
    assert profile.flat is True
    assert "PLAT" in profile.message_fr
    assert FLAG_H_PROFILE_FLAT in result.flags
    assert "PLAT" in JOINT_CALIBRATION_LABELS_FR[FLAG_H_PROFILE_FLAT]
    assert any("PLAT" in warning for warning in result.warnings_fr)
    assert result.success is False


def test_grid_refinement_bias_is_measured_and_reported(recovered):
    """
    The Phase-3 short-end skew bias is neither ignored nor silently corrected.

    The calibration grid is finer than the pipeline default, and what remains is
    quantified as a first-order parameter displacement on a twice-finer grid.
    """
    bias = recovered.grid_bias
    assert bias is not None
    assert bias.n_refined > bias.n_calibration
    assert bias.factor == 2
    assert recovered.config.grid_n_max > GridConfig().n_max  # finer than the default
    assert all(math.isfinite(x) for x in bias.theta_shift)
    assert bias.unmeasured == ()
    payload = bias.to_dict()
    assert "log-Euler" in payload["known_issue_fr"]
    assert set(payload["theta_shift"]) == {"H", "eta", "rho"}
    # The estimate is REPORTED, never applied.
    assert recovered.params.H == pytest.approx(float(recovered.theta[0]))

    # D11(b): "material" is judged against the parameter's OWN scale - the
    # recovery tolerance - not against the bound-to-bound width. Against the
    # width the default threshold needed |dH| > 0.048, |d_eta| > 0.495 and
    # |d_rho| > 0.1998, every one of them LARGER than the tolerance the result
    # is accepted on.
    assert PARAM_SCALE == {"H": TOL_H, "eta": TOL_ETA, "rho": TOL_RHO}
    for name, shift in zip(PARAM_ORDER, bias.theta_shift):
        assert payload["theta_shift_relative"][name] == pytest.approx(
            abs(shift) / PARAM_SCALE[name]
        )
    threshold = recovered.config.grid_bias_material
    for name in PARAM_ORDER:
        firing_shift = threshold * PARAM_SCALE[name]
        lo, hi = DEFAULT_BOUNDS[name]
        assert firing_shift < PARAM_SCALE[name]
        assert firing_shift < threshold * (hi - lo)  # strictly tighter than before
    assert "échelle propre" in payload["threshold_basis_fr"]


def test_grid_bias_reports_nan_when_it_could_not_be_measured(objective):
    """
    D3: "not measured" is NaN, never ``0.0``.

    ``step_H = 0.02 * (0.49 - 0.01) = 0.0096``, so any calibrated ``H < 0.0196``
    puts ``theta - step`` outside the box and the finite difference cannot be
    taken - and that is exactly the small-``H`` rough regime this pipeline
    exists for.  The skipped parameter used to be reported as
    ``theta_shift = 0.0`` with ``material = False``, i.e. "no measurement" read
    as "no bias" (a live run at ``H = 0.01832`` did precisely that), while the
    other skip path (non-positive curvature) already reported NaN.
    """
    theta = [0.012, 1.2, -0.6]  # H below step_H: unmeasurable by construction
    config = JointMCConfig(
        grid_n_max=96,
        stage2_paths=1_000,
        profile_points=3,
        noise_replicates=2,
        refinement_factor=2,
    )
    slim = JointObjective(
        quotes=objective.quotes, xi0=objective.xi0, config=config
    )
    profiles = [
        profile_slice(
            slim,
            theta,
            index=0,
            bounds=DEFAULT_BOUNDS["H"],
            n_points=3,
            n_paths=1_000,
            seed=99,
            noise_floor=1e-9,
        )
    ]
    step_H = 0.02 * (DEFAULT_BOUNDS["H"][1] - DEFAULT_BOUNDS["H"][0])
    assert theta[0] - step_H < DEFAULT_BOUNDS["H"][0]  # the trigger, stated

    bias = grid_refinement_bias(
        slim,
        theta,
        profiles=profiles,
        config=config,
        bounds=DEFAULT_BOUNDS,
        n_paths=1_000,
        seed=99,
    )
    assert bias is not None
    shifts = dict(zip(PARAM_ORDER, bias.theta_shift))
    relatives = dict(zip(PARAM_ORDER, bias.theta_shift_relative))
    for name in PARAM_ORDER:
        assert math.isnan(shifts[name]), f"{name} fabricated a shift of {shifts[name]}"
        assert math.isnan(relatives[name])
    # "Not measured" cannot be read as "no bias".
    assert bias.material is False
    assert set(bias.unmeasured) == set(PARAM_ORDER)
    assert len(bias.unmeasured_reasons) == len(bias.unmeasured)
    assert "NON MESURÉ" in bias.message_fr
    assert "non mesurables" in bias.message_fr
    assert "H" in bias.message_fr
    payload = bias.to_dict()
    assert payload["unmeasured"] == list(bias.unmeasured)
    json.dumps(payload, allow_nan=True)


def test_an_unmeasurable_grid_bias_is_flagged_end_to_end(market_points, xi0_curve):
    """A pinned parameter cannot be profiled, so its bias is NaN and flagged."""
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.12, 1.4, -0.65),
        mc_cfg=_cheap_config(refinement_check=True, refinement_factor=2),
        settings=CalibratorSettings(max_nfev=12, n_starts=1, seed=606_060),
        S0=S0,
        constraints={"rho": -0.65},
    )
    bias = result.grid_bias
    assert bias is not None
    shifts = dict(zip(PARAM_ORDER, bias.theta_shift))
    assert math.isnan(shifts["rho"])
    assert "rho" in bias.unmeasured
    assert FLAG_GRID_BIAS_NOT_MEASURED in result.flags
    assert FLAG_GRID_BIAS_NOT_MEASURED in JOINT_CALIBRATION_LABELS_FR


# ---------------------------------------------------------------------------
# 4bis. The restart count is honoured, or the truncation is reported (D5)
# ---------------------------------------------------------------------------
def test_n_starts_is_honoured_when_stage_one_can_supply_the_starts(
    market_points, xi0_curve
):
    """
    ``top_k`` must not silently cap ``settings.n_starts``.

    Measured before the fix: ``n_starts=4 / n_design=6 / top_k=2`` performed TWO
    local runs while ``details`` reported four restarts and four restart seeds.
    """
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=_cheap_config(n_design=6, top_k=2, stage1_paths=1_000),
        settings=CalibratorSettings(max_nfev=5, n_starts=3, seed=31_337),
        S0=S0,
    )
    assert result.config.top_k == 2  # deliberately smaller than n_starts
    assert len(result.stage2.runs) == 3
    assert result.stage2.n_starts_requested == 3
    assert result.stage2.n_starts_effective == 3
    assert len(result.details["seeds"]["restarts"]) == 3
    assert result.details["seeds"]["restarts_unused"] == []
    assert result.details["settings"]["n_starts_effective"] == 3
    assert FLAG_RESTARTS_TRUNCATED not in result.flags
    # The restarts really are distinct draws and distinct starting points.
    assert len({run.seed for run in result.stage2.runs}) == 3
    assert len({run.x0 for run in result.stage2.runs}) == 3


def test_a_truncated_restart_count_is_reported_not_papered_over(
    market_points, xi0_curve
):
    """
    With Stage 1 disabled there is exactly ONE start, whatever ``n_starts`` says.

    Measured before the fix: ``n_starts=4 / n_design=0`` performed one local run
    and reported four.  A number that did not happen must never be reported.
    """
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=_cheap_config(n_design=0),
        settings=CalibratorSettings(max_nfev=5, n_starts=4, seed=51_515),
        S0=S0,
    )
    assert len(result.stage2.runs) == 1
    assert result.stage2.n_starts_requested == 4
    assert result.stage2.n_starts_effective == 1
    assert FLAG_RESTARTS_TRUNCATED in result.flags
    assert FLAG_RESTARTS_TRUNCATED in JOINT_CALIBRATION_LABELS_FR
    assert result.details["settings"]["n_starts_requested"] == 4
    assert result.details["settings"]["n_starts_effective"] == 1
    assert len(result.details["seeds"]["restarts"]) == 1
    assert len(result.details["seeds"]["restarts_unused"]) == 3
    payload = result.to_dict()
    assert payload["stage2"]["n_runs"] == payload["stage2"]["n_starts_effective"] == 1


# ---------------------------------------------------------------------------
# 5. Determinism of the whole calibration
# ---------------------------------------------------------------------------
def test_same_seed_gives_a_bit_identical_calibration(market_points, xi0_curve):
    config = JointMCConfig(
        grid_n_max=192,
        n_design=4,
        stage1_paths=3_000,
        top_k=1,
        stage2_paths=4_000,
        final_paths=4_000,
        batch_paths=4_000,
        profile_points=4,
        valley_points=4,
        noise_replicates=2,
        refinement_check=False,
    )
    settings = CalibratorSettings(max_nfev=12, n_starts=1, seed=4_242)
    first = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=config,
        settings=settings,
        S0=S0,
    )
    second = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=config,
        settings=settings,
        S0=S0,
    )
    assert np.array_equal(first.theta, second.theta)
    assert first.loss_crn == second.loss_crn
    assert first.loss_fresh == second.loss_fresh
    assert first.loss_initial == second.loss_initial
    assert np.array_equal(first.iv_model, second.iv_model)
    assert first.details["seeds"] == second.details["seeds"]


# ---------------------------------------------------------------------------
# 6. Bounds and the constraints protocol
# ---------------------------------------------------------------------------
def test_default_bounds_are_the_spec_box():
    bounds, pinned = resolve_bounds(None)
    assert pinned == ()
    assert bounds["H"] == (0.01, 0.49)
    assert bounds["eta"] == (0.05, 5.0)
    assert bounds["rho"] == (-0.999, 0.999)


def test_constraints_pin_and_tighten():
    bounds, pinned = resolve_bounds(
        {"H": 0.17, "eta": [0.5, 2.0], "rho": {"min": -0.9, "max": -0.2}}
    )
    assert pinned == ("H",)
    assert bounds["H"] == (0.17, 0.17)
    assert bounds["eta"] == (0.5, 2.0)
    assert bounds["rho"] == (-0.9, -0.2)

    # A RANGE only ever tightens: it is intersected with the hard spec-4.6 box,
    # never widened past it.
    widened, _ = resolve_bounds({"H": [0.001, 0.6]})
    assert widened["H"] == (0.01, 0.49)

    # A PIN outside the model domain is a different statement, and it raises.
    with pytest.raises(RBergomiCalibrationError, match="hors du domaine"):
        resolve_bounds({"H": 0.6})
    with pytest.raises(RBergomiCalibrationError, match="min"):
        resolve_bounds({"eta": [2.0, 1.0]})


def test_calibrated_parameters_respect_the_tightened_bounds(market_points, xi0_curve):
    config = JointMCConfig(
        grid_n_max=256,
        n_design=4,
        stage1_paths=4_000,
        top_k=1,
        stage2_paths=6_000,
        final_paths=6_000,
        batch_paths=6_000,
        profile_points=4,
        valley_points=4,
        noise_replicates=2,
        refinement_check=False,
    )
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(0.2, 1.0, -0.5),
        mc_cfg=config,
        settings=CalibratorSettings(max_nfev=20, n_starts=1, seed=31),
        S0=S0,
        constraints={"H": [0.15, 0.25], "eta": [0.8, 2.5], "rho": [-0.85, -0.55]},
    )
    assert 0.15 <= result.params.H <= 0.25
    assert 0.8 <= result.params.eta <= 2.5
    assert -0.85 <= result.params.rho <= -0.55
    for slice_ in result.identifiability.profiles:
        lower, upper = result.bounds[slice_.parameter]
        assert slice_.values.min() == pytest.approx(lower)
        assert slice_.values.max() == pytest.approx(upper)


# ---------------------------------------------------------------------------
# 7. Quote set, weights and the forward drift
# ---------------------------------------------------------------------------
def test_forward_drift_reproduces_the_market_forwards_exactly(quote_set):
    """
    Oracle: re-accumulate ``S0 exp(sum r_i dt_i)`` here and compare to ``F(T)``.
    """
    grid = build_simulation_grid(
        maturities=MATURITIES, config=GridConfig(n_max=192, min_steps=16)
    )
    rates = forward_step_rates(
        grid=grid,
        maturities=quote_set.maturities,
        forwards=quote_set.forwards,
        S0=quote_set.S0,
    )
    cumulative = np.concatenate([[0.0], np.cumsum(rates * np.asarray(grid.dt))])
    forwards = S0 * np.exp(cumulative)
    market = _market_forwards()
    for T in MATURITIES:
        index = grid.index_of(float(T))
        assert forwards[index] == pytest.approx(market[float(T)], rel=1e-12)


def test_the_forward_is_extrapolated_past_the_last_quoted_maturity(quote_set):
    """
    D6: ``numpy.interp`` CLAMPS, so every step rate beyond the last quote was 0.

    The simulation grid spans the union of the quoted AND the reported
    maturities, and ``RBergomiJointHCalibrator.calibrate`` always hands the UI's
    ``t_grid`` in as the reporting grid, so the tail is genuinely reachable.
    Measured with quotes to 2 y and a reporting maturity at 3 y, the clamped
    forward gave ``F(3 y) = 102.020134`` against a market ``103.045453`` -
    ``-9.95e-03`` relative, about 0.15 volatility point at every strike of that
    maturity.

    Oracle: the market forward curve is ``S0 exp(DRIFT * T)`` with a CONSTANT
    rate, so continuing at the last known rate must reproduce it exactly - the
    test recomputes ``S0 exp(sum r_i dt_i)`` itself.
    """
    beyond = 3.0
    assert beyond > float(max(MATURITIES))
    grid = build_simulation_grid(
        maturities=list(MATURITIES) + [beyond],
        config=GridConfig(n_max=192, min_steps=16),
    )
    rates = forward_step_rates(
        grid=grid,
        maturities=quote_set.maturities,
        forwards=quote_set.forwards,
        S0=quote_set.S0,
    )
    cumulative = np.concatenate([[0.0], np.cumsum(rates * np.asarray(grid.dt))])
    forwards = S0 * np.exp(cumulative)

    market = _market_forwards()
    for T in MATURITIES:  # the quoted maturities are still exact
        index = grid.index_of(float(T))
        assert forwards[index] == pytest.approx(market[float(T)], rel=1e-12)

    index = grid.index_of(beyond)
    expected = S0 * math.exp(DRIFT * beyond)
    assert forwards[index] == pytest.approx(expected, rel=1e-12)
    # Not frozen: the clamped value would have been F(2y).
    assert forwards[index] > market[float(MATURITIES[-1])]
    assert np.all(np.asarray(rates)[grid.t[1:] > float(max(MATURITIES))] > 0.0)


def test_weights_are_normalised_per_maturity(quote_set):
    """
    Per-maturity normalisation, and the weight of a set with NO known spread.

    Every quote here carries an unknown bid-ask spread, so they all receive the
    same ``spread_iv`` and the weights inside a maturity are exactly uniform.
    That is the honest answer when nothing distinguishes the quotes' precision -
    and it is a change: the previous formula multiplied by a vega factor on top
    of the ``1 / spread_iv^2`` that already carries two powers of vega (D7).
    """
    weights = np.asarray(quote_set.weights, dtype=float)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)

    maturities = quote_set.array("T")
    for T in quote_set.maturities:
        share = float(weights[maturities == T].sum())
        assert share == pytest.approx(1.0 / quote_set.n_maturities, rel=1e-9)

    for T in quote_set.maturities:
        rows = [q for q in quote_set.quotes if q.T == T]
        assert all(not math.isfinite(q.spread_abs) for q in rows)
        expected = 1.0 / len(rows) / quote_set.n_maturities
        got = np.asarray([q.weight for q in rows], dtype=float)
        assert np.allclose(got, expected, rtol=1e-9, atol=0.0)


def _spread_rows(spread_of) -> list[dict]:
    """Raw quote mappings with a caller-chosen absolute bid-ask spread."""
    vol = 0.22
    rows: list[dict] = []
    for T in MATURITIES:
        F = S0 * math.exp(DRIFT * float(T))
        D = math.exp(-DISCOUNT_RATE * float(T))
        for j, sd in enumerate(SD_LADDER):
            K = F * math.exp(float(sd) * vol * math.sqrt(float(T)))
            rows.append(
                {
                    "T": float(T),
                    "K": float(K),
                    "F": float(F),
                    "D": float(D),
                    "k": math.log(float(K) / F),
                    "iv": vol,
                    "mid": float("nan"),
                    "spread_abs": float(spread_of(j)),
                    "spread_rel": float("nan"),
                    "option_type": "call" if K >= F else "put",
                }
            )
    return rows


def test_the_weight_exponent_matches_the_documented_intent(quote_set):
    """
    D7: with a strike-independent absolute spread the weight was ``~ vega^3``.

    ``spread_iv = 0.5 s / vega`` divides by the vega, ``(median / spread_iv)^2``
    multiplies it back twice, and the extra ``vega_i / max_T vega`` factor
    multiplied it a THIRD time.  Fitting ``log w`` against ``log vega`` on such a
    surface returned ``p = 3.00`` exactly, and the two most at-the-money quotes
    of each maturity took 43-67 % of that maturity's whole weight (11 strikes, so
    a uniform split is 18 %) - the wings that carry the ``rho`` / skew
    information were stripped.

    The documented intent is a plain inverse variance in vol units,
    ``w ~ vega^2 / spread_price^2``, so the exponent must now be **2**.

    NOTE, recorded rather than silently resolved: the Phase-4 finding asks in the
    same sentence for "``vega^2 / spread_price^2``" *and* for a fitted exponent
    of ~1.  Those two are not the same statement - with a constant absolute
    spread ``vega^2 / spread^2`` IS an exponent of 2 - and the finding's own
    prescription, "do not multiply by vega a third time", removes exactly one of
    the three powers.  This suite pins 2, the inverse-variance value, and reports
    the discrepancy.
    """
    config = WeightConfig(
        spread_iv_floor=1e-9, spread_iv_cap=50.0, spread_rel_max=1e9
    )
    quotes = build_calibration_quotes(
        _spread_rows(lambda j: 0.02), weights_cfg=config, S0=S0
    )
    weights = np.asarray([q.weight for q in quotes.quotes], dtype=float)
    vegas = np.asarray([q.vega for q in quotes.quotes], dtype=float)
    maturities = quotes.array("T")

    # Oracle: the vega is recomputed here from black76_vega, not read back.
    for quote in quotes.quotes:
        assert quote.vega == pytest.approx(
            black76_vega(F=quote.F, K=quote.K, T=quote.T, D=quote.D, vol=quote.iv)
        )

    for T in quotes.maturities:
        rows = maturities == T
        exponent = float(
            np.polyfit(np.log(vegas[rows]), np.log(weights[rows]), 1)[0]
        )
        assert exponent == pytest.approx(2.0, abs=0.02)
        share = weights[rows] / weights[rows].sum()
        # No single quote may run away with its maturity.
        assert share.max() < 0.35
        assert float(np.sort(share)[-2:].sum()) < 0.60


def test_unknown_spreads_are_neutral_against_known_ones(quote_set):
    """
    D8: ``default_spread_iv`` was not neutral at all.

    Measured on a set where half the quotes carry a 0.02 absolute spread and the
    median ``spread_iv`` equals ``default_spread_iv`` exactly, the known-spread
    quotes took **98.89 %** of the total weight (mean weight ratio 117x, overall
    max/min 1.25e7): an at-the-money known-spread quote's ``0.5 * 0.02 / vega``
    falls below ``spread_iv_floor = 2e-3`` and was clipped up to it, receiving
    ``(0.02 / 0.002)^2 = 100``, while every unknown-spread quote received exactly
    ``1``.  The unknown ones now take the median of the known ones, after the
    clip, which is the only assignment that puts both populations on one scale.
    """
    rows = _spread_rows(lambda j: 0.02 if j % 2 == 0 else float("nan"))
    quotes = build_calibration_quotes(rows, S0=S0)
    weights = np.asarray([q.weight for q in quotes.quotes], dtype=float)
    known = np.asarray(
        [math.isfinite(q.spread_abs) and q.spread_abs > 0.0 for q in quotes.quotes]
    )
    assert known.any() and (~known).any()

    known_share = float(weights[known].sum())
    assert 0.25 < known_share < 0.75, known_share
    ratio = float(weights[known].mean() / weights[~known].mean())
    assert 0.2 < ratio < 5.0, ratio
    assert float(weights.max() / weights.min()) < 5.0e3

    # An unknown-spread quote is priced at the median of the KNOWN ones, so a
    # median known quote and an unknown quote weigh the same.
    known_spread_iv = np.asarray([q.spread_iv for q in quotes.quotes])[known]
    unknown_spread_iv = np.asarray([q.spread_iv for q in quotes.quotes])[~known]
    assert np.allclose(unknown_spread_iv, np.median(known_spread_iv))
    assert unknown_spread_iv[0] != pytest.approx(WeightConfig().default_spread_iv)

    # With NO known spread anywhere the documented constant is still the
    # fallback, and being a constant it is genuinely neutral there.
    none_known = build_calibration_quotes(
        _spread_rows(lambda j: float("nan")), S0=S0
    )
    assert all(
        q.spread_iv == pytest.approx(WeightConfig().default_spread_iv)
        for q in none_known.quotes
    )


def test_quotes_are_re_expressed_out_of_the_money(quote_set):
    for quote in quote_set.quotes:
        if quote.K >= quote.F:
            assert quote.option_type == "call"
        else:
            assert quote.option_type == "put"
        assert quote.price > 0.0
        assert quote.vega > 0.0


def test_hard_exclusions_are_recorded_with_a_reason(strikes, market_iv):
    points = _surface_points(strikes, market_iv)
    poisoned = list(points)
    T0 = float(MATURITIES[3])
    F0 = _market_forwards()[T0]
    poisoned.append(
        SurfacePoint(
            T=T0,
            K=float(F0 * math.e ** 2.0),
            k=2.0,
            F=float(F0),
            D=float(_market_discounts()[T0]),
            iv=0.30,
            option_type="call",
            mid=float("nan"),
            call_equivalent_price=float("nan"),
            vendor_iv=float("nan"),
            one_sided=False,
        )
    )
    poisoned.append(
        SurfacePoint(
            T=T0,
            K=float(F0),
            k=0.0,
            F=float(F0),
            D=float(_market_discounts()[T0]),
            iv=float("nan"),
            option_type="call",
            mid=float("nan"),
            call_equivalent_price=float("nan"),
            vendor_iv=float("nan"),
            one_sided=False,
        )
    )
    quotes = build_calibration_quotes(poisoned, S0=S0)
    reasons = {rejection.reason for rejection in quotes.rejections}
    assert REASON_K_TOO_FAR in reasons
    assert REASON_NON_FINITE_IV in reasons
    for rejection in quotes.rejections:
        assert rejection.reason_fr
        assert rejection.to_dict()["reason_fr"]
    assert quotes.n_quotes == len(points)


def test_an_empty_quote_set_refuses_rather_than_fitting_nothing(xi0_curve):
    with pytest.raises(RBergomiCalibrationError, match="insuffisant"):
        build_calibration_quotes([], S0=S0)


def test_two_forwards_on_one_expiry_are_refused(strikes, market_iv):
    """Mixing two forward curves would land straight on (H, eta, rho)."""
    points = _surface_points(strikes, market_iv)
    poisoned = list(points)
    victim = poisoned[0]
    poisoned[0] = SurfacePoint(
        T=victim.T,
        K=victim.K,
        k=victim.k,
        F=victim.F * 1.01,
        D=victim.D,
        iv=victim.iv,
        option_type=victim.option_type,
        mid=victim.mid,
        call_equivalent_price=victim.call_equivalent_price,
        vendor_iv=victim.vendor_iv,
        one_sided=victim.one_sided,
    )
    with pytest.raises(RBergomiCalibrationError, match="incohérentes"):
        build_calibration_quotes(poisoned, S0=S0)


def test_pooling_batches_is_exact(objective):
    """
    Oracle: pooling ``B`` identical batches leaves the mean alone and divides the
    standard error by ``sqrt(B)``. Pure algebra, checked to round-off.
    """
    grid = objective.grid()
    rates = objective.step_rates(grid)
    paths = simulate_rbergomi_xi_curve(
        S0=S0,
        xi_curve=objective.xi0.curve,
        params=RBergomiParams(H_TRUE, ETA_TRUE, RHO_TRUE),
        maturities=[float(T) for T in MATURITIES],
        grid=grid,
        r=rates,
        q=0.0,
        config=SimulationConfig(n_paths=2_000, antithetic=True, seed=5, grid_config=None),
    )
    single = price_call(
        paths,
        strikes=np.full((len(MATURITIES), 1), S0),
        maturities=[float(T) for T in MATURITIES],
        estimator=ESTIMATOR_CONDITIONAL,
    )
    pooled = _pool_price_results([single, single, single, single])
    assert np.allclose(pooled.price, single.price, rtol=0.0, atol=1e-15)
    assert np.allclose(pooled.stderr, single.stderr / 2.0, rtol=1e-12)
    assert pooled.n_paths == 4 * single.n_paths
    assert _pool_price_results([single]) is single


# ---------------------------------------------------------------------------
# 8. Alternative objective
# ---------------------------------------------------------------------------
def test_price_relative_objective_also_recovers(market_points, xi0_curve):
    config = JointMCConfig(
        grid_n_max=256,
        n_design=0,
        stage2_paths=8_000,
        final_paths=8_000,
        batch_paths=8_000,
        profile_points=4,
        valley_points=4,
        noise_replicates=2,
        refinement_check=False,
    )
    result = calibrate_rbergomi(
        market_points,
        xi0_curve,
        RBergomiParams(H=0.20, eta=1.0, rho=-0.50),
        weights_cfg=WeightConfig(objective=OBJECTIVE_PRICE_RELATIVE),
        mc_cfg=config,
        settings=CalibratorSettings(n_starts=1, seed=606),
        S0=S0,
    )
    assert result.quotes.config.objective == OBJECTIVE_PRICE_RELATIVE
    assert abs(result.params.H - H_TRUE) <= TOL_H
    assert abs(result.params.eta - ETA_TRUE) <= TOL_ETA
    assert abs(result.params.rho - RHO_TRUE) <= TOL_RHO


# ---------------------------------------------------------------------------
# 9. Repo integration
# ---------------------------------------------------------------------------
def _surface_grid(
    strikes: np.ndarray, market_iv: np.ndarray, *, extra_t: float | None = None
) -> SurfaceGrid:
    """
    The repo's grid mode, interpolated in log-moneyness from the quotes.

    ``extra_t`` appends a reporting maturity with NO market IV (all NaN), which
    is how the UI's ``t_grid`` can legitimately reach past the quoted range: the
    row is excluded from the fit by ``effective_mask`` but still priced by the
    model, so it exercises the forward extrapolation of D6.
    """
    m_grid = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    t_grid = np.asarray(MATURITIES[:4], dtype=float)
    forwards = _market_forwards()
    iv_grid = np.empty((t_grid.size, m_grid.size), dtype=float)
    for i, T in enumerate(t_grid):
        F = forwards[float(T)]
        iv_grid[i, :] = np.interp(
            np.log(m_grid * S0 / F), np.log(strikes[i] / F), market_iv[i]
        )
    if extra_t is not None:
        t_grid = np.concatenate([t_grid, [float(extra_t)]])
        iv_grid = np.vstack([iv_grid, np.full((1, m_grid.size), np.nan)])
    return SurfaceGrid(
        S0=S0,
        r=DISCOUNT_RATE,
        q=DISCOUNT_RATE - DRIFT,
        m_grid=m_grid,
        t_grid=t_grid,
        iv_market=iv_grid,
        mask=np.isfinite(iv_grid),
    )


def test_calibrator_class_runs_on_a_surface_grid(strikes, market_iv, xi0_curve):
    surface = _surface_grid(strikes, market_iv)
    calibrator = RBergomiJointHCalibrator()
    assert calibrator.model == "rbergomi"
    assert calibrator.method == "joint_h_mc"
    assert calibrator.PARAM_ORDER == ("H", "eta", "rho")
    assert "xi0" not in calibrator.DEFAULT_BOUNDS

    result = calibrator.calibrate(
        surface,
        constraints={
            "xi0_curve": xi0_curve,
            "mc_cfg": {
                "grid_n_max": 256,
                "n_design": 4,
                "stage1_paths": 4_000,
                "top_k": 1,
                "stage2_paths": 6_000,
                "final_paths": 6_000,
                "batch_paths": 6_000,
                "profile_points": 4,
                "valley_points": 4,
                "noise_replicates": 2,
                "refinement_check": False,
                "local_nfev_per_param": 20,
            },
        },
        settings=CalibratorSettings(n_starts=1, seed=21),
    )
    assert result.success is True
    assert set(result.params) == {"H", "eta", "rho"}
    for name, value in result.params.items():
        low, high = DEFAULT_BOUNDS[name]
        assert low <= value <= high
    assert result.iv_model.shape == surface.iv_market.shape
    assert np.isfinite(result.iv_model).all()
    assert np.isfinite(result.metrics["rmse"])
    assert result.vega_weights.shape == surface.iv_market.shape
    # Everything the controller layer will serialise must survive json.
    json.dumps(result.details, allow_nan=True)
    assert result.details["report"]["xi0_frozen"] is True
    assert result.details["initial_params_source"] == "box_centre"
    assert result.details["stage2"]["max_nfev_source"] == "config"
    # Sized by the FREE parameter count, from the calibrator's own config -
    # `local_nfev_per_param` is shrunk here only to keep this test cheap.
    assert result.details["report"]["max_nfev_effective"] == 20 * 3
    assert FLAG_REPORT_BEYOND_QUOTES not in result.details["flags"]


def test_a_reporting_maturity_past_the_last_quote_is_priced_and_flagged(
    strikes, market_iv, xi0_curve
):
    """
    D6 end to end: the UI grid is not constrained to the quoted range.

    The forward is extrapolated at the last known rate instead of being frozen,
    and the caller is told that nothing out there is constrained by a quote.
    """
    beyond = 3.0
    surface = _surface_grid(strikes, market_iv, extra_t=beyond)
    assert beyond > float(max(MATURITIES[:4]))
    result = RBergomiJointHCalibrator().calibrate(
        surface,
        constraints={
            "xi0_curve": xi0_curve,
            "mc_cfg": {
                "grid_n_max": 192,
                "n_design": 0,
                "stage2_paths": 2_000,
                "profile_paths": 2_000,
                "final_paths": 4_000,
                "batch_paths": 4_000,
                "profile_points": 4,
                "valley_points": 4,
                "noise_replicates": 2,
                "refinement_check": False,
            },
        },
        settings=CalibratorSettings(max_nfev=10, n_starts=1, seed=97),
    )
    details = result.details
    assert FLAG_REPORT_BEYOND_QUOTES in details["flags"]
    assert details["details"]["report_maturities_beyond_quotes"] == [beyond]
    assert FLAG_REPORT_BEYOND_QUOTES in JOINT_CALIBRATION_LABELS_FR
    assert any("EXTRAPOL" in w for w in details["warnings_fr"])
    # The far row is priced, finite, and NOT the frozen continuation of the last
    # quoted maturity: a clamped forward would have shifted it by ~0.15 vol pt.
    far = np.asarray(result.iv_model, dtype=float)[-1, :]
    assert np.isfinite(far).all()
    assert np.all(far > 0.0)


def test_calibrator_refuses_without_a_forward_variance_curve(strikes, market_iv):
    surface = _surface_grid(strikes, market_iv)
    result = RBergomiJointHCalibrator().calibrate(surface, constraints={})
    assert result.success is False
    assert "variance forward" in result.message
    assert result.params == {}


def test_calibrator_refuses_a_xi0_constraint(strikes, market_iv, xi0_curve):
    surface = _surface_grid(strikes, market_iv)
    result = RBergomiJointHCalibrator().calibrate(
        surface, constraints={"xi0_curve": xi0_curve, "xi0": 0.04}
    )
    assert result.success is False
    assert "calibrable" in result.message


def test_calibrator_fits_real_quotes_when_they_are_supplied(
    strikes, market_iv, market_points, xi0_curve
):
    """
    Spec 4.10: the reference objective uses the REAL quotes, not grid-snapped
    values; the ``SurfaceGrid`` is then only a reporting canvas.
    """
    surface = _surface_grid(strikes, market_iv)
    result = RBergomiJointHCalibrator().calibrate(
        surface,
        constraints={
            "xi0_curve": xi0_curve,
            "option_surface": market_points,
            "initial_params": RBergomiParams(0.2, 1.2, -0.6),
            "mc_cfg": {
                "grid_n_max": 256,
                "n_design": 0,
                "stage2_paths": 6_000,
                "final_paths": 6_000,
                "batch_paths": 6_000,
                "profile_points": 4,
                "valley_points": 4,
                "noise_replicates": 2,
                "refinement_check": False,
                "local_nfev_per_param": 20,
            },
        },
        settings=CalibratorSettings(n_starts=1, seed=77),
    )
    assert result.success is True
    assert result.details["quote_source"] == "option_surface"
    assert result.details["quotes"]["n_quotes"] == len(market_points)
    # The reported surface still covers the whole reporting grid.
    assert result.iv_model.shape == surface.iv_market.shape
    assert np.isfinite(result.iv_model).all()


def test_calibrator_degrades_gracefully_on_an_unbuildable_grid(
    market_points, xi0_curve
):
    """
    A reporting maturity indistinguishable from a quoted one makes the joint
    covariance singular; that must surface as an explicit failed result, not as
    an exception escaping into the controller layer.
    """
    colliding = float(MATURITIES[0]) * (1.0 + 1e-12)
    surface = SurfaceGrid(
        S0=S0,
        r=DISCOUNT_RATE,
        q=DISCOUNT_RATE - DRIFT,
        m_grid=np.array([0.95, 1.0, 1.05]),
        t_grid=np.array([colliding]),
        iv_market=np.full((1, 3), 0.20),
        mask=np.ones((1, 3), dtype=bool),
    )
    result = RBergomiJointHCalibrator().calibrate(
        surface,
        constraints={"xi0_curve": xi0_curve, "option_surface": market_points},
        settings=CalibratorSettings(max_nfev=4, n_starts=1, seed=5),
    )
    assert result.success is False
    assert "impossible" in result.message
    assert result.params == {}


def test_report_payload_is_json_safe(recovered):
    payload = recovered.to_dict()
    json.dumps(payload, allow_nan=True)
    summary = calibration_report(recovered)
    json.dumps(summary, allow_nan=True)
    assert summary["xi0_frozen"] is True
    assert summary["n_quotes"] == recovered.quotes.n_quotes
    assert summary["mean_evaluation_seconds"] > 0.0
    assert payload["method"] == "joint_h_mc"
    assert payload["stage2"]["method"] in ("nelder-mead", "powell", "least_squares")
    assert payload["identifiability"]["noise_floor"]["value"] >= 0.0
    # Everything the Phase-4 panel asked to be reported must reach the payload.
    assert payload["identifiability"]["noise_floor"]["difference_value"] > 0.0
    assert payload["identifiability"]["H_standard_error"] > 0.0
    assert set(payload["identifiability"]["standard_errors"]) == {"H", "eta", "rho"}
    assert payload["fresh_seed_gap"]["n_paths_crn"] == payload["fresh_seed_gap"][
        "n_paths_fresh"
    ]
    assert payload["stage2"]["n_starts_effective"] == payload["stage2"]["n_runs"]
    assert summary["H_standard_error"] > 0.0
    assert summary["max_nfev_effective"] >= 150
    assert summary["n_starts_effective"] == 1
    assert summary["noise_floor"] == pytest.approx(
        payload["identifiability"]["noise_floor"]["difference_value"]
    )
