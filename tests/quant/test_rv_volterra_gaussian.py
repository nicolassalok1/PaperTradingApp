"""
Tests for the exact joint Gaussian construction of the rough Bergomi driver pair.

Target: app/model/volatility_models/rbergomi/volterra_gaussian.py
  - build_simulation_grid                 -> grid policy (quoted maturities exact)
  - volterra_autocovariance               -> Cov(W~_u, W~_v), 2F1 form
  - volterra_brownian_cross_covariance    -> Cov(W~_t, B_b - B_a)
  - build_joint_covariance                -> the 2n x 2n Sigma
  - cholesky_factor                       -> L + jitter policy + LRU cache
  - draw_joint_gaussian                   -> (W~, dB, dB^perp), antithetic pairs

ORACLES (all independent of the code under test):
  * ``_autocovariance_quad`` - brute-force ``scipy.integrate.quad`` of
    ``2H * int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds`` with the integrable
    endpoint singularity declared through ``points=``.  The module evaluates the
    same quantity through ``scipy.special.hyp2f1``; the two share no code path.
  * ``ORACLE_AUTOCOVARIANCE`` - spot values derived and verified independently
    by the orchestrator (MATH_ORACLE.md sec. 7), hard-coded here so a silent
    change of representation cannot slip through.
  * ``_cross_covariance_direct`` - the closed form written with the DIRECT
    subtraction of two powers; the module uses a cancellation-free
    ``expm1``/``log1p`` factorisation of the same expression, so the inline
    version is a genuine second implementation.  ``_cross_covariance_quad``
    integrates the Ito-isometry integrand numerically as a third opinion.
  * The exact degeneracy at ``H = 1/2``: the Riemann-Liouville kernel collapses
    to 1, so ``W~ == B`` and ``Cov(W~_u, W~_v) == min(u, v)``.  That is a
    property of the target process, not of the algorithm.
  * Structural invariants that hold sample-wise and are therefore exact under a
    fixed seed: exact symmetry of Sigma, exact float equality of the quoted
    maturities on the grid, bit-identical antithetic mirroring, bit-identical
    reproducibility, cache identity.

STATISTICAL TOLERANCES: every statistical assertion averages an unbiased
per-path estimator over independent paths, so the standard error is estimated
from the SAME sample (std over paths / sqrt(n_paths)) and the band is
``4 * stderr + <small absolute floor>`` - the same discipline as
tests/quant/test_mc_pricing.py.  Seeds are fixed, so the outcome is
deterministic.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.integrate import quad

from app.model.volatility_models.rbergomi import volterra_gaussian as vg
from app.model.volatility_models.rbergomi.volterra_gaussian import (
    CovarianceFactorizationError,
    GridConfig,
    GridConfigurationError,
    SimulationGrid,
    build_joint_covariance,
    build_simulation_grid,
    cholesky_factor,
    clear_factor_cache,
    draw_joint_gaussian,
    factor_cache_info,
    set_factor_cache_maxsize,
    volterra_autocovariance,
    volterra_brownian_cross_covariance,
)

pytestmark = pytest.mark.unit


# Realistic listed-option maturities (calendar days / 365, the repo day count).
QUOTED_MATURITIES = tuple(
    d / 365.0 for d in (7.0, 14.0, 30.0, 60.0, 91.0, 182.0, 365.0, 730.0)
)

# Rough-vol Hurst values plus one at the smooth end, to exercise the 2F1 branch
# on both sides of the Brownian case.
HURSTS = (0.05, 0.12, 0.25, 0.40, 0.70)

# Statistical block: 12-step grid, 60k paths -> ~0.6% relative standard error on
# a second moment, and every draw stays in the millisecond range.
STAT_PATHS = 60_000


# ---------------------------------------------------------------------------
# Independent oracles (no shared code path with the module under test)
# ---------------------------------------------------------------------------
def _autocovariance_quad(u: float, v: float, H: float) -> float:
    """``2H * int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds`` by quadrature."""
    lo, hi = (float(u), float(v)) if u <= v else (float(v), float(u))
    if lo <= 0.0:
        return 0.0

    def integrand(s: float) -> float:
        return (hi - s) ** (H - 0.5) * (lo - s) ** (H - 0.5)

    value, _ = quad(integrand, 0.0, lo, points=[lo], limit=400)
    return 2.0 * H * value


def _cross_covariance_direct(t: float, a: float, b: float, H: float) -> float:
    """``Cov(W~_t, B_b - B_a)`` written with the DIRECT power subtraction."""
    if a >= t:
        return 0.0
    upper = min(b, t)
    return (
        math.sqrt(2.0 * H)
        / (H + 0.5)
        * ((t - a) ** (H + 0.5) - (t - upper) ** (H + 0.5))
    )


def _cross_covariance_quad(t: float, a: float, b: float, H: float) -> float:
    """``sqrt(2H) * int_a^{min(b,t)} (t-s)^{H-1/2} ds`` by quadrature (Ito isometry)."""
    if a >= t:
        return 0.0
    upper = min(b, t)
    value, _ = quad(
        lambda s: (t - s) ** (H - 0.5), a, upper, points=[t], limit=400
    )
    return math.sqrt(2.0 * H) * value


def _stderr(sample: np.ndarray) -> np.ndarray:
    """Standard error of the mean of a per-path estimator, from the sample itself."""
    return np.std(sample, axis=0, ddof=1) / math.sqrt(sample.shape[0])


# MATH_ORACLE.md sec. 7 - (H, u, v, Cov(W~_u, W~_v)), derived and verified
# independently of this implementation.
ORACLE_AUTOCOVARIANCE = (
    (0.10, 0.02, 1.0, 3.2039548504e-02),
    (0.10, 0.30, 0.9, 1.8617589491e-01),
    (0.25, 0.02, 1.0, 3.5557538975e-02),
    (0.25, 0.30, 0.9, 2.9318880146e-01),
    (0.40, 0.30, 0.9, 3.1013265940e-01),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_factor_cache():
    """The Cholesky LRU is module-level state: isolate every test from it."""
    clear_factor_cache()
    previous = factor_cache_info()["maxsize"]
    yield
    set_factor_cache_maxsize(previous)
    clear_factor_cache()


@pytest.fixture(scope="module")
def stat_sample() -> tuple[SimulationGrid, float, vg.JointGaussianDraw]:
    """One shared 60k-path draw reused by every statistical assertion."""
    grid = build_simulation_grid(
        maturities=[0.08, 0.25, 1.0],
        config=GridConfig(n_max=12, min_steps=4),
    )
    hurst = 0.15
    factor = cholesky_factor(H=hurst, grid=grid, use_cache=False)
    draw = draw_joint_gaussian(factor=factor, n_paths=STAT_PATHS, seed=20260820)
    return grid, hurst, draw


def _small_grid(**overrides) -> SimulationGrid:
    config = GridConfig(**{"n_max": 24, "min_steps": 5, **overrides})
    return build_simulation_grid(maturities=[0.05, 0.5, 1.5], config=config)


# ---------------------------------------------------------------------------
# 1. Autocovariance: the 2F1 form against the orchestrator's oracle and quad
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H, u, v, expected", ORACLE_AUTOCOVARIANCE)
def test_autocovariance_matches_orchestrator_spot_values(H, u, v, expected):
    got = float(volterra_autocovariance(u, v, H=H))
    assert got == pytest.approx(expected, rel=1e-9)
    # Symmetry of the closed form itself, bit for bit.
    assert float(volterra_autocovariance(v, u, H=H)) == got


@pytest.mark.parametrize("H", HURSTS)
@pytest.mark.parametrize(
    "u, v",
    [
        (0.1, 0.1),
        (0.02, 1.0),
        (0.5, 0.5001),
        (0.3, 0.9),
        (1e-3, 2.0),
        (1.0, 1.0),
        (2.0, 0.004),
    ],
)
def test_autocovariance_matches_independent_quadrature(H, u, v):
    got = float(volterra_autocovariance(u, v, H=H))
    reference = _autocovariance_quad(u, v, H)
    assert got == pytest.approx(reference, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("H", HURSTS)
@pytest.mark.parametrize("t", [0.004, 0.05, 0.25, 1.0, 3.0])
def test_autocovariance_diagonal_is_t_to_the_2h(H, t):
    """Gauss's theorem collapses the 2F1 to (H+1/2)/(2H) at u = v."""
    got = float(volterra_autocovariance(t, t, H=H))
    assert got == pytest.approx(t ** (2.0 * H), rel=1e-12, abs=1e-15)


def test_autocovariance_degenerates_to_brownian_at_half_hurst():
    """At H = 1/2 the RL kernel is 1, so W~ == B and Cov = min(u, v)."""
    u = np.array([0.01, 0.25, 1.0, 2.0])
    v = np.array([1.0, 0.25, 0.3, 0.5])
    got = np.asarray(volterra_autocovariance(u, v, H=0.5))
    np.testing.assert_allclose(got, np.minimum(u, v), rtol=1e-13, atol=0.0)


def test_autocovariance_is_zero_at_the_origin_and_rejects_negative_times():
    assert float(volterra_autocovariance(0.0, 1.0, H=0.2)) == 0.0
    assert float(volterra_autocovariance(0.0, 0.0, H=0.2)) == 0.0
    with pytest.raises(ValueError):
        volterra_autocovariance(-0.1, 1.0, H=0.2)
    with pytest.raises(ValueError):
        volterra_autocovariance(0.1, 1.0, H=0.0)
    with pytest.raises(ValueError):
        volterra_autocovariance(0.1, 1.0, H=1.0)


def test_autocovariance_broadcasts_and_stays_exactly_symmetric():
    times = np.array([0.01, 0.1, 0.4, 1.0, 2.0])
    matrix = np.asarray(
        volterra_autocovariance(times[:, None], times[None, :], H=0.13)
    )
    assert matrix.shape == (5, 5)
    assert np.array_equal(matrix, matrix.T)


# ---------------------------------------------------------------------------
# 2. Cross-covariance driver <-> Brownian increment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", HURSTS)
@pytest.mark.parametrize(
    "t, a, b",
    [
        (1.0, 0.0, 0.01),
        (1.0, 0.0, 1.0),
        (1.0, 0.5, 0.51),
        (1.0, 0.99, 1.0),
        (0.02, 0.0, 0.02),
        (0.25, 0.24, 0.30),  # b beyond t -> clipped at t
        (1.0, 1.0, 1.01),  # a == t -> exactly 0
        (1.0, 1.5, 2.0),  # increment strictly in the future -> exactly 0
    ],
)
def test_cross_covariance_matches_the_closed_form_and_quadrature(H, t, a, b):
    got = float(volterra_brownian_cross_covariance(t, a, b, H=H))
    direct = _cross_covariance_direct(t, a, b, H)
    integrated = _cross_covariance_quad(t, a, b, H)
    assert got == pytest.approx(direct, rel=1e-11, abs=1e-16)
    assert got == pytest.approx(integrated, rel=1e-8, abs=1e-16)


def test_cross_covariance_is_causal_and_additive_over_increments():
    """Splitting [a, b] must split the covariance: the integrand is deterministic."""
    H, t, a, mid, b = 0.17, 1.0, 0.1, 0.4, 0.7
    whole = float(volterra_brownian_cross_covariance(t, a, b, H=H))
    left = float(volterra_brownian_cross_covariance(t, a, mid, H=H))
    right = float(volterra_brownian_cross_covariance(t, mid, b, H=H))
    assert whole == pytest.approx(left + right, rel=1e-13)
    assert float(volterra_brownian_cross_covariance(t, t, t + 1.0, H=H)) == 0.0


def test_cross_covariance_variance_consistency_at_half_hurst():
    """At H = 1/2, Cov(W~_t, B_b - B_a) = min(b, t) - a for a < t."""
    got = float(volterra_brownian_cross_covariance(1.0, 0.2, 0.7, H=0.5))
    assert got == pytest.approx(0.5, rel=1e-14)


def test_cross_covariance_rejects_inverted_increments():
    with pytest.raises(ValueError):
        volterra_brownian_cross_covariance(1.0, 0.5, 0.2, H=0.2)
    with pytest.raises(ValueError):
        volterra_brownian_cross_covariance(-1.0, 0.0, 0.2, H=0.2)


# ---------------------------------------------------------------------------
# 3. Sigma assembly: diagonal, symmetry, positive semi-definiteness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", HURSTS)
@pytest.mark.parametrize(
    "maturities, config",
    [
        ([0.05, 0.5, 1.5], GridConfig(n_max=24, min_steps=5)),
        (list(QUOTED_MATURITIES), GridConfig(n_max=40, min_steps=8)),
        ([1.0], GridConfig(n_max=32, min_steps=16)),
    ],
)
def test_sigma_diagonal_equals_t_to_the_2h(H, maturities, config):
    grid = build_simulation_grid(maturities=maturities, config=config)
    sigma = build_joint_covariance(H=H, grid=grid)
    n = grid.n
    assert sigma.shape == (2 * n, 2 * n)
    np.testing.assert_allclose(
        np.diag(sigma)[:n], grid.times ** (2.0 * H), rtol=0.0, atol=1e-10
    )
    # The Brownian block carries the step lengths exactly.
    np.testing.assert_array_equal(np.diag(sigma)[n:], grid.dt)


@pytest.mark.parametrize("H", HURSTS)
def test_sigma_is_exactly_symmetric_and_positive_semidefinite(H):
    grid = _small_grid()
    sigma = build_joint_covariance(H=H, grid=grid)
    assert np.array_equal(sigma, sigma.T)
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert eigenvalues.min() >= -1e-10 * eigenvalues.max()


def test_sigma_blocks_match_the_closed_forms_entry_by_entry():
    """Spot-check the assembly wiring against the inline oracles, not the module."""
    H = 0.19
    grid = build_simulation_grid(
        maturities=[0.1, 0.6], config=GridConfig(n_max=10, min_steps=3)
    )
    sigma = build_joint_covariance(H=H, grid=grid)
    n = grid.n
    times = np.asarray(grid.times)
    for i in range(n):
        for j in range(n):
            assert sigma[i, j] == pytest.approx(
                _autocovariance_quad(times[i], times[j], H), rel=1e-9, abs=1e-15
            )
            assert sigma[i, n + j] == pytest.approx(
                _cross_covariance_direct(times[i], grid.t[j], grid.t[j + 1], H),
                rel=1e-11,
                abs=1e-16,
            )
            expected_increment = grid.dt[i] if i == j else 0.0
            assert sigma[n + i, n + j] == expected_increment
    # Causality: the driver at t_i is blind to increments after t_i.
    upper = np.triu(sigma[:n, n:], k=1)
    assert np.array_equal(upper, np.zeros_like(upper))


def test_sigma_rejects_an_invalid_hurst():
    grid = _small_grid()
    with pytest.raises(ValueError):
        build_joint_covariance(H=1.5, grid=grid)


# ---------------------------------------------------------------------------
# 4. Grid policy
# ---------------------------------------------------------------------------
def test_every_quoted_maturity_lies_exactly_on_the_grid():
    config = GridConfig(n_max=256, min_steps=16)
    grid = build_simulation_grid(maturities=list(QUOTED_MATURITIES), config=config)
    for maturity in QUOTED_MATURITIES:
        index = grid.index_of(maturity)
        assert grid.t[index] == maturity  # EXACT float equality: no snapping
    assert grid.t[0] == 0.0
    assert np.all(np.diff(grid.t) > 0.0)
    np.testing.assert_array_equal(grid.dt, np.diff(grid.t))
    assert grid.quoted_maturities == tuple(sorted(QUOTED_MATURITIES))


def test_short_maturities_receive_at_least_min_steps_points():
    for min_steps in (4, 16, 32):
        config = GridConfig(n_max=256, min_steps=min_steps)
        grid = build_simulation_grid(
            maturities=list(QUOTED_MATURITIES), config=config
        )
        shortest = min(QUOTED_MATURITIES)
        assert grid.index_of(shortest) >= min_steps
        assert int(np.sum((grid.t > 0.0) & (grid.t <= shortest))) >= min_steps
        assert grid.t[1] < shortest  # the block really is BELOW the first quote


@pytest.mark.parametrize("n_max", (16, 32, 64, 128, 256))
def test_grid_never_exceeds_n_max(n_max):
    grid = build_simulation_grid(
        maturities=list(QUOTED_MATURITIES),
        config=GridConfig(n_max=n_max, min_steps=8),
    )
    assert grid.n <= n_max
    assert grid.t.size == grid.n + 1
    for maturity in QUOTED_MATURITIES:
        assert grid.t[grid.index_of(maturity)] == maturity


def test_grid_spends_its_whole_budget_when_the_range_allows_it():
    grid = build_simulation_grid(
        maturities=list(QUOTED_MATURITIES),
        config=GridConfig(n_max=64, min_steps=8),
    )
    assert grid.n == 64
    assert grid.n_dropped_fill == 0


def test_grid_fill_is_log_spaced_and_dense_near_zero():
    grid = build_simulation_grid(
        maturities=[1.0], config=GridConfig(n_max=40, min_steps=40)
    )
    steps = np.diff(grid.t)
    # Beyond the origin cell the log-spaced fill has monotonically growing steps.
    assert np.all(np.diff(steps[1:]) > 0.0)
    # The origin cell spans [0, short_end_ratio * T_min]: a geometric block
    # cannot reach 0, so that first cell is the grading floor. It must still be
    # far finer than the long end.
    assert steps[0] == pytest.approx(
        grid.config.short_end_ratio * min(grid.quoted_maturities), rel=1e-12
    )
    assert steps[0] < steps[-1] / 10.0


def test_grid_rejects_a_budget_that_cannot_honour_min_steps():
    with pytest.raises(GridConfigurationError):
        build_simulation_grid(
            maturities=list(QUOTED_MATURITIES),
            config=GridConfig(n_max=10, min_steps=8),
        )


def test_grid_rejects_indistinguishable_maturities():
    with pytest.raises(GridConfigurationError):
        build_simulation_grid(maturities=[1.0, 1.0 + 1e-13])


def test_grid_collapses_exact_duplicate_maturities():
    grid = build_simulation_grid(
        maturities=[0.25, 1.0, 0.25], config=GridConfig(n_max=32, min_steps=4)
    )
    assert grid.quoted_maturities == (0.25, 1.0)


def test_grid_rejects_degenerate_maturity_inputs():
    with pytest.raises(ValueError):
        build_simulation_grid(maturities=[])
    with pytest.raises(ValueError):
        build_simulation_grid(maturities=[0.5, 0.0])
    with pytest.raises(ValueError):
        build_simulation_grid(maturities=[0.5, -1.0])
    with pytest.raises(ValueError):
        build_simulation_grid(maturities=[0.5, np.nan])


def test_grid_index_of_refuses_to_snap_an_unquoted_maturity():
    grid = _small_grid()
    with pytest.raises(KeyError):
        grid.index_of(0.05 + 1e-12)


def test_grid_arrays_are_read_only_and_hash_is_content_addressed():
    grid_a = _small_grid()
    grid_b = _small_grid()
    assert grid_a.grid_hash == grid_b.grid_hash
    assert grid_a is not grid_b
    other = build_simulation_grid(
        maturities=[0.05, 0.5, 1.5], config=GridConfig(n_max=25, min_steps=5)
    )
    assert other.grid_hash != grid_a.grid_hash
    with pytest.raises(ValueError):
        grid_a.t[0] = 1.0
    with pytest.raises(ValueError):
        grid_a.dt[0] = 1.0
    diagnostics = grid_a.diagnostics()
    assert diagnostics["n"] == grid_a.n
    assert diagnostics["min_steps_realised"] >= diagnostics["min_steps"]


def test_grid_config_validates_its_knobs():
    for kwargs in (
        {"n_max": 0},
        {"min_steps": 0},
        {"short_end_ratio": 0.0},
        {"short_end_ratio": 1.0},
        {"dedup_rtol": -1e-9},
    ):
        with pytest.raises(ValueError):
            GridConfig(**kwargs)


# ---------------------------------------------------------------------------
# 5. Cholesky factory, cache and jitter
# ---------------------------------------------------------------------------
def test_cholesky_reproduces_sigma():
    H = 0.11
    grid = _small_grid()
    sigma = build_joint_covariance(H=H, grid=grid)
    factor = cholesky_factor(H=H, grid=grid)
    assert factor.jitter_applied == 0.0
    assert factor.jitter_attempts == 0
    np.testing.assert_allclose(factor.L @ factor.L.T, sigma, rtol=0.0, atol=1e-12)
    assert np.array_equal(factor.L, np.tril(factor.L))
    assert factor.L.shape == (2 * grid.n, 2 * grid.n)
    assert factor.pivot_ratio_squared > 1.0
    # Rigorous lower bound on cond_2(Sigma) = cond_2(L)**2, never above it.
    assert factor.pivot_ratio_squared <= np.linalg.cond(sigma) * (1.0 + 1e-9)
    assert factor.diagnostics()["grid_hash"] == grid.grid_hash


def test_cholesky_cache_returns_the_same_object_for_the_same_key():
    grid = _small_grid()
    first = cholesky_factor(H=0.13, grid=grid)
    second = cholesky_factor(H=0.13, grid=grid)
    assert second is first
    info = factor_cache_info()
    assert info == {"hits": 1, "misses": 1, "size": 1, "maxsize": info["maxsize"]}

    # A different H is a different key: recomputed, not served from the cache.
    other_h = cholesky_factor(H=0.29, grid=grid)
    assert other_h is not first
    assert not np.array_equal(other_h.L, first.L)
    assert factor_cache_info()["misses"] == 2

    # A different grid is a different key too.
    other_grid = build_simulation_grid(
        maturities=[0.05, 0.5, 1.5], config=GridConfig(n_max=25, min_steps=5)
    )
    assert cholesky_factor(H=0.13, grid=other_grid) is not first
    assert factor_cache_info()["misses"] == 3

    # A rebuilt but identical grid hits the cache: the key is content-addressed.
    assert cholesky_factor(H=0.13, grid=_small_grid()) is first
    assert factor_cache_info()["hits"] == 2


def test_cholesky_cache_can_be_bypassed_and_evicts_least_recently_used():
    grid = _small_grid()
    uncached_a = cholesky_factor(H=0.13, grid=grid, use_cache=False)
    uncached_b = cholesky_factor(H=0.13, grid=grid, use_cache=False)
    assert uncached_a is not uncached_b
    np.testing.assert_array_equal(uncached_a.L, uncached_b.L)
    assert factor_cache_info() == {
        "hits": 0,
        "misses": 0,
        "size": 0,
        "maxsize": factor_cache_info()["maxsize"],
    }

    set_factor_cache_maxsize(1)
    first = cholesky_factor(H=0.13, grid=grid)
    cholesky_factor(H=0.29, grid=grid)
    assert factor_cache_info()["size"] == 1
    assert cholesky_factor(H=0.13, grid=grid) is not first  # evicted


def test_cholesky_jitter_path_is_exercised_reported_and_logged(caplog):
    """
    H = 1/2 makes Sigma EXACTLY singular: the kernel collapses to 1, so
    W~_t = B_t is a deterministic function of the increments and the joint law
    has rank n, not 2n. numpy rejects it and the documented jitter takes over.
    """
    grid = build_simulation_grid(
        maturities=[0.25, 1.0], config=GridConfig(n_max=32, min_steps=6)
    )
    sigma = build_joint_covariance(H=0.5, grid=grid)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(sigma)  # the injected case really is near-singular

    with caplog.at_level(logging.WARNING, logger=vg.__name__):
        factor = cholesky_factor(H=0.5, grid=grid)

    expected = vg.DEFAULT_JITTER_REL * float(np.max(np.diag(sigma)))
    assert factor.jitter_attempts == 1
    assert factor.jitter_applied == pytest.approx(expected, rel=1e-12)
    assert factor.diagnostics()["jitter_applied"] == factor.jitter_applied
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "jitter" in messages.lower()
    assert f"{factor.jitter_applied:.3e}" in messages
    # The factor reproduces Sigma + jitter*I, and only that.
    np.testing.assert_allclose(
        factor.L @ factor.L.T,
        sigma + factor.jitter_applied * np.eye(sigma.shape[0]),
        rtol=0.0,
        atol=1e-10,
    )


def test_cholesky_raises_instead_of_returning_a_wrong_factor():
    grid = build_simulation_grid(
        maturities=[0.25, 1.0], config=GridConfig(n_max=32, min_steps=6)
    )
    with pytest.raises(CovarianceFactorizationError):
        cholesky_factor(H=0.5, grid=grid, jitter_rel=0.0)
    with pytest.raises(CovarianceFactorizationError):
        cholesky_factor(H=0.5, grid=grid, max_jitter_attempts=0)


def test_cholesky_validates_its_arguments():
    grid = _small_grid()
    with pytest.raises(ValueError):
        cholesky_factor(H=0.0, grid=grid)
    with pytest.raises(ValueError):
        cholesky_factor(H=0.2, grid=grid, jitter_rel=-1.0)
    with pytest.raises(ValueError):
        cholesky_factor(H=0.2, grid=grid, max_jitter_attempts=-1)
    with pytest.raises(TypeError):
        cholesky_factor(H=0.2, grid=object())
    with pytest.raises(ValueError):
        set_factor_cache_maxsize(0)


# ---------------------------------------------------------------------------
# 6. Draws: shapes, antithetic mirroring, reproducibility
# ---------------------------------------------------------------------------
def test_draw_shapes_and_provenance():
    grid = _small_grid()
    factor = cholesky_factor(H=0.14, grid=grid)
    draw = draw_joint_gaussian(factor=factor, n_paths=32, seed=1)
    assert draw.W_tilde.shape == (32, grid.n)
    assert draw.dB.shape == (32, grid.n)
    assert draw.dB_perp.shape == (32, grid.n)
    assert draw.n_paths == 32
    assert draw.n_base_paths == 32
    assert draw.antithetic is False
    assert draw.H == 0.14
    assert draw.grid is grid


def test_draw_is_bit_identical_under_a_fixed_seed():
    grid = _small_grid()
    factor = cholesky_factor(H=0.18, grid=grid)
    first = draw_joint_gaussian(factor=factor, n_paths=64, seed=4242)
    second = draw_joint_gaussian(factor=factor, n_paths=64, seed=4242)
    assert np.array_equal(first.W_tilde, second.W_tilde)
    assert np.array_equal(first.dB, second.dB)
    assert np.array_equal(first.dB_perp, second.dB_perp)

    # A generator seeded the same way gives the same stream.
    third = draw_joint_gaussian(
        factor=factor, n_paths=64, rng=np.random.default_rng(4242)
    )
    assert np.array_equal(first.W_tilde, third.W_tilde)
    assert np.array_equal(first.dB_perp, third.dB_perp)

    other = draw_joint_gaussian(factor=factor, n_paths=64, seed=4243)
    assert not np.array_equal(first.W_tilde, other.W_tilde)


def test_antithetic_second_half_is_the_exact_negation_of_the_first():
    grid = _small_grid()
    factor = cholesky_factor(H=0.16, grid=grid)
    draw = draw_joint_gaussian(factor=factor, n_paths=40, seed=7, antithetic=True)
    assert draw.antithetic is True
    assert draw.n_base_paths == 20
    half = draw.n_paths // 2
    for array in (draw.W_tilde, draw.dB, draw.dB_perp):
        assert np.array_equal(array[half:], -array[:half])  # bit-identical

    # The base half is exactly the plain draw from the same seed.
    plain = draw_joint_gaussian(factor=factor, n_paths=20, seed=7)
    assert np.array_equal(draw.W_tilde[:half], plain.W_tilde)


def test_draw_validates_its_arguments():
    grid = _small_grid()
    factor = cholesky_factor(H=0.16, grid=grid)
    with pytest.raises(ValueError):
        draw_joint_gaussian(factor=factor, n_paths=0)
    with pytest.raises(ValueError):
        draw_joint_gaussian(factor=factor, n_paths=7, antithetic=True)
    with pytest.raises(ValueError):
        draw_joint_gaussian(factor=factor, n_paths=4, seed=1, rng=np.random.default_rng(1))
    with pytest.raises(TypeError):
        draw_joint_gaussian(factor=object(), n_paths=4)


def test_half_hurst_draw_recovers_the_driving_brownian_motion():
    """
    At H = 1/2 the construction must return W~_t = B_t path by path. The only
    deviation allowed is the documented jitter, whose scale is sqrt(jitter).
    """
    grid = build_simulation_grid(
        maturities=[0.5, 1.0], config=GridConfig(n_max=24, min_steps=6)
    )
    factor = cholesky_factor(H=0.5, grid=grid)
    draw = draw_joint_gaussian(factor=factor, n_paths=256, seed=99)
    deviation = np.max(np.abs(draw.W_tilde - np.cumsum(draw.dB, axis=1)))
    assert factor.jitter_applied > 0.0
    assert deviation < 100.0 * math.sqrt(factor.jitter_applied)


# ---------------------------------------------------------------------------
# 7. Statistical validation on drawn paths
# ---------------------------------------------------------------------------
def test_drawn_variance_matches_t_to_the_2h(stat_sample):
    grid, H, draw = stat_sample
    squares = draw.W_tilde**2
    empirical = squares.mean(axis=0)
    band = 4.0 * _stderr(squares) + 1e-12
    theoretical = np.asarray(grid.times) ** (2.0 * H)
    assert np.all(np.abs(empirical - theoretical) <= band)


def test_drawn_autocovariance_matches_the_2f1_form(stat_sample):
    grid, H, draw = stat_sample
    times = np.asarray(grid.times)
    for i, j in ((0, 1), (0, grid.n - 1), (2, 5), (grid.n - 2, grid.n - 1)):
        products = draw.W_tilde[:, i] * draw.W_tilde[:, j]
        empirical = float(products.mean())
        band = 4.0 * float(_stderr(products[:, None])[0]) + 1e-12
        theoretical = float(volterra_autocovariance(times[i], times[j], H=H))
        assert abs(empirical - theoretical) <= band


def test_drawn_cross_covariance_matches_the_closed_form(stat_sample):
    grid, H, draw = stat_sample
    for i in (1, grid.n // 2, grid.n - 1):
        products = draw.W_tilde[:, i][:, None] * draw.dB
        empirical = products.mean(axis=0)
        band = 4.0 * _stderr(products) + 1e-12
        theoretical = np.array(
            [
                _cross_covariance_direct(
                    float(grid.times[i]), float(grid.t[j]), float(grid.t[j + 1]), H
                )
                for j in range(grid.n)
            ]
        )
        assert np.all(np.abs(empirical - theoretical) <= band)


def test_drawn_increments_are_independent_with_variance_dt(stat_sample):
    grid, _H, draw = stat_sample
    squares = draw.dB**2
    band = 4.0 * _stderr(squares) + 1e-14
    assert np.all(np.abs(squares.mean(axis=0) - np.asarray(grid.dt)) <= band)

    # Off-diagonal increments and the perpendicular block are uncorrelated.
    for i, j in ((0, 1), (0, grid.n - 1), (2, 4)):
        cross = draw.dB[:, i] * draw.dB[:, j]
        assert abs(float(cross.mean())) <= 4.0 * float(_stderr(cross[:, None])[0])
    perp_squares = draw.dB_perp**2
    perp_band = 4.0 * _stderr(perp_squares) + 1e-14
    assert np.all(np.abs(perp_squares.mean(axis=0) - np.asarray(grid.dt)) <= perp_band)
    mixed = draw.dB * draw.dB_perp
    assert np.all(np.abs(mixed.mean(axis=0)) <= 4.0 * _stderr(mixed) + 1e-14)
    mixed_driver = draw.W_tilde * draw.dB_perp
    assert np.all(
        np.abs(mixed_driver.mean(axis=0)) <= 4.0 * _stderr(mixed_driver) + 1e-14
    )


def test_spot_vol_cross_moment_is_realised_through_the_shared_driver(stat_sample):
    """
    MATH_ORACLE sec. 5: with W^S = rho*B + sqrt(1-rho^2)*B^perp,
    E[W^S_t * W~_t] = rho * sqrt(2H)/(H+1/2) * t^{H+1/2}.
    This is the whole point of the joint construction - the correlation lives in
    the driving noise, never in a correlation imposed on the marginals.
    """
    grid, H, draw = stat_sample
    rho = -0.7
    w_spot = rho * np.cumsum(draw.dB, axis=1) + math.sqrt(
        1.0 - rho * rho
    ) * np.cumsum(draw.dB_perp, axis=1)
    products = w_spot * draw.W_tilde
    empirical = products.mean(axis=0)
    band = 4.0 * _stderr(products) + 1e-12
    theoretical = (
        rho
        * math.sqrt(2.0 * H)
        / (H + 0.5)
        * np.asarray(grid.times) ** (H + 0.5)
    )
    assert np.all(np.abs(empirical - theoretical) <= band)
