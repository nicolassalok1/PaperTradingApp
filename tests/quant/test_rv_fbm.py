"""
Tests for the Davies-Harte fractional Gaussian noise / fractional Brownian
motion utility.

Target: app/model/volatility_models/rbergomi/fbm.py
  - fgn_autocovariance     -> gamma(j) of standard fGn
  - fgn_davies_harte       -> (n_paths, n) standard fGn
  - fgn_with_diagnostics   -> (sample, FgnDiagnostics)
  - fbm_davies_harte       -> (n_paths, n+1) standard fBm on [0, T], starting at 0
  - CirculantEmbeddingError + the exact Cholesky fallback

ORACLES (all independent of the code under test):
  - The textbook fGn autocovariance, written inline with the DIRECT formula
    gamma(j) = 0.5*(|j+1|^{2H} + |j-1|^{2H} - 2|j|^{2H}).  The module evaluates
    the same quantity through a stabilised expm1/log1p factorisation, so the
    inline version is a genuine second implementation, not a mirror.
  - The self-similarity law Var(B^H_t) = t^{2H}, which is a property of the
    target process, not of the algorithm.
  - Structural invariants that hold sample-wise and are therefore exact under a
    fixed seed: fbm[:, 0] == 0, the dt^H scaling between two horizons T1 != T2,
    bit-identical reproducibility, shape/dtype.

STATISTICAL TOLERANCES: every statistical assertion averages an unbiased
per-path estimator over independent paths, so the standard error is estimated
from the SAME sample (std over paths / sqrt(n_paths)) and the band is
`4 * stderr + <small absolute floor>` - the same discipline as
tests/quant/test_mc_pricing.py.  Seeds are fixed, so the outcome is
deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.model.volatility_models.rbergomi import fbm as fbm_module
from app.model.volatility_models.rbergomi.fbm import (
    CirculantEmbeddingError,
    FractionalNoiseError,
    fbm_davies_harte,
    fbm_with_diagnostics,
    fgn_autocovariance,
    fgn_davies_harte,
    fgn_with_diagnostics,
)

pytestmark = pytest.mark.unit


# Modest but stable: n=32 lags with 8000 independent paths keeps every test
# below ~0.1 s while giving a ~0.3-0.5% standard error on gamma(j).
N_STEPS = 32
N_PATHS = 8000
HURSTS = (0.2, 0.5, 0.8)


# ---------------------------------------------------------------------------
# Independent oracle + estimators
# ---------------------------------------------------------------------------
def gamma_oracle(j: int, H: float) -> float:
    """Textbook fGn autocovariance, direct (unstabilised) evaluation."""
    two_h = 2.0 * H
    return 0.5 * (
        abs(j + 1) ** two_h + abs(j - 1) ** two_h - 2.0 * abs(j) ** two_h
    )


def empirical_autocovariance(
    sample: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unbiased empirical autocovariance and its standard error.

    For each path, `mean_i X_i X_{i+j}` is an unbiased estimator of gamma(j)
    (the process is centred and stationary, so no mean subtraction).  Paths are
    independent, hence the cross-path mean has stderr = std / sqrt(n_paths).
    """
    n_paths, n = sample.shape
    means = np.empty(max_lag + 1, dtype=float)
    errs = np.empty(max_lag + 1, dtype=float)
    for j in range(max_lag + 1):
        per_path = np.mean(sample[:, : n - j] * sample[:, j:], axis=1)
        means[j] = float(per_path.mean())
        errs[j] = float(per_path.std(ddof=1)) / math.sqrt(n_paths)
    return means, errs


def mean_and_stderr(values: np.ndarray) -> tuple[float, float]:
    """Cross-path mean and its standard error."""
    return (
        float(values.mean()),
        float(values.std(ddof=1)) / math.sqrt(values.size),
    )


# ---------------------------------------------------------------------------
# Autocovariance (deterministic, no sampling involved)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_autocovariance_matches_textbook_formula(H: float) -> None:
    gamma = fgn_autocovariance(n_lags=40, H=H)
    assert gamma.shape == (40,)
    assert gamma[0] == 1.0
    for j in range(40):
        assert gamma[j] == pytest.approx(gamma_oracle(j, H), rel=1e-9, abs=1e-14)


def test_autocovariance_at_half_is_exactly_white() -> None:
    """H = 1/2 IS white noise: the module must return exact zeros, not 1e-17."""
    gamma = fgn_autocovariance(n_lags=16, H=0.5)
    assert gamma[0] == 1.0
    assert np.array_equal(gamma[1:], np.zeros(15))


def test_autocovariance_sign_structure() -> None:
    """H < 1/2 anti-persistent (gamma(1) < 0), H > 1/2 persistent (gamma(1) > 0)."""
    assert fgn_autocovariance(n_lags=2, H=0.25)[1] < 0.0
    assert fgn_autocovariance(n_lags=2, H=0.75)[1] > 0.0


# ---------------------------------------------------------------------------
# Empirical law of the generated fGn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", HURSTS)
def test_empirical_fgn_autocovariance_matches_gamma(H: float) -> None:
    sample, diagnostics = fgn_with_diagnostics(
        n=N_STEPS, H=H, n_paths=N_PATHS, seed=20260820
    )
    assert sample.shape == (N_PATHS, N_STEPS)
    assert diagnostics.method == "davies_harte"

    max_lag = 6
    means, errs = empirical_autocovariance(sample, max_lag)
    for j in range(max_lag + 1):
        target = gamma_oracle(j, H)
        tol = 4.0 * errs[j] + 5e-3
        assert abs(means[j] - target) <= tol, (H, j, means[j], target, errs[j])

    # Centred: the sample mean of the whole (n_paths, n) block is ~0.
    flat = sample.mean(axis=1)
    mean, stderr = mean_and_stderr(flat)
    assert abs(mean) <= 4.0 * stderr + 5e-3


def test_h_one_half_is_iid_white_noise() -> None:
    """
    H = 1/2 -> increments are i.i.d. N(0, 1): gamma(0) = 1, gamma(j) ~ 0, j >= 1.
    """
    sample = fgn_davies_harte(n=N_STEPS, H=0.5, n_paths=N_PATHS, seed=11)
    means, errs = empirical_autocovariance(sample, 8)

    assert abs(means[0] - 1.0) <= 4.0 * errs[0] + 5e-3
    for j in range(1, 9):
        assert abs(means[j]) <= 4.0 * errs[j] + 5e-3, (j, means[j], errs[j])

    # Marginal law: unit variance and near-zero excess kurtosis (Gaussian).
    flat = sample.reshape(-1)
    assert float(flat.var()) == pytest.approx(1.0, abs=0.05)
    kurtosis = float(np.mean(flat**4)) / float(np.mean(flat**2)) ** 2
    assert kurtosis == pytest.approx(3.0, abs=0.15)


# ---------------------------------------------------------------------------
# fBm: variance law, self-similarity, structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", HURSTS)
def test_fbm_variance_follows_t_power_two_h(H: float) -> None:
    """Var(B^H_t) = t^{2H} along the grid, for several t."""
    path = fbm_davies_harte(n=N_STEPS, H=H, T=1.0, n_paths=N_PATHS, seed=4242)
    assert path.shape == (N_PATHS, N_STEPS + 1)
    dt = 1.0 / N_STEPS

    for k in (4, 8, 16, N_STEPS):
        est, stderr = mean_and_stderr(path[:, k] ** 2)
        target = (k * dt) ** (2.0 * H)
        assert abs(est - target) <= 4.0 * stderr + 1e-3, (H, k, est, target)


@pytest.mark.parametrize("H", HURSTS)
def test_fbm_variance_at_horizon_for_non_unit_T(H: float) -> None:
    """Exact self-similarity across horizons: Var(B^H_T) = T^{2H} for T != 1."""
    for T in (0.25, 2.5):
        path = fbm_davies_harte(n=N_STEPS, H=H, T=T, n_paths=N_PATHS, seed=909)
        est, stderr = mean_and_stderr(path[:, -1] ** 2)
        target = T ** (2.0 * H)
        assert abs(est - target) <= 4.0 * stderr + 1e-3, (H, T, est, target)


@pytest.mark.parametrize("H", HURSTS)
def test_fbm_scaling_in_T_is_exact_under_a_fixed_seed(H: float) -> None:
    """
    Structural (non-statistical) invariant: on a fixed seed the ONLY difference
    between two horizons is the dt^H factor, so B(T2) == (T2/T1)^H * B(T1).
    """
    base = fbm_davies_harte(n=N_STEPS, H=H, T=1.0, n_paths=64, seed=7)
    scaled = fbm_davies_harte(n=N_STEPS, H=H, T=4.0, n_paths=64, seed=7)
    np.testing.assert_allclose(scaled, (4.0**H) * base, rtol=1e-12, atol=1e-12)


def test_fbm_starts_at_zero_and_increments_are_the_fgn() -> None:
    """fbm[:, 0] is exactly 0 and diff(fbm) == dt^H * fgn on the same seed."""
    H, T = 0.3, 1.7
    path, diagnostics = fbm_with_diagnostics(
        n=N_STEPS, H=H, T=T, n_paths=32, seed=5150
    )
    noise = fgn_davies_harte(n=N_STEPS, H=H, n_paths=32, seed=5150)

    assert np.array_equal(path[:, 0], np.zeros(32))
    assert diagnostics.method == "davies_harte"
    dt = T / N_STEPS
    np.testing.assert_allclose(
        np.diff(path, axis=1), (dt**H) * noise, rtol=1e-13, atol=1e-15
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_seed_reproducibility_is_bit_identical() -> None:
    first = fgn_davies_harte(n=64, H=0.35, n_paths=16, seed=123456)
    second = fgn_davies_harte(n=64, H=0.35, n_paths=16, seed=123456)
    other = fgn_davies_harte(n=64, H=0.35, n_paths=16, seed=123457)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)

    # An explicit Generator seeded identically must give the same stream.
    via_rng = fgn_davies_harte(
        n=64, H=0.35, n_paths=16, rng=np.random.default_rng(123456)
    )
    assert np.array_equal(first, via_rng)

    # fBm inherits reproducibility from its driving fGn.
    assert np.array_equal(
        fbm_davies_harte(n=64, H=0.35, T=2.0, n_paths=16, seed=999),
        fbm_davies_harte(n=64, H=0.35, T=2.0, n_paths=16, seed=999),
    )


def test_cholesky_path_is_reproducible_and_distinct_from_fft_path() -> None:
    kwargs = {"n": 24, "H": 0.4, "n_paths": 8, "seed": 2024}
    chol_a = fgn_davies_harte(method="cholesky", **kwargs)
    chol_b = fgn_davies_harte(method="cholesky", **kwargs)
    dh = fgn_davies_harte(method="davies_harte", **kwargs)

    assert np.array_equal(chol_a, chol_b)
    # Same law, different construction -> different numbers on the same seed.
    assert not np.array_equal(chol_a, dh)


# ---------------------------------------------------------------------------
# Cholesky fallback: forced, and statistically equivalent to the FFT path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", HURSTS)
def test_forced_cholesky_matches_the_fft_path_in_law(H: float) -> None:
    n = 24
    chol, chol_diag = fgn_with_diagnostics(
        n=n, H=H, n_paths=N_PATHS, seed=31337, method="cholesky"
    )
    dh, dh_diag = fgn_with_diagnostics(
        n=n, H=H, n_paths=N_PATHS, seed=31337, method="davies_harte"
    )
    assert chol_diag.method == "cholesky"
    assert chol_diag.circulant_size == 0
    assert chol_diag.fallback_reason is None
    assert dh_diag.method == "davies_harte"

    max_lag = 5
    chol_means, chol_errs = empirical_autocovariance(chol, max_lag)
    dh_means, dh_errs = empirical_autocovariance(dh, max_lag)
    for j in range(max_lag + 1):
        target = gamma_oracle(j, H)
        assert abs(chol_means[j] - target) <= 4.0 * chol_errs[j] + 5e-3, (H, j)
        # ... and the two samplers agree with each other within their joint band.
        joint = 4.0 * math.hypot(chol_errs[j], dh_errs[j]) + 5e-3
        assert abs(chol_means[j] - dh_means[j]) <= joint, (H, j)


def test_forced_negative_eigenvalue_raises_then_falls_back(monkeypatch) -> None:
    """
    Davies-Harte embedding is provably valid for fGn with H in (0, 1), so no
    admissible H can produce a genuinely negative eigenvalue.  We therefore take
    the second option allowed by the spec and INJECT a covariance sequence via
    the module's single autocovariance entry point.

    gamma = (1, 0.9, 0.7) is a legitimate positive-definite Toeplitz covariance
    for n = 3 (leading minors 1, 0.19, 0.024 > 0) whose circulant embedding of
    size m = 4 has eigenvalues (1 + 2a + b, 1 - b, 1 - 2a + b, 1 - b) =
    (3.5, 0.3, -0.1, 0.3): non-embeddable, yet exactly representable by the
    Cholesky fallback.  That is precisely the situation the fallback exists for.
    """
    gamma = np.array([1.0, 0.9, 0.7])
    monkeypatch.setattr(
        fbm_module, "_autocovariance", lambda n_lags, H: gamma[:n_lags].copy()
    )

    # Strict Davies-Harte: typed error, no silent clipping.
    with pytest.raises(CirculantEmbeddingError) as excinfo:
        fgn_davies_harte(n=3, H=0.3, n_paths=4, seed=1, method="davies_harte")
    assert "not non-negative definite" in str(excinfo.value)
    assert isinstance(excinfo.value, FractionalNoiseError)

    # "auto": documented fallback, reason recorded, law still exact.
    sample, diagnostics = fgn_with_diagnostics(
        n=3, H=0.3, n_paths=N_PATHS, seed=1, method="auto"
    )
    assert diagnostics.method == "cholesky"
    assert diagnostics.fallback_reason is not None
    assert "min eigenvalue" in diagnostics.fallback_reason
    assert diagnostics.circulant_size == 4

    means, errs = empirical_autocovariance(sample, 2)
    for j in range(3):
        assert abs(means[j] - gamma[j]) <= 4.0 * errs[j] + 5e-3, (j, means[j])


def test_tiny_negative_eigenvalues_are_tolerated_and_reported(monkeypatch) -> None:
    """
    Round-off-sized negatives must be clamped AND counted, never hidden.
    For n = 3 the m = 4 spectrum is (1 + 2a + b, 1 - b, 1 - 2a + b, 1 - b) with
    gamma = (1, a, b); injecting a = 0.75 + delta, b = 0.5 puts exactly one
    eigenvalue at -2*delta, far inside the tolerance 1e-10 * max|lambda| = 3e-10.
    """
    delta = 1e-13
    gamma = np.array([1.0, 0.75 + delta, 0.5])
    monkeypatch.setattr(
        fbm_module, "_autocovariance", lambda n_lags, H: gamma[:n_lags].copy()
    )
    sample, diagnostics = fgn_with_diagnostics(
        n=3, H=0.3, n_paths=4, seed=2, method="davies_harte"
    )
    assert diagnostics.method == "davies_harte"
    assert diagnostics.n_clamped_eigenvalues == 1
    assert -1e-12 < diagnostics.min_eigenvalue < 0.0
    assert diagnostics.eigenvalue_tolerance > 0.0
    assert np.all(np.isfinite(sample))


# ---------------------------------------------------------------------------
# Embedding size / padding
# ---------------------------------------------------------------------------
def test_padding_choice_reports_the_right_circulant_size() -> None:
    n = 25  # minimal 2*(n-1) = 48 is NOT a power of two -> 64
    _, minimal = fgn_with_diagnostics(n=n, H=0.4, seed=1, padding="minimal")
    _, padded = fgn_with_diagnostics(n=n, H=0.4, seed=1, padding="power_of_two")
    assert minimal.circulant_size == 48
    assert padded.circulant_size == 64
    for diagnostics in (minimal, padded):
        assert diagnostics.min_eigenvalue > 0.0
        assert diagnostics.n_clamped_eigenvalues == 0
        assert diagnostics.max_eigenvalue >= diagnostics.min_eigenvalue


@pytest.mark.parametrize("padding", ["minimal", "power_of_two"])
def test_padding_keeps_the_first_n_lags_exact(padding: str) -> None:
    """Power-of-two padding must not perturb the covariance we asked for."""
    H, n = 0.7, 25
    sample = fgn_davies_harte(
        n=n, H=H, n_paths=N_PATHS, seed=606, padding=padding  # type: ignore[arg-type]
    )
    means, errs = empirical_autocovariance(sample, 5)
    for j in range(6):
        target = gamma_oracle(j, H)
        assert abs(means[j] - target) <= 4.0 * errs[j] + 5e-3, (padding, j)


def test_degenerate_lengths_are_supported() -> None:
    """n = 1 and n = 2 exercise the minimal embedding m = 2."""
    one, diag_one = fgn_with_diagnostics(n=1, H=0.6, n_paths=N_PATHS, seed=17)
    assert one.shape == (N_PATHS, 1)
    assert diag_one.circulant_size == 2
    est, stderr = mean_and_stderr(one[:, 0] ** 2)
    assert abs(est - 1.0) <= 4.0 * stderr + 1e-3

    two = fgn_davies_harte(n=2, H=0.6, n_paths=N_PATHS, seed=18)
    assert two.shape == (N_PATHS, 2)
    means, errs = empirical_autocovariance(two, 1)
    assert abs(means[1] - gamma_oracle(1, 0.6)) <= 4.0 * errs[1] + 5e-3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("H", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_hurst_outside_the_open_unit_interval_is_rejected(H: float) -> None:
    with pytest.raises(ValueError, match="H must lie strictly inside"):
        fgn_davies_harte(n=8, H=H)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        fgn_davies_harte(n=0, H=0.3)
    with pytest.raises(ValueError, match="n_paths must be >= 1"):
        fgn_davies_harte(n=8, H=0.3, n_paths=0)
    with pytest.raises(ValueError, match="eig_rtol"):
        fgn_davies_harte(n=8, H=0.3, eig_rtol=-1.0)
    with pytest.raises(ValueError, match="method must be"):
        fgn_davies_harte(n=8, H=0.3, method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="padding must be"):
        fgn_davies_harte(n=8, H=0.3, padding="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not both"):
        fgn_davies_harte(n=8, H=0.3, seed=1, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="T must be"):
        fbm_davies_harte(n=8, H=0.3, T=0.0)
    with pytest.raises(ValueError, match="n_lags must be >= 1"):
        fgn_autocovariance(n_lags=0, H=0.3)


def test_outputs_are_finite_float_arrays() -> None:
    sample = fgn_davies_harte(n=16, H=0.15, n_paths=8, seed=3)
    path = fbm_davies_harte(n=16, H=0.15, n_paths=8, seed=3)
    assert sample.dtype == np.float64
    assert path.dtype == np.float64
    assert np.all(np.isfinite(sample))
    assert np.all(np.isfinite(path))
