"""
Tests for the short-maturity ATM skew and the initial Hurst estimate.

Target: app/model/calibration/rough_vol/hurst_estimator.py (spec 4.5)

ORACLES (all independent of the code under test):
  - The synthetic surfaces are built **forward** from a closed form. A chosen
    ``H`` and amplitude ``A`` define ``psi(T) = -A * T^{H - 1/2}`` exactly, and
    the implied vol of every strike is written as the exact local quadratic
    ``sigma(k) = sigma_atm + psi(T) * k + c * k^2``. The slope, the curvature
    and the exponent are therefore known **by construction** and are never read
    back from the module.
  - The weighted least squares is re-derived inline from the textbook normal
    equations (``beta = (X'WX)^{-1} X'W y``, ``cov = s^2 (X'WX)^{-1}``) with no
    rescaling and no weight normalisation, so the module's conditioning tricks
    are checked to be exactly neutral.
  - Theil-Sen is cross-checked against ``scipy.stats.theilslopes`` (with
    ``method="joint"``, the ``median(y - slope x)`` intercept the module
    documents; scipy's default is the ``"separate"`` intercept).
  - Black-76 vega is cross-checked against a central finite difference of
    ``forward_curve.black76_call_price``.
  - The end-to-end test runs the real Phase-1 chain: cleaned quotes ->
    put-call-parity forward -> OTM surface with **re-inverted** implied vols,
    so the estimator is exercised on the exact objects the pipeline produces.

Determinism: no RNG, no Monte-Carlo, no network. Every "corruption" is a fixed
multiplicative factor applied to one expiry, not a random draw.

Numerical note: on exactly-quadratic input the local regression has a zero
residual, so ``SE(psi)`` collapses to rounding noise; the module floors it for
weighting (``HurstConfig.se_floor``) and the tests assert the flag rather than
a fake precision.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from scipy.stats import theilslopes

from app.model.calibration.rough_vol.chain_cleaning import CleaningConfig, clean_expiry_chain
from app.model.calibration.rough_vol.forward_curve import (
    SurfacePoint,
    black76_call_price,
    build_forward_point,
    build_otm_surface,
)
from app.model.calibration.rough_vol.hurst_estimator import (
    FALLBACK_H0,
    FLAG_COARSE_LADDER_IN_WINDOW,
    FLAG_FORWARD_MISMATCH,
    FLAG_NO_FORWARD_POINT,
    FLAG_ONE_SIDED_IN_WINDOW,
    FLAG_PSI_NEAR_ZERO_DROPPED,
    FLAG_ROBUST_DISAGREEMENT,
    FLAG_SE_FLOORED,
    FLAG_SIGN_FLIP,
    REASON_H0_OUT_OF_RANGE,
    REASON_INVALID_MATURITY,
    REASON_LOW_R2,
    REASON_NO_USABLE_QUOTE,
    REASON_SE_TOO_LARGE,
    REASON_TOO_FEW_EXPIRIES,
    REASON_TOO_FEW_STRIKES,
    REASON_UNBALANCED_WINGS,
    Z95,
    HurstConfig,
    SkewConfig,
    SkewFailure,
    SkewPoint,
    black76_vega,
    build_skew_curve,
    build_spread_lookup,
    estimate_atm_skew,
    estimate_hurst_from_skew,
    hurst_report,
    psi_from_strike_slope,
    strike_slope_from_psi,
    theil_sen_slope,
    weighted_polynomial_fit,
)
from app.model.calibration.rough_vol.variance_swap import (
    FLAG_COARSE_STRIKE_LADDER,
    VarianceSwapDiagnostics,
    VarianceSwapPoint,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic-surface builders (the oracle side)
# ---------------------------------------------------------------------------

F_REF = 100.0
D_REF = 0.995
SIGMA_ATM = 0.30
CURV = 0.8  # d2 sigma / dk2, so the quadratic coefficient is CURV / 2
H_TRUE = 0.12
A_TRUE = 0.35

#: log-moneyness ladder: 1 % steps out to +/- 8 %.
K_LADDER = tuple(round(0.01 * i, 10) for i in range(-8, 9))


def psi_true(T: float, *, H: float = H_TRUE, A: float = A_TRUE) -> float:
    """The oracle skew: ``psi(T) = -A * T^{H - 1/2}``."""
    return -A * T ** (H - 0.5)


def iv_exact(k: float, psi: float, *, sigma_atm: float = SIGMA_ATM, curv: float = CURV) -> float:
    """``sigma(k) = sigma_atm + psi k + (curv/2) k^2`` — exact, by construction."""
    return sigma_atm + psi * k + 0.5 * curv * k * k


def make_expiry(
    T: float,
    psi: float,
    *,
    ks=K_LADDER,
    sigma_atm: float = SIGMA_ATM,
    curv: float = CURV,
    one_sided: bool = False,
) -> list[SurfacePoint]:
    """One expiry of :class:`SurfacePoint`, written straight from the oracle."""
    points: list[SurfacePoint] = []
    for k in ks:
        points.append(
            SurfacePoint(
                T=float(T),
                K=F_REF * math.exp(k),
                k=float(k),
                F=F_REF,
                D=D_REF,
                iv=iv_exact(k, psi, sigma_atm=sigma_atm, curv=curv),
                option_type="put" if k < 0.0 else "call",
                mid=1.0,
                call_equivalent_price=1.0,
                vendor_iv=float("nan"),
                one_sided=bool(one_sided),
                contract_symbol=None,
                flags=(),
            )
        )
    return points


def make_power_law_surface(maturities, *, multipliers=None, H: float = H_TRUE) -> list[SurfacePoint]:
    """Flat surface whose every expiry follows the oracle power law exactly."""
    surface: list[SurfacePoint] = []
    for i, T in enumerate(maturities):
        psi = psi_true(T, H=H)
        if multipliers is not None:
            psi *= float(multipliers[i])
        surface.extend(make_expiry(T, psi))
    return surface


MATURITIES_7 = (
    7.0 / 365.0,
    14.0 / 365.0,
    21.0 / 365.0,
    30.0 / 365.0,
    45.0 / 365.0,
    60.0 / 365.0,
    90.0 / 365.0,
)
MATURITIES_9 = (
    7.0 / 365.0,
    10.0 / 365.0,
    14.0 / 365.0,
    21.0 / 365.0,
    30.0 / 365.0,
    45.0 / 365.0,
    60.0 / 365.0,
    75.0 / 365.0,
    90.0 / 365.0,
)


def textbook_wls(x, y, w, degree):
    """
    Independent weighted least squares: raw normal equations, no rescaling.

    ``beta = (X'WX)^{-1} X'W y`` and ``cov = (r'Wr / (n - p)) (X'WX)^{-1}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = x.size
    p = degree + 1
    design = np.column_stack([x**j for j in range(p)])
    wmat = np.diag(w)
    normal = design.T @ wmat @ design
    beta = np.linalg.solve(normal, design.T @ wmat @ y)
    resid = y - design @ beta
    s2 = float(resid @ wmat @ resid) / (n - p)
    cov = s2 * np.linalg.inv(normal)
    return beta, np.sqrt(np.diag(cov)), math.sqrt(s2)


def assert_plain_python(obj, path="root"):
    """Everything must already be a JSON-native type before ``_json_safe``."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        assert not isinstance(obj, np.generic), f"{path} is a numpy scalar"
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str), f"{path} has a non-str key {key!r}"
            assert_plain_python(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            assert_plain_python(value, f"{path}[{i}]")
        return
    raise AssertionError(f"{path} is a {type(obj).__name__}, not a JSON-native type")


# ---------------------------------------------------------------------------
# 1. The weighted regression machinery
# ---------------------------------------------------------------------------


def test_weighted_polynomial_fit_matches_the_textbook_normal_equations():
    """Rescaling the abscissa and normalising the weights must be exactly neutral."""
    x = np.array([-0.08, -0.05, -0.02, 0.0, 0.01, 0.04, 0.07], dtype=float)
    y = np.array([0.372, 0.351, 0.331, 0.318, 0.312, 0.297, 0.281], dtype=float)
    w = np.array([1.0, 4.0, 9.0, 16.0, 9.0, 4.0, 1.0], dtype=float)

    fit = weighted_polynomial_fit(x, y, w, degree=2)
    beta, se, sigma_resid = textbook_wls(x, y, w, 2)

    assert fit.n == 7
    assert fit.dof == 4
    np.testing.assert_allclose(np.asarray(fit.coeffs), beta, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.asarray(fit.se), se, rtol=1e-9, atol=1e-16)
    assert fit.sigma_resid == pytest.approx(sigma_resid, rel=1e-10)
    assert 0.0 <= fit.r_squared <= 1.0

    # the unweighted residual RMSE is the interpretable, weight-scale-free one
    design = np.column_stack([x**j for j in range(3)])
    resid = y - design @ beta
    assert fit.resid_rmse == pytest.approx(math.sqrt(float(resid @ resid) / x.size), rel=1e-10)


def test_weighted_polynomial_fit_covariance_is_invariant_to_weight_scaling():
    x = np.linspace(-0.06, 0.06, 9)
    y = 0.3 - 0.9 * x + 0.4 * x**2 + 1e-4 * np.array([1, -1, 1, -1, 0, 1, -1, 1, -1], dtype=float)
    w = np.linspace(1.0, 5.0, 9)

    base = weighted_polynomial_fit(x, y, w, degree=2)
    scaled = weighted_polynomial_fit(x, y, 1234.5 * w, degree=2)

    np.testing.assert_allclose(np.asarray(scaled.coeffs), np.asarray(base.coeffs), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(scaled.se), np.asarray(base.se), rtol=1e-10)
    assert scaled.resid_rmse == pytest.approx(base.resid_rmse, rel=1e-12)
    # ...whereas the *weighted* residual scale is by definition weight-scaled
    assert scaled.sigma_resid == pytest.approx(math.sqrt(1234.5) * base.sigma_resid, rel=1e-10)


def test_weighted_polynomial_fit_refuses_a_fit_without_residual_degrees_of_freedom():
    with pytest.raises(ValueError):
        weighted_polynomial_fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], degree=2)


def test_theil_sen_matches_scipy_theilslopes():
    x = np.array([-4.0, -3.2, -2.5, -1.9, -1.4], dtype=float)
    y = np.array([1.52, 1.20, 0.99, 3.10, 0.61], dtype=float)  # one gross outlier

    slope, intercept = theil_sen_slope(x, y)
    reference = theilslopes(y, x, method="joint")

    assert slope == pytest.approx(float(reference[0]), rel=1e-12, abs=1e-12)
    assert intercept == pytest.approx(float(reference[1]), rel=1e-12, abs=1e-12)
    # the slope is the median of the pairwise slopes, unmoved by one gross outlier
    assert slope == pytest.approx(float(theilslopes(y, x)[0]), rel=1e-12)


def test_black76_vega_matches_a_central_finite_difference():
    F, K, T, D, vol = 100.0, 105.0, 0.25, 0.995, 0.31
    h = 1e-6
    up = black76_call_price(F=F, K=K, T=T, D=D, vol=vol + h)
    down = black76_call_price(F=F, K=K, T=T, D=D, vol=vol - h)
    numerical = (up - down) / (2.0 * h)

    assert black76_vega(F=F, K=K, T=T, D=D, vol=vol) == pytest.approx(numerical, rel=1e-6)
    assert black76_vega(F=F, K=K, T=0.0, D=D, vol=vol) == 0.0
    assert black76_vega(F=F, K=K, T=T, D=D, vol=0.0) == 0.0


# ---------------------------------------------------------------------------
# 2. The per-expiry local quadratic fit
# ---------------------------------------------------------------------------


def test_local_quadratic_fit_recovers_a_known_b_and_c():
    """Exact parabola in k -> psi = b and curvature = 2c to machine precision."""
    T = 0.05
    psi = -0.77
    curvature = 1.4  # d2 sigma / dk2
    points = make_expiry(T, psi, ks=[0.015 * i for i in range(-4, 5)], curv=curvature)

    skew = estimate_atm_skew(points)

    assert isinstance(skew, SkewPoint)
    assert skew.psi == pytest.approx(psi, rel=1e-11, abs=1e-13)
    assert skew.curvature == pytest.approx(curvature, rel=1e-9, abs=1e-11)
    assert skew.quad_coeff == pytest.approx(curvature / 2.0, rel=1e-9, abs=1e-11)
    assert skew.sigma_atm == pytest.approx(SIGMA_ATM, rel=1e-12)
    assert skew.r_squared == pytest.approx(1.0, abs=1e-9)
    assert skew.dof == skew.n_strikes - 3


def test_atm_window_is_the_spec_half_width_and_the_fixed_point_converges():
    T = 0.08
    points = make_expiry(T, psi_true(T))
    cfg = SkewConfig()

    skew = estimate_atm_skew(points, config=cfg)

    assert isinstance(skew, SkewPoint)
    assert skew.converged is True
    assert skew.iterations >= 1
    # sigma_ATM is the fitted k = 0 intercept, and the window is rebuilt from it.
    expected_half = cfg.window_c * skew.sigma_atm * math.sqrt(T)
    assert skew.half_width == pytest.approx(expected_half, rel=1e-3)
    assert skew.window == (-skew.half_width, skew.half_width)
    assert abs(skew.k_min) <= skew.half_width * (1.0 + 1e-12)
    assert abs(skew.k_max) <= skew.half_width * (1.0 + 1e-12)
    # every strike outside the window really was excluded
    assert skew.n_strikes == sum(1 for p in points if abs(p.k) <= skew.half_width * (1.0 + 1e-12))
    assert skew.n_left >= 2 and skew.n_right >= 2


def test_expiry_with_fewer_than_five_strikes_is_skipped():
    T = 0.10
    # 4 strikes, 2 per side, all comfortably inside the 1.5-sigma-sqrt(T) window
    points = make_expiry(T, psi_true(T), ks=[-0.10, -0.05, 0.05, 0.10])

    result = estimate_atm_skew(points)

    assert isinstance(result, SkewFailure)
    assert result.reason == REASON_TOO_FEW_STRIKES
    assert "5" in result.message_fr
    assert result.to_dict()["reason_fr"]


def test_expiry_with_fewer_than_two_strikes_on_one_side_is_skipped():
    T = 0.10
    # 6 strikes but only one strictly below the money
    points = make_expiry(T, psi_true(T), ks=[-0.05, 0.02, 0.05, 0.08, 0.11, 0.13])

    result = estimate_atm_skew(points)

    assert isinstance(result, SkewFailure)
    assert result.reason == REASON_UNBALANCED_WINGS
    assert result.detail["n_left"] == 1.0


def test_strike_and_log_moneyness_conventions_are_exact_inverses():
    T = 0.05
    psi = -0.9
    curvature = 1.2
    points = make_expiry(T, psi, ks=[0.015 * i for i in range(-4, 5)], curv=curvature)
    skew = estimate_atm_skew(points)
    assert isinstance(skew, SkewPoint)

    # round trip
    assert psi_from_strike_slope(strike_slope_from_psi(psi, F=F_REF), F=F_REF) == pytest.approx(psi)
    assert skew.dsigma_dK == pytest.approx(psi / F_REF, rel=1e-11)

    # d2 sigma / dK2 against a numerical second derivative of sigma(ln(K/F))
    def sigma_of_K(K: float) -> float:
        return iv_exact(math.log(K / F_REF), psi, curv=curvature)

    h = 1e-3 * F_REF
    numerical = (sigma_of_K(F_REF + h) - 2.0 * sigma_of_K(F_REF) + sigma_of_K(F_REF - h)) / (h * h)
    assert skew.d2sigma_dK2 == pytest.approx(numerical, rel=1e-5)


def test_weight_scheme_falls_back_to_vega_without_spreads_and_uses_spreads_with_them():
    T = 0.08
    points = make_expiry(T, psi_true(T))

    without = estimate_atm_skew(points)
    assert isinstance(without, SkewPoint)
    assert without.weight_scheme == "vega"

    lookup = {
        ("strike", p.option_type, round(p.K, 10), round(p.T, 10)): 0.05 + 0.01 * abs(p.k) * 100.0
        for p in points
    }
    with_spreads = estimate_atm_skew(points, spread_lookup=lookup)
    assert isinstance(with_spreads, SkewPoint)
    assert with_spreads.weight_scheme == "iv_spread"
    # weights change the fit only through the residuals; the input is exact, so
    # psi must be identical either way.
    assert with_spreads.psi == pytest.approx(without.psi, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. The Hurst regression — recovery
# ---------------------------------------------------------------------------


def test_known_power_law_recovers_H_exactly():
    surface = make_power_law_surface(MATURITIES_7)

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is False
    assert estimate.n_expiries == len(MATURITIES_7)
    assert estimate.H0 == pytest.approx(H_TRUE, abs=1e-10)
    assert estimate.diagnostics["H0_estimated"] == pytest.approx(H_TRUE, abs=1e-10)
    assert estimate.diagnostics["slope_wls"] == pytest.approx(H_TRUE - 0.5, abs=1e-10)
    assert estimate.diagnostics["amplitude_A"] == pytest.approx(A_TRUE, rel=1e-9)
    assert estimate.r2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("H", [0.05, 0.10, 0.20, 0.35, 0.45])
def test_recovery_is_exact_across_the_admissible_H_range(H):
    surface = make_power_law_surface(MATURITIES_7, H=H)

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is False
    assert estimate.H0 == pytest.approx(H, abs=1e-10)


def test_per_expiry_skew_matches_the_closed_form_psi():
    surface = make_power_law_surface(MATURITIES_7)
    points, failures = build_skew_curve(surface, None)

    assert failures == []
    assert len(points) == len(MATURITIES_7)
    for sp in points:
        assert sp.psi == pytest.approx(psi_true(sp.T), rel=1e-10, abs=1e-12)
        assert sp.curvature == pytest.approx(CURV, rel=1e-8, abs=1e-10)


def test_se_ci95_and_r2_are_internally_consistent():
    # a mildly scattered surface, so SE is a real number rather than fp noise
    multipliers = [1.0 + 0.05 * (1 if i % 2 == 0 else -1) for i in range(len(MATURITIES_7))]
    surface = make_power_law_surface(MATURITIES_7, multipliers=multipliers)

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is False
    assert estimate.se > 0.0
    assert estimate.se == pytest.approx(estimate.diagnostics["slope_se"], rel=1e-15)
    assert estimate.ci95[0] == pytest.approx(estimate.H0 - Z95 * estimate.se, rel=1e-12)
    assert estimate.ci95[1] == pytest.approx(estimate.H0 + Z95 * estimate.se, rel=1e-12)
    assert estimate.ci95[0] < estimate.H0 < estimate.ci95[1]
    assert 0.0 <= estimate.r2 <= 1.0
    assert estimate.diagnostics["dof"] == estimate.n_expiries - 2
    # the Student-t interval of a 5-dof regression is strictly wider than 1.96 SE
    ci_t = estimate.diagnostics["ci95_student_t"]
    assert ci_t[0] < estimate.ci95[0] and ci_t[1] > estimate.ci95[1]


def test_short_maturity_window_is_configurable():
    surface = make_power_law_surface(MATURITIES_7)

    wide = estimate_hurst_from_skew(surface, None, (5.0 / 365.0, 0.25))
    narrow = estimate_hurst_from_skew(surface, None, (5.0 / 365.0, 0.10))

    assert wide.n_expiries == 7
    assert narrow.n_expiries == 4  # 7, 14, 21 and 30 days
    assert narrow.window == (5.0 / 365.0, 0.10)
    assert wide.window == (5.0 / 365.0, 0.25)
    assert narrow.diagnostics["T_used"] == pytest.approx([T for T in MATURITIES_7 if T <= 0.10])
    # the power law is exact, so trimming the window must not move the estimate
    assert narrow.H0 == pytest.approx(H_TRUE, abs=1e-9)
    assert wide.H0 == pytest.approx(H_TRUE, abs=1e-9)


def test_window_bounds_are_rejected_when_degenerate():
    surface = make_power_law_surface(MATURITIES_7)
    with pytest.raises(ValueError):
        estimate_hurst_from_skew(surface, None, (0.25, 0.05))
    with pytest.raises(ValueError):
        estimate_hurst_from_skew(surface, None, (0.0, 0.25))


# ---------------------------------------------------------------------------
# 4. Guard rails and rejection paths (one assertion block each)
# ---------------------------------------------------------------------------


def test_reject_fewer_than_three_expiries():
    surface = make_power_law_surface(MATURITIES_7[:2])

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is True
    assert estimate.H0 == FALLBACK_H0
    assert estimate.n_expiries == 2
    assert REASON_TOO_FEW_EXPIRIES in estimate.diagnostics["rejection_reasons"]
    assert math.isnan(estimate.se)
    assert math.isnan(estimate.ci95[0]) and math.isnan(estimate.ci95[1])
    assert math.isnan(estimate.diagnostics["H0_estimated"])


def test_reject_r2_below_minimum():
    multipliers = [1.0 + 0.45 * (1 if i % 2 == 0 else -1) for i in range(len(MATURITIES_9))]
    surface = make_power_law_surface(MATURITIES_9, multipliers=multipliers)

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is True
    assert estimate.H0 == FALLBACK_H0
    assert estimate.diagnostics["rejection_reasons"] == [REASON_LOW_R2]
    assert estimate.r2 < HurstConfig().r2_min
    assert estimate.se <= HurstConfig().se_max  # isolated: only R^2 fired


def test_reject_standard_error_above_maximum():
    # three expiries over a very narrow log-T span: any scatter blows the SE up
    maturities = (30.0 / 365.0, 35.0 / 365.0, 40.0 / 365.0)
    multipliers = [1.02, 0.98, 1.02]
    surface = make_power_law_surface(maturities, multipliers=multipliers)

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is True
    assert estimate.H0 == FALLBACK_H0
    assert estimate.diagnostics["rejection_reasons"] == [REASON_SE_TOO_LARGE]
    assert estimate.se > HurstConfig().se_max
    assert estimate.r2 >= HurstConfig().r2_min  # isolated: only the SE fired


def test_reject_H0_outside_the_admissible_range():
    # psi ~ T^{+0.2} => slope = +0.2 => H0 = 0.7, far above 0.49
    surface: list[SurfacePoint] = []
    for T in MATURITIES_7:
        surface.extend(make_expiry(T, -A_TRUE * T**0.2))

    estimate = estimate_hurst_from_skew(surface, None)

    assert estimate.unstable is True
    assert estimate.H0 == FALLBACK_H0
    assert estimate.diagnostics["rejection_reasons"] == [REASON_H0_OUT_OF_RANGE]
    assert estimate.diagnostics["H0_estimated"] == pytest.approx(0.70, abs=1e-9)
    assert estimate.diagnostics["H0_clipped"] == pytest.approx(0.49, abs=1e-12)
    assert estimate.r2 == pytest.approx(1.0, abs=1e-12)  # a perfect fit, wrong exponent


def test_sign_flip_is_flagged_and_the_fit_falls_back_to_abs_psi():
    maturities = MATURITIES_7[:5]
    surface: list[SurfacePoint] = []
    for i, T in enumerate(maturities):
        psi = psi_true(T)
        if i == 2:
            psi = -psi  # same magnitude, opposite sign
        surface.extend(make_expiry(T, psi))

    estimate = estimate_hurst_from_skew(surface, None)

    assert FLAG_SIGN_FLIP in estimate.diagnostics["flags"]
    assert estimate.diagnostics["sign_consistent"] is False
    assert estimate.diagnostics["n_psi_positive"] == 1
    assert estimate.diagnostics["n_psi_negative"] == 4
    # |psi| still lies on the oracle power law, so H is recovered exactly
    assert estimate.n_expiries == 5
    assert estimate.H0 == pytest.approx(H_TRUE, abs=1e-9)


def test_log_of_zero_guard_drops_near_zero_skews():
    maturities = MATURITIES_7[:6]
    surface: list[SurfacePoint] = []
    for i, T in enumerate(maturities):
        psi = 0.0 if i == 2 else psi_true(T)
        surface.extend(make_expiry(T, psi))

    estimate = estimate_hurst_from_skew(surface, None)

    dropped = estimate.diagnostics["dropped_psi_near_zero"]
    assert len(dropped) == 1
    assert dropped[0]["T"] == pytest.approx(maturities[2])
    assert abs(dropped[0]["psi"]) < HurstConfig().psi_floor
    assert FLAG_PSI_NEAR_ZERO_DROPPED in estimate.diagnostics["flags"]
    assert estimate.n_expiries == 5
    assert estimate.diagnostics["n_in_window"] == 6
    # the surviving five still lie on the power law
    assert estimate.unstable is False
    assert estimate.H0 == pytest.approx(H_TRUE, abs=1e-9)


def test_unstable_fallback_is_never_presented_as_a_result():
    surface: list[SurfacePoint] = []
    for T in MATURITIES_7:
        surface.extend(make_expiry(T, -A_TRUE * T**0.2))

    estimate = estimate_hurst_from_skew(surface, None)
    payload = estimate.to_dict()
    report = hurst_report(estimate)

    assert estimate.H0 == FALLBACK_H0 == 0.1
    assert estimate.unstable is True
    assert estimate.is_usable is False
    assert payload["H0_is_fallback"] is True
    assert report["H0_is_fallback"] is True
    assert "repli" in estimate.message_fr
    # the rejected measurement is preserved, not overwritten by the fallback
    assert estimate.diagnostics["H0_estimated"] != estimate.H0
    assert report["H0_estimated"] == pytest.approx(0.70, abs=1e-9)
    assert "jamais un résultat de calibration" in report["warning_fr"]


# ---------------------------------------------------------------------------
# 5. WLS vs robust cross-check
# ---------------------------------------------------------------------------


def test_wls_and_robust_agree_on_clean_data():
    surface = make_power_law_surface(MATURITIES_7)

    estimate = estimate_hurst_from_skew(surface, None)
    diagnostics = estimate.diagnostics

    assert diagnostics["slope_wls"] == pytest.approx(H_TRUE - 0.5, abs=1e-10)
    assert diagnostics["slope_robust"] == pytest.approx(H_TRUE - 0.5, abs=1e-10)
    assert diagnostics["H0_robust"] == pytest.approx(H_TRUE, abs=1e-10)
    assert diagnostics["robust_disagreement"] is False
    assert FLAG_ROBUST_DISAGREEMENT not in diagnostics["flags"]
    # the exact-fit SE floor is what keeps the flag from firing on rounding noise
    assert FLAG_SE_FLOORED in diagnostics["flags"]


def test_robust_fit_resists_one_corrupted_expiry_and_the_disagreement_flag_fires():
    """
    One expiry (the shortest, i.e. the highest-leverage point of the log-T
    regression) has its skew inflated by 15 %. Theil-Sen is unmoved; WLS is
    dragged by more than one SE, and the flag fires **before** any rejection
    threshold — the estimate here is not even rejected.
    """
    multipliers = [1.15] + [1.0] * (len(MATURITIES_7) - 1)
    clean = estimate_hurst_from_skew(make_power_law_surface(MATURITIES_7), None)
    corrupted = estimate_hurst_from_skew(
        make_power_law_surface(MATURITIES_7, multipliers=multipliers), None
    )
    diagnostics = corrupted.diagnostics

    # robust stays on the truth, WLS does not
    assert diagnostics["H0_robust"] == pytest.approx(H_TRUE, abs=1e-9)
    assert abs(diagnostics["H0_estimated"] - H_TRUE) > 0.05
    assert abs(diagnostics["H0_estimated"] - H_TRUE) > 10.0 * abs(clean.H0 - H_TRUE)

    # the flag fires, and it is not a by-product of a rejection
    assert diagnostics["robust_disagreement"] is True
    assert FLAG_ROBUST_DISAGREEMENT in diagnostics["flags"]
    assert diagnostics["slope_gap"] > diagnostics["robust_disagreement_threshold"]
    assert corrupted.unstable is False
    assert diagnostics["rejection_reasons"] == []


def test_a_gross_corruption_moves_wls_but_not_theil_sen():
    multipliers = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
    estimate = estimate_hurst_from_skew(
        make_power_law_surface(MATURITIES_7, multipliers=multipliers), None
    )
    diagnostics = estimate.diagnostics

    assert diagnostics["H0_robust"] == pytest.approx(H_TRUE, abs=1e-9)
    assert abs(diagnostics["H0_estimated"] - H_TRUE) > 0.15
    assert FLAG_ROBUST_DISAGREEMENT in diagnostics["flags"]
    assert estimate.unstable is True


# ---------------------------------------------------------------------------
# 6. Input shapes, Phase-2 carry-over, JSON safety
# ---------------------------------------------------------------------------


def test_flat_nested_and_mapping_inputs_are_equivalent():
    per_expiry = [make_expiry(T, psi_true(T)) for T in MATURITIES_7]
    flat = [p for group in per_expiry for p in group]
    mapping = {T: group for T, group in zip(MATURITIES_7, per_expiry)}

    a = estimate_hurst_from_skew(flat, None)
    b = estimate_hurst_from_skew(per_expiry, None)
    c = estimate_hurst_from_skew(mapping, None)

    assert a.H0 == pytest.approx(b.H0, rel=1e-15)
    assert a.H0 == pytest.approx(c.H0, rel=1e-15)
    assert a.n_expiries == b.n_expiries == c.n_expiries == len(MATURITIES_7)


def test_prebuilt_skew_points_are_accepted_unchanged():
    surface = make_power_law_surface(MATURITIES_7)
    points, _ = build_skew_curve(surface, None)

    estimate = estimate_hurst_from_skew(points, None)

    assert estimate.unstable is False
    assert estimate.H0 == pytest.approx(H_TRUE, abs=1e-10)
    assert estimate.n_expiries == len(MATURITIES_7)


def test_coarse_strike_ladder_is_surfaced_from_the_variance_curve():
    surface = make_power_law_surface(MATURITIES_7)
    flagged_T = MATURITIES_7[0]
    variance_points = [
        VarianceSwapPoint(
            T=flagged_T,
            k_var=0.09,
            k_var_trunc=0.089,
            n_puts=5,
            n_calls=5,
            F=F_REF,
            D=D_REF,
            diagnostics=VarianceSwapDiagnostics(
                discretisation_bias=0.004,
                flags=(FLAG_COARSE_STRIKE_LADDER,),
            ),
        )
    ]

    estimate = estimate_hurst_from_skew(surface, None, variance_curve=variance_points)
    coarse = estimate.diagnostics["coarse_strike_ladder"]

    assert len(coarse) == 1
    assert coarse[0]["T"] == pytest.approx(flagged_T)
    assert coarse[0]["discretisation_bias"] == pytest.approx(0.004)
    flagged = [sp for sp in estimate.diagnostics["skew_points"] if FLAG_COARSE_LADDER_IN_WINDOW in sp["flags"]]
    assert len(flagged) == 1
    assert flagged[0]["T"] == pytest.approx(flagged_T)


def test_output_is_json_native_and_survives_json_safe():
    surface = make_power_law_surface(MATURITIES_7)
    estimate = estimate_hurst_from_skew(surface, None)

    payload = estimate.to_dict()
    assert_plain_python(payload)
    assert_plain_python(hurst_report(estimate))
    # NaN is permitted by the repo's json encoder settings; the point is that no
    # numpy scalar, tuple or dataclass survives into the payload.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# 7. End-to-end on the real Phase-1 objects
# ---------------------------------------------------------------------------


def _chain_rows(T: float, psi: float, *, expiry_ts: int):
    """Raw Yahoo-shaped rows priced from the oracle smile, with real spreads."""
    D = math.exp(-0.03 * T)
    F = 100.0 / D
    rows = []
    for i, k in enumerate(K_LADDER):
        K = F * math.exp(k)
        vol = iv_exact(k, psi)
        call = black76_call_price(F=F, K=K, T=T, D=D, vol=vol)
        put = call - D * (F - K)
        for option_type, mid in (("call", call), ("put", put)):
            half = 0.005 * mid
            rows.append(
                {
                    "type": option_type,
                    "strike": K,
                    "T": T,
                    "bid": mid - half,
                    "ask": mid + half,
                    "lastPrice": mid,
                    "iv": vol,
                    "volume": 100.0,
                    "openInterest": 500.0,
                    "S0": 100.0,
                    "expiry_ts": expiry_ts,
                    "contractSymbol": f"XYZ{expiry_ts}{option_type[0].upper()}{i:03d}",
                    "underlying": "XYZ",
                }
            )
    return rows, F, D


def test_end_to_end_on_phase1_chains_forward_and_otm_surface():
    maturities = MATURITIES_7[:5]
    chains = []
    forwards = []
    surfaces = []
    for idx, T in enumerate(maturities):
        rows, F, D = _chain_rows(T, psi_true(T), expiry_ts=1_700_000_000 + 86_400 * (idx + 1))
        chain = clean_expiry_chain(rows, config=CleaningConfig(), T=T)
        point = build_forward_point(chain, D=D, S0=100.0)
        assert point is not None
        assert point.F == pytest.approx(F, rel=1e-9)
        surface, _rejections = build_otm_surface(chain, point)
        chains.append(chain)
        forwards.append(point)
        surfaces.append(surface)

    estimate = estimate_hurst_from_skew(
        surfaces, forwards, clean_chains=chains, short_maturity_window=(5.0 / 365.0, 0.25)
    )

    assert estimate.unstable is False
    assert estimate.n_expiries == len(maturities)
    # implied vols are re-inverted with Brent at xtol=1e-6, so the recovery is
    # accurate but not exact; 5e-3 on H is far tighter than any calibration use.
    assert estimate.H0 == pytest.approx(H_TRUE, abs=5e-3)
    assert estimate.diagnostics["H0_robust"] == pytest.approx(H_TRUE, abs=5e-3)
    # spreads came from the cleaned chains, so the spec's 1/spread^2 weights ran
    assert {sp["weight_scheme"] for sp in estimate.diagnostics["skew_points"]} == {"iv_spread"}
    assert_plain_python(estimate.to_dict())


def test_build_spread_lookup_recovers_the_quote_spreads():
    T = 30.0 / 365.0
    rows, _F, _D = _chain_rows(T, psi_true(T), expiry_ts=1_700_000_000)
    chain = clean_expiry_chain(rows, config=CleaningConfig(), T=T)

    lookup = build_spread_lookup([chain])

    assert lookup
    quote = chain.calls[0]
    assert lookup[("symbol", quote.contract_symbol)] == pytest.approx(quote.spread_abs)
    assert lookup[
        ("strike", quote.option_type, round(quote.strike, 10), round(chain.T, 10))
    ] == pytest.approx(quote.spread_abs)


# ---------------------------------------------------------------------------
# 8. Remaining branches: forward matching, one-sided quotes, config guards
# ---------------------------------------------------------------------------


def test_forward_mismatch_and_missing_forward_point_are_flagged():
    T = 30.0 / 365.0
    rows, F_chain, D_chain = _chain_rows(T, psi_true(T), expiry_ts=1_700_000_000)
    chain = clean_expiry_chain(rows, config=CleaningConfig(), T=T)
    real_point = build_forward_point(chain, D=D_chain, S0=100.0)
    assert real_point is not None
    assert real_point.F != pytest.approx(F_REF, rel=1e-6)  # the fixture forward is not 100

    # synthetic surface points carry F = F_REF, the curve says otherwise
    mismatched = build_skew_curve(make_expiry(T, psi_true(T)), [real_point])[0]
    assert len(mismatched) == 1
    assert FLAG_FORWARD_MISMATCH in mismatched[0].flags
    assert mismatched[0].F == pytest.approx(real_point.F)

    # a curve that has no point at this maturity: fall back to the surface's F
    orphan = build_skew_curve(make_expiry(0.5, psi_true(0.5)), [real_point])[0]
    assert len(orphan) == 1
    assert FLAG_NO_FORWARD_POINT in orphan[0].flags
    assert orphan[0].F == pytest.approx(F_REF)


def test_one_sided_quotes_are_flagged_when_kept_and_dropped_when_excluded():
    T = 0.08
    points = make_expiry(T, psi_true(T))
    # mark the four furthest strikes of each wing as zero-bid quotes
    points = [
        dataclasses.replace(p, one_sided=True) if abs(p.k) >= 0.05 else p for p in points
    ]

    kept = estimate_atm_skew(points, config=SkewConfig(include_one_sided=True))
    assert isinstance(kept, SkewPoint)
    assert FLAG_ONE_SIDED_IN_WINDOW in kept.flags

    dropped = estimate_atm_skew(points, config=SkewConfig(include_one_sided=False))
    assert isinstance(dropped, SkewPoint)
    assert FLAG_ONE_SIDED_IN_WINDOW not in dropped.flags
    assert dropped.n_strikes < kept.n_strikes
    # the input is an exact parabola, so removing points must not move psi
    assert dropped.psi == pytest.approx(kept.psi, rel=1e-9)


def test_invalid_configuration_values_raise():
    points = make_expiry(0.08, psi_true(0.08))
    with pytest.raises(ValueError):
        estimate_atm_skew(points, config=SkewConfig(weight_scheme="nope"))
    with pytest.raises(ValueError):
        estimate_hurst_from_skew(
            make_power_law_surface(MATURITIES_7),
            None,
            hurst_config=HurstConfig(log_weight_mode="nope"),
        )


def test_empty_and_invalid_expiries_return_failures_not_numbers():
    empty = estimate_atm_skew([])
    assert isinstance(empty, SkewFailure)
    assert empty.reason == REASON_NO_USABLE_QUOTE

    zero_maturity = estimate_atm_skew(make_expiry(0.0, -1.0))
    assert isinstance(zero_maturity, SkewFailure)
    assert zero_maturity.reason == REASON_INVALID_MATURITY
    assert zero_maturity.to_dict()["reason_fr"]


def test_literal_spec_weight_mode_is_available_and_agrees_on_exact_data():
    surface = make_power_law_surface(MATURITIES_7)

    delta = estimate_hurst_from_skew(surface, None)
    literal = estimate_hurst_from_skew(
        surface, None, hurst_config=HurstConfig(log_weight_mode="raw_psi")
    )
    uniform = estimate_hurst_from_skew(
        surface, None, hurst_config=HurstConfig(log_weight_mode="uniform")
    )

    for estimate in (delta, literal, uniform):
        assert estimate.unstable is False
        assert estimate.H0 == pytest.approx(H_TRUE, abs=1e-9)
    assert delta.diagnostics["log_weight_mode"] == "delta"
    assert literal.diagnostics["log_weight_mode"] == "raw_psi"
