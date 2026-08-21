"""
Tests for the rough Bergomi parameter initialiser.

Target: app/model/volatility_models/rbergomi/initializer.py (spec 4.9)

ORACLES (all independent of the code under test):
  - ``c(H)``: the closed form is re-written literally inline from
    ``MATH_ORACLE.md`` section 8 (``0.5 * sqrt(2H) / ((H+1/2)(H+3/2))``) and,
    separately, checked against the **measured** table produced by the
    orchestrator's ``measure_cH.py`` run against this repo's own simulator
    (60k antithetic paths, conditional estimator, flat ``xi0 = 0.04``). The
    stated envelope there is ~2 % for ``eta <= 1``; the table is asserted at
    3 %, which is the worst cell (``H = 0.05``, ratio 0.974).
  - ``a(H)`` (the rho-free ATM curvature level): cross-checked against
    **Hagan's beta = 1 SABR expansion**, written out inline. At ``H = 1/2``
    rBergomi *is* lognormal SABR with ``nu = eta/2``, and Hagan gives
    ``d^2 sigma/dk^2|_{k=0} = (2 - 3 rho^2) nu^2 / (6 alpha)``. This is an
    entirely different formula from the module's
    ``(eta^2 T^{2H-1}/sigma)(a(H) - 2 c(H)^2 rho^2)`` and pins both the level
    and the ``rho^2`` law exactly. ``a(H)`` is also checked against the
    measured ``rho = 0`` curvature run (150k paths).
  - Every skew curve fed to the initialiser is built **forward** from the
    closed forms with a chosen ``(H, eta, rho)``, so the parameters the module
    must recover are known by construction and never read back from it.

Determinism: closed forms only. No RNG, no Monte-Carlo, no network, no I/O.
"""

from __future__ import annotations

import json
import math

import pytest

from app.model.calibration.rough_vol.hurst_estimator import HurstEstimate, SkewPoint
from app.model.volatility_models.rbergomi.initializer import (
    ETA_DISAGREEMENT_FACTOR,
    ETA_INIT_MAX,
    ETA_INIT_MIN,
    FLAG_CURVATURE_ILL_CONDITIONED,
    FLAG_ETA_CLIPPED,
    FLAG_ETA_DISAGREEMENT,
    FLAG_HURST_UNSTABLE,
    FLAG_H_CLIPPED,
    FLAG_NO_CURVATURE_ESTIMATE,
    FLAG_PSI_SIGN_AMBIGUOUS,
    FLAG_SIGN_DISAGREES_WITH_CURVE,
    FLAG_T_REF_OUTSIDE_HURST_WINDOW,
    RHO_PRIOR_ABS,
    T_REF_RULE_EXPLICIT,
    T_REF_RULE_MIN_RELATIVE_SE,
    T_REF_RULE_SHORTEST,
    InitializerConfig,
    RBergomiInitializationError,
    atm_curvature_model,
    atm_skew_model,
    c_of_H,
    curvature_coefficient,
    curvature_information_share,
    curvature_rho_coefficient,
    curvature_sign_change_rho_abs,
    eta_from_curvature,
    eta_from_skew,
    initial_rbergomi_params,
    initialize_rbergomi_parameters,
    initializer_report,
    select_t_ref,
)
from app.model.volatility_models.rbergomi.simulator_xi_curve import (
    H_MAX,
    H_MIN,
    RBergomiParams,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

#: MATH_ORACLE.md section 8, verdict line, transcribed literally.
def c_of_H_oracle(H: float) -> float:
    return 0.5 * math.sqrt(2.0 * H) / ((H + 0.5) * (H + 1.5))


#: MATH_ORACLE.md section 8, measured table: H -> c_hat at eta = 0.5, measured
#: against this repo's simulator. Stated envelope ~2 % (worst cell 2.6 %).
C_HAT_MEASURED = {
    0.05: 0.18070,
    0.10: 0.22867,
    0.20: 0.26188,
    0.35: 0.26201,
    0.45: 0.25188,
}

#: rho = 0 ATM curvature coefficient measured against this repo's simulator
#: (60k antithetic paths, n_max = 256, flat xi0 = 0.04, eta = 0.5, rho = 0,
#: T in {20,40,80}/365). Envelope ~3 %.
A_HAT_MEASURED = {
    0.10: 0.06333,
    0.30: 0.08851,
    0.45: 0.08377,
}


def hagan_sabr_beta1_smile(k: float, *, alpha: float, nu: float, rho: float) -> float:
    """
    Hagan's beta = 1 SABR implied vol, leading (T-independent) shape only.

    ``sigma(k) = alpha * z / x(z)``, ``z = -(nu/alpha) k``,
    ``x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho))``.
    Written out here so the module's curvature formula is checked against a
    completely different closed form.
    """
    z = -(nu / alpha) * k
    if abs(z) < 1e-14:
        return alpha
    x = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    return alpha * z / x


def hagan_atm_curvature(*, alpha: float, nu: float, rho: float, h: float = 1e-3) -> float:
    """
    Central second difference of :func:`hagan_sabr_beta1_smile` at ``k = 0``,
    Richardson-extrapolated (``(4 D(h) - D(2h)) / 3``) so the ``O(h^2)``
    truncation of the plain difference does not masquerade as a formula error.
    """

    def second_difference(step: float) -> float:
        up = hagan_sabr_beta1_smile(step, alpha=alpha, nu=nu, rho=rho)
        mid = hagan_sabr_beta1_smile(0.0, alpha=alpha, nu=nu, rho=rho)
        dn = hagan_sabr_beta1_smile(-step, alpha=alpha, nu=nu, rho=rho)
        return (up - 2.0 * mid + dn) / (step * step)

    return (4.0 * second_difference(h) - second_difference(2.0 * h)) / 3.0


# ---------------------------------------------------------------------------
# Synthetic skew curves (built forward from the closed forms)
# ---------------------------------------------------------------------------

SIGMA_ATM = 0.25
F_REF = 100.0
MATURITIES = (7.0 / 365.0, 14.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0)
HURST_WINDOW = (5.0 / 365.0, 0.25)


def make_skew_point(
    T: float,
    *,
    H: float,
    eta: float,
    rho: float,
    sigma_atm: float = SIGMA_ATM,
    se_rel: float = 0.02,
    curvature: float | None = None,
) -> SkewPoint:
    """One expiry whose psi and curvature are the exact leading-order forms."""
    psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
    kappa = (
        atm_curvature_model(H=H, eta=eta, rho=rho, T=T, sigma_atm=sigma_atm)
        if curvature is None
        else float(curvature)
    )
    return SkewPoint(
        T=float(T),
        psi=psi,
        se=abs(psi) * se_rel,
        n_strikes=9,
        window=(-0.05, 0.05),
        curvature=kappa,
        se_curvature=abs(kappa) * se_rel,
        quad_coeff=kappa / 2.0,
        sigma_atm=sigma_atm,
        n_left=4,
        n_right=4,
        r_squared=0.999,
        converged=True,
        weight_scheme="vega",
        F=F_REF,
    )


def make_skew_curve(
    *,
    H: float,
    eta: float,
    rho: float,
    maturities=MATURITIES,
    sigma_atm: float = SIGMA_ATM,
    se_rel: float = 0.02,
) -> list[SkewPoint]:
    return [
        make_skew_point(T, H=H, eta=eta, rho=rho, sigma_atm=sigma_atm, se_rel=se_rel)
        for T in maturities
    ]


def make_hurst(
    H: float,
    points,
    *,
    unstable: bool = False,
    window=HURST_WINDOW,
    dominant_sign: int | None = None,
    amplitude: float | None = None,
) -> HurstEstimate:
    diagnostics = {
        "skew_points": [p.to_dict() for p in points],
        "sign_consistent": True,
        "rejection_reasons": ["low_r2"] if unstable else [],
        "rejection_reasons_fr": ["R² trop faible"] if unstable else [],
    }
    if dominant_sign is not None:
        diagnostics["dominant_sign"] = int(dominant_sign)
    if amplitude is not None:
        diagnostics["amplitude_A"] = float(amplitude)
    return HurstEstimate(
        H0=float(H),
        se=0.02,
        ci95=(H - 0.04, H + 0.04),
        r2=0.35 if unstable else 0.98,
        n_expiries=len(points),
        window=tuple(float(v) for v in window),
        unstable=bool(unstable),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# c(H)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", [0.01, 0.05, 0.1, 0.2, 0.3, 0.42, 0.49, 0.5])
def test_c_of_H_matches_the_closed_form_exactly(H):
    assert c_of_H(H) == pytest.approx(c_of_H_oracle(H), rel=0.0, abs=1e-15)


@pytest.mark.parametrize("H, c_hat", sorted(C_HAT_MEASURED.items()))
def test_c_of_H_matches_the_measured_table(H, c_hat):
    """
    The closed form is within the envelope MATH_ORACLE section 8 states: ~2 %
    for eta <= 1, worst cell 2.6 % at H = 0.05. Asserted at 3 %.
    """
    assert c_of_H(H) == pytest.approx(c_hat, rel=0.03)


@pytest.mark.parametrize("H", sorted(C_HAT_MEASURED))
def test_the_halving_factor_is_the_one_the_measurement_settled(H):
    """
    The measurement pinned c_hat / [unhalved literature form] at 0.487-0.493.
    The module must carry the halved form, not the unhalved one.
    """
    unhalved = math.sqrt(2.0 * H) / ((H + 0.5) * (H + 1.5))
    assert C_HAT_MEASURED[H] / unhalved == pytest.approx(0.49, abs=0.01)
    assert c_of_H(H) / unhalved == pytest.approx(0.5, abs=1e-15)


def test_c_of_H_is_nan_outside_its_domain():
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        assert math.isnan(c_of_H(bad))


# ---------------------------------------------------------------------------
# a(H) and the curvature law
# ---------------------------------------------------------------------------


def test_curvature_coefficient_matches_the_closed_form():
    for H in (0.05, 0.1, 0.25, 0.4, 0.5):
        expected = H / (4.0 * (H + 0.5) ** 2 * (H + 1.0))
        assert curvature_coefficient(H) == pytest.approx(expected, abs=1e-15)


def test_curvature_coefficient_at_half_is_one_twelfth():
    """Hagan's beta = 1 SABR at rho = 0 gives eta^2 / (12 alpha)."""
    assert curvature_coefficient(0.5) == pytest.approx(1.0 / 12.0, abs=1e-15)


@pytest.mark.parametrize("rho", [-0.9, -0.7, -0.4, 0.0, 0.3, 0.8])
@pytest.mark.parametrize("eta", [0.4, 1.0, 2.0])
def test_atm_curvature_matches_hagan_sabr_at_H_one_half(eta, rho):
    """
    At H = 1/2 the rBergomi variance is lognormal with vol-of-vol ``eta``, so
    the *volatility* is lognormal with vol-of-vol ``nu = eta / 2`` — i.e. SABR
    with beta = 1. Hagan's expansion, differentiated numerically, must
    reproduce the module's curvature formula including its rho^2 law.
    """
    alpha = SIGMA_ATM
    model = atm_curvature_model(H=0.5, eta=eta, rho=rho, T=1.0, sigma_atm=alpha)
    hagan = hagan_atm_curvature(alpha=alpha, nu=eta / 2.0, rho=rho)
    assert model == pytest.approx(hagan, rel=1e-5)
    # and the closed form Hagan's series expands to
    closed = (2.0 - 3.0 * rho * rho) * eta * eta / (24.0 * alpha)
    assert model == pytest.approx(closed, abs=1e-14)


@pytest.mark.parametrize("H, a_hat", sorted(A_HAT_MEASURED.items()))
def test_curvature_coefficient_matches_the_measured_rho_zero_run(H, a_hat):
    """Measured against this repo's simulator at rho = 0; envelope ~3 %."""
    assert curvature_coefficient(H) == pytest.approx(a_hat, rel=0.03)


def test_curvature_rho_coefficient_is_twice_c_squared():
    for H in (0.05, 0.2, 0.45, 0.5):
        assert curvature_rho_coefficient(H) == pytest.approx(
            2.0 * c_of_H(H) ** 2, abs=1e-16
        )


def test_curvature_sign_change_brackets_the_equity_prior():
    """
    The leading-order ATM curvature vanishes at |rho| = sqrt(a / 2c^2), which
    sits just above the 0.7 equity prior across the whole rough range. That is
    exactly why the module inverts the rho-free combination instead.
    """
    for H in (0.05, 0.1, 0.2, 0.35, 0.45):
        rho_star = curvature_sign_change_rho_abs(H)
        assert 0.7 < rho_star < 0.85
        assert atm_curvature_model(
            H=H, eta=1.0, rho=-rho_star, T=0.1, sigma_atm=SIGMA_ATM
        ) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# The two inversions, in isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", [0.05, 0.12, 0.3, 0.45])
@pytest.mark.parametrize("eta, rho", [(0.6, -0.8), (1.5, -0.4), (2.2, 0.55)])
def test_eta_from_skew_is_the_exact_inverse(H, eta, rho):
    T = 21.0 / 365.0
    psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
    assert eta_from_skew(psi, H=H, rho_abs=abs(rho), T=T) == pytest.approx(eta, rel=1e-13)


@pytest.mark.parametrize("H", [0.05, 0.12, 0.3, 0.45])
@pytest.mark.parametrize("eta, rho", [(0.6, -0.8), (1.5, -0.4), (2.2, 0.55), (1.0, 0.0)])
def test_eta_from_curvature_is_the_exact_inverse_and_needs_no_rho(H, eta, rho):
    T = 21.0 / 365.0
    psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
    kappa = atm_curvature_model(H=H, eta=eta, rho=rho, T=T, sigma_atm=SIGMA_ATM)
    recovered = eta_from_curvature(kappa, psi=psi, H=H, T=T, sigma_atm=SIGMA_ATM)
    assert recovered == pytest.approx(eta, rel=1e-12)


def test_eta_from_curvature_is_blind_to_the_sign_of_rho():
    """The whole point of the cross-check: only rho^2 enters."""
    H, eta, T = 0.14, 1.3, 30.0 / 365.0
    for rho in (0.65, -0.65):
        psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
        kappa = atm_curvature_model(H=H, eta=eta, rho=rho, T=T, sigma_atm=SIGMA_ATM)
        assert eta_from_curvature(
            kappa, psi=psi, H=H, T=T, sigma_atm=SIGMA_ATM
        ) == pytest.approx(eta, rel=1e-12)


def test_eta_from_curvature_refuses_an_inconsistent_smile():
    """A curvature so negative that kappa*sigma + 2 psi^2 <= 0 yields nan."""
    assert math.isnan(
        eta_from_curvature(-1e6, psi=0.1, H=0.1, T=0.05, sigma_atm=SIGMA_ATM)
    )
    assert math.isnan(
        eta_from_curvature(float("nan"), psi=0.1, H=0.1, T=0.05, sigma_atm=SIGMA_ATM)
    )


# ---------------------------------------------------------------------------
# eta0 inverts the skew relation exactly, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H, eta, rho", [(0.08, 1.2, -0.7), (0.2, 2.0, -0.7), (0.4, 0.9, 0.7)])
def test_eta0_inverts_the_skew_relation_exactly(H, eta, rho):
    """
    Build psi consistent with (H, eta, rho) via the formula, feed it in with a
    matching |rho0| prior, and recover eta0 == eta (the value is inside the
    clip range, so no clip interferes).
    """
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    estimate = make_hurst(H, points)
    cfg = InitializerConfig(rho_prior_abs=abs(rho))
    H0, eta0, rho0, diag = initialize_rbergomi_parameters(estimate, None, config=cfg)

    assert H0 == pytest.approx(H, abs=1e-15)
    assert eta0 == pytest.approx(eta, rel=1e-12)
    assert diag["eta0_skew_unclipped"] == pytest.approx(eta, rel=1e-12)
    assert rho0 == pytest.approx(rho, rel=1e-15)
    assert diag["eta_source"] == "skew"
    assert not any(c["applied"] for c in diag["clips"])


def test_eta0_uses_the_module_c_of_H_and_the_recorded_t_ref():
    """The reported (c(H), T_ref, psi) must reproduce eta0 exactly."""
    H, eta, rho = 0.13, 1.6, -0.55
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, rho0, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None
    )
    rebuilt = abs(diag["psi_at_t_ref"]) / (
        diag["c_of_H"] * abs(rho0) * diag["T_ref"] ** (diag["H0"] - 0.5)
    )
    assert rebuilt == pytest.approx(eta0, rel=1e-12)


# ---------------------------------------------------------------------------
# rho0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho, expected_sign", [(-0.6, -1), (0.6, +1)])
def test_rho0_sign_follows_the_sign_of_psi(rho, expected_sign):
    points = make_skew_curve(H=0.12, eta=1.4, rho=rho)
    _, _, rho0, diag = initialize_rbergomi_parameters(make_hurst(0.12, points), None)
    assert math.copysign(1.0, rho0) == expected_sign
    assert diag["psi_sign"] == expected_sign
    assert abs(rho0) == pytest.approx(RHO_PRIOR_ABS, abs=1e-15)
    assert math.copysign(1.0, diag["psi_at_t_ref"]) == expected_sign


def test_rho0_magnitude_prior_is_configurable():
    points = make_skew_curve(H=0.12, eta=1.4, rho=-0.6)
    estimate = make_hurst(0.12, points)
    for prior in (0.3, 0.5, 0.7, 0.95):
        _, _, rho0, diag = initialize_rbergomi_parameters(
            estimate, None, config=InitializerConfig(rho_prior_abs=prior)
        )
        assert rho0 == pytest.approx(-prior, abs=1e-15)
        assert diag["rho_prior_abs"] == pytest.approx(prior, abs=1e-15)
        assert diag["rho_eta_degeneracy"]["rho_is_a_prior"] is True


def test_a_zero_skew_falls_back_to_the_equity_prior_sign_and_says_so():
    H = 0.12
    points = [
        make_skew_point(T, H=H, eta=1.0, rho=-0.5) for T in MATURITIES[1:]
    ]
    flat = SkewPoint(
        T=MATURITIES[0],
        psi=0.0,
        se=0.0,
        n_strikes=9,
        window=(-0.05, 0.05),
        curvature=1.0,
        sigma_atm=SIGMA_ATM,
        F=F_REF,
    )
    # psi = 0 is below the floor, so the flat expiry is not eligible as T_ref;
    # force it by pinning t_ref explicitly.
    _, _, rho0, diag = initialize_rbergomi_parameters(
        make_hurst(H, [flat] + points),
        None,
        skew_curve=[flat],
        config=InitializerConfig(t_ref=MATURITIES[0], psi_floor=0.0),
    )
    assert rho0 < 0.0
    assert FLAG_PSI_SIGN_AMBIGUOUS in diag["flags"]
    assert diag["psi_sign"] == 0


def test_a_sign_that_contradicts_the_curve_is_flagged():
    H = 0.12
    points = make_skew_curve(H=H, eta=1.2, rho=-0.6)
    estimate = make_hurst(H, points, dominant_sign=+1)
    _, _, _, diag = initialize_rbergomi_parameters(estimate, None)
    assert FLAG_SIGN_DISAGREES_WITH_CURVE in diag["flags"]
    assert diag["dominant_psi_sign"] == 1
    assert diag["psi_sign"] == -1


def test_an_out_of_range_rho_prior_is_clipped_and_recorded():
    points = make_skew_curve(H=0.12, eta=1.4, rho=-0.6)
    _, _, rho0, diag = initialize_rbergomi_parameters(
        make_hurst(0.12, points), None, config=InitializerConfig(rho_prior_abs=1.4)
    )
    assert abs(rho0) < 1.0
    clip = next(c for c in diag["clips"] if c["parameter"] == "rho0_abs")
    assert clip["applied"] is True
    assert clip["raw"] == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# The [0.5, 3.5] clip
# ---------------------------------------------------------------------------


def test_eta0_upper_clip_fires_and_is_recorded():
    H, eta, rho = 0.12, 5.0, -0.95
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)

    raw = eta * abs(rho) / RHO_PRIOR_ABS
    assert raw > ETA_INIT_MAX
    assert eta0 == pytest.approx(ETA_INIT_MAX, abs=1e-15)
    assert diag["eta0_skew_unclipped"] == pytest.approx(raw, rel=1e-12)
    assert FLAG_ETA_CLIPPED in diag["flags"]
    clip = next(c for c in diag["clips"] if c["parameter"] == "eta0")
    assert clip["applied"] is True
    assert clip["bound"] == "max"
    assert clip["raw"] == pytest.approx(raw, rel=1e-12)
    assert clip["value"] == pytest.approx(ETA_INIT_MAX, abs=1e-15)
    assert clip["min"] == pytest.approx(ETA_INIT_MIN)
    assert clip["max"] == pytest.approx(ETA_INIT_MAX)
    assert diag["rho_eta_degeneracy"]["product_preserved_after_clip"] is False


def test_eta0_lower_clip_fires_and_is_recorded():
    H, eta, rho = 0.12, 0.2, -0.3
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)

    raw = eta * abs(rho) / RHO_PRIOR_ABS
    assert raw < ETA_INIT_MIN
    assert eta0 == pytest.approx(ETA_INIT_MIN, abs=1e-15)
    assert FLAG_ETA_CLIPPED in diag["flags"]
    clip = next(c for c in diag["clips"] if c["parameter"] == "eta0")
    assert clip["applied"] is True
    assert clip["bound"] == "min"


def test_the_clip_bounds_are_configurable_and_reported():
    H, eta, rho = 0.12, 4.0, -0.7
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None, config=InitializerConfig(eta_min=0.2, eta_max=4.5)
    )
    assert eta0 == pytest.approx(4.0, rel=1e-12)
    assert diag["eta_min"] == pytest.approx(0.2)
    assert diag["eta_max"] == pytest.approx(4.5)
    assert FLAG_ETA_CLIPPED not in diag["flags"]


def test_H0_outside_the_simulator_bounds_is_clipped_and_recorded():
    points = make_skew_curve(H=0.2, eta=1.0, rho=-0.6)
    estimate = make_hurst(0.9, points)  # duck-typed, deliberately out of range
    H0, _, _, diag = initialize_rbergomi_parameters(estimate, None)
    assert H0 == pytest.approx(H_MAX, abs=1e-15)
    assert FLAG_H_CLIPPED in diag["flags"]
    clip = next(c for c in diag["clips"] if c["parameter"] == "H0")
    assert clip["applied"] is True
    assert clip["raw"] == pytest.approx(0.9)
    assert H_MIN <= H0 <= H_MAX


# ---------------------------------------------------------------------------
# The curvature cross-check
# ---------------------------------------------------------------------------


def test_the_curvature_estimate_is_reported_and_agrees_when_the_data_agrees():
    H, eta, rho = 0.14, 1.5, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None, config=InitializerConfig(rho_prior_abs=abs(rho))
    )
    assert diag["eta0_curvature"] == pytest.approx(eta, rel=1e-12)
    assert diag["eta0_skew"] == pytest.approx(eta, rel=1e-12)
    assert diag["eta_ratio"] == pytest.approx(1.0, rel=1e-10)
    assert diag["eta_disagreement"] is False
    assert FLAG_ETA_DISAGREEMENT not in diag["flags"]
    assert FLAG_NO_CURVATURE_ESTIMATE not in diag["flags"]
    assert diag["eta_disagreement_factor"] == pytest.approx(ETA_DISAGREEMENT_FACTOR)


def test_the_curvature_estimate_recovers_eta_whatever_the_rho_prior_is():
    """
    eta0 (skew) is conditional on |rho0|; eta0_curvature is not. Sweeping the
    prior must move one and leave the other alone.
    """
    H, eta, rho = 0.14, 1.5, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    estimate = make_hurst(H, points)
    for prior in (0.4, 0.6, 0.8):
        _, _, _, diag = initialize_rbergomi_parameters(
            estimate, None, config=InitializerConfig(rho_prior_abs=prior)
        )
        assert diag["eta0_curvature"] == pytest.approx(eta, rel=1e-12)
        assert diag["eta0_skew"] == pytest.approx(eta * abs(rho) / prior, rel=1e-12)
        assert diag["implied_abs_rho_from_curvature"] == pytest.approx(abs(rho), rel=1e-12)


def test_the_factor_two_disagreement_flag_fires_when_they_disagree():
    H, eta, rho, T = 0.14, 1.5, -0.6, MATURITIES[0]
    target = 3.0 * eta  # the curvature says 3x the skew-implied eta
    psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
    a = curvature_coefficient(H)
    kappa = (a * target * target * T ** (2.0 * H - 1.0) - 2.0 * psi * psi) / SIGMA_ATM
    corrupted = make_skew_point(T, H=H, eta=eta, rho=rho, curvature=kappa)

    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, [corrupted]),
        None,
        config=InitializerConfig(rho_prior_abs=abs(rho)),
    )
    assert diag["eta0_curvature"] == pytest.approx(target, rel=1e-11)
    assert diag["eta_ratio"] == pytest.approx(3.0, rel=1e-10)
    assert diag["eta_disagreement"] is True
    assert FLAG_ETA_DISAGREEMENT in diag["flags"]
    assert any("facteur 2" in w for w in diag["warnings_fr"])


def test_the_disagreement_flag_stays_quiet_just_below_the_factor():
    H, eta, rho, T = 0.14, 1.5, -0.6, MATURITIES[0]
    target = 1.9 * eta
    psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
    a = curvature_coefficient(H)
    kappa = (a * target * target * T ** (2.0 * H - 1.0) - 2.0 * psi * psi) / SIGMA_ATM
    point = make_skew_point(T, H=H, eta=eta, rho=rho, curvature=kappa)

    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, [point]), None, config=InitializerConfig(rho_prior_abs=abs(rho))
    )
    assert diag["eta_ratio"] == pytest.approx(1.9, rel=1e-10)
    assert diag["eta_disagreement"] is False
    assert FLAG_ETA_DISAGREEMENT not in diag["flags"]


def test_a_missing_curvature_is_reported_not_faked():
    H, eta, rho = 0.14, 1.5, -0.6
    point = make_skew_point(MATURITIES[0], H=H, eta=eta, rho=rho, curvature=float("nan"))
    _, eta0, _, diag = initialize_rbergomi_parameters(make_hurst(H, [point]), None)
    assert math.isnan(diag["eta0_curvature"])
    assert math.isnan(diag["eta_ratio"])
    assert diag["eta_disagreement"] is False
    assert FLAG_NO_CURVATURE_ESTIMATE in diag["flags"]
    assert eta0 > 0.0  # the skew estimate still stands


def test_the_ill_conditioned_corner_is_flagged_not_hidden():
    """
    At |rho| = sqrt(a / 2c^2) the leading-order ATM curvature is exactly zero:
    the curvature carries no information beyond psi. The estimate still comes
    out right (the combination is then pure 2 psi^2) but must be flagged.
    """
    H, eta = 0.14, 1.5
    rho = -curvature_sign_change_rho_abs(H)
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None, config=InitializerConfig(rho_prior_abs=abs(rho))
    )
    assert diag["curvature_at_t_ref"] == pytest.approx(0.0, abs=1e-13)
    assert diag["curvature_information_share"] == pytest.approx(0.0, abs=1e-12)
    assert FLAG_CURVATURE_ILL_CONDITIONED in diag["flags"]
    assert diag["eta0_curvature"] == pytest.approx(eta, rel=1e-10)


def test_curvature_information_share_is_between_zero_and_one():
    H, eta, T = 0.14, 1.2, 0.05
    for rho in (0.0, -0.3, -0.7, -0.95):
        psi = atm_skew_model(H=H, eta=eta, rho=rho, T=T)
        kappa = atm_curvature_model(H=H, eta=eta, rho=rho, T=T, sigma_atm=SIGMA_ATM)
        share = curvature_information_share(kappa, psi=psi, sigma_atm=SIGMA_ATM)
        assert 0.0 <= share <= 1.0
    assert curvature_information_share(
        atm_curvature_model(H=H, eta=eta, rho=0.0, T=T, sigma_atm=SIGMA_ATM),
        psi=0.0,
        sigma_atm=SIGMA_ATM,
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# An unstable Hurst estimate
# ---------------------------------------------------------------------------


def test_an_unstable_hurst_estimate_propagates_and_does_not_crash():
    points = make_skew_curve(H=0.3, eta=1.2, rho=-0.6)
    estimate = make_hurst(0.1, points, unstable=True)  # 0.1 is the spec-4.5 fallback
    H0, eta0, rho0, diag = initialize_rbergomi_parameters(estimate, None)

    assert H0 == pytest.approx(0.1, abs=1e-15)
    assert diag["hurst_unstable"] is True
    assert diag["H0_is_fallback"] is True
    assert FLAG_HURST_UNSTABLE in diag["flags"]
    assert diag["hurst_rejection_reasons"] == ["low_r2"]
    assert math.isfinite(eta0) and eta0 > 0.0
    assert math.isfinite(rho0)
    # the amplitude-based alternative is withheld when the regression was rejected
    assert math.isnan(diag["eta0_from_regression_amplitude"])


def test_a_stable_estimate_exposes_the_amplitude_alternative():
    H, eta, rho = 0.13, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    amplitude = abs(c_of_H(H) * rho * eta)  # exp(intercept) of the log|psi| fit
    estimate = make_hurst(H, points, amplitude=amplitude)
    _, _, _, diag = initialize_rbergomi_parameters(
        estimate, None, config=InitializerConfig(rho_prior_abs=abs(rho))
    )
    assert diag["eta0_from_regression_amplitude"] == pytest.approx(eta, rel=1e-12)


# ---------------------------------------------------------------------------
# The rho-eta degeneracy statement
# ---------------------------------------------------------------------------


def test_the_degeneracy_is_stated_in_diagnostics():
    H, eta, rho = 0.12, 1.5, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)

    deg = diag["rho_eta_degeneracy"]
    assert deg["identified_quantity"] == "rho * eta"
    assert deg["rho_is_a_prior"] is True
    assert deg["eta_is_conditional_on_rho_prior"] is True
    assert "4.10" in deg["broken_by"]
    assert "rho*eta" in deg["message_fr"] or "rho·eta" in deg["message_fr"]
    assert deg["independent_check"].lower().startswith("atm curvature")


def test_the_degeneracy_statement_is_truthful_about_the_product():
    """
    The claim: the data identifies rho * eta and nothing finer. Sweeping the
    |rho0| prior must leave rho0 * eta0 bit-for-bit invariant (before clipping)
    and equal to psi(T_ref) / (c(H0) T_ref^{H0-1/2}).
    """
    H, eta, rho = 0.12, 1.5, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    estimate = make_hurst(H, points)

    products = []
    for prior in (0.35, 0.5, 0.7, 0.9):
        _, eta0, rho0, diag = initialize_rbergomi_parameters(
            estimate, None, config=InitializerConfig(rho_prior_abs=prior)
        )
        deg = diag["rho_eta_degeneracy"]
        # the recorded product is the one the returned pair carries
        assert deg["product_rho_eta"] == pytest.approx(rho0 * eta0, rel=1e-13)
        # and it equals the model-free quantity the skew actually identifies
        expected = diag["psi_at_t_ref"] / (
            diag["c_of_H"] * diag["T_ref"] ** (diag["H0"] - 0.5)
        )
        assert deg["identified_value"] == pytest.approx(expected, rel=1e-12)
        assert deg["identified_value"] == pytest.approx(rho * eta, rel=1e-12)
        products.append(deg["identified_value"])

    assert max(products) == pytest.approx(min(products), rel=1e-12)


def test_the_product_claim_is_marked_broken_when_the_clip_fires():
    H, eta, rho = 0.12, 5.0, -0.95
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, eta0, rho0, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    deg = diag["rho_eta_degeneracy"]
    assert deg["product_preserved_after_clip"] is False
    assert deg["product_rho_eta"] == pytest.approx(rho0 * eta0, rel=1e-13)
    assert abs(deg["identified_value"]) > abs(deg["product_rho_eta"])


# ---------------------------------------------------------------------------
# T_ref selection
# ---------------------------------------------------------------------------


def test_default_rule_picks_the_best_measured_maturity_not_the_shortest():
    H, eta, rho = 0.12, 1.4, -0.6
    points = [
        make_skew_point(MATURITIES[0], H=H, eta=eta, rho=rho, se_rel=0.30),
        make_skew_point(MATURITIES[1], H=H, eta=eta, rho=rho, se_rel=0.05),
        make_skew_point(MATURITIES[2], H=H, eta=eta, rho=rho, se_rel=0.01),
        make_skew_point(MATURITIES[3], H=H, eta=eta, rho=rho, se_rel=0.12),
    ]
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    assert diag["T_ref"] == pytest.approx(MATURITIES[2])
    assert diag["T_ref_rule"] == T_REF_RULE_MIN_RELATIVE_SE
    selection = diag["T_ref_selection"]
    assert selection["criterion"] == "min SE(psi)/|psi|, ties to the shorter maturity"
    assert selection["rationale_fr"]
    assert len(selection["candidates"]) == 4
    assert all(c["usable"] for c in selection["candidates"])
    best = min(selection["candidates"], key=lambda c: c["relative_se"])
    assert best["T"] == pytest.approx(MATURITIES[2])


def test_ties_on_relative_se_break_toward_the_shorter_maturity():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho, se_rel=0.02)
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    assert diag["T_ref"] == pytest.approx(min(MATURITIES))


def test_the_shortest_rule_is_available():
    H, eta, rho = 0.12, 1.4, -0.6
    points = [
        make_skew_point(MATURITIES[0], H=H, eta=eta, rho=rho, se_rel=0.30),
        make_skew_point(MATURITIES[2], H=H, eta=eta, rho=rho, se_rel=0.01),
    ]
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None, config=InitializerConfig(t_ref_rule=T_REF_RULE_SHORTEST)
    )
    assert diag["T_ref"] == pytest.approx(MATURITIES[0])
    assert diag["T_ref_rule"] == T_REF_RULE_SHORTEST


def test_an_explicit_t_ref_is_honoured_and_reported():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, points), None, config=InitializerConfig(t_ref=MATURITIES[3])
    )
    assert diag["T_ref"] == pytest.approx(MATURITIES[3])
    assert diag["T_ref_rule"] == T_REF_RULE_EXPLICIT
    assert diag["T_ref_days"] == pytest.approx(MATURITIES[3] * 365.0)


def test_an_unmatched_explicit_t_ref_raises_instead_of_snapping():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    with pytest.raises(RBergomiInitializationError, match="approximation silencieuse"):
        initialize_rbergomi_parameters(
            make_hurst(H, points), None, config=InitializerConfig(t_ref=0.123456)
        )


def test_t_ref_stays_inside_the_hurst_regression_window():
    H, eta, rho = 0.12, 1.4, -0.6
    inside = make_skew_point(30.0 / 365.0, H=H, eta=eta, rho=rho, se_rel=0.20)
    outside = make_skew_point(1.5, H=H, eta=eta, rho=rho, se_rel=0.001)
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, [inside, outside], window=(5.0 / 365.0, 0.25)), None
    )
    assert diag["T_ref"] == pytest.approx(30.0 / 365.0)
    entry = next(
        c for c in diag["T_ref_selection"]["candidates"] if c["T"] > 1.0
    )
    assert entry["in_hurst_window"] is False
    assert entry["usable"] is False


def test_no_candidate_inside_the_window_falls_back_and_says_so():
    H, eta, rho = 0.12, 1.4, -0.6
    far = [make_skew_point(T, H=H, eta=eta, rho=rho) for T in (1.2, 1.6)]
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, far, window=(5.0 / 365.0, 0.25)), None
    )
    assert FLAG_T_REF_OUTSIDE_HURST_WINDOW in diag["flags"]
    assert diag["T_ref"] == pytest.approx(1.2)


def test_the_window_restriction_can_be_switched_off():
    H, eta, rho = 0.12, 1.4, -0.6
    inside = make_skew_point(30.0 / 365.0, H=H, eta=eta, rho=rho, se_rel=0.20)
    outside = make_skew_point(1.5, H=H, eta=eta, rho=rho, se_rel=0.001)
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, [inside, outside]),
        None,
        config=InitializerConfig(restrict_to_hurst_window=False),
    )
    assert diag["T_ref"] == pytest.approx(1.5)


def test_select_t_ref_is_usable_on_its_own():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    chosen, selection, flags = select_t_ref(
        points, config=InitializerConfig(), hurst_window=HURST_WINDOW
    )
    assert chosen.T == pytest.approx(min(MATURITIES))
    assert selection["n_usable"] == len(points)
    assert flags == []


# ---------------------------------------------------------------------------
# Input plumbing
# ---------------------------------------------------------------------------


def test_an_explicit_skew_curve_wins_over_the_hurst_diagnostics():
    H, eta, rho = 0.12, 1.4, -0.6
    recorded = make_skew_curve(H=H, eta=eta, rho=rho)
    explicit = [make_skew_point(45.0 / 365.0, H=H, eta=3.0, rho=rho)]
    _, eta0, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, recorded),
        None,
        skew_curve=explicit,
        config=InitializerConfig(rho_prior_abs=abs(rho)),
    )
    assert diag["skew_source"] == "skew_curve"
    assert diag["T_ref"] == pytest.approx(45.0 / 365.0)
    assert eta0 == pytest.approx(3.0, rel=1e-12)


def test_the_hurst_diagnostics_path_rehydrates_skew_points_faithfully():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    assert diag["skew_source"] == "hurst_diagnostics"
    assert diag["n_skew_points"] == len(points)
    assert diag["skew_fit"] == points[0].to_dict()


def test_a_plain_mapping_hurst_result_is_accepted():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    estimate = make_hurst(H, points)
    from_object = initialize_rbergomi_parameters(estimate, None)
    from_mapping = initialize_rbergomi_parameters(estimate.to_dict(), None)
    assert from_object[:3] == pytest.approx(from_mapping[:3])
    assert from_mapping[3]["T_ref"] == pytest.approx(from_object[3]["T_ref"])


def test_no_skew_information_at_all_refuses_rather_than_inventing():
    empty = HurstEstimate(
        H0=0.1,
        se=float("nan"),
        ci95=(float("nan"), float("nan")),
        r2=float("nan"),
        n_expiries=0,
        window=HURST_WINDOW,
        unstable=True,
        diagnostics={},
    )
    with pytest.raises(RBergomiInitializationError, match="rho\\*eta"):
        initialize_rbergomi_parameters(empty, None)


def test_a_missing_hurst_result_refuses():
    with pytest.raises(RBergomiInitializationError):
        initialize_rbergomi_parameters(None, None)


def test_sigma_atm_falls_back_to_the_xi0_curve():
    from app.model.calibration.rough_vol.forward_variance import (
        build_forward_variance_curve,
    )
    from app.model.calibration.rough_vol.variance_swap import VarianceSwapPoint

    H, eta, rho = 0.12, 1.4, -0.6
    xi0 = SIGMA_ATM * SIGMA_ATM
    curve = build_forward_variance_curve(
        [
            VarianceSwapPoint(T=T, k_var=xi0, k_var_trunc=xi0, n_puts=6, n_calls=6,
                              F=F_REF, D=1.0)
            for T in MATURITIES
        ]
    )
    point = make_skew_point(MATURITIES[0], H=H, eta=eta, rho=rho)
    blind = SkewPoint(
        T=point.T,
        psi=point.psi,
        se=point.se,
        n_strikes=point.n_strikes,
        window=point.window,
        curvature=point.curvature,
        sigma_atm=float("nan"),
        F=F_REF,
    )
    _, _, _, diag = initialize_rbergomi_parameters(
        make_hurst(H, [blind]),
        None,
        xi0_curve=curve,
        config=InitializerConfig(rho_prior_abs=abs(rho)),
    )
    assert diag["sigma_atm_source"] == "xi0_curve"
    assert diag["sigma_atm"] == pytest.approx(SIGMA_ATM, rel=1e-12)
    assert diag["eta0_curvature"] == pytest.approx(eta, rel=1e-10)


# ---------------------------------------------------------------------------
# Contracts and reporting
# ---------------------------------------------------------------------------


def test_initial_rbergomi_params_reuses_the_phase_three_contract():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    params, diag = initial_rbergomi_params(
        make_hurst(H, points), None, config=InitializerConfig(rho_prior_abs=abs(rho))
    )
    assert isinstance(params, RBergomiParams)
    assert params.H == pytest.approx(H, abs=1e-15)
    assert params.eta == pytest.approx(eta, rel=1e-12)
    assert params.rho == pytest.approx(rho, rel=1e-15)
    assert params.to_dict() == {
        "H": diag["H0"],
        "eta": diag["eta0"],
        "rho": diag["rho0"],
    }


def test_every_returned_triple_is_a_legal_simulator_input():
    """Whatever the input, the clips must leave RBergomiParams constructible."""
    for H, eta, rho in ((0.9, 8.0, -0.9), (0.001, 0.01, 0.2), (0.3, 2.0, -0.7)):
        points = make_skew_curve(H=max(H, 0.01), eta=eta, rho=rho)
        H0, eta0, rho0, _ = initialize_rbergomi_parameters(
            make_hurst(H, points), None
        )
        RBergomiParams(H=H0, eta=eta0, rho=rho0)


def test_diagnostics_survive_json_serialisation():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    round_tripped = json.loads(json.dumps(diag))
    assert round_tripped["T_ref"] == pytest.approx(diag["T_ref"])
    assert round_tripped["rho_eta_degeneracy"]["identified_quantity"] == "rho * eta"


def test_the_report_names_the_provenance_and_the_status():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    _, _, _, diag = initialize_rbergomi_parameters(make_hurst(H, points), None)
    report = initializer_report(diag)
    assert report["H0"] == pytest.approx(diag["H0"])
    assert report["eta0"] == pytest.approx(diag["eta0"])
    assert report["rho0"] == pytest.approx(diag["rho0"])
    assert report["T_ref_rule"] == T_REF_RULE_MIN_RELATIVE_SE
    assert "jamais un résultat de calibration" in report["warning_fr"]
    assert report["identified_product_rho_eta"] == pytest.approx(
        diag["rho_eta_degeneracy"]["identified_value"]
    )
    assert json.loads(json.dumps(report))
    assert diag["is_initial_guess"] is True
    assert "c_of_H_provenance" in diag
    assert "measured" in diag["c_of_H_provenance"]
    assert "INITIALISER ONLY" in diag["c_of_H_provenance"]


def test_the_configuration_rejects_impossible_settings():
    with pytest.raises(ValueError):
        InitializerConfig(rho_prior_abs=0.0)
    with pytest.raises(ValueError):
        InitializerConfig(eta_min=0.0)
    with pytest.raises(ValueError):
        InitializerConfig(eta_min=2.0, eta_max=1.0)
    with pytest.raises(ValueError):
        InitializerConfig(eta_max=99.0)
    with pytest.raises(ValueError):
        InitializerConfig(disagreement_factor=1.0)
    with pytest.raises(ValueError):
        InitializerConfig(t_ref_rule="nearest")
    with pytest.raises(ValueError):
        InitializerConfig(t_ref=-1.0)


def test_the_initialiser_does_not_mutate_its_inputs():
    H, eta, rho = 0.12, 1.4, -0.6
    points = make_skew_curve(H=H, eta=eta, rho=rho)
    estimate = make_hurst(H, points)
    before = json.dumps(estimate.to_dict(), sort_keys=True)
    initialize_rbergomi_parameters(estimate, None)
    assert json.dumps(estimate.to_dict(), sort_keys=True) == before
