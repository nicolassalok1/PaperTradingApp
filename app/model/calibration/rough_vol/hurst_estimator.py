"""
Short-maturity ATM skew and the **initial** Hurst estimate (spec 4.5).

.. warning::

   **THIS IS AN ASYMPTOTIC *INITIAL* ESTIMATE, NOT A CALIBRATION RESULT.**

   The power law ``psi(T) ~ A * T^{H - 1/2}`` is the *short-maturity asymptotic*
   behaviour of the ATM skew of a rough volatility model
   (Alos-Leon-Vives 2007, *Finance & Stochastics* 11(4); Fukasawa 2011,
   *Finance & Stochastics* 15(4); Bayer-Friz-Gatheral 2016, *Quantitative
   Finance* 16(6)). It holds as ``T -> 0``. Every listed maturity is finite, the
   window used here ends at a quarter of a year, and the observed exponent is
   therefore contaminated by the sub-leading terms of the expansion, by the
   term structure of the forward variance, and by quote noise. The number this
   module returns is a **starting point for the joint optimiser of spec 4.8**
   and a **diagnostic**; it must never be reported as "the" Hurst exponent of
   the surface, and it must never be pinned as a calibrated parameter.

   Constraint enforced everywhere below: ``0 < H < 1/2``.

What the module does
--------------------
1. **Per expiry** (:func:`estimate_atm_skew`) — a *weighted local quadratic*
   regression of the recomputed market implied vol against log-forward
   moneyness ``k = ln(K / F)``::

       sigma(k) ~ a + b * k + c * k^2

   evaluated at the money: ``psi(T) = d sigma / dk |_{k=0} = b`` and the ATM
   curvature ``d^2 sigma / dk^2 |_{k=0} = 2c``. The fit window is
   ``|k| <= window_c * sigma_ATM(T) * sqrt(T)`` with ``window_c = 1.5`` by
   default, and an expiry is **skipped** unless the window holds at least
   ``5`` strikes with at least ``2`` strictly on each side of the money.
   ``SE(b)`` comes from the weighted-regression covariance
   ``sigma_resid^2 * (X' W X)^{-1}`` — computed, never assumed.

2. **Across expiries** (:func:`fit_hurst_from_skew_points`) — a weighted least
   squares of ``log|psi(T)|`` on ``log T`` over a configurable short-maturity
   window (default ``[5/365, 0.25]``), cross-checked against an
   outlier-resistant Theil-Sen fit. ``H0 = slope + 1/2``.

Conventions and the two input conventions for the skew
------------------------------------------------------
The pipeline convention is the **log-moneyness** one:
``psi(T) = d sigma_BS(k, T) / dk |_{k=0}`` with ``k = ln(K / F)``.
The strike convention ``d sigma / dK |_{K=F}`` is accepted and converted
exactly, in both directions, by :func:`psi_from_strike_slope` and
:func:`strike_slope_from_psi`::

    k = ln(K / F)  =>  d sigma/dK = (d sigma/dk) / K
                   =>  psi = F * d sigma/dK |_{K=F}
                   =>  d^2 sigma/dK^2 |_{K=F} = (2c - b) / F^2

Both derivatives are also exposed directly on :class:`SkewPoint`
(:attr:`SkewPoint.dsigma_dK`, :attr:`SkewPoint.d2sigma_dK2`), so a consumer
that thinks in strikes never has to redo the chain rule. The regression itself
is *always* run in ``k``: it is dimensionless, centred at 0 by construction and
far better conditioned than raw strikes.

Resolving the self-referential fit window
-----------------------------------------
The window half-width needs ``sigma_ATM(T)``, and ``sigma_ATM(T)`` is what the
fit produces. This is resolved by a **robust seed plus a fixed-point pass**:

* seed: the implied vol of the quote **nearest** ``k = 0``. It needs no fit, it
  cannot fail on a well-formed expiry, and it is within a few skew-percent of
  the truth because it is by construction the closest strike to the money;
* pass 1: build the window from the seed, fit, and take ``sigma_ATM = a``, the
  fitted ``k = 0`` intercept — *not* the nearest quote. The intercept is the
  model's own ATM level and is the quantity consistent with ``psi = b``, which
  is also defined at ``k = 0``; the nearest quote is biased by the skew
  whenever the nearest strike is not exactly at the money;
* pass 2 (``SkewConfig.max_window_iterations = 2``): rebuild the window from
  ``a`` and refit. If the relative move of ``sigma_ATM`` is below
  ``SkewConfig.window_rtol`` the point is marked converged; otherwise the last
  fit is still reported, flagged :data:`FLAG_WINDOW_NOT_CONVERGED`.

The reported ``(fit, window)`` pair is always *consistent*: the reported
coefficients are the ones produced by the reported window. If a refined window
no longer satisfies the strike-count requirements, the previous valid fit is
kept and flagged :data:`FLAG_WINDOW_REVERTED` rather than silently widened.

Weights
-------
Spec 4.5 asks for vega-based weights or ``1 / spread^2``. The regression is on
**implied vol**, so the only dimensionally consistent reading of
``1 / spread^2`` is *in vol units*: a price half-spread ``s`` maps to an IV
uncertainty ``s / vega`` (first order), hence

    w_i = 1 / spread_iv^2 = (vega_i / spread_i)^2                ("iv_spread")

That is the default **when spreads are available**. Phase-1
:class:`~app.model.calibration.rough_vol.forward_curve.SurfacePoint` does not
carry a spread (only ``mid``), so spreads must be supplied by handing the
cleaned chains to :func:`build_skew_curve` /
:func:`estimate_hurst_from_skew` via ``clean_chains=``; the lookup is keyed on
the contract symbol first, then on ``(option_type, strike, T)``. Without them
the documented fallback is

    w_i = vega_i^2                                               ("vega")

which is the same weighting under the assumption of a homoskedastic price
noise. ``"price_spread"`` (the literal ``1 / spread^2``, in price units) and
``"uniform"`` are available for sensitivity work. Weights are normalised to
mean 1 before the solve — the covariance formula
``sigma_resid^2 (X'WX)^{-1}`` is exactly invariant to that rescaling, so the
reported standard errors do not depend on the weight normalisation.

Guard rails (spec 4.5)
----------------------
* ``|psi| < psi_floor`` (default ``1e-6``) is dropped before any log is taken;
* the skew sign must be constant across the window — a flip is flagged
  (:data:`FLAG_SIGN_FLIP`) and the fit continues on ``|psi|`` with the sign
  census in the diagnostics;
* at least ``3`` usable expiries;
* ``SE(H0)``, a 95 % CI and ``R^2`` are always reported;
* the WLS slope is cross-checked against a **Theil-Sen** slope (median of the
  pairwise slopes, implemented in-module and exact for the 3-8 points this
  regression sees); both are reported and a gap larger than one ``SE`` raises
  :data:`FLAG_ROBUST_DISAGREEMENT` **before** the rejection thresholds run;
* rejection as UNSTABLE when ``R^2 < r2_min`` (0.6), ``SE(H0) > 0.15`` or
  ``H0`` outside ``(0.01, 0.49)``. An UNSTABLE estimate returns
  ``H0 = 0.1`` with ``unstable=True``: a **fallback default for the optimiser
  start**, never a result. The measured value stays visible in
  ``diagnostics["H0_estimated"]``.

Phase-2 carry-over
------------------
``VarianceSwapPoint.diagnostics.discretisation_bias`` is a strike-grid bias
that decays like ``1/T`` and is therefore largest exactly inside this
regression window. ``K_var`` is not consumed here, but an expiry whose
variance-swap point carries ``FLAG_COARSE_STRIKE_LADDER`` gets
:data:`FLAG_COARSE_LADDER_IN_WINDOW` on its :class:`SkewPoint` and is listed in
``diagnostics["coarse_strike_ladder"]`` so the Phase-4 report can see that the
``H`` regression leaned on a coarse ladder.

Layering / purity: model layer only, no side effects, no network, no RNG.
Everything returned is plain ``float`` / ``int`` / ``bool`` / ``str`` / ``list``
/ ``dict`` so it survives the controller's ``_json_safe`` unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from app.model.calibration.rough_vol.chain_cleaning import CleanChain
from app.model.calibration.rough_vol.forward_curve import ForwardPoint, SurfacePoint
from app.model.calibration.rough_vol.variance_swap import (
    FLAG_COARSE_STRIKE_LADDER,
    VarianceSwapCurve,
    VarianceSwapPoint,
)

# ---------------------------------------------------------------------------
# Constants, reason codes and flags
# ---------------------------------------------------------------------------

#: Fallback starting value handed to the optimiser when the estimate is
#: rejected. **Never** a measurement — always paired with ``unstable=True``.
FALLBACK_H0 = 0.1

#: Hard admissibility interval of the rough-volatility Hurst exponent.
H_LOWER_BOUND = 0.0
H_UPPER_BOUND = 0.5

#: Normal quantile used for the reported 95 % interval. Spec 4.5 pins ``1.96``;
#: the exact quantile is ``1.959964``. The Student-t interval that a 3-8 point
#: regression really deserves is reported alongside, in
#: ``diagnostics["ci95_student_t"]``.
Z95 = 1.96

# --- per-expiry skew failures ---------------------------------------------
REASON_INVALID_MATURITY = "invalid_maturity"
REASON_NO_USABLE_QUOTE = "no_usable_quote"
REASON_TOO_FEW_STRIKES = "too_few_strikes_in_window"
REASON_UNBALANCED_WINGS = "fewer_than_two_strikes_on_one_side"
REASON_SINGULAR_DESIGN = "singular_local_design_matrix"
REASON_NON_FINITE_FIT = "non_finite_local_fit"

SKEW_REASON_LABELS_FR: dict[str, str] = {
    REASON_INVALID_MATURITY: "Maturité invalide ou nulle",
    REASON_NO_USABLE_QUOTE: "Aucune cotation exploitable sur cette échéance",
    REASON_TOO_FEW_STRIKES: "Moins de 5 strikes dans la fenêtre ATM",
    REASON_UNBALANCED_WINGS: "Moins de 2 strikes d'un côté de la monnaie",
    REASON_SINGULAR_DESIGN: "Matrice de régression locale singulière",
    REASON_NON_FINITE_FIT: "Ajustement quadratique local non fini",
}

# --- Hurst-regression rejections -------------------------------------------
REASON_TOO_FEW_EXPIRIES = "too_few_usable_expiries"
REASON_DEGENERATE_REGRESSION = "degenerate_regression"
REASON_LOW_R2 = "r_squared_below_minimum"
REASON_SE_TOO_LARGE = "standard_error_above_maximum"
REASON_H0_OUT_OF_RANGE = "hurst_outside_admissible_range"

HURST_REASON_LABELS_FR: dict[str, str] = {
    REASON_TOO_FEW_EXPIRIES: "Moins de 3 échéances exploitables dans la fenêtre courte",
    REASON_DEGENERATE_REGRESSION: "Régression log-log dégénérée",
    REASON_LOW_R2: "R² de la régression log-log sous le seuil",
    REASON_SE_TOO_LARGE: "Erreur-type de H0 au-dessus du seuil",
    REASON_H0_OUT_OF_RANGE: "H0 hors de l'intervalle admissible (0.01, 0.49)",
}

# --- flags ------------------------------------------------------------------
#: The refined ATM window did not settle within ``window_rtol``.
FLAG_WINDOW_NOT_CONVERGED = "atm_window_not_converged"
#: A refined window lost the strike-count requirement; the previous fit is kept.
FLAG_WINDOW_REVERTED = "atm_window_reverted"
#: At least one zero-bid (one-sided) quote entered the local fit.
FLAG_ONE_SIDED_IN_WINDOW = "one_sided_quote_in_window"
#: Spreads were requested but unavailable; vega weights were used instead.
FLAG_WEIGHTS_FELL_BACK_TO_VEGA = "weights_fell_back_to_vega"
#: The forward carried by the surface points disagrees with the forward curve.
FLAG_FORWARD_MISMATCH = "forward_point_mismatch"
#: No forward-curve point matched this maturity; the surface's own F was used.
FLAG_NO_FORWARD_POINT = "no_matching_forward_point"
#: Phase-2 carry-over: this expiry's variance swap ran on a coarse strike ladder.
FLAG_COARSE_LADDER_IN_WINDOW = "coarse_strike_ladder_in_window"
#: The skew changed sign inside the short-maturity window.
FLAG_SIGN_FLIP = "skew_sign_flip"
#: At least one expiry was dropped by the ``|psi| < psi_floor`` log guard.
FLAG_PSI_NEAR_ZERO_DROPPED = "psi_near_zero_dropped"
#: WLS and the robust fit disagree by more than one SE.
FLAG_ROBUST_DISAGREEMENT = "wls_robust_disagreement"
#: At least one ``SE(psi)`` was below ``se_floor`` and was floored for weighting.
FLAG_SE_FLOORED = "skew_se_floored"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkewConfig:
    """
    Thresholds of the per-expiry local quadratic skew fit.

    Attributes
    ----------
    window_c:
        Half-width multiplier: the fit window is
        ``|k| <= window_c * sigma_ATM(T) * sqrt(T)``. ``1.5`` (spec default)
        keeps roughly the 1.5-standard-deviation core of the distribution,
        where the quadratic expansion of ``sigma(k)`` is legitimate.
    min_strikes:
        Minimum number of strikes inside the window (spec: 5).
    min_strikes_per_side:
        Minimum number of strikes **strictly** on each side of ``k = 0``
        (spec: 2). A strike exactly at ``k = 0`` counts towards
        ``min_strikes`` but towards neither side.
    weight_scheme:
        ``"auto"`` (default) resolves to ``"iv_spread"`` when a spread is known
        for every usable quote of the expiry, else to ``"vega"``. The explicit
        values are ``"iv_spread"``, ``"price_spread"``, ``"vega"`` and
        ``"uniform"``. See the module docstring.
    spread_floor:
        Floor on the price spread used in the weights, in quote currency. Keeps
        a zero-spread synthetic chain well-posed.
    min_weight_rel:
        Weights are floored at ``min_weight_rel * max(weight)`` so a deep-wing
        quote with a vanishing vega cannot produce an exactly singular design.
    max_window_iterations:
        Number of local fits allowed by the window fixed point (``2`` = seed
        fit plus one refinement, the spec's "one fixed-point pass").
    window_rtol:
        Relative move of ``sigma_ATM`` below which the window is converged.
    min_half_width:
        Absolute floor on the window half-width in ``k`` units, so a
        near-expiry with a very low ATM vol still admits a window.
    include_one_sided:
        Keep zero-bid quotes (flagged upstream). They are flagged here too, via
        :data:`FLAG_ONE_SIDED_IN_WINDOW`.
    forward_match_rtol:
        Relative maturity tolerance when pairing a surface expiry with a
        :class:`~app.model.calibration.rough_vol.forward_curve.ForwardPoint`.
    forward_mismatch_rtol:
        Relative gap between the surface's ``F`` and the forward curve's ``F``
        above which :data:`FLAG_FORWARD_MISMATCH` is raised. Diagnostic only.
    """

    window_c: float = 1.5
    min_strikes: int = 5
    min_strikes_per_side: int = 2
    weight_scheme: str = "auto"
    spread_floor: float = 1e-4
    min_weight_rel: float = 1e-8
    max_window_iterations: int = 2
    window_rtol: float = 1e-3
    min_half_width: float = 1e-4
    include_one_sided: bool = True
    forward_match_rtol: float = 1e-6
    forward_mismatch_rtol: float = 1e-6


@dataclass(frozen=True)
class HurstConfig:
    """
    Thresholds of the ``log|psi(T)|`` vs ``log T`` regression.

    Attributes
    ----------
    short_maturity_window:
        ``(T_min, T_max)`` in years, inclusive on both ends. Default
        ``(5/365, 0.25)`` — the asymptotic regime is a short-maturity
        statement, and 5 calendar days is the shortest maturity whose quotes
        are not dominated by the pin risk of the expiring series.
    psi_floor:
        ``|psi| < psi_floor`` is dropped before any logarithm (spec: ``1e-6``).
    min_expiries:
        Minimum number of usable expiries (spec: 3).
    r2_min, se_max:
        Rejection thresholds on the regression quality (spec: 0.6 and 0.15).
    h_min, h_max:
        Admissible open interval for ``H0`` (spec: ``(0.01, 0.49)``). Outside
        it the estimate is rejected; the value is also clipped into
        ``[h_min, h_max]`` and reported as ``H0_clipped`` in the diagnostics.
    fallback_H0:
        Value returned when the estimate is rejected. Always accompanied by
        ``unstable=True``.
    log_weight_mode:
        ``"delta"`` (default) uses the delta-method transform of ``SE(psi)``
        into the log space the regression actually lives in,
        ``SE(log|psi|) = SE(psi) / |psi|``, hence
        ``w = (|psi| / SE(psi))^2``. ``"raw_psi"`` is the literal
        ``w = 1 / SE(psi)^2`` of the spec, kept for sensitivity work;
        ``"uniform"`` disables weighting.
    se_floor:
        Floor on ``SE(psi)`` used for the weights only. A regression SE of
        exactly ``0`` (an expiry whose quotes lie on an exact parabola) is not
        infinite precision, it is an absent residual; without the floor it
        would receive an infinite weight.
    robust_disagreement_atol:
        Absolute floor, in slope units, on the WLS/robust disagreement test.
        With an exact fit ``SE(slope)`` collapses to rounding noise and any
        float difference would fire the flag; ``1e-8`` in slope units is
        ``1e-8`` in ``H``, far below anything meaningful.
    """

    short_maturity_window: tuple[float, float] = (5.0 / 365.0, 0.25)
    psi_floor: float = 1e-6
    min_expiries: int = 3
    r2_min: float = 0.6
    se_max: float = 0.15
    h_min: float = 0.01
    h_max: float = 0.49
    fallback_H0: float = FALLBACK_H0
    log_weight_mode: str = "delta"
    se_floor: float = 1e-8
    robust_disagreement_atol: float = 1e-8


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard normal density (inlined; scipy is not needed for one pdf)."""
    return _INV_SQRT_TWO_PI * math.exp(-0.5 * x * x)


def black76_vega(*, F: float, K: float, T: float, D: float, vol: float) -> float:
    """
    Black-76 vega ``dC/dsigma = D * F * phi(d1) * sqrt(T)``.

    Returns ``0.0`` on a degenerate input rather than raising: a vega is only
    ever used here as a *weight*, and a zero weight is the correct statement
    about a quote that carries no vol information.
    """
    F = float(F)
    K = float(K)
    T = float(T)
    D = float(D)
    vol = float(vol)
    if not (
        math.isfinite(F)
        and math.isfinite(K)
        and math.isfinite(T)
        and math.isfinite(D)
        and math.isfinite(vol)
    ):
        return 0.0
    if F <= 0.0 or K <= 0.0 or T <= 0.0 or vol <= 0.0 or D <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    return D * F * _norm_pdf(d1) * sqrt_t


def psi_from_strike_slope(dsigma_dK: float, *, F: float) -> float:
    """
    Convert the strike-convention ATM skew into the pipeline convention.

    ``k = ln(K / F)`` gives ``d sigma/dK = (d sigma/dk) / K``, so at ``K = F``
    the log-moneyness skew is ``psi = F * d sigma/dK|_{K=F}``. Exact, not an
    approximation.
    """
    slope = float(dsigma_dK)
    fwd = float(F)
    if not (math.isfinite(slope) and math.isfinite(fwd)) or fwd <= 0.0:
        return float("nan")
    return fwd * slope


def strike_slope_from_psi(psi: float, *, F: float) -> float:
    """Inverse of :func:`psi_from_strike_slope`: ``d sigma/dK|_{K=F} = psi / F``."""
    value = float(psi)
    fwd = float(F)
    if not (math.isfinite(value) and math.isfinite(fwd)) or fwd <= 0.0:
        return float("nan")
    return value / fwd


def theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """
    Exact Theil-Sen estimator: median of the pairwise slopes.

    Implemented in-module rather than delegated to ``scipy.stats.theilslopes``
    (which does exist in the pinned scipy 1.11.4): the estimator is three lines
    for the 3-8 points this module regresses, and an in-module implementation
    keeps the numbers independent of any scipy behaviour change. The intercept
    is the **joint** one, ``median(y - slope * x)`` — the convention
    ``scipy.stats.theilslopes(..., method="joint")`` implements, not the
    ``"separate"`` default ``median(y) - slope * median(x)``.

    Returns ``(nan, nan)`` when fewer than two distinct abscissae are given.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = int(xa.size)
    if n != int(ya.size) or n < 2:
        return (float("nan"), float("nan"))
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xa[j] - xa[i]
            if dx == 0.0:
                continue
            slopes.append(float((ya[j] - ya[i]) / dx))
    if not slopes:
        return (float("nan"), float("nan"))
    slope = float(np.median(np.asarray(slopes, dtype=float)))
    intercept = float(np.median(ya - slope * xa))
    return (slope, intercept)


@dataclass(frozen=True)
class WeightedFit:
    """
    Result of :func:`weighted_polynomial_fit`, in the original ``x`` units.

    ``sigma_resid`` is the textbook weighted residual scale
    ``sqrt(r' W r / (n - p))`` computed with the **caller's own** weights: it is
    the ``sigma_resid`` of ``cov = sigma_resid^2 (X'WX)^{-1}`` and, like any
    weighted residual scale, it moves with the arbitrary scale of ``W``.
    ``resid_rmse`` is the plain unweighted ``sqrt(mean(r^2))``, in the units of
    ``y`` — the interpretable one, and the one reported downstream.
    """

    coeffs: tuple[float, ...]
    se: tuple[float, ...]
    cov: tuple[tuple[float, ...], ...]
    sigma_resid: float
    resid_rmse: float
    r_squared: float
    dof: int
    n: int
    condition_number: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coeffs": [float(c) for c in self.coeffs],
            "se": [float(s) for s in self.se],
            "sigma_resid": float(self.sigma_resid),
            "resid_rmse": float(self.resid_rmse),
            "r_squared": float(self.r_squared),
            "dof": int(self.dof),
            "n": int(self.n),
            "condition_number": float(self.condition_number),
        }


def weighted_polynomial_fit(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    degree: int = 2,
    max_condition_number: float = 1e12,
) -> WeightedFit:
    """
    Weighted least squares of ``y`` on a polynomial of ``x``, with covariance.

    The covariance is the textbook ``sigma_resid^2 * (X' W X)^{-1}`` with
    ``sigma_resid^2 = (r' W r) / (n - p)``; nothing is faked and nothing is
    approximated. Two numerical precautions:

    * the abscissa is rescaled to ``u = x / max|x|`` before the solve and the
      coefficients and covariance are mapped back exactly with the diagonal
      similarity ``S = diag(scale^-j)``. Raw ``k`` values of order ``1e-2``
      make the ``[1, k, k^2]`` design matrix span eight orders of magnitude;
      the rescaled one is well conditioned;
    * the weights are normalised to mean 1. Both ``sigma_resid^2`` and
      ``(X'WX)^{-1}`` scale with the weight normalisation, in opposite
      directions, so the covariance is *exactly* invariant — the normalisation
      only buys numerical range.

    Raises
    ------
    ValueError
        On non-finite input, non-positive weights, or fewer points than
        ``degree + 2`` (a zero-residual-degree fit has no ``sigma_resid``).
    numpy.linalg.LinAlgError
        When the normal matrix is singular or worse conditioned than
        ``max_condition_number``.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = int(xa.size)
    p = int(degree) + 1
    if ya.size != n:
        raise ValueError("x and y must have the same length")
    if weights is None:
        wa = np.ones(n, dtype=float)
    else:
        wa = np.asarray(weights, dtype=float)
        if wa.size != n:
            raise ValueError("weights must have the same length as x")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya)) and np.all(np.isfinite(wa))):
        raise ValueError("weighted_polynomial_fit: non-finite input")
    if np.any(wa <= 0.0):
        raise ValueError("weighted_polynomial_fit: weights must be strictly positive")
    dof = n - p
    if dof <= 0:
        raise ValueError(
            f"weighted_polynomial_fit: {n} points cannot support a degree-{degree} fit "
            "with a residual variance"
        )

    scale = float(np.max(np.abs(xa)))
    if not (math.isfinite(scale) and scale > 0.0):
        scale = 1.0
    u = xa / scale
    design = np.vander(u, p, increasing=True)

    w_mean = float(np.mean(wa))
    w_norm = wa / w_mean
    xtw = design.T * w_norm
    normal = xtw @ design
    condition_number = float(np.linalg.cond(normal))
    if not math.isfinite(condition_number) or condition_number > float(max_condition_number):
        raise np.linalg.LinAlgError(
            f"weighted_polynomial_fit: ill-conditioned normal matrix (cond={condition_number:.3e})"
        )

    beta_u = np.linalg.solve(normal, xtw @ ya)
    resid = ya - design @ beta_u
    rss_w = float(resid @ (w_norm * resid))
    s2 = rss_w / float(dof)
    cov_u = s2 * np.linalg.inv(normal)
    # ``s2`` above is expressed in normalised-weight units; ``cov`` is invariant
    # because ``(X'WX)^{-1}`` carries the inverse factor, but the *reported*
    # residual scale is put back on the caller's own weight scale so it matches
    # the textbook ``sqrt(r' W r / (n - p))``.
    s2_raw = s2 * w_mean
    resid_rmse = float(math.sqrt(float(resid @ resid) / float(n)))

    back = np.array([scale ** (-j) for j in range(p)], dtype=float)
    beta = beta_u * back
    cov = cov_u * np.outer(back, back)
    diag = np.diag(cov)
    se = np.sqrt(np.where(diag >= 0.0, diag, np.nan))

    y_bar = float(np.sum(w_norm * ya) / np.sum(w_norm))
    ss_tot = float(np.sum(w_norm * (ya - y_bar) ** 2))
    if ss_tot > 0.0:
        r2 = 1.0 - rss_w / ss_tot
    else:
        # Constant response: the fit is perfect and explains a zero variance.
        r2 = 1.0 if rss_w <= 1e-24 else 0.0
    r2 = float(min(1.0, r2))

    return WeightedFit(
        coeffs=tuple(float(v) for v in beta),
        se=tuple(float(v) for v in se),
        cov=tuple(tuple(float(v) for v in row) for row in cov),
        sigma_resid=float(math.sqrt(s2_raw)) if s2_raw >= 0.0 else float("nan"),
        resid_rmse=resid_rmse,
        r_squared=r2,
        dof=int(dof),
        n=n,
        condition_number=condition_number,
    )


# ---------------------------------------------------------------------------
# Spec-6 data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkewPoint:
    """
    One expiry's ATM skew (spec 6 data contract).

    ``psi`` is ``d sigma/dk|_{k=0}`` (the ``b`` coefficient of the local
    quadratic), ``se`` its regression standard error, ``n_strikes`` the number
    of strikes that entered the fit, and ``window`` the ``(-h, +h)``
    log-moneyness window actually used.

    ``curvature`` is ``d^2 sigma/dk^2|_{k=0} = 2c``, exposed for the Phase-4
    stage that wants it; ``quad_coeff`` is the raw ``c``.
    """

    T: float
    psi: float
    se: float
    n_strikes: int
    window: tuple[float, float]
    curvature: float = float("nan")
    se_curvature: float = float("nan")
    quad_coeff: float = float("nan")
    sigma_atm: float = float("nan")
    se_sigma_atm: float = float("nan")
    sigma_atm_seed: float = float("nan")
    half_width: float = float("nan")
    n_left: int = 0
    n_right: int = 0
    k_min: float = float("nan")
    k_max: float = float("nan")
    r_squared: float = float("nan")
    sigma_resid: float = float("nan")
    resid_rmse: float = float("nan")
    dof: int = 0
    iterations: int = 0
    converged: bool = False
    weight_scheme: str = "unknown"
    F: float = float("nan")
    flags: tuple[str, ...] = ()

    @property
    def dsigma_dK(self) -> float:
        """Strike-convention skew ``d sigma/dK|_{K=F} = psi / F``."""
        return strike_slope_from_psi(self.psi, F=self.F)

    @property
    def d2sigma_dK2(self) -> float:
        """Strike-convention curvature ``(2c - b) / F^2`` at ``K = F``."""
        if not (math.isfinite(self.F) and self.F > 0.0):
            return float("nan")
        if not (math.isfinite(self.curvature) and math.isfinite(self.psi)):
            return float("nan")
        return (self.curvature - self.psi) / (self.F * self.F)

    @property
    def se_log_abs_psi(self) -> float:
        """Delta-method SE of ``log|psi|``: ``SE(psi) / |psi|``."""
        if not (math.isfinite(self.se) and math.isfinite(self.psi)) or self.psi == 0.0:
            return float("nan")
        return float(self.se) / abs(float(self.psi))

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "psi": float(self.psi),
            "se": float(self.se),
            "n_strikes": int(self.n_strikes),
            "window": [float(self.window[0]), float(self.window[1])],
            "curvature": float(self.curvature),
            "se_curvature": float(self.se_curvature),
            "quad_coeff": float(self.quad_coeff),
            "sigma_atm": float(self.sigma_atm),
            "se_sigma_atm": float(self.se_sigma_atm),
            "sigma_atm_seed": float(self.sigma_atm_seed),
            "half_width": float(self.half_width),
            "n_left": int(self.n_left),
            "n_right": int(self.n_right),
            "k_min": float(self.k_min),
            "k_max": float(self.k_max),
            "r_squared": float(self.r_squared),
            "sigma_resid": float(self.sigma_resid),
            "resid_rmse": float(self.resid_rmse),
            "dof": int(self.dof),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "weight_scheme": str(self.weight_scheme),
            "F": float(self.F),
            "dsigma_dK": float(self.dsigma_dK),
            "d2sigma_dK2": float(self.d2sigma_dK2),
            "se_log_abs_psi": float(self.se_log_abs_psi),
            "flags": [str(f) for f in self.flags],
        }


@dataclass(frozen=True)
class SkewFailure:
    """A skipped expiry: a reason code, a French message and what was known."""

    T: float
    reason: str
    message_fr: str
    detail: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "reason": str(self.reason),
            "reason_fr": SKEW_REASON_LABELS_FR.get(self.reason, self.reason),
            "message_fr": str(self.message_fr),
            "detail": {str(k): float(v) for k, v in dict(self.detail or {}).items()},
        }


@dataclass(frozen=True)
class HurstEstimate:
    """
    The initial Hurst estimate (spec 6 data contract).

    ``H0`` is the estimate **only when** ``unstable`` is ``False``. When
    ``unstable`` is ``True`` the field carries
    :data:`FALLBACK_H0` — a starting value for the optimiser, not a
    measurement — and the rejected estimate lives in
    ``diagnostics["H0_estimated"]``.

    ``se`` / ``ci95`` / ``r2`` are the *measured* regression quantities: they
    are the evidence on which an unstable verdict rests, so they are reported
    in both cases. When ``unstable`` is ``True`` they describe
    ``diagnostics["H0_estimated"]``, **not** ``H0``.
    """

    H0: float
    se: float
    ci95: tuple[float, float]
    r2: float
    n_expiries: int
    window: tuple[float, float]
    unstable: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """``True`` only for an accepted estimate."""
        return not bool(self.unstable)

    @property
    def message_fr(self) -> str:
        """Short French verdict for the Phase-5 report."""
        if not self.unstable:
            return (
                f"Estimation initiale H0 = {self.H0:.4f} "
                f"(IC95 [{self.ci95[0]:.4f}, {self.ci95[1]:.4f}], R² = {self.r2:.3f}) "
                f"sur {self.n_expiries} échéances — estimation asymptotique de départ, "
                "pas une calibration."
            )
        reasons = self.diagnostics.get("rejection_reasons_fr") or []
        detail = " ; ".join(str(x) for x in reasons) if reasons else "raison non renseignée"
        return (
            f"Estimation de H instable — valeur de repli H0 = {self.H0:.2f} "
            f"utilisée uniquement comme point de départ ({detail})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "H0": float(self.H0),
            "se": float(self.se),
            "ci95": [float(self.ci95[0]), float(self.ci95[1])],
            "r2": float(self.r2),
            "n_expiries": int(self.n_expiries),
            "window": [float(self.window[0]), float(self.window[1])],
            "unstable": bool(self.unstable),
            "H0_is_fallback": bool(self.unstable),
            "is_usable": bool(self.is_usable),
            "estimator": "asymptotic_atm_skew_power_law",
            "message_fr": self.message_fr,
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _group_by_maturity(
    points: Sequence[SurfacePoint],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> list[list[SurfacePoint]]:
    """Split a flat list of surface points into per-expiry buckets, sorted by T."""
    usable = [p for p in points if math.isfinite(float(p.T))]
    usable.sort(key=lambda p: (float(p.T), float(p.k), float(p.K)))
    groups: list[list[SurfacePoint]] = []
    for point in usable:
        t = float(point.T)
        if groups:
            t_ref = float(groups[-1][0].T)
            if abs(t - t_ref) <= max(atol, rtol * abs(t_ref)):
                groups[-1].append(point)
                continue
        groups.append([point])
    return groups


def _normalise_option_surface(option_surface: Any) -> list[list[SurfacePoint]]:
    """
    Accept every shape Phase 1/2 naturally produces and return expiry buckets.

    Supported inputs:

    * a flat ``Sequence[SurfacePoint]`` (several expiries mixed together);
    * a ``Mapping[T, Sequence[SurfacePoint]]`` (keys are ignored, the maturity
      carried by the points is authoritative);
    * a ``Sequence[Sequence[SurfacePoint]]``, i.e. one
      ``build_otm_surface(...)[0]`` per expiry.

    In every case the points are regrouped by their own ``T``, so a caller
    cannot accidentally merge two expiries by passing the wrong nesting.
    """
    if option_surface is None:
        return []
    if isinstance(option_surface, Mapping):
        buckets: list[Any] = list(option_surface.values())
    else:
        buckets = list(option_surface)
    flat: list[SurfacePoint] = []
    for item in buckets:
        if isinstance(item, SurfacePoint):
            flat.append(item)
            continue
        if isinstance(item, (str, bytes)) or not hasattr(item, "__iter__"):
            raise TypeError(
                "option_surface must hold SurfacePoint objects, or sequences/mappings of them; "
                f"got {type(item).__name__}"
            )
        for sub in item:
            if not isinstance(sub, SurfacePoint):
                raise TypeError(
                    "option_surface must hold SurfacePoint objects; "
                    f"got {type(sub).__name__}"
                )
            flat.append(sub)
    return _group_by_maturity(flat)


def _forward_points(forward_curve: Any) -> list[ForwardPoint]:
    """Accept a sequence or a mapping of :class:`ForwardPoint`, or ``None``."""
    if forward_curve is None:
        return []
    if isinstance(forward_curve, ForwardPoint):
        return [forward_curve]
    values = list(forward_curve.values()) if isinstance(forward_curve, Mapping) else list(forward_curve)
    points: list[ForwardPoint] = []
    for value in values:
        if not isinstance(value, ForwardPoint):
            raise TypeError(f"forward_curve must hold ForwardPoint objects; got {type(value).__name__}")
        points.append(value)
    points.sort(key=lambda p: float(p.T))
    return points


def _match_forward(T: float, points: Sequence[ForwardPoint], *, rtol: float) -> ForwardPoint | None:
    """Nearest forward point within a relative maturity tolerance, else ``None``."""
    best: ForwardPoint | None = None
    best_gap = float("inf")
    for point in points:
        gap = abs(float(point.T) - float(T))
        if gap < best_gap:
            best_gap = gap
            best = point
    if best is None:
        return None
    tol = max(1e-12, float(rtol) * max(abs(float(T)), 1e-12))
    return best if best_gap <= tol else None


def build_spread_lookup(clean_chains: Sequence[CleanChain] | None) -> dict[Any, float]:
    """
    Map cleaned quotes to their absolute price spread, for ``1/spread^2`` weights.

    Phase-1 ``SurfacePoint`` carries the mid but not the spread, so the spread
    has to come back from the chains the surface was built from. Two keys are
    emitted per quote: the contract symbol (primary, unique and stable) and
    ``("strike", option_type, round(K, 10), round(T, 10))`` where ``T`` is the
    **chain** maturity — the same value ``build_forward_point`` stamps on the
    ``ForwardPoint`` and hence on every ``SurfacePoint`` of the expiry.
    """
    lookup: dict[Any, float] = {}
    if not clean_chains:
        return lookup
    for chain in clean_chains:
        chain_t = float(chain.T)
        for quote in tuple(chain.calls) + tuple(chain.puts):
            spread = float(quote.spread_abs)
            if not math.isfinite(spread) or spread < 0.0:
                continue
            symbol = quote.contract_symbol
            if symbol:
                lookup[("symbol", str(symbol))] = spread
            lookup[("strike", str(quote.option_type), round(float(quote.strike), 10), round(chain_t, 10))] = spread
    return lookup


def _lookup_spread(point: SurfacePoint, lookup: Mapping[Any, float] | None) -> float:
    if not lookup:
        return float("nan")
    symbol = point.contract_symbol
    if symbol:
        value = lookup.get(("symbol", str(symbol)))
        if value is not None:
            return float(value)
    value = lookup.get(
        ("strike", str(point.option_type), round(float(point.K), 10), round(float(point.T), 10))
    )
    return float(value) if value is not None else float("nan")


def coarse_ladder_maturities(variance_curve: Any) -> list[tuple[float, float]]:
    """
    Phase-2 carry-over: maturities whose variance swap ran on a coarse ladder.

    Accepts a :class:`~app.model.calibration.rough_vol.variance_swap.VarianceSwapCurve`,
    a sequence of :class:`~app.model.calibration.rough_vol.variance_swap.VarianceSwapPoint`,
    or ``None``. Returns ``[(T, discretisation_bias), ...]``.
    """
    if variance_curve is None:
        return []
    if isinstance(variance_curve, VarianceSwapCurve):
        points: Sequence[Any] = variance_curve.points
    else:
        points = list(variance_curve)
    out: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, VarianceSwapPoint):
            continue
        if FLAG_COARSE_STRIKE_LADDER not in tuple(point.flags):
            continue
        out.append((float(point.T), float(point.diagnostics.discretisation_bias)))
    out.sort(key=lambda item: item[0])
    return out


# ---------------------------------------------------------------------------
# Stage 1 — the per-expiry local quadratic skew
# ---------------------------------------------------------------------------


def _quote_weights(
    points: Sequence[SurfacePoint],
    *,
    config: SkewConfig,
    spread_lookup: Mapping[Any, float] | None,
) -> tuple[np.ndarray, str, bool]:
    """
    Per-quote regression weights. Returns ``(weights, scheme_used, fell_back)``.

    See the module docstring for the interpretation of ``1/spread^2`` in vol
    units. Zero-vega quotes would produce a zero weight and, in the limit, a
    singular design; weights are therefore floored relative to their maximum.
    """
    scheme = str(config.weight_scheme)
    vega = np.array(
        [black76_vega(F=p.F, K=p.K, T=p.T, D=p.D, vol=p.iv) for p in points], dtype=float
    )
    spreads = np.array([_lookup_spread(p, spread_lookup) for p in points], dtype=float)
    have_spreads = bool(np.all(np.isfinite(spreads)))

    fell_back = False
    if scheme == "auto":
        scheme = "iv_spread" if have_spreads else "vega"
    elif scheme in ("iv_spread", "price_spread") and not have_spreads:
        scheme = "vega"
        fell_back = True

    floor = max(float(config.spread_floor), 1e-12)
    if scheme == "iv_spread":
        raw = (vega / np.maximum(spreads, floor)) ** 2
    elif scheme == "price_spread":
        raw = 1.0 / np.maximum(spreads, floor) ** 2
    elif scheme == "vega":
        raw = vega**2
    elif scheme == "uniform":
        raw = np.ones(len(points), dtype=float)
    else:
        raise ValueError(
            "SkewConfig.weight_scheme must be one of "
            "'auto', 'iv_spread', 'price_spread', 'vega', 'uniform'; "
            f"got {config.weight_scheme!r}"
        )

    raw = np.where(np.isfinite(raw), raw, 0.0)
    top = float(np.max(raw)) if raw.size else 0.0
    if not (math.isfinite(top) and top > 0.0):
        # Every quote carries a zero vega (or a degenerate spread): the only
        # honest weighting left is uniform, and it is reported as such.
        return (np.ones(len(points), dtype=float), "uniform", True)
    return (np.maximum(raw, float(config.min_weight_rel) * top), scheme, fell_back)


def estimate_atm_skew(
    surface_points: Sequence[SurfacePoint],
    *,
    F: float | None = None,
    config: SkewConfig | None = None,
    spread_lookup: Mapping[Any, float] | None = None,
    extra_flags: Sequence[str] = (),
) -> SkewPoint | SkewFailure:
    """
    Weighted local quadratic fit of ``sigma(k)`` for **one** expiry.

    Returns a :class:`SkewPoint` or, when the expiry does not meet spec 4.5's
    strike requirements, a :class:`SkewFailure` — never a number that pretends
    the expiry was usable.
    """
    cfg = config or SkewConfig()
    points = list(surface_points)
    if not points:
        return SkewFailure(
            T=float("nan"),
            reason=REASON_NO_USABLE_QUOTE,
            message_fr="Échéance vide : aucune cotation transmise.",
        )

    T = float(points[0].T)
    if not (math.isfinite(T) and T > 0.0):
        return SkewFailure(
            T=T,
            reason=REASON_INVALID_MATURITY,
            message_fr="Maturité invalide : la fenêtre ATM ne peut pas être construite.",
            detail={"T": T},
        )

    usable = [
        p
        for p in points
        if math.isfinite(float(p.iv))
        and float(p.iv) > 0.0
        and math.isfinite(float(p.k))
        and (cfg.include_one_sided or not bool(p.one_sided))
    ]
    if len(usable) < int(cfg.min_strikes):
        return SkewFailure(
            T=T,
            reason=REASON_TOO_FEW_STRIKES if usable else REASON_NO_USABLE_QUOTE,
            message_fr=(
                f"Échéance ignorée : seulement {len(usable)} cotation(s) exploitable(s), "
                f"{cfg.min_strikes} requises."
            ),
            detail={"n_usable": float(len(usable)), "min_strikes": float(cfg.min_strikes)},
        )

    forward = float(F) if F is not None else float(usable[0].F)

    k_all = np.array([float(p.k) for p in usable], dtype=float)
    iv_all = np.array([float(p.iv) for p in usable], dtype=float)
    weights_all, scheme_used, fell_back = _quote_weights(
        usable, config=cfg, spread_lookup=spread_lookup
    )

    # --- seed of the self-referential window: the quote nearest the money ---
    seed_index = int(np.argmin(np.abs(k_all)))
    sigma_seed = float(iv_all[seed_index])
    if not (math.isfinite(sigma_seed) and sigma_seed > 0.0):
        return SkewFailure(
            T=T,
            reason=REASON_NO_USABLE_QUOTE,
            message_fr="Aucune cotation proche de la monnaie exploitable pour amorcer la fenêtre.",
            detail={"T": T},
        )

    sqrt_t = math.sqrt(T)
    sigma_current = sigma_seed
    max_iter = max(1, int(cfg.max_window_iterations))

    best: tuple[WeightedFit, float, np.ndarray] | None = None
    iterations = 0
    converged = False
    flags: list[str] = list(extra_flags)
    last_failure: SkewFailure | None = None

    for _ in range(max_iter):
        half_width = max(float(cfg.window_c) * sigma_current * sqrt_t, float(cfg.min_half_width))
        selector = np.abs(k_all) <= half_width * (1.0 + 1e-12)
        n_sel = int(np.count_nonzero(selector))
        n_left = int(np.count_nonzero(selector & (k_all < 0.0)))
        n_right = int(np.count_nonzero(selector & (k_all > 0.0)))

        if n_sel < int(cfg.min_strikes):
            last_failure = SkewFailure(
                T=T,
                reason=REASON_TOO_FEW_STRIKES,
                message_fr=(
                    f"Échéance ignorée : {n_sel} strike(s) dans la fenêtre ATM "
                    f"|k| <= {half_width:.4f}, {cfg.min_strikes} requis."
                ),
                detail={
                    "n_strikes": float(n_sel),
                    "half_width": float(half_width),
                    "min_strikes": float(cfg.min_strikes),
                },
            )
            break
        if n_left < int(cfg.min_strikes_per_side) or n_right < int(cfg.min_strikes_per_side):
            last_failure = SkewFailure(
                T=T,
                reason=REASON_UNBALANCED_WINGS,
                message_fr=(
                    f"Échéance ignorée : {n_left} strike(s) sous la monnaie et {n_right} au-dessus, "
                    f"{cfg.min_strikes_per_side} requis de chaque côté."
                ),
                detail={
                    "n_left": float(n_left),
                    "n_right": float(n_right),
                    "half_width": float(half_width),
                    "min_per_side": float(cfg.min_strikes_per_side),
                },
            )
            break

        try:
            fit = weighted_polynomial_fit(
                k_all[selector], iv_all[selector], weights_all[selector], degree=2
            )
        except np.linalg.LinAlgError as exc:
            last_failure = SkewFailure(
                T=T,
                reason=REASON_SINGULAR_DESIGN,
                message_fr=f"Régression locale impossible : {exc}",
                detail={"n_strikes": float(n_sel), "half_width": float(half_width)},
            )
            break
        except ValueError as exc:
            last_failure = SkewFailure(
                T=T,
                reason=REASON_NON_FINITE_FIT,
                message_fr=f"Régression locale impossible : {exc}",
                detail={"n_strikes": float(n_sel), "half_width": float(half_width)},
            )
            break

        iterations += 1
        best = (fit, half_width, selector)
        sigma_new = float(fit.coeffs[0])
        if not (math.isfinite(sigma_new) and sigma_new > 0.0):
            last_failure = SkewFailure(
                T=T,
                reason=REASON_NON_FINITE_FIT,
                message_fr="Volatilité ATM ajustée non finie ou négative.",
                detail={"sigma_atm": sigma_new, "half_width": float(half_width)},
            )
            best = None
            break
        if abs(sigma_new - sigma_current) <= float(cfg.window_rtol) * sigma_current:
            sigma_current = sigma_new
            converged = True
            break
        sigma_current = sigma_new

    if best is None:
        return last_failure or SkewFailure(
            T=T,
            reason=REASON_NON_FINITE_FIT,
            message_fr="Ajustement local de la nappe impossible sur cette échéance.",
            detail={"T": T},
        )

    if last_failure is not None:
        # A refined window lost the strike requirement; the previous fit stands.
        flags.append(FLAG_WINDOW_REVERTED)
    if not converged:
        flags.append(FLAG_WINDOW_NOT_CONVERGED)
    if fell_back:
        flags.append(FLAG_WEIGHTS_FELL_BACK_TO_VEGA)

    fit, half_width, selector = best
    selected = [p for p, keep in zip(usable, selector) if bool(keep)]
    if any(bool(p.one_sided) for p in selected):
        flags.append(FLAG_ONE_SIDED_IN_WINDOW)

    k_sel = k_all[selector]
    a, b, c = (float(fit.coeffs[0]), float(fit.coeffs[1]), float(fit.coeffs[2]))
    se_a, se_b, se_c = (float(fit.se[0]), float(fit.se[1]), float(fit.se[2]))

    return SkewPoint(
        T=T,
        psi=b,
        se=se_b,
        n_strikes=int(k_sel.size),
        window=(-float(half_width), float(half_width)),
        curvature=2.0 * c,
        se_curvature=2.0 * se_c,
        quad_coeff=c,
        sigma_atm=a,
        se_sigma_atm=se_a,
        sigma_atm_seed=sigma_seed,
        half_width=float(half_width),
        n_left=int(np.count_nonzero(k_sel < 0.0)),
        n_right=int(np.count_nonzero(k_sel > 0.0)),
        k_min=float(np.min(k_sel)),
        k_max=float(np.max(k_sel)),
        r_squared=float(fit.r_squared),
        sigma_resid=float(fit.sigma_resid),
        resid_rmse=float(fit.resid_rmse),
        dof=int(fit.dof),
        iterations=int(iterations),
        converged=bool(converged),
        weight_scheme=str(scheme_used),
        F=forward,
        flags=tuple(dict.fromkeys(flags)),
    )


def build_skew_curve(
    option_surface: Any,
    forward_curve: Any = None,
    *,
    config: SkewConfig | None = None,
    clean_chains: Sequence[CleanChain] | None = None,
    variance_curve: Any = None,
) -> tuple[list[SkewPoint], list[SkewFailure]]:
    """
    Run :func:`estimate_atm_skew` on every expiry of an OTM market surface.

    ``option_surface`` accepts the shapes listed in
    :func:`_normalise_option_surface`, or an already-built
    ``Sequence[SkewPoint]`` (returned untouched — useful when a caller has its
    own skew source, and the way spec 4.5's "either input convention" is
    honoured end-to-end).

    Returns ``(skew_points sorted by T, failures sorted by T)``.
    """
    cfg = config or SkewConfig()

    items = None
    if option_surface is not None and not isinstance(option_surface, Mapping):
        items = list(option_surface)
        if items and all(isinstance(it, SkewPoint) for it in items):
            return (sorted(items, key=lambda sp: float(sp.T)), [])
    del items

    groups = _normalise_option_surface(option_surface)
    forwards = _forward_points(forward_curve)
    spread_lookup = build_spread_lookup(clean_chains)
    coarse = coarse_ladder_maturities(variance_curve)

    points: list[SkewPoint] = []
    failures: list[SkewFailure] = []
    for group in groups:
        T = float(group[0].T)
        extra_flags: list[str] = []
        forward_point = _match_forward(T, forwards, rtol=cfg.forward_match_rtol)
        if forwards and forward_point is None:
            extra_flags.append(FLAG_NO_FORWARD_POINT)
        F = float(forward_point.F) if forward_point is not None else float(group[0].F)
        if forward_point is not None:
            surface_F = float(group[0].F)
            if math.isfinite(surface_F) and surface_F > 0.0:
                if abs(F - surface_F) > float(cfg.forward_mismatch_rtol) * abs(surface_F):
                    extra_flags.append(FLAG_FORWARD_MISMATCH)
        for t_coarse, _bias in coarse:
            if abs(t_coarse - T) <= max(1e-12, cfg.forward_match_rtol * max(abs(T), 1e-12)):
                extra_flags.append(FLAG_COARSE_LADDER_IN_WINDOW)
                break

        result = estimate_atm_skew(
            group,
            F=F,
            config=cfg,
            spread_lookup=spread_lookup,
            extra_flags=extra_flags,
        )
        if isinstance(result, SkewPoint):
            points.append(result)
        else:
            failures.append(result)

    points.sort(key=lambda sp: float(sp.T))
    failures.sort(key=lambda f: (float(f.T) if math.isfinite(float(f.T)) else float("inf")))
    return (points, failures)


# ---------------------------------------------------------------------------
# Stage 2 — the log|psi| vs log T regression
# ---------------------------------------------------------------------------


def _student_t_quantile(dof: int) -> float:
    """Two-sided 97.5 % Student-t quantile; falls back to :data:`Z95`."""
    try:
        from scipy.stats import t as _student_t

        return float(_student_t.ppf(0.975, int(dof)))
    except Exception:  # pragma: no cover - scipy is a hard dependency of the repo
        return float(Z95)


def _empty_diagnostics(config: HurstConfig, window: tuple[float, float]) -> dict[str, Any]:
    return {
        "estimator": "asymptotic_atm_skew_power_law",
        "reference": "Alos-Leon-Vives 2007; Fukasawa 2011; Bayer-Friz-Gatheral 2016",
        "is_asymptotic_initial_estimate": True,
        "short_maturity_window": [float(window[0]), float(window[1])],
        "psi_floor": float(config.psi_floor),
        "r2_min": float(config.r2_min),
        "se_max": float(config.se_max),
        "h_bounds": [float(config.h_min), float(config.h_max)],
        "fallback_H0": float(config.fallback_H0),
        "log_weight_mode": str(config.log_weight_mode),
        "robust_method": "theil_sen",
    }


def fit_hurst_from_skew_points(
    skew_points: Sequence[SkewPoint],
    *,
    config: HurstConfig | None = None,
    short_maturity_window: tuple[float, float] | None = None,
    skew_failures: Sequence[SkewFailure] | None = None,
    coarse_ladder: Sequence[tuple[float, float]] | None = None,
) -> HurstEstimate:
    """
    Fit ``log|psi(T)| = (H - 1/2) log T + c`` over the short-maturity window.

    Weighted least squares (weights from ``SE(psi)``, see
    :class:`HurstConfig`), cross-checked against an unweighted Theil-Sen slope.
    The robust cross-check is deliberately **unweighted**: the weights derive
    from the very standard errors a corrupted expiry misreports, so weighting
    the robust fit would import the contamination it exists to resist.

    Never raises on bad data: a rejected fit comes back as
    ``unstable=True`` with ``H0 = HurstConfig.fallback_H0``.
    """
    cfg = config or HurstConfig()
    window = tuple(
        float(v)
        for v in (short_maturity_window if short_maturity_window is not None else cfg.short_maturity_window)
    )
    if len(window) != 2:
        raise ValueError("short_maturity_window must be a (T_min, T_max) pair")
    t_min, t_max = float(window[0]), float(window[1])
    if not (math.isfinite(t_min) and math.isfinite(t_max)) or t_min <= 0.0 or t_max <= t_min:
        raise ValueError(
            f"short_maturity_window must satisfy 0 < T_min < T_max; got ({t_min}, {t_max})"
        )
    window_pair = (t_min, t_max)

    diagnostics = _empty_diagnostics(cfg, window_pair)
    diagnostics["skew_points"] = [sp.to_dict() for sp in skew_points]
    diagnostics["skew_failures"] = [f.to_dict() for f in (skew_failures or [])]
    diagnostics["coarse_strike_ladder"] = [
        {"T": float(t), "discretisation_bias": float(bias)} for t, bias in (coarse_ladder or [])
    ]
    diagnostics["n_candidate_expiries"] = int(len(skew_points))

    flags: list[str] = []
    rejections: list[str] = []

    in_window = [sp for sp in skew_points if t_min <= float(sp.T) <= t_max]
    in_window.sort(key=lambda sp: float(sp.T))
    diagnostics["n_in_window"] = int(len(in_window))
    diagnostics["T_in_window"] = [float(sp.T) for sp in in_window]

    used: list[SkewPoint] = []
    dropped_near_zero: list[dict[str, float]] = []
    dropped_non_finite: list[dict[str, float]] = []
    for sp in in_window:
        psi = float(sp.psi)
        se = float(sp.se)
        if not math.isfinite(psi) or not math.isfinite(se) or se < 0.0:
            dropped_non_finite.append({"T": float(sp.T), "psi": psi, "se": se})
            continue
        if abs(psi) < float(cfg.psi_floor):
            dropped_near_zero.append({"T": float(sp.T), "psi": psi})
            continue
        used.append(sp)

    diagnostics["dropped_psi_near_zero"] = dropped_near_zero
    diagnostics["dropped_non_finite"] = dropped_non_finite
    if dropped_near_zero:
        flags.append(FLAG_PSI_NEAR_ZERO_DROPPED)

    n_neg = int(sum(1 for sp in used if float(sp.psi) < 0.0))
    n_pos = int(sum(1 for sp in used if float(sp.psi) > 0.0))
    sign_consistent = bool(n_neg == 0 or n_pos == 0)
    diagnostics["n_psi_negative"] = n_neg
    diagnostics["n_psi_positive"] = n_pos
    diagnostics["sign_consistent"] = sign_consistent
    diagnostics["dominant_sign"] = -1 if n_neg >= n_pos else 1
    if not sign_consistent:
        flags.append(FLAG_SIGN_FLIP)

    n_used = int(len(used))
    diagnostics["n_used"] = n_used
    diagnostics["T_used"] = [float(sp.T) for sp in used]
    diagnostics["psi_used"] = [float(sp.psi) for sp in used]
    diagnostics["se_psi_used"] = [float(sp.se) for sp in used]

    if n_used < int(cfg.min_expiries):
        rejections.append(REASON_TOO_FEW_EXPIRIES)
        diagnostics["rejection_reasons"] = rejections
        diagnostics["rejection_reasons_fr"] = [HURST_REASON_LABELS_FR[r] for r in rejections]
        diagnostics["flags"] = flags
        diagnostics["H0_estimated"] = float("nan")
        diagnostics["H0_clipped"] = float("nan")
        return HurstEstimate(
            H0=float(cfg.fallback_H0),
            se=float("nan"),
            ci95=(float("nan"), float("nan")),
            r2=float("nan"),
            n_expiries=n_used,
            window=window_pair,
            unstable=True,
            diagnostics=diagnostics,
        )

    log_t = np.array([math.log(float(sp.T)) for sp in used], dtype=float)
    abs_psi = np.array([abs(float(sp.psi)) for sp in used], dtype=float)
    log_abs_psi = np.log(abs_psi)
    se_raw = np.array([float(sp.se) for sp in used], dtype=float)
    se_eff = np.maximum(se_raw, float(cfg.se_floor))
    if bool(np.any(se_raw < float(cfg.se_floor))):
        flags.append(FLAG_SE_FLOORED)

    mode = str(cfg.log_weight_mode)
    if mode == "delta":
        weights = (abs_psi / se_eff) ** 2
    elif mode == "raw_psi":
        weights = 1.0 / se_eff**2
    elif mode == "uniform":
        weights = np.ones(n_used, dtype=float)
    else:
        raise ValueError(
            f"HurstConfig.log_weight_mode must be 'delta', 'raw_psi' or 'uniform'; got {mode!r}"
        )

    diagnostics["log_T"] = [float(v) for v in log_t]
    diagnostics["log_abs_psi"] = [float(v) for v in log_abs_psi]
    diagnostics["weights"] = [float(v) for v in weights]

    try:
        fit = weighted_polynomial_fit(log_t, log_abs_psi, weights, degree=1)
    except (ValueError, np.linalg.LinAlgError) as exc:
        rejections.append(REASON_DEGENERATE_REGRESSION)
        diagnostics["regression_error"] = str(exc)
        diagnostics["rejection_reasons"] = rejections
        diagnostics["rejection_reasons_fr"] = [HURST_REASON_LABELS_FR[r] for r in rejections]
        diagnostics["flags"] = flags
        diagnostics["H0_estimated"] = float("nan")
        diagnostics["H0_clipped"] = float("nan")
        return HurstEstimate(
            H0=float(cfg.fallback_H0),
            se=float("nan"),
            ci95=(float("nan"), float("nan")),
            r2=float("nan"),
            n_expiries=n_used,
            window=window_pair,
            unstable=True,
            diagnostics=diagnostics,
        )

    intercept = float(fit.coeffs[0])
    slope = float(fit.coeffs[1])
    se_slope = float(fit.se[1])
    r2 = float(fit.r_squared)

    slope_robust, intercept_robust = theil_sen_slope(log_t, log_abs_psi)

    h0_raw = slope + 0.5
    h0_robust = slope_robust + 0.5 if math.isfinite(slope_robust) else float("nan")
    se_h0 = se_slope
    ci95 = (h0_raw - Z95 * se_h0, h0_raw + Z95 * se_h0)
    t_quantile = _student_t_quantile(int(fit.dof))
    ci95_t = (h0_raw - t_quantile * se_h0, h0_raw + t_quantile * se_h0)

    slope_gap = (
        abs(slope - slope_robust) if math.isfinite(slope_robust) else float("nan")
    )
    disagreement_threshold = max(
        se_h0 if math.isfinite(se_h0) else 0.0, float(cfg.robust_disagreement_atol)
    )
    disagreement = bool(math.isfinite(slope_gap) and slope_gap > disagreement_threshold)
    if disagreement:
        flags.append(FLAG_ROBUST_DISAGREEMENT)

    diagnostics.update(
        {
            "intercept_wls": intercept,
            "slope_wls": slope,
            "slope_se": se_h0,
            "slope_robust": float(slope_robust),
            "intercept_robust": float(intercept_robust),
            "slope_gap": float(slope_gap),
            "robust_disagreement": disagreement,
            "robust_disagreement_threshold": float(disagreement_threshold),
            "H0_estimated": float(h0_raw),
            "H0_robust": float(h0_robust),
            "residual_rmse": float(fit.resid_rmse),
            "residual_scale_weighted": float(fit.sigma_resid),
            "dof": int(fit.dof),
            "condition_number": float(fit.condition_number),
            "ci95_student_t": [float(ci95_t[0]), float(ci95_t[1])],
            "student_t_quantile": float(t_quantile),
            "amplitude_A": float(math.exp(intercept)) if math.isfinite(intercept) else float("nan"),
        }
    )

    h_min = float(cfg.h_min)
    h_max = float(cfg.h_max)
    h0_clipped = float(min(max(h0_raw, h_min), h_max)) if math.isfinite(h0_raw) else float("nan")
    diagnostics["H0_clipped"] = h0_clipped

    if not math.isfinite(h0_raw) or not math.isfinite(se_h0):
        rejections.append(REASON_DEGENERATE_REGRESSION)
    if not math.isfinite(r2) or r2 < float(cfg.r2_min):
        rejections.append(REASON_LOW_R2)
    if not math.isfinite(se_h0) or se_h0 > float(cfg.se_max):
        rejections.append(REASON_SE_TOO_LARGE)
    if not (math.isfinite(h0_raw) and h_min < h0_raw < h_max):
        rejections.append(REASON_H0_OUT_OF_RANGE)

    rejections = list(dict.fromkeys(rejections))
    diagnostics["rejection_reasons"] = rejections
    diagnostics["rejection_reasons_fr"] = [
        HURST_REASON_LABELS_FR.get(r, r) for r in rejections
    ]
    diagnostics["flags"] = list(dict.fromkeys(flags))

    unstable = bool(rejections)
    return HurstEstimate(
        H0=float(cfg.fallback_H0) if unstable else float(h0_raw),
        se=float(se_h0),
        ci95=(float(ci95[0]), float(ci95[1])),
        r2=float(r2),
        n_expiries=n_used,
        window=window_pair,
        unstable=unstable,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Public entry point (spec 5)
# ---------------------------------------------------------------------------


def estimate_hurst_from_skew(
    option_surface: Any,
    forward_curve: Any,
    short_maturity_window: tuple[float, float] | None = None,
    *,
    skew_config: SkewConfig | None = None,
    hurst_config: HurstConfig | None = None,
    clean_chains: Sequence[CleanChain] | None = None,
    variance_curve: Any = None,
) -> HurstEstimate:
    """
    Initial Hurst estimate from the short-maturity ATM skew (spec 4.5 / 5).

    Parameters
    ----------
    option_surface:
        The OTM market surface produced by Phase 1. Any of:
        a flat ``Sequence[SurfacePoint]``, a
        ``Mapping[T, Sequence[SurfacePoint]]``, a
        ``Sequence[Sequence[SurfacePoint]]`` (one
        ``build_otm_surface(chain, point)[0]`` per expiry), or an
        already-built ``Sequence[SkewPoint]``.
    forward_curve:
        ``Sequence[ForwardPoint]`` (or a mapping of them, or ``None``). Used to
        pin ``F(T)`` for the strike-convention conversions and to cross-check
        the forward the surface points carry; a maturity with no matching
        forward point falls back to the surface's own ``F`` and is flagged
        :data:`FLAG_NO_FORWARD_POINT`.
    short_maturity_window:
        ``(T_min, T_max)`` in years, inclusive. ``None`` uses
        ``HurstConfig.short_maturity_window`` (``5/365`` to ``0.25``).
    clean_chains:
        The Phase-1 :class:`~app.model.calibration.rough_vol.chain_cleaning.CleanChain`
        list the surface was built from. Supplying it enables the
        ``1 / spread^2`` weights; without it the local fits fall back to vega
        weights (see the module docstring).
    variance_curve:
        The Phase-2 variance-swap curve (or its points). Only used to surface
        :data:`FLAG_COARSE_LADDER_IN_WINDOW` for the Phase-4 report — ``K_var``
        itself is never consumed here.

    Returns
    -------
    HurstEstimate
        With ``unstable=True`` and ``H0 = 0.1`` when the estimate is rejected.
        **The returned number is an asymptotic starting point, not a
        calibration result** — see the module docstring.
    """
    points, failures = build_skew_curve(
        option_surface,
        forward_curve,
        config=skew_config,
        clean_chains=clean_chains,
        variance_curve=variance_curve,
    )
    return fit_hurst_from_skew_points(
        points,
        config=hurst_config,
        short_maturity_window=short_maturity_window,
        skew_failures=failures,
        coarse_ladder=coarse_ladder_maturities(variance_curve),
    )


def hurst_report(estimate: HurstEstimate) -> dict[str, Any]:
    """Plain-python summary for the Phase-5 report; ``_json_safe`` compatible."""
    diagnostics = dict(estimate.diagnostics)
    return {
        "H0": float(estimate.H0),
        "H0_is_fallback": bool(estimate.unstable),
        "H0_estimated": float(diagnostics.get("H0_estimated", float("nan"))),
        "H0_robust": float(diagnostics.get("H0_robust", float("nan"))),
        "se": float(estimate.se),
        "ci95": [float(estimate.ci95[0]), float(estimate.ci95[1])],
        "r2": float(estimate.r2),
        "n_expiries": int(estimate.n_expiries),
        "window": [float(estimate.window[0]), float(estimate.window[1])],
        "unstable": bool(estimate.unstable),
        "flags": [str(f) for f in diagnostics.get("flags", [])],
        "rejection_reasons": [str(r) for r in diagnostics.get("rejection_reasons", [])],
        "rejection_reasons_fr": [str(r) for r in diagnostics.get("rejection_reasons_fr", [])],
        "coarse_strike_ladder": list(diagnostics.get("coarse_strike_ladder", [])),
        "n_skew_failures": int(len(diagnostics.get("skew_failures", []))),
        "message_fr": estimate.message_fr,
        "warning_fr": (
            "Estimation asymptotique initiale du coefficient de Hurst (skew ATM court terme) — "
            "point de départ de l'optimiseur, jamais un résultat de calibration."
        ),
    }


__all__ = [
    "FALLBACK_H0",
    "FLAG_COARSE_LADDER_IN_WINDOW",
    "FLAG_FORWARD_MISMATCH",
    "FLAG_NO_FORWARD_POINT",
    "FLAG_ONE_SIDED_IN_WINDOW",
    "FLAG_PSI_NEAR_ZERO_DROPPED",
    "FLAG_ROBUST_DISAGREEMENT",
    "FLAG_SE_FLOORED",
    "FLAG_SIGN_FLIP",
    "FLAG_WEIGHTS_FELL_BACK_TO_VEGA",
    "FLAG_WINDOW_NOT_CONVERGED",
    "FLAG_WINDOW_REVERTED",
    "HURST_REASON_LABELS_FR",
    "H_LOWER_BOUND",
    "H_UPPER_BOUND",
    "HurstConfig",
    "HurstEstimate",
    "REASON_DEGENERATE_REGRESSION",
    "REASON_H0_OUT_OF_RANGE",
    "REASON_INVALID_MATURITY",
    "REASON_LOW_R2",
    "REASON_NON_FINITE_FIT",
    "REASON_NO_USABLE_QUOTE",
    "REASON_SE_TOO_LARGE",
    "REASON_SINGULAR_DESIGN",
    "REASON_TOO_FEW_EXPIRIES",
    "REASON_TOO_FEW_STRIKES",
    "REASON_UNBALANCED_WINGS",
    "SKEW_REASON_LABELS_FR",
    "SkewConfig",
    "SkewFailure",
    "SkewPoint",
    "WeightedFit",
    "Z95",
    "black76_vega",
    "build_skew_curve",
    "build_spread_lookup",
    "coarse_ladder_maturities",
    "estimate_atm_skew",
    "estimate_hurst_from_skew",
    "fit_hurst_from_skew_points",
    "hurst_report",
    "psi_from_strike_slope",
    "strike_slope_from_psi",
    "theil_sen_slope",
    "weighted_polynomial_fit",
]
