"""
Initial ``(H, eta, rho)`` for the rough Bergomi joint calibration (spec 4.9).

.. warning::

   **EVERYTHING THIS MODULE RETURNS IS A STARTING POINT, NEVER A RESULT.**

   The three numbers are the seed of the optimiser of spec 4.10. They are built
   from *leading-order asymptotics* whose constants are known to a few percent
   at best, and the short-dated ATM skew that carries most of the information
   identifies only the **product** ``rho * eta`` — not ``rho`` and ``eta``
   separately. Reporting any of them as a calibrated parameter would be a lie.

What the module does
--------------------
1. **``H0``** comes straight from spec 4.5
   (:func:`~app.model.calibration.rough_vol.hurst_estimator.estimate_hurst_from_skew`).
   Its ``unstable`` flag is propagated verbatim into the diagnostics: an
   unstable Hurst estimate does not stop the initialisation, it labels it.

2. **``rho0`` and ``eta0`` jointly, from the skew level.** At leading order the
   short-maturity ATM skew of rBergomi is

   .. math::  \\psi(T) = \\frac{\\partial\\sigma_{BS}}{\\partial k}\\Big|_{k=0}
              \\;\\simeq\\; c(H)\\,\\rho\\,\\eta\\,T^{H-1/2}

   which is a function of the **product** ``rho * eta`` only. The degeneracy is
   broken here by a prior, not by data: ``rho0 = sign(psi) * 0.7`` (the
   equity-index prior, configurable through
   :attr:`InitializerConfig.rho_prior_abs`), and then

   .. math::  \\eta_0 = \\mathrm{clip}\\!\\left(
              \\frac{|\\psi(T_{ref})|}{c(H_0)\\,|\\rho_0|\\,T_{ref}^{H_0-1/2}},
              0.5,\\;3.5\\right)

   The degeneracy is stated explicitly in ``diagnostics["rho_eta_degeneracy"]``
   together with the invariant it implies: changing ``|rho0|`` rescales
   ``eta0`` by exactly the inverse factor and leaves ``rho0 * eta0`` unchanged.
   **Stage 1 of spec 4.10 is what actually breaks it.**

3. **An independent ``eta`` from the ATM curvature** (see below), reported
   alongside, never used as ``eta0``, and compared against it: a disagreement
   beyond a factor of :data:`ETA_DISAGREEMENT_FACTOR` raises
   :data:`FLAG_ETA_DISAGREEMENT` — a strong hint that Stage 1 needs a wider
   ``eta`` bracket (and/or that the ``|rho0|`` prior is wrong for this name).

``c(H)`` — MEASURED, NOT ASSUMED
--------------------------------
Spec 4.9 explicitly forbids trusting the literature constant blindly and
requires it to be calibrated numerically **once** against our own simulator
before use. That measurement was run (60 000 antithetic paths, conditional
estimator, flat ``xi0 = 0.04``, central difference ``dk = 0.01``,
``T in {5,10,20,40,80}/365``) over
``H in {0.05,0.10,0.20,0.35,0.45} x eta in {0.5,1,2} x rho in {-0.9,-0.7,-0.4}``
and settles the convention factor at **1/2**:

.. math::  c(H) = \\tfrac12 \\frac{\\sqrt{2H}}{(H+\\tfrac12)(H+\\tfrac32)}

Evidence recorded with the measurement:

* ``c_hat / [sqrt(2H)/((H+1/2)(H+3/2))]`` lands in **0.487 - 0.493 at
  ``eta = 0.5`` for every ``H`` tested**, i.e. it converges on ``1/2`` as
  ``eta -> 0``, which is the leading-order limit the formula describes;
* ``c_hat`` is **independent of ``rho`` to three decimals**
  (``H = 0.05``, ``eta = 0.5``: 0.18070 / 0.18079 / 0.18097 at
  ``rho = -0.9 / -0.7 / -0.4``) — a direct confirmation that the short skew
  identifies only the product ``rho * eta``;
* ``c_hat`` does drift with ``eta`` (a higher-order correction), most at small
  ``H``: ``H = 0.05`` gives 0.1807 / 0.1723 / 0.1416 at ``eta = 0.5 / 1 / 2``.

Accuracy of the closed form against that measurement:

===== ============== ========== =======
``H`` ``c_hat`` eta=0.5 ``0.5*lit`` ratio
===== ============== ========== =======
0.05  0.18070        0.18547    0.974
0.10  0.22867        0.23292    0.982
0.20  0.26188        0.26574    0.985
0.35  0.26201        0.26603    0.985
0.45  0.25188        0.25605    0.984
===== ============== ========== =======

So: **~2 % for ``eta <= 1``, degrading to ~10-15 % at ``eta = 2``, worst at
small ``H``.** Ample for a seed, useless as a result. ``c(H)`` is exposed as
:func:`c_of_H` so the calibrator, this module and the tests share exactly one
definition.

The curvature cross-check
-------------------------
Spec 4.9 asks for a second, *independent* ``eta`` from the ATM smile curvature
``d^2 sigma / dk^2|_{k=0}``, which the spec-4.5 local quadratic fit already
produces (:attr:`SkewPoint.curvature`). The leading-order curvature of rBergomi
used here is

.. math::  \\kappa(T) \\;=\\; \\frac{\\eta^2 T^{2H-1}}{\\sigma_{ATM}}
           \\Big[a(H) - 2\\,c(H)^2\\rho^2\\Big],
           \\qquad a(H) = \\frac{H}{4\\,(H+\\tfrac12)^2 (H+1)}

**``a(H)`` is derived, not borrowed.** At ``rho = 0`` the mixing (Romano-Touzi)
representation gives the implied *total* variance
``w(k) = w + \\tfrac12 Var(I)[k^2/(2w^2) - 1/(2w) - 1/8] + O(eta^3)`` with
``I = int_0^T V_t dt`` and ``w = xi0 T``. Since ``d^2 w/dk^2|_0 = Var(I)/(2w^2)``
and ``sigma = sqrt(w/T)`` with a vanishing ATM slope at ``rho = 0``, this gives
``kappa = Var(I) / (4 w^{5/2} sqrt(T))`` and, using the exact
Riemann-Liouville covariance
``Cov(W_u, W_v) = 2H int_0^{u^v} (u-s)^{H-1/2}(v-s)^{H-1/2} ds``,
``Var(I) = xi0^2 eta^2 H T^{2H+2} / ((H+1/2)^2 (H+1))``, which collapses to
``a(H)`` above. Two independent confirmations:

* at ``H = 1/2`` rBergomi *is* lognormal SABR with ``nu = eta/2``, and Hagan's
  ``beta = 1`` expansion gives ``d^2 sigma/dk^2 = (2 - 3 rho^2) nu^2/(6 alpha)
  = (2 - 3 rho^2) eta^2 / (24 sqrt(xi0))``. Setting ``rho = 0`` gives
  ``eta^2/(12 sqrt(xi0))`` and ``a(1/2) = 1/12``. **Exact match**, and the
  ``rho`` law ``(2-3rho^2)/24 = a - 2c(1/2)^2 rho^2`` fixes the ``rho^2``
  coefficient at ``2 c(H)^2`` at ``H = 1/2``;
* measured against this repo's own simulator at ``rho = 0``, where
  ``g_hat / a(H)`` is 1.034 / 1.006 / 0.993 / 0.986 / 0.978 / 0.972 at
  ``H = 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.45`` (150 000 antithetic paths,
  ``n_max = 256``, flat ``xi0 = 0.04``, ``T in {20,40,80}/365``) — the level
  ``a(H)`` is therefore good to ~3 %;
* the ``rho^2`` law was measured as the *ratio* ``g(rho)/g(0)`` at matched
  ``(H, T, eta, grid)``, which cancels the grid and higher-order biases:
  ``b(H)/a(H)`` comes out 1.60 / 1.64 / 1.61 / 1.53 / 1.40 / 1.31 averaged over
  the whole ``(T, eta)`` grid and 1.63 / 1.68 / 1.67 / 1.65 / 1.63 / 1.63 on the
  best-resolved cells (``T = 80 d``, where the ATM second difference is largest
  relative to its noise). Against ``2c(H)^2/a(H)`` = 1.75 / 1.72 / 1.66 / 1.61 /
  1.55 / 1.53 the residual RMS on those cells is 0.073; against a flat 1.5 — the
  rival ``b = 1.5 a(H)`` law, which is *also* exact at ``H = 1/2`` — it is 0.150
  and biased low at every ``H``. Both laws sit within ~10 % of the measurement;
  ``2 c(H)^2`` is the better of the two and is the one implemented.

**The form actually used avoids the ``rho`` prior entirely.** Because
``psi^2 = c(H)^2 rho^2 eta^2 T^{2H-1}`` exactly, the identity above rearranges
into a relation with no ``rho`` in it at all:

.. math::  \\kappa\\,\\sigma_{ATM} + 2\\psi^2 \\;=\\; a(H)\\,\\eta^2\\,T^{2H-1}
           \\;\\Longrightarrow\\;
           \\eta_{curv} = \\sqrt{\\frac{(\\kappa\\,\\sigma_{ATM} + 2\\psi^2)\\,
           T^{1-2H}}{a(H)}}

This is what :func:`eta_from_curvature` computes. Using it rather than plugging
``rho0`` into ``[a - 2c^2 rho^2]`` matters twice over: (i) an estimate meant to
*test* the ``rho0`` prior must not consume it, and (ii) the bracket
``a - 2c^2 rho^2`` **changes sign** at
``|rho| = sqrt(a(H)/(2c(H)^2))`` (:func:`curvature_sign_change_rho_abs`, about
0.76 at ``H = 0.1`` and 0.81 at ``H = 0.45``), so at the equity prior
``|rho0| = 0.7`` it is a small difference of two comparable numbers and
inverting it directly is badly conditioned. The ``rho``-free form is a *sum* of
the two and is well conditioned.

End-to-end verification of ``eta_curv`` against the simulator (150 000 paths,
``n_max = 256``, ``T in {10,20,40,80}/365``, ``xi0 = 0.04``, over
``H in {0.05,0.10,0.15,0.20,0.30,0.35,0.45}``, ``eta in {0.3,0.5,1,2}``,
``rho in {0,+-0.4,+-0.7}``): ``eta_curv / eta_true`` sits in **0.95 - 1.09** in
every cell with ``eta >= 1`` or ``|rho| <= 0.4``. In the near-cancellation
corner — ``|rho| ~ 0.7`` together with ``eta <= 0.5``, where the ATM curvature
is itself a small difference of two comparable numbers — single cells ran from
0.48 to 1.45. That corner is not hidden: ``diagnostics`` reports
``curvature_information_share = |kappa*sigma| / (|kappa*sigma| + 2 psi^2)`` and
raises :data:`FLAG_CURVATURE_ILL_CONDITIONED` below
:attr:`InitializerConfig.curvature_information_min`. For a typical equity index
(``|rho| ~ 0.7``) that share is naturally only ~0.16 at ``H = 0.1`` and ~0.25 at
``H = 0.45``, so the curvature genuinely carries little independent information
there — which is exactly why it is a *cross-check with a factor-2 tolerance*
and not a second calibration.

``T_ref``
---------
``T_ref`` is chosen among the skew points the Hurst regression actually used
(inside :attr:`HurstEstimate.window`, finite ``psi`` above the floor) as the one
with the **smallest relative standard error** ``SE(psi)/|psi|``, ties broken
toward the shorter maturity (:data:`T_REF_RULE_MIN_RELATIVE_SE`, the default).

Rationale: the two competing concerns are asymptotic validity (which favours
the shortest maturity, since ``psi ~ c(H) rho eta T^{H-1/2}`` is a ``T -> 0``
statement) and measurement precision (which favours the longest, since the
shortest expiries carry the widest spreads, the coarsest strike ladders and the
zero-bid wall of spec 4.1). The ``c(H)`` measurement above settles the
arbitration: the closed form is accurate to ~2 % roughly *uniformly* over 5-80
days, so asymptotic validity is not the binding constraint inside the window —
precision is. :data:`T_REF_RULE_SHORTEST` is available for the opposite policy
and an explicit ``t_ref`` pins a maturity outright (it must match a candidate;
it is never snapped silently).

Layering / purity: model layer only. No side effects, no network, no RNG, no
mutation of any input. Everything in ``diagnostics`` is plain
``float``/``int``/``bool``/``str``/``list``/``dict`` so it survives the
controller's ``_json_safe`` unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.model.calibration.rough_vol.hurst_estimator import (
    SkewConfig,
    SkewPoint,
    build_skew_curve,
)
from app.model.volatility_models.rbergomi.simulator_xi_curve import (
    ETA_MAX,
    H_MAX,
    H_MIN,
    RHO_ABS_MAX,
    RBergomiParams,
)

# ---------------------------------------------------------------------------
# Constants (spec 4.9)
# ---------------------------------------------------------------------------

#: Equity-index prior on ``|rho|``. A prior, not a measurement: the short-dated
#: ATM skew cannot separate ``rho`` from ``eta``.
RHO_PRIOR_ABS: float = 0.7

#: Clip applied to ``eta0`` (spec 4.9).
ETA_INIT_MIN: float = 0.5
ETA_INIT_MAX: float = 3.5

#: Ratio between the two ``eta`` estimates above which they are called
#: inconsistent (spec 4.9: "beyond a factor of 2").
ETA_DISAGREEMENT_FACTOR: float = 2.0

#: ``T_ref`` selection rules.
T_REF_RULE_MIN_RELATIVE_SE: str = "min_relative_se"
T_REF_RULE_SHORTEST: str = "shortest"
T_REF_RULE_EXPLICIT: str = "explicit"

#: Closed form of ``c(H)``, kept as a string so a report can print the formula
#: next to the number it used.
C_OF_H_FORMULA: str = "0.5 * sqrt(2H) / ((H + 1/2) * (H + 3/2))"

#: Provenance of ``c(H)`` — repeated in every diagnostics payload so the number
#: can never be mistaken for a fitted quantity.
C_OF_H_PROVENANCE: str = (
    "measured against this repo's own rBergomi simulator (60k antithetic paths, "
    "conditional estimator, flat xi0=0.04, dk=0.01, T in {5,10,20,40,80}/365, "
    "H in {0.05,0.10,0.20,0.35,0.45} x eta in {0.5,1,2} x rho in {-0.9,-0.7,-0.4}); "
    "convention factor 1/2 confirmed (ratio 0.487-0.493 to the unhalved form at "
    "eta=0.5 for every H); accurate to ~2% for eta<=1 and ~10-15% at eta=2, worst "
    "at small H. INITIALISER ONLY - never a calibration result."
)

#: Provenance of ``a(H)`` (the rho-free curvature level coefficient).
CURVATURE_COEFFICIENT_PROVENANCE: str = (
    "derived from the mixing representation at rho=0 "
    "(a(H) = H / (4 (H+1/2)^2 (H+1))), cross-checked exactly against Hagan's "
    "beta=1 SABR at H=1/2 (a(1/2)=1/12 and the rho^2 law (2-3rho^2)/24), and "
    "measured against this repo's simulator at rho=0: g_hat/a(H) = 1.034 / 1.006 "
    "/ 0.993 / 0.986 / 0.978 / 0.972 at H = 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / "
    "0.45 (150k antithetic paths, n_max=256, flat xi0=0.04). The rho^2 "
    "coefficient b(H)=2c(H)^2 is measured to within ~10%. INITIALISER ONLY."
)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

#: ``H0`` had to be clipped into ``[H_MIN, H_MAX]``.
FLAG_H_CLIPPED = "H0_clipped"
#: ``eta0`` hit one of the ``[eta_min, eta_max]`` bounds.
FLAG_ETA_CLIPPED = "eta0_clipped"
#: ``|rho0|`` had to be clipped into ``(0, RHO_ABS_MAX]``.
FLAG_RHO_CLIPPED = "rho0_clipped"
#: The Hurst estimate came back ``unstable`` — ``H0`` is its fallback value.
FLAG_HURST_UNSTABLE = "hurst_estimate_unstable"
#: The skew-based and curvature-based ``eta`` differ by more than the factor.
FLAG_ETA_DISAGREEMENT = "eta_estimates_disagree"
#: No curvature-based ``eta`` could be formed at ``T_ref``.
FLAG_NO_CURVATURE_ESTIMATE = "no_curvature_eta_estimate"
#: The curvature carries almost no information beyond the skew at this ``rho``.
FLAG_CURVATURE_ILL_CONDITIONED = "curvature_ill_conditioned"
#: ``psi(T_ref)`` is (numerically) zero, so its sign cannot pin ``sign(rho0)``.
FLAG_PSI_SIGN_AMBIGUOUS = "psi_sign_ambiguous"
#: ``sign(psi(T_ref))`` disagrees with the dominant sign of the skew curve.
FLAG_SIGN_DISAGREES_WITH_CURVE = "psi_sign_disagrees_with_curve"
#: No candidate maturity fell inside the Hurst regression window.
FLAG_T_REF_OUTSIDE_HURST_WINDOW = "t_ref_outside_hurst_window"
#: ``sigma_ATM(T_ref)`` came from ``xi0_curve`` rather than from the skew fit.
FLAG_SIGMA_ATM_FROM_XI0 = "sigma_atm_from_xi0_curve"
#: ``sigma_ATM`` from the skew fit and from ``xi0_curve`` disagree materially.
FLAG_SIGMA_ATM_MISMATCH = "sigma_atm_mismatch"

INITIALIZER_FLAG_LABELS_FR: dict[str, str] = {
    FLAG_H_CLIPPED: "H0 ramené dans les bornes admissibles",
    FLAG_ETA_CLIPPED: "eta0 ramené dans l'intervalle d'initialisation",
    FLAG_RHO_CLIPPED: "rho0 ramené dans les bornes admissibles",
    FLAG_HURST_UNSTABLE: "estimation de H instable — H0 est une valeur de repli",
    FLAG_ETA_DISAGREEMENT: (
        "les deux estimations de eta divergent de plus d'un facteur 2 — "
        "élargir l'intervalle de eta à l'étape 1"
    ),
    FLAG_NO_CURVATURE_ESTIMATE: "aucune estimation de eta par la courbure disponible",
    FLAG_CURVATURE_ILL_CONDITIONED: (
        "la courbure ATM n'apporte quasiment pas d'information indépendante "
        "à ce niveau de rho"
    ),
    FLAG_PSI_SIGN_AMBIGUOUS: "signe du skew indéterminé — a priori actions-indices retenu",
    FLAG_SIGN_DISAGREES_WITH_CURVE: (
        "le signe du skew à T_ref contredit le signe dominant de la courbe"
    ),
    FLAG_T_REF_OUTSIDE_HURST_WINDOW: (
        "aucune échéance candidate dans la fenêtre de régression de H"
    ),
    FLAG_SIGMA_ATM_FROM_XI0: "sigma ATM reconstruit depuis la courbe de variance forward",
    FLAG_SIGMA_ATM_MISMATCH: (
        "écart matériel entre le sigma ATM du fit et celui de la courbe xi0"
    ),
}


class RBergomiInitializationError(ValueError):
    """
    No skew information at all: the product ``rho * eta`` is unidentified.

    Raised rather than returning a fabricated number — an initialiser with no
    data behind it is worse than a refusal, and the caller (spec 5) can catch
    this and report the surface as unusable.
    """


# ---------------------------------------------------------------------------
# The asymptotic constants
# ---------------------------------------------------------------------------


def c_of_H(H: float) -> float:
    """
    The ATM-skew constant of ``psi(T) = c(H) * rho * eta * T^{H-1/2}``.

    ``c(H) = 0.5 * sqrt(2H) / ((H + 1/2) * (H + 3/2))`` — see the module
    docstring for the measurement that fixes the ``1/2`` and for the accuracy
    envelope (~2 % for ``eta <= 1``, ~10-15 % at ``eta = 2``).

    **This is an initialiser, never a result.**

    Returns ``nan`` for a non-finite or non-positive ``H``.
    """
    h = float(H)
    if not math.isfinite(h) or h <= 0.0:
        return float("nan")
    return 0.5 * math.sqrt(2.0 * h) / ((h + 0.5) * (h + 1.5))


def curvature_coefficient(H: float) -> float:
    """
    ``a(H) = H / (4 (H + 1/2)^2 (H + 1))`` — the rho-free ATM curvature level.

    Defined by ``kappa * sigma_ATM + 2 psi^2 = a(H) * eta^2 * T^{2H-1}``.
    Derived from the mixing representation, exact against Hagan's ``beta = 1``
    SABR at ``H = 1/2`` (``a(1/2) = 1/12``) and measured to within 3 % against
    this repo's simulator — see the module docstring.

    Returns ``nan`` for a non-finite or non-positive ``H``.
    """
    h = float(H)
    if not math.isfinite(h) or h <= 0.0:
        return float("nan")
    return h / (4.0 * (h + 0.5) ** 2 * (h + 1.0))


def curvature_rho_coefficient(H: float) -> float:
    """
    ``b(H) = 2 c(H)^2`` — the ``rho^2`` coefficient of the ATM curvature.

    ``kappa = (eta^2 T^{2H-1} / sigma_ATM) * [a(H) - b(H) rho^2]``. Exact at
    ``H = 1/2`` (Hagan: ``(2 - 3 rho^2) / 24``); measured to within ~8 % over
    ``H in [0.05, 0.45]``.
    """
    c = c_of_H(H)
    if not math.isfinite(c):
        return float("nan")
    return 2.0 * c * c


def curvature_sign_change_rho_abs(H: float) -> float:
    """
    ``|rho|`` at which the leading-order ATM curvature vanishes.

    ``sqrt(a(H) / (2 c(H)^2))`` — about 0.76 at ``H = 0.1`` and 0.81 at
    ``H = 0.45``. The equity-index prior ``|rho0| = 0.7`` sits just below it,
    which is why the ``rho``-free inversion of :func:`eta_from_curvature` is
    used instead of inverting ``[a - 2 c^2 rho^2]`` directly.
    """
    a = curvature_coefficient(H)
    b = curvature_rho_coefficient(H)
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0.0 or a <= 0.0:
        return float("nan")
    return math.sqrt(a / b)


def atm_skew_model(*, H: float, eta: float, rho: float, T: float) -> float:
    """Leading-order model skew ``c(H) * rho * eta * T^{H-1/2}`` (forward direction)."""
    c = c_of_H(H)
    t = float(T)
    if not (math.isfinite(c) and math.isfinite(t)) or t <= 0.0:
        return float("nan")
    return c * float(rho) * float(eta) * t ** (float(H) - 0.5)


def atm_curvature_model(
    *, H: float, eta: float, rho: float, T: float, sigma_atm: float
) -> float:
    """
    Leading-order model ATM curvature ``d^2 sigma / dk^2|_{k=0}``.

    ``(eta^2 T^{2H-1} / sigma_ATM) * [a(H) - 2 c(H)^2 rho^2]``. The exact
    inverse of :func:`eta_from_curvature`, and the forward direction the tests
    use as their oracle.
    """
    a = curvature_coefficient(H)
    b = curvature_rho_coefficient(H)
    t = float(T)
    s = float(sigma_atm)
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(t) and math.isfinite(s)):
        return float("nan")
    if t <= 0.0 or s <= 0.0:
        return float("nan")
    e = float(eta)
    r = float(rho)
    return (e * e * t ** (2.0 * float(H) - 1.0) / s) * (a - b * r * r)


def eta_from_skew(psi: float, *, H: float, rho_abs: float, T: float) -> float:
    """
    Invert ``psi(T) = c(H) rho eta T^{H-1/2}`` for ``eta`` at a pinned ``|rho|``.

    ``eta = |psi| / (c(H) * |rho| * T^{H-1/2})``. Exact inverse of
    :func:`atm_skew_model`; ``nan`` when any input makes it undefined.
    """
    c = c_of_H(H)
    t = float(T)
    r = abs(float(rho_abs))
    p = abs(float(psi))
    if not (math.isfinite(c) and math.isfinite(t) and math.isfinite(r) and math.isfinite(p)):
        return float("nan")
    if t <= 0.0 or r <= 0.0 or c <= 0.0:
        return float("nan")
    return p / (c * r * t ** (float(H) - 0.5))


def eta_from_curvature(
    curvature: float, *, psi: float, H: float, T: float, sigma_atm: float
) -> float:
    """
    ``eta`` from the ATM curvature, **without using ``rho``**.

    Uses ``kappa * sigma_ATM + 2 psi^2 = a(H) eta^2 T^{2H-1}``, an exact
    rearrangement of the leading-order curvature that absorbs the whole
    ``rho^2`` dependence into the measured ``psi^2``. See the module docstring
    for the derivation, the Hagan/SABR cross-check and the measured envelope.

    Returns ``nan`` when the combination
    ``kappa * sigma_ATM + 2 psi^2`` is non-positive — the market smile is then
    inconsistent with the leading order and no ``eta`` can be read from it.
    """
    a = curvature_coefficient(H)
    t = float(T)
    s = float(sigma_atm)
    kap = float(curvature)
    p = float(psi)
    if not (math.isfinite(a) and math.isfinite(t) and math.isfinite(s) and math.isfinite(kap)):
        return float("nan")
    if not math.isfinite(p) or t <= 0.0 or s <= 0.0 or a <= 0.0:
        return float("nan")
    combination = kap * s + 2.0 * p * p
    if combination <= 0.0:
        return float("nan")
    return math.sqrt(combination * t ** (1.0 - 2.0 * float(H)) / a)


def curvature_information_share(
    curvature: float, *, psi: float, sigma_atm: float
) -> float:
    """
    Share of :func:`eta_from_curvature`'s input carried by the curvature itself.

    ``|kappa sigma| / (|kappa sigma| + 2 psi^2)``. Near 0 the combination is
    dominated by ``psi^2`` and the "independent" estimate degenerates into the
    skew estimate at the implicit ``|rho| = sqrt(a/(2c^2))`` — it is then not
    independent evidence at all. ``nan`` when undefined.
    """
    kap = float(curvature)
    s = float(sigma_atm)
    p = float(psi)
    if not (math.isfinite(kap) and math.isfinite(s) and math.isfinite(p)):
        return float("nan")
    num = abs(kap * s)
    den = num + 2.0 * p * p
    if den <= 0.0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitializerConfig:
    """
    Knobs of :func:`initialize_rbergomi_parameters`.

    Attributes
    ----------
    rho_prior_abs:
        ``|rho0|``. **A prior, not a measurement** — the short-dated ATM skew
        identifies only ``rho * eta``. ``0.7`` is the equity-index value of
        spec 4.9; a single-name or a commodity deserves a different one.
    eta_min, eta_max:
        The ``eta0`` clip of spec 4.9 (``0.5`` / ``3.5``). Every clip that
        fires is recorded in ``diagnostics["clips"]``.
    disagreement_factor:
        Ratio between the two ``eta`` estimates above which
        :data:`FLAG_ETA_DISAGREEMENT` is raised (spec 4.9: 2).
    t_ref_rule:
        :data:`T_REF_RULE_MIN_RELATIVE_SE` (default),
        :data:`T_REF_RULE_SHORTEST` or :data:`T_REF_RULE_EXPLICIT`.
    t_ref:
        Explicit maturity in years. Implies :data:`T_REF_RULE_EXPLICIT`; it
        must match a candidate maturity within ``t_ref_rtol`` — an unmatched
        ``t_ref`` raises rather than snapping silently.
    t_ref_rtol:
        Relative tolerance of the explicit ``t_ref`` match.
    restrict_to_hurst_window:
        Restrict the ``T_ref`` candidates to the maturities inside the Hurst
        regression window, so ``eta0`` is read where ``H0`` was fitted.
    psi_floor:
        ``|psi| < psi_floor`` disqualifies a maturity as ``T_ref`` (same floor
        as :class:`~app.model.calibration.rough_vol.hurst_estimator.HurstConfig`).
    curvature_information_min:
        Threshold on :func:`curvature_information_share` below which
        :data:`FLAG_CURVATURE_ILL_CONDITIONED` is raised. For an equity index
        (``|rho| ~ 0.7``) the share is naturally only ~0.16 at ``H = 0.1``, so
        the default sits below that: the flag marks a genuinely uninformative
        curvature, not the ordinary equity case.
    sigma_atm_mismatch_rtol:
        Relative gap between the fitted ``sigma_ATM`` and the one implied by
        ``xi0_curve`` above which :data:`FLAG_SIGMA_ATM_MISMATCH` is raised.
    """

    rho_prior_abs: float = RHO_PRIOR_ABS
    eta_min: float = ETA_INIT_MIN
    eta_max: float = ETA_INIT_MAX
    disagreement_factor: float = ETA_DISAGREEMENT_FACTOR
    t_ref_rule: str = T_REF_RULE_MIN_RELATIVE_SE
    t_ref: float | None = None
    t_ref_rtol: float = 1e-6
    restrict_to_hurst_window: bool = True
    psi_floor: float = 1e-6
    curvature_information_min: float = 0.10
    sigma_atm_mismatch_rtol: float = 0.25

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rho_prior_abs) and 0.0 < abs(self.rho_prior_abs)):
            raise ValueError(
                f"rho_prior_abs must be finite and non-zero; got {self.rho_prior_abs!r}"
            )
        if not (math.isfinite(self.eta_min) and self.eta_min > 0.0):
            raise ValueError(f"eta_min must be finite and > 0; got {self.eta_min!r}")
        if not (math.isfinite(self.eta_max) and self.eta_max > self.eta_min):
            raise ValueError(
                f"eta_max must be finite and > eta_min; got {self.eta_max!r}"
            )
        if self.eta_max > ETA_MAX:
            raise ValueError(
                f"eta_max must not exceed the simulator bound ETA_MAX={ETA_MAX}; "
                f"got {self.eta_max!r}"
            )
        if not (math.isfinite(self.disagreement_factor) and self.disagreement_factor > 1.0):
            raise ValueError(
                f"disagreement_factor must be finite and > 1; got {self.disagreement_factor!r}"
            )
        rules = (T_REF_RULE_MIN_RELATIVE_SE, T_REF_RULE_SHORTEST, T_REF_RULE_EXPLICIT)
        if self.t_ref_rule not in rules:
            raise ValueError(f"t_ref_rule must be one of {rules}; got {self.t_ref_rule!r}")
        if self.t_ref is not None and not (
            math.isfinite(float(self.t_ref)) and float(self.t_ref) > 0.0
        ):
            raise ValueError(f"t_ref must be finite and > 0 when given; got {self.t_ref!r}")

    @property
    def effective_t_ref_rule(self) -> str:
        """The rule actually applied: an explicit ``t_ref`` always wins."""
        if self.t_ref is not None:
            return T_REF_RULE_EXPLICIT
        return str(self.t_ref_rule)


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _read_hurst(hurst_result: Any) -> dict[str, Any]:
    """
    Accept a ``HurstEstimate``, its ``to_dict()`` or any object exposing the
    same attributes, and return the plain fields this module needs.
    """
    if hurst_result is None:
        raise RBergomiInitializationError(
            "Initialisation rBergomi impossible : aucune estimation de H fournie."
        )

    def _get(name: str, default: Any) -> Any:
        if isinstance(hurst_result, Mapping):
            return hurst_result.get(name, default)
        return getattr(hurst_result, name, default)

    diagnostics = _get("diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    window = _get("window", None)
    window_pair: tuple[float, float] | None = None
    if window is not None:
        try:
            values = [float(v) for v in window]
        except (TypeError, ValueError):
            values = []
        if len(values) == 2 and math.isfinite(values[0]) and math.isfinite(values[1]):
            window_pair = (values[0], values[1])

    h0 = _get("H0", None)
    if h0 is None:
        raise RBergomiInitializationError(
            "Initialisation rBergomi impossible : l'estimation de H ne porte pas de H0."
        )
    return {
        "H0": float(h0),
        "unstable": bool(_get("unstable", False)),
        "window": window_pair,
        "se": float(_get("se", float("nan"))),
        "r2": float(_get("r2", float("nan"))),
        "n_expiries": int(_get("n_expiries", 0) or 0),
        "diagnostics": dict(diagnostics),
    }


def _skew_point_from_mapping(payload: Mapping[str, Any]) -> SkewPoint | None:
    """Rehydrate a :class:`SkewPoint` from its ``to_dict()`` payload."""
    names = {f.name for f in dataclasses.fields(SkewPoint)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in names:
            continue
        if key == "window":
            try:
                pair = [float(v) for v in value]
            except (TypeError, ValueError):
                continue
            if len(pair) != 2:
                continue
            kwargs[key] = (pair[0], pair[1])
        elif key == "flags":
            kwargs[key] = tuple(str(f) for f in (value or ()))
        elif key in {"n_strikes", "n_left", "n_right", "dof", "iterations"}:
            kwargs[key] = int(value)
        elif key == "converged":
            kwargs[key] = bool(value)
        elif key == "weight_scheme":
            kwargs[key] = str(value)
        else:
            kwargs[key] = float(value)
    if "T" not in kwargs or "psi" not in kwargs:
        return None
    kwargs.setdefault("se", float("nan"))
    kwargs.setdefault("n_strikes", 0)
    kwargs.setdefault("window", (float("nan"), float("nan")))
    return SkewPoint(**kwargs)


def _resolve_skew_points(
    hurst: Mapping[str, Any],
    option_surface: Any,
    *,
    skew_curve: Sequence[Any] | None,
    forward_curve: Any,
    skew_config: SkewConfig | None,
    clean_chains: Sequence[Any] | None,
    variance_curve: Any,
) -> tuple[list[SkewPoint], str]:
    """
    Resolve the skew curve, preferring the points the Hurst fit actually saw.

    Order: an explicit ``skew_curve``; then the ``skew_points`` recorded in the
    Hurst diagnostics (rehydrated — this guarantees ``eta0`` is read off the
    very points ``H0`` was fitted on); then a fresh
    :func:`~app.model.calibration.rough_vol.hurst_estimator.build_skew_curve`
    over ``option_surface``.
    """
    if skew_curve is not None:
        points = []
        for item in skew_curve:
            if isinstance(item, SkewPoint):
                points.append(item)
            elif isinstance(item, Mapping):
                rehydrated = _skew_point_from_mapping(item)
                if rehydrated is not None:
                    points.append(rehydrated)
        if points:
            points.sort(key=lambda sp: float(sp.T))
            return points, "skew_curve"

    recorded = hurst["diagnostics"].get("skew_points")
    if isinstance(recorded, Sequence) and not isinstance(recorded, (str, bytes)):
        points = []
        for item in recorded:
            if isinstance(item, SkewPoint):
                points.append(item)
            elif isinstance(item, Mapping):
                rehydrated = _skew_point_from_mapping(item)
                if rehydrated is not None:
                    points.append(rehydrated)
        if points:
            points.sort(key=lambda sp: float(sp.T))
            return points, "hurst_diagnostics"

    if option_surface is not None:
        built, _failures = build_skew_curve(
            option_surface,
            forward_curve,
            config=skew_config,
            clean_chains=clean_chains,
            variance_curve=variance_curve,
        )
        if built:
            return list(built), "option_surface"

    raise RBergomiInitializationError(
        "Initialisation rBergomi impossible : aucun point de skew exploitable "
        "(ni skew_curve, ni diagnostics de Hurst, ni surface d'options). "
        "Le produit rho*eta n'est identifié par aucune donnée."
    )


# ---------------------------------------------------------------------------
# T_ref selection
# ---------------------------------------------------------------------------


def _relative_se(point: SkewPoint) -> float:
    psi = abs(float(point.psi))
    se = float(point.se)
    if psi <= 0.0 or not math.isfinite(psi):
        return float("inf")
    if not math.isfinite(se) or se < 0.0:
        return float("inf")
    return se / psi


def select_t_ref(
    points: Sequence[SkewPoint],
    *,
    config: InitializerConfig,
    hurst_window: tuple[float, float] | None,
) -> tuple[SkewPoint, dict[str, Any], list[str]]:
    """
    Pick the reference maturity and explain the choice.

    Returns ``(chosen_point, selection_diagnostics, flags)``. See the module
    docstring for the rationale behind the default rule.
    """
    flags: list[str] = []
    rule = config.effective_t_ref_rule
    window = hurst_window if config.restrict_to_hurst_window else None

    candidates: list[dict[str, Any]] = []
    usable: list[SkewPoint] = []
    for sp in sorted(points, key=lambda p: float(p.T)):
        T = float(sp.T)
        psi = float(sp.psi)
        in_window = True
        if window is not None:
            in_window = bool(window[0] <= T <= window[1])
        finite = math.isfinite(T) and T > 0.0 and math.isfinite(psi)
        above_floor = finite and abs(psi) >= float(config.psi_floor)
        is_usable = bool(finite and above_floor and in_window)
        candidates.append(
            {
                "T": T,
                "psi": psi,
                "se": float(sp.se),
                "relative_se": float(_relative_se(sp)),
                "n_strikes": int(sp.n_strikes),
                "curvature": float(sp.curvature),
                "sigma_atm": float(sp.sigma_atm),
                "in_hurst_window": bool(in_window),
                "usable": is_usable,
            }
        )
        if is_usable:
            usable.append(sp)

    if not usable and window is not None:
        # Fall back to the whole curve rather than refusing, but say so.
        flags.append(FLAG_T_REF_OUTSIDE_HURST_WINDOW)
        for entry, sp in zip(candidates, sorted(points, key=lambda p: float(p.T))):
            T = float(sp.T)
            psi = float(sp.psi)
            finite = math.isfinite(T) and T > 0.0 and math.isfinite(psi)
            is_usable = bool(finite and abs(psi) >= float(config.psi_floor))
            entry["usable"] = is_usable
            if is_usable:
                usable.append(sp)

    if not usable:
        raise RBergomiInitializationError(
            "Initialisation rBergomi impossible : aucune échéance ne porte un skew "
            "ATM exploitable (|psi| au-dessus du plancher, T > 0). "
            "Le produit rho*eta n'est identifié par aucune donnée."
        )

    if rule == T_REF_RULE_EXPLICIT:
        target = float(config.t_ref) if config.t_ref is not None else float("nan")
        rtol = float(config.t_ref_rtol)
        matches = [
            sp
            for sp in usable
            if abs(float(sp.T) - target) <= rtol * max(abs(target), 1e-12)
        ]
        if not matches:
            raise RBergomiInitializationError(
                f"T_ref explicite {target!r} ne correspond à aucune échéance exploitable "
                f"({[round(float(sp.T), 10) for sp in usable]}) à la tolérance "
                f"relative {rtol}. Aucune approximation silencieuse n'est faite."
            )
        chosen = matches[0]
        criterion = "explicit"
    elif rule == T_REF_RULE_SHORTEST:
        chosen = min(usable, key=lambda sp: float(sp.T))
        criterion = "shortest maturity"
    else:
        chosen = min(usable, key=lambda sp: (_relative_se(sp), float(sp.T)))
        criterion = "min SE(psi)/|psi|, ties to the shorter maturity"

    selection = {
        "rule": rule,
        "criterion": criterion,
        "restricted_to_hurst_window": bool(config.restrict_to_hurst_window),
        "hurst_window": [float(window[0]), float(window[1])] if window else None,
        "n_candidates": int(len(candidates)),
        "n_usable": int(len(usable)),
        "candidates": candidates,
        "chosen_T": float(chosen.T),
        "rationale_fr": (
            "T_ref est l'échéance de la fenêtre de régression de H dont le skew est "
            "le mieux mesuré (plus faible SE(psi)/|psi|) : la constante c(H) est "
            "précise à ~2 % de façon quasi uniforme sur 5-80 jours, donc c'est la "
            "précision de mesure, et non la validité asymptotique, qui contraint."
            if rule == T_REF_RULE_MIN_RELATIVE_SE
            else (
                "T_ref est l'échéance la plus courte exploitable (relation "
                "asymptotique T -> 0)."
                if rule == T_REF_RULE_SHORTEST
                else "T_ref a été imposée par l'appelant."
            )
        ),
    }
    return chosen, selection, flags


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


def _apply_clip(
    name: str,
    value: float,
    lo: float,
    hi: float,
    clips: list[dict[str, Any]],
) -> tuple[float, bool]:
    """Clip, record and report whether the clip fired."""
    raw = float(value)
    if not math.isfinite(raw):
        clips.append(
            {
                "parameter": name,
                "raw": raw,
                "value": raw,
                "min": float(lo),
                "max": float(hi),
                "applied": False,
                "reason": "non_finite_input",
            }
        )
        return raw, False
    clipped = min(max(raw, float(lo)), float(hi))
    applied = clipped != raw
    clips.append(
        {
            "parameter": name,
            "raw": raw,
            "value": float(clipped),
            "min": float(lo),
            "max": float(hi),
            "applied": bool(applied),
            "bound": ("min" if applied and clipped == float(lo) else
                      "max" if applied and clipped == float(hi) else None),
        }
    )
    return float(clipped), bool(applied)


# ---------------------------------------------------------------------------
# Public entry point (spec 4.9 / 5)
# ---------------------------------------------------------------------------


def initialize_rbergomi_parameters(
    hurst_result: Any,
    option_surface: Any,
    *,
    skew_curve: Sequence[Any] | None = None,
    xi0_curve: Any = None,
    forward_curve: Any = None,
    config: InitializerConfig | None = None,
    skew_config: SkewConfig | None = None,
    clean_chains: Sequence[Any] | None = None,
    variance_curve: Any = None,
) -> tuple[float, float, float, dict[str, Any]]:
    """
    Seed the rough Bergomi joint calibration: ``(H0, eta0, rho0, diagnostics)``.

    Parameters
    ----------
    hurst_result:
        The spec-4.5
        :class:`~app.model.calibration.rough_vol.hurst_estimator.HurstEstimate`
        (or its ``to_dict()``). Its ``unstable`` flag is propagated, never
        swallowed: an unstable estimate yields a labelled initialisation, not
        an exception.
    option_surface:
        The Phase-1 OTM surface, in any shape
        :func:`~app.model.calibration.rough_vol.hurst_estimator.build_skew_curve`
        accepts. Used only when no skew curve can be recovered from
        ``skew_curve`` or from the Hurst diagnostics; may be ``None`` then.
    skew_curve:
        An explicit ``Sequence[SkewPoint]`` (or their dict payloads). Highest
        precedence.
    xi0_curve:
        The Phase-2 forward-variance curve. Used to reconstruct
        ``sigma_ATM(T_ref)`` when the skew fit did not produce one, and always
        reported as an independent cross-check of that level.
    forward_curve, skew_config, clean_chains, variance_curve:
        Passed through to ``build_skew_curve`` on the fallback path only.
    config:
        :class:`InitializerConfig`. Defaults are the spec-4.9 values.

    Returns
    -------
    (H0, eta0, rho0, diagnostics)
        ``H0`` clipped into the simulator's ``[H_MIN, H_MAX]``, ``eta0`` into
        ``[eta_min, eta_max]``, ``rho0 = sign(psi) * rho_prior_abs``. Every clip
        that fired is listed in ``diagnostics["clips"]``.

    Raises
    ------
    RBergomiInitializationError
        When no usable skew point exists at all: ``rho * eta`` is then
        identified by no data and no honest number can be returned.

    Notes
    -----
    **The returned triple is a starting point.** ``rho0`` in particular is a
    prior; ``diagnostics["rho_eta_degeneracy"]`` states exactly what the data
    did and did not identify.
    """
    cfg = config or InitializerConfig()
    hurst = _read_hurst(hurst_result)
    points, skew_source = _resolve_skew_points(
        hurst,
        option_surface,
        skew_curve=skew_curve,
        forward_curve=forward_curve,
        skew_config=skew_config,
        clean_chains=clean_chains,
        variance_curve=variance_curve,
    )

    flags: list[str] = []
    clips: list[dict[str, Any]] = []

    # ---- H0 -------------------------------------------------------------
    h0_raw = float(hurst["H0"])
    H0, h_clipped = _apply_clip("H0", h0_raw, H_MIN, H_MAX, clips)
    if h_clipped:
        flags.append(FLAG_H_CLIPPED)
    if hurst["unstable"]:
        flags.append(FLAG_HURST_UNSTABLE)

    # ---- T_ref ----------------------------------------------------------
    point, selection, t_ref_flags = select_t_ref(
        points, config=cfg, hurst_window=hurst["window"]
    )
    flags.extend(t_ref_flags)
    T_ref = float(point.T)
    psi = float(point.psi)

    # ---- rho0 -----------------------------------------------------------
    if psi > 0.0:
        psi_sign = 1
    elif psi < 0.0:
        psi_sign = -1
    else:
        psi_sign = 0
    if psi_sign == 0:
        flags.append(FLAG_PSI_SIGN_AMBIGUOUS)
        sign_used = -1  # equity-index prior: a negative skew, hence rho < 0
    else:
        sign_used = psi_sign
    dominant_sign = hurst["diagnostics"].get("dominant_sign")
    dominant_sign = int(dominant_sign) if isinstance(dominant_sign, (int, float)) else None
    if dominant_sign is not None and psi_sign != 0 and dominant_sign != psi_sign:
        flags.append(FLAG_SIGN_DISAGREES_WITH_CURVE)

    rho_abs_raw = abs(float(cfg.rho_prior_abs))
    rho_abs, rho_clipped = _apply_clip(
        "rho0_abs", rho_abs_raw, 1e-12, RHO_ABS_MAX, clips
    )
    if rho_clipped:
        flags.append(FLAG_RHO_CLIPPED)
    rho0 = float(sign_used) * rho_abs

    # ---- eta0 from the skew (the spec-4.9 estimate) ---------------------
    c_H = c_of_H(H0)
    eta_skew_raw = eta_from_skew(psi, H=H0, rho_abs=rho_abs, T=T_ref)
    eta0, eta_clipped = _apply_clip(
        "eta0", eta_skew_raw, cfg.eta_min, cfg.eta_max, clips
    )
    if eta_clipped:
        flags.append(FLAG_ETA_CLIPPED)
    if not math.isfinite(eta0):
        raise RBergomiInitializationError(
            "Initialisation rBergomi impossible : le skew de référence ne permet pas "
            f"d'inverser eta (psi={psi!r}, T_ref={T_ref!r}, H0={H0!r})."
        )

    # ---- sigma_ATM(T_ref) ----------------------------------------------
    sigma_fit = float(point.sigma_atm)
    sigma_xi0 = float("nan")
    if xi0_curve is not None:
        try:
            total_variance = float(xi0_curve.integrated(T_ref))
        except (AttributeError, TypeError, ValueError):
            total_variance = float("nan")
        if math.isfinite(total_variance) and total_variance > 0.0 and T_ref > 0.0:
            sigma_xi0 = math.sqrt(total_variance / T_ref)
    if math.isfinite(sigma_fit) and sigma_fit > 0.0:
        sigma_atm = sigma_fit
        sigma_source = "skew_fit"
        if math.isfinite(sigma_xi0) and sigma_xi0 > 0.0:
            if abs(sigma_fit - sigma_xi0) > float(cfg.sigma_atm_mismatch_rtol) * sigma_xi0:
                flags.append(FLAG_SIGMA_ATM_MISMATCH)
    elif math.isfinite(sigma_xi0) and sigma_xi0 > 0.0:
        sigma_atm = sigma_xi0
        sigma_source = "xi0_curve"
        flags.append(FLAG_SIGMA_ATM_FROM_XI0)
    else:
        sigma_atm = float("nan")
        sigma_source = "unavailable"

    # ---- eta from the curvature (the independent cross-check) -----------
    curvature = float(point.curvature)
    eta_curvature = eta_from_curvature(
        curvature, psi=psi, H=H0, T=T_ref, sigma_atm=sigma_atm
    )
    info_share = curvature_information_share(
        curvature, psi=psi, sigma_atm=sigma_atm
    )
    eta_ratio = float("nan")
    disagreement = False
    if math.isfinite(eta_curvature) and eta_curvature > 0.0 and eta_skew_raw > 0.0:
        eta_ratio = max(eta_skew_raw / eta_curvature, eta_curvature / eta_skew_raw)
        disagreement = bool(eta_ratio > float(cfg.disagreement_factor))
        if disagreement:
            flags.append(FLAG_ETA_DISAGREEMENT)
    else:
        flags.append(FLAG_NO_CURVATURE_ESTIMATE)
    if math.isfinite(info_share) and info_share < float(cfg.curvature_information_min):
        flags.append(FLAG_CURVATURE_ILL_CONDITIONED)

    #: |rho| implied by pairing the measured skew with the curvature-based eta.
    #: Not used to set rho0 — reported because it is the most direct read on how
    #: far the equity prior is from this surface, and Stage 1 can bracket on it.
    implied_rho_abs = float("nan")
    if math.isfinite(eta_curvature) and eta_curvature > 0.0 and math.isfinite(c_H) and c_H > 0.0:
        implied_rho_abs = abs(psi) / (c_H * eta_curvature * T_ref ** (H0 - 0.5))

    # ---- the regression-amplitude alternative (diagnostic only) ---------
    amplitude = hurst["diagnostics"].get("amplitude_A")
    eta_amplitude = float("nan")
    if (
        not hurst["unstable"]
        and isinstance(amplitude, (int, float))
        and math.isfinite(float(amplitude))
        and math.isfinite(c_H)
        and c_H > 0.0
        and rho_abs > 0.0
    ):
        eta_amplitude = float(amplitude) / (c_H * rho_abs)

    # ---- diagnostics ----------------------------------------------------
    product_identified = rho0 * eta_skew_raw
    warnings_fr = [
        INITIALIZER_FLAG_LABELS_FR.get(f, f) for f in dict.fromkeys(flags)
    ]
    diagnostics: dict[str, Any] = {
        "H0": float(H0),
        "H0_input": float(h0_raw),
        "H0_is_fallback": bool(hurst["unstable"]),
        "hurst_unstable": bool(hurst["unstable"]),
        "hurst_window": (
            [float(hurst["window"][0]), float(hurst["window"][1])]
            if hurst["window"]
            else None
        ),
        "hurst_se": float(hurst["se"]),
        "hurst_r2": float(hurst["r2"]),
        "hurst_n_expiries": int(hurst["n_expiries"]),
        "hurst_rejection_reasons": [
            str(r) for r in hurst["diagnostics"].get("rejection_reasons", [])
        ],
        "hurst_rejection_reasons_fr": [
            str(r) for r in hurst["diagnostics"].get("rejection_reasons_fr", [])
        ],
        "T_ref": float(T_ref),
        "T_ref_days": float(T_ref * 365.0),
        "T_ref_rule": str(cfg.effective_t_ref_rule),
        "T_ref_selection": selection,
        "skew_source": str(skew_source),
        "n_skew_points": int(len(points)),
        "skew_fit": point.to_dict(),
        "c_of_H": float(c_H),
        "c_of_H_formula": C_OF_H_FORMULA,
        "c_of_H_provenance": C_OF_H_PROVENANCE,
        "curvature_coefficient_a": float(curvature_coefficient(H0)),
        "curvature_rho_coefficient_b": float(curvature_rho_coefficient(H0)),
        "curvature_coefficient_provenance": CURVATURE_COEFFICIENT_PROVENANCE,
        "curvature_sign_change_rho_abs": float(curvature_sign_change_rho_abs(H0)),
        "sigma_atm": float(sigma_atm),
        "sigma_atm_source": str(sigma_source),
        "sigma_atm_from_fit": float(sigma_fit),
        "sigma_atm_from_xi0": float(sigma_xi0),
        "psi_at_t_ref": float(psi),
        "psi_se_at_t_ref": float(point.se),
        "psi_sign": int(psi_sign),
        "dominant_psi_sign": dominant_sign,
        "sign_consistent": bool(hurst["diagnostics"].get("sign_consistent", True)),
        "curvature_at_t_ref": float(curvature),
        "eta0": float(eta0),
        "eta0_skew": float(eta_skew_raw),
        "eta0_skew_unclipped": float(eta_skew_raw),
        "eta0_curvature": float(eta_curvature),
        "eta0_from_regression_amplitude": float(eta_amplitude),
        "eta_source": "skew",
        "eta_ratio": float(eta_ratio),
        "eta_disagreement": bool(disagreement),
        "eta_disagreement_factor": float(cfg.disagreement_factor),
        "curvature_information_share": float(info_share),
        "curvature_information_min": float(cfg.curvature_information_min),
        "curvature_ill_conditioned": bool(
            math.isfinite(info_share) and info_share < float(cfg.curvature_information_min)
        ),
        "implied_abs_rho_from_curvature": float(implied_rho_abs),
        "rho0": float(rho0),
        "rho_prior_abs": float(rho_abs),
        "rho_prior_abs_input": float(rho_abs_raw),
        "eta_min": float(cfg.eta_min),
        "eta_max": float(cfg.eta_max),
        "clips": clips,
        "rho_eta_degeneracy": {
            "identified_quantity": "rho * eta",
            "identified_value": float(product_identified),
            "product_rho_eta": float(rho0 * eta0),
            "product_rho_eta_unclipped": float(product_identified),
            "product_preserved_after_clip": bool(not eta_clipped),
            "rho_is_a_prior": True,
            "rho_prior_abs": float(rho_abs),
            "eta_is_conditional_on_rho_prior": True,
            "invariant": (
                "eta0 * |rho0| is fixed by the data; rescaling |rho0| rescales "
                "eta0 by the exact inverse factor and leaves rho0 * eta0 unchanged "
                "(before the [eta_min, eta_max] clip)."
            ),
            "independent_check": "ATM curvature (rho-free, see eta0_curvature)",
            "broken_by": "spec 4.10 Stage 1 (joint surface fit)",
            "message_fr": (
                "Le skew ATM court terme n'identifie que le PRODUIT rho*eta "
                f"({product_identified:.4f}) : rho0 = {rho0:+.2f} est un a priori "
                "(actions-indices), eta0 en découle. L'étape 1 de la calibration "
                "jointe (spec 4.10) est ce qui lève la dégénérescence."
            ),
        },
        "flags": list(dict.fromkeys(flags)),
        "warnings_fr": warnings_fr,
        "is_initial_guess": True,
        "message_fr": (
            f"Initialisation rBergomi : H0 = {H0:.4f}"
            f"{' (repli, estimation instable)' if hurst['unstable'] else ''}, "
            f"eta0 = {eta0:.3f}, rho0 = {rho0:+.2f}, "
            f"T_ref = {T_ref * 365.0:.1f} j. "
            "Point de départ de l'optimiseur — jamais un résultat de calibration."
        ),
    }

    return float(H0), float(eta0), float(rho0), diagnostics


def initial_rbergomi_params(
    hurst_result: Any,
    option_surface: Any,
    **kwargs: Any,
) -> tuple[RBergomiParams, dict[str, Any]]:
    """
    :func:`initialize_rbergomi_parameters` packed into the spec-6 data contract.

    Reuses the Phase-3
    :class:`~app.model.volatility_models.rbergomi.simulator_xi_curve.RBergomiParams`
    rather than introducing a near-duplicate ``RoughBergomiParams``: it already
    is ``(H, eta, rho)`` with the spec-4.6 bounds enforced and ``xi0``
    deliberately absent, which is exactly the contract spec 6 describes.
    """
    H0, eta0, rho0, diagnostics = initialize_rbergomi_parameters(
        hurst_result, option_surface, **kwargs
    )
    return RBergomiParams(H=H0, eta=eta0, rho=rho0), diagnostics


def initializer_report(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Plain-python summary for the Phase-5 report; ``_json_safe`` compatible."""
    degeneracy = dict(diagnostics.get("rho_eta_degeneracy", {}) or {})
    return {
        "H0": float(diagnostics.get("H0", float("nan"))),
        "eta0": float(diagnostics.get("eta0", float("nan"))),
        "rho0": float(diagnostics.get("rho0", float("nan"))),
        "H0_is_fallback": bool(diagnostics.get("H0_is_fallback", False)),
        "T_ref": float(diagnostics.get("T_ref", float("nan"))),
        "T_ref_days": float(diagnostics.get("T_ref_days", float("nan"))),
        "T_ref_rule": str(diagnostics.get("T_ref_rule", "")),
        "c_of_H": float(diagnostics.get("c_of_H", float("nan"))),
        "eta0_skew": float(diagnostics.get("eta0_skew", float("nan"))),
        "eta0_curvature": float(diagnostics.get("eta0_curvature", float("nan"))),
        "eta_ratio": float(diagnostics.get("eta_ratio", float("nan"))),
        "eta_disagreement": bool(diagnostics.get("eta_disagreement", False)),
        "curvature_information_share": float(
            diagnostics.get("curvature_information_share", float("nan"))
        ),
        "implied_abs_rho_from_curvature": float(
            diagnostics.get("implied_abs_rho_from_curvature", float("nan"))
        ),
        "clips_applied": [
            str(c.get("parameter"))
            for c in diagnostics.get("clips", [])
            if c.get("applied")
        ],
        "rho_eta_degeneracy_fr": str(degeneracy.get("message_fr", "")),
        "identified_product_rho_eta": float(
            degeneracy.get("identified_value", float("nan"))
        ),
        "flags": [str(f) for f in diagnostics.get("flags", [])],
        "warnings_fr": [str(w) for w in diagnostics.get("warnings_fr", [])],
        "message_fr": str(diagnostics.get("message_fr", "")),
        "warning_fr": (
            "Valeurs initiales pour l'optimiseur rBergomi (skew ATM court terme et "
            "courbure ATM, constantes asymptotiques mesurées) — jamais un résultat "
            "de calibration."
        ),
    }


__all__ = [
    "CURVATURE_COEFFICIENT_PROVENANCE",
    "C_OF_H_FORMULA",
    "C_OF_H_PROVENANCE",
    "ETA_DISAGREEMENT_FACTOR",
    "ETA_INIT_MAX",
    "ETA_INIT_MIN",
    "FLAG_CURVATURE_ILL_CONDITIONED",
    "FLAG_ETA_CLIPPED",
    "FLAG_ETA_DISAGREEMENT",
    "FLAG_HURST_UNSTABLE",
    "FLAG_H_CLIPPED",
    "FLAG_NO_CURVATURE_ESTIMATE",
    "FLAG_PSI_SIGN_AMBIGUOUS",
    "FLAG_RHO_CLIPPED",
    "FLAG_SIGMA_ATM_FROM_XI0",
    "FLAG_SIGMA_ATM_MISMATCH",
    "FLAG_SIGN_DISAGREES_WITH_CURVE",
    "FLAG_T_REF_OUTSIDE_HURST_WINDOW",
    "INITIALIZER_FLAG_LABELS_FR",
    "InitializerConfig",
    "RBergomiInitializationError",
    "RHO_PRIOR_ABS",
    "T_REF_RULE_EXPLICIT",
    "T_REF_RULE_MIN_RELATIVE_SE",
    "T_REF_RULE_SHORTEST",
    "atm_curvature_model",
    "atm_skew_model",
    "c_of_H",
    "curvature_coefficient",
    "curvature_information_share",
    "curvature_rho_coefficient",
    "curvature_sign_change_rho_abs",
    "eta_from_curvature",
    "eta_from_skew",
    "initial_rbergomi_params",
    "initialize_rbergomi_parameters",
    "initializer_report",
    "select_t_ref",
]
