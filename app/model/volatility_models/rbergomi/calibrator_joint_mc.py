"""
Joint ``(H, eta, rho)`` rough Bergomi calibration with ``xi0`` FROZEN (spec 4.10)
plus the identifiability diagnostics of spec 4.11.

This module is the last numerical stage of the rough-volatility pipeline.  The
four preceding separations are already done and are **not** re-litigated here:

===== ============================================ ==================================
Stage What it produces                             Module
===== ============================================ ==================================
A     ``xi0`` from the variance-swap curve         ``rough_vol/variance_swap.py`` +
                                                   ``rough_vol/forward_variance.py``
B     ``H0`` from the short-maturity ATM skew      ``rough_vol/hurst_estimator.py``
C     ``(eta0, rho0)`` seeds                       ``rbergomi/initializer.py``
D     the joint fit of ``(H, eta, rho)``           **this module**
===== ============================================ ==================================

WHAT IS OPTIMISED, AND WHAT CANNOT BE
=====================================
The optimizer moves a **three**-component vector ``theta = (H, eta, rho)`` inside
``0.01 <= H <= 0.49``, ``0.05 <= eta <= 5.0``, ``-0.999 <= rho <= 0.999``.
``xi0`` is not a component of ``theta``, is not a key of the bounds dictionary,
and is not reachable from the objective's argument: it enters as a
:class:`FrozenXi0` holder wrapping the spec-4.4
:class:`~app.model.calibration.rough_vol.forward_variance.ForwardVarianceCurve`,
which is itself a frozen dataclass of tuples with non-writeable cached arrays.
The holder records the curve's SHA-256 content fingerprint at construction and
:meth:`FrozenXi0.verify` re-checks it after the optimisation; the very same
object is handed back in :attr:`JointCalibrationResult.xi0_curve`, so a caller
can assert *identity*, not merely equality.  Passing ``"xi0"`` through the
constraints protocol of :class:`RBergomiJointHCalibrator` is **rejected with an
explicit French error**, never silently ignored.

COMMON RANDOM NUMBERS - WHY THE OBJECTIVE IS DETERMINISTIC AND SMOOTH
=====================================================================
Every evaluation of the objective at a given stage uses the *same* underlying
standard normals.  Concretely:

* the simulation grid is built **once** (from the union of the quoted
  maturities) and reused for every ``theta``, so the draw's shape
  ``(n_base_paths, 3n)`` never changes;
* ``simulate_rbergomi_xi_curve`` is called with a **fixed**
  ``SimulationConfig.seed`` for the whole stage, and the underlying
  ``draw_joint_gaussian`` seeds a fresh ``numpy.random.default_rng(seed)`` and
  consumes exactly ``2n`` then ``n`` normals per base path.  Same seed + same
  shape => **bit-identical Z**, whatever ``(H, eta, rho)`` is;
* only the Cholesky factor ``L(H)`` changes with ``H``; it is memoised by
  ``(H, grid hash)`` in ``volterra_gaussian.cholesky_factor``.  ``W~ = Z L(H)^T``
  is a continuous function of ``H`` because the Cholesky map is continuous on
  the positive-definite cone, so the loss is a *smooth* function of ``theta`` on
  one draw - which is exactly what lets a derivative-free local method (and
  finite differences) work on a Monte-Carlo objective.

This is non-negotiable and it is the single largest difference with the legacy
:mod:`app.model.volatility_models.rbergomi.calibrator_mc_surrogate`, which draws
an **independent** seed per design point.  With independent seeds the objective
is a noisy function whose differences are dominated by sampling noise; with CRN
the noise cancels to first order in every *difference* of losses, which is all
an optimizer, a profile slice or a flat-parameter test ever looks at.

MATCHING THE MODEL FORWARD TO THE MARKET FORWARD
================================================
The market implied volatilities of spec 4.2 are inverted on the put-call-parity
forward ``F(T)`` and the curve discount ``D(T)``.  Comparing them to a model
whose forward is ``S0 exp(int (r - q))`` for some unrelated ``(r, q)`` would put
a pure forward mismatch straight into ``(H, eta, rho)``.  So the drift is built
**from the market forwards themselves**: with ``q = 0`` the per-step rates are

.. code-block:: text

    r_i = ( ln F(t_{i+1}) - ln F(t_i) ) / dt_i ,     ln F interpolated linearly
                                                     in T through (0, ln S0)

which makes ``paths.model_forward(T) == F_market(T)`` **exactly** at every
quoted maturity.  The path set's own discount factor is then
``D_paths(T) = S0 / F(T)``, which is *not* the market discount - and it does not
have to be: an implied volatility is invariant under a common rescaling of the
price and of ``F*D`` (``implied_vol_surface`` inverts through the Black-76
substitution), so the IV objective is unaffected.  The price-based objective is
not, so there every model price is rescaled by ``D_market(T) / D_paths(T)``
before being compared - an exact algebraic rescale, not an approximation.

THE OBJECTIVE
=============
Default (:data:`OBJECTIVE_IV`), on the **cleaned quote set**, in volatility
units:

.. code-block:: text

    Loss(theta) = sum_i w_i * ( IV_model(K_i,T_i;theta) - IV_market(K_i,T_i) )^2 ,
    sum_i w_i = 1

:data:`OBJECTIVE_PRICE_RELATIVE` replaces the residual by the repo's precedent
``(px_model - px_market) / max(price_floor, px_market)``.

Each quote is priced **out of the money** (a put below the forward, a call above
it) because put-call parity is exact on a shared sample: the two carry the same
information but the OTM leg carries far less Monte-Carlo variance, and
``implied_vol_surface`` converts the put back to a call with that exact sample
parity before inverting.

Weights combine the three ingredients spec 4.10 asks for, in the only
dimensionally consistent way:

.. code-block:: text

    s_iv_i         = clip( 0.5 * (ask - bid)_i / vega_i , s_floor , s_cap )  # vol units
    w_i            = ( median_j s_iv_j / s_iv_i )^2 , then normalised so that
                     every maturity contributes exactly 1 / n_maturities

The bid-ask spread is converted into **volatility** units through the vega
before it is inverted and squared, so ``w_i`` is a plain inverse variance in the
units the residual lives in.  The literal ``1 / spread^2`` of the spec is
dimensionally inconsistent with a vol-space objective and, worse, it *rewards*
the cheap deep wings whose absolute spread is a half tick - precisely the "do
not overweight illiquid deep-OTM options" failure the spec warns about.  The
per-maturity normalisation stops a densely quoted expiry from drowning the
others.

There is deliberately **no separate vega factor**.  Substituting the definition
gives ``w ~ vega^2 / spread^2``: the vega already enters twice, which is the
whole content of "in vol units".  Multiplying by ``vega_i / max_T vega`` on top
of that - as an earlier version did - made the measured exponent ``w ~ vega^3``
on a constant-absolute-spread surface and handed 43-67 % of every maturity's
weight to its two most at-the-money quotes.  See :class:`WeightConfig`.

Hard exclusions, applied before any weighting: ``|k| > k_max``, relative spread
above ``spread_rel_max``, a non-finite or non-positive market IV, a maturity left
with fewer than ``min_quotes_per_maturity`` quotes.  A **standardised** cut
``|k| > standardised_k_max * sigma_ATM(T) sqrt(T)`` is applied too - it is an
addition to the letter of the spec (documented, and disabled with ``None``),
because an absolute ``k`` threshold means ten standard deviations at one week
and two at two years.

TWO STAGES
==========
*Stage 1 (coarse)* - a Latin-hypercube design over the box (reusing
``optimizers.latin_hypercube_samples``), with the initial point of spec 4.9
always inserted as an anchor, evaluated with reduced paths and the conditional
estimator on ONE common draw.  The best ``top_k`` points survive.

*Stage 2 (local)* - a derivative-free local search from each of the
``settings.n_starts`` best Stage-1 points.  :data:`LOCAL_NELDER_MEAD` (default)
and :data:`LOCAL_POWELL` go through ``scipy.optimize.minimize`` with box bounds;
:data:`LOCAL_LEAST_SQUARES` goes through the repo's
``optimizers.multi_start_least_squares`` on the weighted residual vector
``sqrt(w_i) e_i`` (whose ``cost`` is exactly ``Loss / 2``).  Each restart gets
its own CRN draw (spec: *one Z per restart*), and every restart optimum is then
**re-evaluated on one common selection draw** so the winner is picked from
comparable numbers rather than from three different noise realisations.
``top_k`` is widened to ``max(top_k, n_starts)`` so the caller's restart count is
honoured; when Stage 1 is disabled there is only ever one start and
:data:`FLAG_RESTARTS_TRUNCATED` says so.

THE EVALUATION BUDGET IS THIS MODULE'S OWN
==========================================
``CalibratorSettings.max_nfev`` defaults to ``80`` across the whole repo.  On
this problem that number **truncates**: measured on the reference surface
(8 maturities 7 d ... 2 y, 88 quotes, truth ``H = 0.10, eta = 1.5, rho = -0.70``)
Nelder-Mead needs 115-162 evaluations over three free parameters, and stopping at
80 returned ``H = 0.0757 / 0.0899 / 0.1013`` across three seeds against
``0.1015 / 0.1039 / 0.0994`` at 400 - a truncation cost of up to ``dH = -0.026``
and ``d_rho = -0.09``, about ten times the grid systematic this module measures
elsewhere.  So the per-run budget defaults to
:meth:`JointMCConfig.local_max_nfev` (``55`` evaluations per **free** parameter,
i.e. 165 for the full problem) and ``settings.max_nfev`` is used only when the
caller changed it from that shared default.  The shared dataclass is read, never
modified.

WHAT ``success`` MEANS
======================
``success`` is a verdict computed from the diagnostics, never a constant.  It is
``False`` when the result carries no information about ``H``:
:data:`FLAG_H_PROFILE_FLAT`, :data:`FLAG_NO_IMPROVEMENT`, or
:data:`FLAG_PROFILE_NOT_STATIONARY` (the module's own stationarity invariant -
``theta*`` must be the cheapest point of its own profile, up to the measured
noise floor).  Everything else is a warning carried in ``flags``.  This is not
cosmetic: ``apply_degeneracy_guard`` only trips on an all-NaN surface, so a
hard-coded ``success = True`` shipped a meaningless ``H`` to the Phase-5
controller as a calibrated result - against the repo guardrail "NEVER hard-code
``H`` (or any calibrated parameter) as an output".

FINAL REPRICING
===============
One high-accuracy repricing at the optimum with a **fresh** seed and
``final_paths`` paths (default 100 000), run in batches of ``batch_paths`` and
pooled exactly (equal batches => the pooled mean is the mean of the batch means
and the pooled variance is ``sum SE_b^2 / B^2``, the batches being independent).
Both losses are reported: the in-sample CRN loss and the fresh-seed loss.

The over-fitting test (:data:`FLAG_FRESH_SEED_LOSS_GAP`) is **not** made between
those two, because they are not the same estimator: ``loss_fresh`` uses
``final_paths`` while ``loss_crn`` uses ``stage2_paths``, and measured
``E[L(12k)] = 6.647e-06`` against ``E[L(100k)] = 5.163e-06`` - a ratio of 0.777 -
so the fresh loss is structurally *smaller* and the flag needed a genuine 3.9x
over-fit before it could fire.  Instead ``theta*`` is repriced on a fresh seed at
the **matched** path count (``stage2_paths``) and compared to the loss on the
draw the local stage actually fitted; those two are reported as
``loss_crn_matched`` / ``loss_fresh_matched``.

IDENTIFIABILITY (spec 4.11)
===========================
* 1-D profile slices of the loss along ``H``, ``eta`` and ``rho``, each spanning
  the **bound-to-bound** interval with the optimum inserted;
* the ``(eta, rho)`` valley at fixed ``rho * eta`` - the degeneracy the
  short-dated ATM skew leaves behind (``MATH_ORACLE`` section 8: ``c_hat`` is
  independent of ``rho`` to three decimals, so the leading-order skew identifies
  only the product) - which Stage 1 is supposed to break;
* ``H0`` versus ``H_calibrated`` with the ``H0`` 95 % confidence interval of
  spec 4.5;
* a **measured** Monte-Carlo noise floor.  Every quantity this module tests is a
  difference of two losses taken inside ONE shared draw, and CRN cancels the
  sampling noise in exactly those differences - so the floor is measured on the
  differences themselves: the whole CRN set is re-drawn ``noise_replicates``
  times and the run-to-run standard deviation of the *same* one-step difference
  is taken (measured up to **25x** smaller than the scatter of the loss level
  across seeds, which is what the earlier version used).  The level-based
  quantity is still reported, for the one comparison that is genuinely made
  across draws;
* :data:`FLAG_H_PROFILE_FLAT` when the total loss variation along the H slice is
  **below** that floor: the surface then does not identify ``H`` at all;
* **standard errors read off the profile curvature**,
  ``SE(p) = sqrt(2 sigma_L / (d2L/dp2))``, and the weak-identification flags
  built on them.  This is the number that answers "is ``H`` identified?": the
  bound-to-bound span is dominated by the far arms of the slice and is nearly
  independent of how sharply the optimum is resolved - on a quote set restricted
  to 1 y and 2 y, three seeds returned ``H = 0.1260 / 0.0447 / 0.1780`` (spread
  0.133, i.e. 2.7x ``TOL_H``) while the span sat 250-320x above the floor in all
  three and :data:`FLAG_H_PROFILE_FLAT` never fired.  The span diagnostic is
  kept, as the secondary one;
* :data:`FLAG_PROFILE_NOT_STATIONARY` when ``theta*`` is not the cheapest point
  of its own profile up to the noise floor.

THE LOG-EULER SHORT-END SKEW BIAS (carried over from Phase 3, NOT ignored)
=========================================================================
The log-Euler scheme freezes the variance at the left endpoint of each cell, so
it under-states the model's own ATM skew, and it does so **more at the short end
than at the long end**.  Phase 3 measured roughly 9.7 % at 7 days against 3.0 %
at 2 years on the default ``GridConfig(n_max=256)``, which displaces the model's
own ``log|psi| vs log T`` exponent by about 0.02 - a systematic error on the very
parameter being calibrated.

Two things are done about it here.

1. **A finer grid is the default for calibration**: :attr:`JointMCConfig.grid_n_max`
   is ``384``, one and a half times the pipeline default of 256.  Measured on the
   8-maturity term structure of the test suite (7 d ... 2 y, 60 000 antithetic
   paths, conditional estimator, ``H=0.10, eta=1.5, rho=-0.70``, central
   difference ``dk = 0.01``), the model ATM skew ``psi(T)`` at ``n_max = 384``
   agrees with its ``n_max = 1024`` value to 0.3-1.0 % out to six months and to
   2.8 % at one and two years - the latter being of the order of the Monte-Carlo
   noise of that very measurement - whereas ``n_max = 256`` still sits about 2 %
   low at the short end (``psi(7 d) = -1.0914`` against ``-1.1111``), which is the
   maturity-dependent tilt that lands on ``H``.  ``grid_min_steps`` is *not* the
   lever: the short-end block already receives about half the node budget, so the
   shortest maturity sits at index ~188 out of 384 whatever ``min_steps`` says.
   The cost is roughly linear in ``n`` for the simulation and cubic for the
   one-off Cholesky: at 12 000 paths and 88 quotes over 8 maturities, one full
   objective evaluation measured 0.32 s at ``n_max = 256`` and 0.43 s at
   ``n_max = 384`` on the reference machine - a ~33 % surcharge per evaluation.
2. **The residual bias is measured and reported, in parameter units.**  When
   :attr:`JointMCConfig.refinement_check` is on, the loss gradient is recomputed
   at the optimum on a grid ``refinement_factor`` times finer.  At the optimum
   the gradient on the *calibration* grid is zero by construction, so the
   gradient on the refined grid is exactly the grid-induced tilt; combined with
   the diagonal of the Hessian read off the profile slices it yields a
   first-order Newton estimate of how far ``theta`` would move on the finer grid.
   That triple is reported as :class:`GridBiasReport` and, past
   :attr:`JointMCConfig.grid_bias_material` in relative terms, raises
   :data:`FLAG_GRID_BIAS_MATERIAL`.  It is an *estimate of the residual bias, not
   a correction*: nothing is silently subtracted from the calibrated parameters.
   On the reference synthetic surface the estimate came out at
   ``(dH, d_eta, d_rho) = (+0.0019, -0.0002, +0.0068)`` between ``n = 384`` and
   ``n = 768`` - an order of magnitude below the recovery tolerances, which is
   the evidence that ``n_max = 384`` is enough for this term structure.

REPO INTEGRATION
================
:class:`RBergomiJointHCalibrator` is a ``BaseSurfaceCalibrator`` with
``model = "rbergomi"`` and ``method = "joint_h_mc"``, ``PARAM_ORDER =
("H", "eta", "rho")``.  Because ``SurfaceGrid`` is frozen with *scalar* ``r`` and
``q``, everything curve-shaped rides in ``constraints``:
``constraints["xi0_curve"]`` (**required** - there is no honest way to manufacture
a forward-variance curve out of an IV grid, so its absence is an explicit
failure, not a silent fallback), ``constraints["option_surface"]`` (the Phase-1
quote set; when present the fit uses the *real* quotes and the grid is only used
for reporting), ``constraints["clean_chains"]``, ``constraints["initial_params"]``,
``constraints["mc_cfg"]``, ``constraints["weights_cfg"]``.  ``settings.seed``
makes the whole run bit-identical.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - SciPy is a hard dependency of the repo
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - defensive
    minimize = None  # type: ignore[assignment]

from app.model.calibration.base_calibrator import (
    BaseSurfaceCalibrator,
    CalibratorSettings,
    SurfaceCalibrationResult,
    SurfaceGrid,
    apply_degeneracy_guard,
)
from app.model.calibration.loss_surface import (
    compute_bs_vega_grid,
    effective_mask,
    iv_error_metrics,
    iv_error_metrics_weighted,
)
from app.model.calibration.optimizers import (
    latin_hypercube_samples,
    multi_start_least_squares,
)
from app.model.calibration.rough_vol.chain_cleaning import CleanChain
from app.model.calibration.rough_vol.forward_curve import black76_call_price
from app.model.calibration.rough_vol.forward_variance import ForwardVarianceCurve
from app.model.calibration.rough_vol.hurst_estimator import (
    black76_vega,
    build_spread_lookup,
)
from app.model.calibration.rough_vol.variance_swap import black76_put_price
from app.model.volatility_models.rbergomi.pricing import (
    ESTIMATOR_CONDITIONAL,
    ESTIMATOR_PLAIN,
    PriceResult,
    implied_vol_surface,
    price_call,
    price_put,
)
from app.model.volatility_models.rbergomi.simulator_xi_curve import (
    ETA_MAX,
    ETA_WORKING_MIN,
    H_MAX,
    H_MIN,
    RHO_ABS_MAX,
    RBergomiParams,
    RBergomiSimulationError,
    SimulationConfig,
    curve_fingerprint,
    simulate_rbergomi_xi_curve,
)
from app.model.volatility_models.rbergomi.volterra_gaussian import (
    GridConfig,
    SimulationGrid,
    VolterraGaussianError,
    build_simulation_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Weighted implied-volatility loss - the spec-4.10 reference objective.
OBJECTIVE_IV: str = "iv"
#: Weighted relative price loss, repo precedent ``residual / max(floor, px_mkt)``.
OBJECTIVE_PRICE_RELATIVE: str = "price_relative"
#: The two supported objectives.
OBJECTIVES: tuple[str, ...] = (OBJECTIVE_IV, OBJECTIVE_PRICE_RELATIVE)

#: Derivative-free simplex search (default).
LOCAL_NELDER_MEAD: str = "nelder-mead"
#: Derivative-free conjugate-direction search.
LOCAL_POWELL: str = "powell"
#: SciPy ``least_squares`` on the weighted residual vector, via the repo helper.
LOCAL_LEAST_SQUARES: str = "least_squares"
#: The three supported local methods.
LOCAL_METHODS: tuple[str, ...] = (LOCAL_NELDER_MEAD, LOCAL_POWELL, LOCAL_LEAST_SQUARES)

#: Calibrated parameter names, in optimisation order. ``xi0`` is NOT one of them.
PARAM_ORDER: tuple[str, ...] = ("H", "eta", "rho")

#: Hard box of spec 4.10.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "H": (H_MIN, H_MAX),
    "eta": (ETA_WORKING_MIN, ETA_MAX),
    "rho": (-RHO_ABS_MAX, RHO_ABS_MAX),
}

#: Two-sided 95 % normal quantile, for the ``H0`` confidence interval.
Z95: float = 1.959963984540054

#: The shared repo default of ``CalibratorSettings.max_nfev``, read once so this
#: module can tell "the caller set a budget" from "the caller passed nothing".
#: ``CalibratorSettings`` is used by every calibrator in the repo and is **not**
#: modified here - see :attr:`JointMCConfig.local_nfev_per_param`.
_SETTINGS_DEFAULT_MAX_NFEV: int = int(CalibratorSettings().max_nfev)

#: Vol-unit residual charged for a quote whose model price cannot be inverted.
UNINVERTIBLE_PENALTY_VOL: float = 0.5

#: Characteristic scale of each calibrated parameter, in its OWN units.
#:
#: These are the Monte-Carlo recovery tolerances the slow suite itself judges a
#: calibration against (``TOL_H = 0.05``, ``TOL_ETA = 0.35``, ``TOL_RHO = 0.12``
#: in ``tests/quant/test_rv_rbergomi_calibration.py``).  Every "is this
#: displacement / this standard error material?" verdict of this module is
#: expressed as a fraction of them, **not** as a fraction of the bound-to-bound
#: width: the box is ``0.48`` wide in ``H`` and ``4.95`` wide in ``eta``, so a
#: threshold defined against the width would need ``|dH| > 0.048`` and
#: ``|d_eta| > 0.495`` to fire - i.e. it would only ever complain about errors
#: already larger than the tolerance the result is accepted on.
PARAM_SCALE: dict[str, float] = {"H": 0.05, "eta": 0.35, "rho": 0.12}

FLAG_H_PROFILE_FLAT = "h_profile_flat"
FLAG_ETA_PROFILE_FLAT = "eta_profile_flat"
FLAG_RHO_PROFILE_FLAT = "rho_profile_flat"
FLAG_ETA_RHO_VALLEY_FLAT = "eta_rho_valley_flat"
FLAG_H_OUTSIDE_H0_CI = "h_calibrated_outside_h0_ci"
FLAG_PARAMETER_AT_BOUND = "parameter_at_bound"
FLAG_NO_IMPROVEMENT = "no_improvement_over_initial"
FLAG_FRESH_SEED_LOSS_GAP = "fresh_seed_loss_gap"
FLAG_GRID_BIAS_MATERIAL = "grid_discretisation_bias_material"
FLAG_LOCAL_NOT_CONVERGED = "local_stage_not_converged"
FLAG_UNINVERTIBLE_MODEL_PRICE = "uninvertible_model_price"
FLAG_H0_FALLBACK = "h0_is_fallback"
FLAG_QUOTES_DROPPED = "quotes_dropped"
FLAG_SINGLE_MATURITY = "single_maturity_quote_set"
FLAG_PARAMETER_PINNED = "parameter_pinned"
FLAG_PROFILE_NOT_STATIONARY = "optimum_not_stationary_on_profile"
FLAG_H_WEAKLY_IDENTIFIED = "h_weakly_identified"
FLAG_ETA_WEAKLY_IDENTIFIED = "eta_weakly_identified"
FLAG_RHO_WEAKLY_IDENTIFIED = "rho_weakly_identified"
FLAG_RESTARTS_TRUNCATED = "restart_count_truncated"
FLAG_REPORT_BEYOND_QUOTES = "report_maturity_beyond_quotes"
FLAG_GRID_BIAS_NOT_MEASURED = "grid_bias_not_measured"

#: Flags that mean the returned ``theta`` carries **no usable information** about
#: ``H``.  ``success`` is ``False`` whenever any of them is raised - see the
#: module docstring, section "WHAT ``success`` MEANS".
BLOCKING_FLAGS: tuple[str, ...] = (
    FLAG_H_PROFILE_FLAT,
    FLAG_NO_IMPROVEMENT,
    FLAG_PROFILE_NOT_STATIONARY,
)

#: French label for every flag this module can raise (UI / report layer).
JOINT_CALIBRATION_LABELS_FR: dict[str, str] = {
    FLAG_H_PROFILE_FLAT: (
        "Profil de H PLAT : la variation de la fonction de coût le long de H, "
        "d'une borne à l'autre, reste sous le plancher de bruit Monte-Carlo — "
        "cette surface n'identifie pas H. La valeur rendue est celle où "
        "l'optimiseur s'est arrêté sur une surface plate ; elle ne mesure rien "
        "et n'est PAS l'initialisation (le simplexe se déplace même sans "
        "signal). Calibration en échec : ne pas consommer ce H."
    ),
    FLAG_ETA_PROFILE_FLAT: (
        "Profil de eta plat : la surface n'identifie pas eta au-delà du bruit MC."
    ),
    FLAG_RHO_PROFILE_FLAT: (
        "Profil de rho plat : la surface n'identifie pas rho au-delà du bruit MC."
    ),
    FLAG_ETA_RHO_VALLEY_FLAT: (
        "Vallée (eta, rho) à produit rho*eta constant PLATE : la dégénérescence "
        "du skew court terme n'est pas levée par la surface complète."
    ),
    FLAG_H_OUTSIDE_H0_CI: (
        "H calibré hors de l'intervalle de confiance à 95 % de H0 (spec 4.5) : "
        "l'estimation asymptotique et l'ajustement joint ne concordent pas."
    ),
    FLAG_PARAMETER_AT_BOUND: (
        "Au moins un paramètre calibré est collé à une borne : l'optimum est "
        "contraint, pas intérieur."
    ),
    FLAG_NO_IMPROVEMENT: (
        "L'optimum n'améliore pas la fonction de coût au-delà du plancher de "
        "bruit Monte-Carlo par rapport au point initial."
    ),
    FLAG_FRESH_SEED_LOSS_GAP: (
        "Écart important entre le coût en échantillon (nombres aléatoires "
        "communs) et le coût hors échantillon (nouvelle graine) : l'optimiseur "
        "a en partie ajusté son propre tirage."
    ),
    FLAG_GRID_BIAS_MATERIAL: (
        "Biais de discrétisation résiduel non négligeable : sur une grille plus "
        "fine, l'optimum se déplacerait de plus que le seuil configuré."
    ),
    FLAG_LOCAL_NOT_CONVERGED: (
        "L'étape locale n'a pas convergé dans le budget max_nfev imparti."
    ),
    FLAG_UNINVERTIBLE_MODEL_PRICE: (
        "Au moins un prix modèle n'a pas pu être inversé en volatilité "
        "implicite ; le résidu correspondant est linéarisé par le véga marché."
    ),
    FLAG_H0_FALLBACK: (
        "H0 est une valeur de repli (estimation de Hurst instable) : la "
        "comparaison H0 / H calibré n'a pas de valeur probante."
    ),
    FLAG_QUOTES_DROPPED: "Des cotations ont été exclues du jeu de calibration.",
    FLAG_SINGLE_MATURITY: (
        "Une seule échéance dans le jeu de calibration : H n'est pas "
        "identifiable par construction (il vit dans la structure par terme)."
    ),
    FLAG_PARAMETER_PINNED: "Au moins un paramètre est figé par les contraintes.",
    FLAG_PROFILE_NOT_STATIONARY: (
        "Optimum NON STATIONNAIRE : sur au moins un profil, le coût au point "
        "rendu dépasse le minimum du profil de plus que le plancher de bruit "
        "mesuré sur les différences à tirage commun. L'étape locale s'est "
        "arrêtée avant d'atteindre un point stationnaire (budget max_nfev trop "
        "court, ou arrêt prématuré). Calibration en échec."
    ),
    FLAG_H_WEAKLY_IDENTIFIED: (
        "H faiblement identifié : l'erreur type déduite de la courbure du "
        "profil et du bruit Monte-Carlo mesuré dépasse le seuil configuré — la "
        "surface contraint H, mais trop lâchement pour que la valeur rendue "
        "soit exploitable au niveau de précision attendu."
    ),
    FLAG_ETA_WEAKLY_IDENTIFIED: (
        "eta faiblement identifié : erreur type issue de la courbure du profil "
        "au-delà du seuil configuré."
    ),
    FLAG_RHO_WEAKLY_IDENTIFIED: (
        "rho faiblement identifié : erreur type issue de la courbure du profil "
        "au-delà du seuil configuré."
    ),
    FLAG_RESTARTS_TRUNCATED: (
        "Nombre de redémarrages effectivement exécutés inférieur au n_starts "
        "demandé : l'étape 1 n'a pas fourni assez de points de départ distincts "
        "(design coarse désactivé). Le nombre RÉEL est reporté dans les détails."
    ),
    FLAG_REPORT_BEYOND_QUOTES: (
        "Au moins une échéance de la grille de restitution dépasse la dernière "
        "échéance cotée : le forward y est EXTRAPOLÉ au dernier taux connu et "
        "aucune cotation ne contraint le modèle au-delà — surface indicative."
    ),
    FLAG_GRID_BIAS_NOT_MEASURED: (
        "Biais de discrétisation non mesurable sur au moins un paramètre (le "
        "pas de différence finie ne tient pas dans les bornes, ou la courbure "
        "est nulle ou négative) : la valeur est rendue NaN, jamais 0."
    ),
}

REASON_NON_FINITE_IV = "non_finite_market_iv"
REASON_INVALID_MATURITY = "invalid_maturity"
REASON_INVALID_STRIKE = "invalid_strike"
REASON_K_TOO_FAR = "log_moneyness_out_of_range"
REASON_K_STANDARDISED = "standardised_moneyness_out_of_range"
REASON_SPREAD_TOO_WIDE = "relative_spread_too_wide"
REASON_ZERO_VEGA = "zero_vega"
REASON_TOO_FEW_PER_MATURITY = "too_few_quotes_at_maturity"

#: French label for every exclusion reason.
QUOTE_REASON_LABELS_FR: dict[str, str] = {
    REASON_NON_FINITE_IV: "volatilité implicite marché non finie ou négative",
    REASON_INVALID_MATURITY: "maturité invalide",
    REASON_INVALID_STRIKE: "strike, forward ou discount invalide",
    REASON_K_TOO_FAR: "log-moneyness au-delà de k_max",
    REASON_K_STANDARDISED: "moneyness standardisée au-delà du seuil",
    REASON_SPREAD_TOO_WIDE: "écart bid-ask relatif trop large",
    REASON_ZERO_VEGA: "véga nul : la cotation ne porte aucune information de vol",
    REASON_TOO_FEW_PER_MATURITY: "trop peu de cotations retenues sur cette échéance",
}


class RBergomiCalibrationError(RuntimeError):
    """A joint calibration could not be set up (bad inputs, or a moved ``xi0``)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightConfig:
    """
    Quote selection and weighting policy of the spec-4.10 objective.

    Attributes
    ----------
    objective:
        :data:`OBJECTIVE_IV` (default, the reference objective) or
        :data:`OBJECTIVE_PRICE_RELATIVE`.
    k_max:
        Hard exclusion on ``|k| = |ln(K/F)|``. ``1.0`` is deliberately permissive:
        it is the spec's absolute cut and it is *not* the one that does the work.
    standardised_k_max:
        Hard exclusion on ``|k| / (sigma_ATM(T) sqrt(T))``, i.e. moneyness in
        standard deviations. ``None`` disables it. An **addition** to the letter
        of spec 4.10, kept because an absolute ``k`` threshold is ten standard
        deviations at one week and two at two years.
    spread_rel_max:
        Hard exclusion on the relative bid-ask spread, when it is known.
    spread_iv_floor, spread_iv_cap:
        Clip applied to the half-spread expressed in volatility units before it
        is inverted and squared. The floor is what stops a zero-spread synthetic
        (or a half-tick deep wing) from receiving an unbounded weight; the cap is
        what stops a deep wing from receiving an exactly zero one (its weight can
        never fall below ``(median / spread_iv_cap)^2`` of the median quote's).
    default_spread_iv:
        Half-spread in vol units used **only when no quote in the set has a
        known spread at all**. A quote whose own spread is unknown while others
        are known is assigned the *median of the known ones* instead, which is
        the only assignment that makes it genuinely neutral - see the
        "MIXING KNOWN AND UNKNOWN SPREADS" note below.
    vega_floor_rel:
        **No longer part of the weight formula** (kept so existing configs keep
        constructing). It used to floor a separate ``vega_i / max_T vega``
        factor; that factor was removed because it multiplied the weight by
        ``vega`` a *third* time - see "THE WEIGHT EXPONENT" below. The role it
        played (no quote can silently disappear) is now played by
        ``spread_iv_cap``.

    THE WEIGHT EXPONENT (measured, and why the vega factor is gone)
    --------------------------------------------------------------
    The weight is a plain inverse variance in the objective's own units::

        s_iv_i = clip( 0.5 * (ask - bid)_i / vega_i , floor , cap )   # vol units
        w_i    = ( median_j s_iv_j / s_iv_i )^2                       # ~ 1 / s_iv^2

    then normalised per maturity.  Substituting ``s_iv = 0.5 s / vega`` gives
    ``w ~ vega^2 / s^2``, which is exactly the documented intent: the vol-space
    uncertainty of a quote is ``s / (2 vega)``, so its inverse variance is
    ``4 vega^2 / s^2``.

    The previous formula multiplied this by a further ``vega_i / max_T vega``
    factor.  With a strike-independent absolute spread - the normal market case -
    the effective weight was therefore ``w ~ vega^3``: fitting ``log w`` against
    ``log vega`` on such a surface returned ``p = 3.00`` exactly wherever no clip
    binds, and the two largest quotes of each maturity took 43-67 % of that
    maturity's entire weight (11 strikes, so a uniform split would be 18 %).
    That strips the wings, which are what carry the ``rho``/skew information, and
    it contradicted the docstring's own claim that the vega factor merely
    "tapers" them.  At ``p = 2`` the same measurement gives a top-two share of
    ~50 % and a single-quote maximum of ~28 %.

    MIXING KNOWN AND UNKNOWN SPREADS
    --------------------------------
    ``default_spread_iv`` used to be applied to every unknown-spread quote as a
    constant.  Measured on a set where half the quotes carry a 0.02 absolute
    spread and the median ``s_iv`` equals ``default_spread_iv`` exactly, the
    known-spread quotes took **98.89 %** of the total weight (mean weight ratio
    117x, overall max/min 1.25e7): an at-the-money known-spread quote has
    ``0.5 * 0.02 / vega`` far below ``spread_iv_floor``, so it was clipped up to
    the floor and received ``(0.02 / 0.002)^2 = 100``, while every unknown-spread
    quote received exactly ``1``.  Assigning the unknown ones the **median of the
    known ones (after the clip)** puts both populations on the same scale, which
    is what "neutral" has to mean.
    per_maturity_normalisation:
        Normalise so every maturity contributes exactly ``1 / n_maturities``.
    price_floor:
        ``max(price_floor, px_market)`` denominator of the price-relative loss.
    min_quotes, min_quotes_per_maturity:
        Viability thresholds. A maturity left with fewer than
        ``min_quotes_per_maturity`` quotes is dropped whole - a single quote per
        expiry would otherwise receive the entire ``1 / n_maturities`` share.
    """

    objective: str = OBJECTIVE_IV
    k_max: float = 1.0
    standardised_k_max: float | None = 3.0
    spread_rel_max: float = 0.25
    spread_iv_floor: float = 2e-3
    spread_iv_cap: float = 0.5
    default_spread_iv: float = 2e-2
    vega_floor_rel: float = 1e-3
    per_maturity_normalisation: bool = True
    price_floor: float = 1e-4
    min_quotes: int = 6
    min_quotes_per_maturity: int = 2

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES!r}; got {self.objective!r}."
            )
        for name in (
            "k_max",
            "spread_rel_max",
            "spread_iv_floor",
            "spread_iv_cap",
            "default_spread_iv",
            "vega_floor_rel",
            "price_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {value!r}.")
        if float(self.spread_iv_cap) <= float(self.spread_iv_floor):
            raise ValueError("spread_iv_cap must exceed spread_iv_floor.")
        if self.standardised_k_max is not None:
            value = float(self.standardised_k_max)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"standardised_k_max must be finite and > 0 (or None); got {value!r}."
                )
        if int(self.min_quotes) < 1 or int(self.min_quotes_per_maturity) < 1:
            raise ValueError("min_quotes and min_quotes_per_maturity must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": str(self.objective),
            "k_max": float(self.k_max),
            "standardised_k_max": (
                None if self.standardised_k_max is None else float(self.standardised_k_max)
            ),
            "spread_rel_max": float(self.spread_rel_max),
            "spread_iv_floor": float(self.spread_iv_floor),
            "spread_iv_cap": float(self.spread_iv_cap),
            "default_spread_iv": float(self.default_spread_iv),
            "vega_floor_rel": float(self.vega_floor_rel),
            "per_maturity_normalisation": bool(self.per_maturity_normalisation),
            "price_floor": float(self.price_floor),
            "min_quotes": int(self.min_quotes),
            "min_quotes_per_maturity": int(self.min_quotes_per_maturity),
        }


@dataclass(frozen=True)
class JointMCConfig:
    """
    Monte-Carlo, optimiser and diagnostics knobs of :func:`calibrate_rbergomi`.

    Attributes
    ----------
    grid_n_max, grid_min_steps:
        Simulation-grid policy. ``384`` is **finer than the pipeline default of
        256 on purpose** - see the module docstring, section on the log-Euler
        short-end skew bias.
    n_design, stage1_paths, top_k:
        Stage-1 Latin-hypercube design size (``0`` skips Stage 1 entirely and
        starts the local stage from the spec-4.9 point), its (reduced) path
        count, and how many survivors are handed to Stage 2.
    local_method, stage2_paths, xatol, fatol:
        Stage-2 local search. ``settings.n_starts`` is honoured from the caller's
        ``CalibratorSettings``; ``settings.max_nfev`` is honoured **only when the
        caller set it explicitly** - see ``local_nfev_per_param``.
    local_nfev_per_param:
        Evaluation budget of ONE local run, **per free parameter**.  The repo's
        shared ``CalibratorSettings.max_nfev`` defaults to ``80``, which is a
        sensible number for the analytic calibrators that dataclass was written
        for and a **truncating** one here.  Measured on the 8-maturity reference
        surface (7 d ... 2 y, 88 quotes, truth ``H = 0.10, eta = 1.5,
        rho = -0.70``), Nelder-Mead over the three free parameters needs 115-162
        evaluations to stop on its own convergence test:

        ===========  ==========================  ==================  =========
        ``max_nfev`` ``H`` (3 seeds)             ``loss_crn``        converged
        ===========  ==========================  ==================  =========
        80           0.0757 / 0.0899 / 0.1013    6.3e-6 ... 3.5e-6   **False**
        400          0.1015 / 0.1039 / 0.0994    7.4e-7 ... 3.4e-6   True
        ===========  ==========================  ==================  =========

        Truncation cost up to ``dH = -0.026`` and ``d_rho = -0.09``, about ten
        times the grid systematic this module measures elsewhere, and it broke
        the module's own stationarity invariant.  So the calibrator carries its
        own default, ``local_nfev_per_param * n_free`` (``55 * 3 = 165`` for the
        full problem), and falls back to ``settings.max_nfev`` only when that
        field differs from the ``CalibratorSettings`` class default - a caller
        who deliberately passes ``max_nfev=80`` is therefore indistinguishable
        from one who passed nothing, and gets the calibrator's budget. That
        ambiguity is the price of not touching a shared repo dataclass.
    crn_per_restart:
        ``True`` (spec-literal): one common-random-number draw per restart.
        Restart optima are always re-scored on ONE common *selection* draw before
        the winner is picked, so the comparison is never made across draws.
    final_paths, batch_paths:
        High-accuracy repricing at the optimum with a fresh seed, run in batches
        of at most ``batch_paths`` paths and pooled exactly.
    estimator:
        ``ESTIMATOR_CONDITIONAL`` by default - the Romano-Touzi mixed estimator,
        which is what makes a 10 000-path objective usable.
    antithetic:
        Antithetic pairing (default on). Standard errors are pair-aware.
    profile_points, profile_paths, valley_points:
        Spec-4.11 diagnostics resolution. ``profile_paths`` defaults to
        ``stage2_paths``.
    noise_replicates, noise_sigma_multiplier:
        Independent CRN **re-draws** used to *measure* the Monte-Carlo noise
        floor, and the multiplier applied to the resulting standard deviation.
        See :class:`NoiseFloor`: the floor that every span, gradient and
        improvement is judged against is the run-to-run standard deviation of a
        *difference* taken inside one draw, not of the loss level across draws.
    se_material_ratio, se_vs_h0_factor:
        Weak-identification thresholds of spec 4.11.  A parameter is flagged
        when the standard error read off its own profile curvature exceeds
        ``se_material_ratio * PARAM_SCALE[p]``; ``H`` is additionally flagged
        when its standard error exceeds ``se_vs_h0_factor`` times the spec-4.5
        ``H0`` standard error.  On the reference surface the measured values are
        ``SE(H) = 0.015`` against ``H0_se = 0.0061`` - a ratio of 2.5, well
        inside the default factor of 10.
    refinement_check, refinement_factor, grid_bias_material:
        Residual-discretisation-bias estimate on a ``refinement_factor``-times
        finer grid, and the threshold past which it is flagged.  The threshold
        is expressed as a fraction of :data:`PARAM_SCALE` - the parameter's own
        recovery tolerance - **not** of the bound-to-bound width: against the
        width, ``0.10`` needed ``|dH| > 0.048``, ``|d_eta| > 0.495`` and
        ``|d_rho| > 0.1998``, every one of them larger than the tolerance the
        calibration is accepted on (``TOL_H`` 0.05 / ``TOL_ETA`` 0.35 /
        ``TOL_RHO`` 0.12).  ``0.25`` therefore means "a quarter of the recovery
        tolerance", i.e. ``|dH| > 0.0125``.
    fresh_seed_gap_ratio:
        ``loss_fresh_matched > fresh_seed_gap_ratio * loss_crn_matched`` raises
        :data:`FLAG_FRESH_SEED_LOSS_GAP`.  **Both losses are evaluated at the
        same path count** (``stage2_paths``), one on the draw the local stage
        actually fitted and one on the final fresh seed.  Comparing
        ``loss_fresh`` at ``final_paths = 100 000`` against ``loss_crn`` at
        ``stage2_paths = 12 000``, as an earlier version did, is comparing two
        different estimators: measured ``E[L(12k)] = 6.647e-06`` against
        ``E[L(100k)] = 5.163e-06``, a ratio of 0.777, so the flag needed a
        genuine 3.9x over-fit before it could fire at ``ratio = 3`` and ~14x on a
        tighter fit.  At matched path counts a converged CRN fit of three
        parameters on ~12 000 paths still sits several times below its
        out-of-sample loss - that is real in-sample chasing of the draw, not an
        artefact - so the default is sized from the measurement (see the module
        docstring) rather than left at 3.
    bound_atol_rel:
        Relative distance to a bound below which a parameter counts as "at" it.
    """

    grid_n_max: int = 384
    grid_min_steps: int = 16
    n_design: int = 24
    stage1_paths: int = 8_000
    top_k: int = 3
    local_method: str = LOCAL_NELDER_MEAD
    stage2_paths: int = 12_000
    local_nfev_per_param: int = 55
    xatol: float = 1e-3
    fatol: float = 1e-10
    crn_per_restart: bool = True
    final_paths: int = 100_000
    batch_paths: int = 20_000
    estimator: str = ESTIMATOR_CONDITIONAL
    antithetic: bool = True
    profile_points: int = 7
    profile_paths: int | None = None
    valley_points: int = 7
    noise_replicates: int = 3
    noise_sigma_multiplier: float = 2.0
    se_material_ratio: float = 1.0
    se_vs_h0_factor: float = 10.0
    refinement_check: bool = True
    refinement_factor: int = 2
    grid_bias_material: float = 0.25
    fresh_seed_gap_ratio: float = 12.0
    bound_atol_rel: float = 1e-3

    def __post_init__(self) -> None:
        if self.local_method not in LOCAL_METHODS:
            raise ValueError(
                f"local_method must be one of {LOCAL_METHODS!r}; got {self.local_method!r}."
            )
        if self.estimator not in (ESTIMATOR_PLAIN, ESTIMATOR_CONDITIONAL):
            raise ValueError(f"Unsupported estimator {self.estimator!r}.")
        for name in (
            "grid_n_max",
            "grid_min_steps",
            "stage1_paths",
            "top_k",
            "stage2_paths",
            "local_nfev_per_param",
            "final_paths",
            "batch_paths",
            "valley_points",
            "noise_replicates",
            "refinement_factor",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1; got {getattr(self, name)!r}.")
        for name in ("se_material_ratio", "se_vs_h0_factor"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0; got {value!r}.")
        if int(self.n_design) < 0:
            raise ValueError(
                f"n_design must be >= 0 (0 skips Stage 1); got {self.n_design!r}."
            )
        if int(self.profile_points) < 3:
            raise ValueError("profile_points must be >= 3 to describe a slice.")
        if self.profile_paths is not None and int(self.profile_paths) < 1:
            raise ValueError("profile_paths must be >= 1 when given.")
        if not (math.isfinite(self.grid_bias_material) and self.grid_bias_material > 0.0):
            raise ValueError("grid_bias_material must be finite and > 0.")

    @property
    def effective_profile_paths(self) -> int:
        """Path count used by the profiles, the valley and the noise floor."""
        return int(self.stage2_paths if self.profile_paths is None else self.profile_paths)

    def local_max_nfev(self, n_free: int) -> int:
        """
        This module's own Stage-2 evaluation budget, sized by the free-parameter count.

        ``local_nfev_per_param * max(1, n_free)``; ``55 * 3 = 165`` for the full
        three-parameter problem, against the 115-162 evaluations Nelder-Mead was
        measured to need there.  Used whenever the caller left
        ``CalibratorSettings.max_nfev`` at its class default.
        """
        return int(self.local_nfev_per_param) * max(1, int(n_free))

    def grid_config(self, *, n_max: int | None = None) -> GridConfig:
        """The ``GridConfig`` this configuration describes."""
        return GridConfig(
            n_max=int(self.grid_n_max if n_max is None else n_max),
            min_steps=int(self.grid_min_steps),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_n_max": int(self.grid_n_max),
            "grid_min_steps": int(self.grid_min_steps),
            "n_design": int(self.n_design),
            "stage1_paths": int(self.stage1_paths),
            "top_k": int(self.top_k),
            "local_method": str(self.local_method),
            "stage2_paths": int(self.stage2_paths),
            "local_nfev_per_param": int(self.local_nfev_per_param),
            "crn_per_restart": bool(self.crn_per_restart),
            "final_paths": int(self.final_paths),
            "batch_paths": int(self.batch_paths),
            "estimator": str(self.estimator),
            "antithetic": bool(self.antithetic),
            "profile_points": int(self.profile_points),
            "profile_paths": int(self.effective_profile_paths),
            "valley_points": int(self.valley_points),
            "noise_replicates": int(self.noise_replicates),
            "noise_sigma_multiplier": float(self.noise_sigma_multiplier),
            "se_material_ratio": float(self.se_material_ratio),
            "se_vs_h0_factor": float(self.se_vs_h0_factor),
            "refinement_check": bool(self.refinement_check),
            "refinement_factor": int(self.refinement_factor),
            "grid_bias_material": float(self.grid_bias_material),
            "fresh_seed_gap_ratio": float(self.fresh_seed_gap_ratio),
        }


# ---------------------------------------------------------------------------
# The frozen xi0 holder - separation A, made structural
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FrozenXi0:
    """
    The forward-variance curve, sealed for the duration of a calibration.

    The curve itself is already immutable (a frozen dataclass of tuples whose
    cached numpy views are non-writeable).  This holder adds the *auditable* part
    of the guarantee: the SHA-256 content fingerprint recorded at freeze time,
    and :meth:`verify`, which recomputes it.  The optimizer never sees this
    object - it only ever moves a length-3 ``theta``.

    Attributes
    ----------
    curve:
        The very object the caller handed in. Returned by identity in
        :attr:`JointCalibrationResult.xi0_curve`.
    fingerprint:
        ``curve_fingerprint(curve)`` at freeze time.
    """

    curve: ForwardVarianceCurve
    fingerprint: str

    @classmethod
    def freeze(cls, curve: Any) -> "FrozenXi0":
        """Seal ``curve``; rejects anything that is not a spec-4.4 curve."""
        if not isinstance(curve, ForwardVarianceCurve):
            raise RBergomiCalibrationError(
                "ξ₀ doit être une ForwardVarianceCurve (spec 4.4) figée : "
                f"reçu {type(curve).__name__}."
            )
        return cls(curve=curve, fingerprint=curve_fingerprint(curve))

    def verify(self) -> None:
        """Re-hash the curve; raise if a single float moved."""
        current = curve_fingerprint(self.curve)
        if current != self.fingerprint:
            raise RBergomiCalibrationError(
                "La courbe de variance forward a été modifiée pendant la "
                "calibration : ξ₀ est une donnée figée (spec 4.10, séparation A)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": str(self.fingerprint),
            "method": str(self.curve.method),
            "n_knots": int(len(self.curve)),
            "T_knots": [float(x) for x in self.curve.T_knots],
            "levels": [float(x) for x in self.curve.levels],
            "T_max": float(self.curve.T_max),
            "is_frozen_data": True,
            "message_fr": (
                "ξ₀ provient des swaps de variance (spec 4.3/4.4) et reste une "
                "donnée figée : l'optimiseur n'a structurellement pas accès à "
                "cette courbe (theta = (H, eta, rho) uniquement)."
            ),
        }


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationQuote:
    """
    One market observation the joint fit is scored against.

    ``option_type`` is the **out-of-the-money** leg (``"put"`` below the forward,
    ``"call"`` at or above it), whatever the quote was originally.  ``iv`` is the
    market implied volatility on ``(F, D)``, ``price`` the Black-76 price of that
    OTM leg at ``iv``, and ``vega`` its Black-76 vega - all three consistent by
    construction so the weights and the residuals never mix conventions.
    """

    T: float
    K: float
    k: float
    F: float
    D: float
    iv: float
    option_type: str
    price: float
    vega: float
    spread_abs: float
    spread_rel: float
    spread_iv: float
    weight: float
    source: str = "surface_point"
    contract_symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "K": float(self.K),
            "k": float(self.k),
            "F": float(self.F),
            "D": float(self.D),
            "iv": float(self.iv),
            "option_type": str(self.option_type),
            "price": float(self.price),
            "vega": float(self.vega),
            "spread_abs": float(self.spread_abs),
            "spread_rel": float(self.spread_rel),
            "spread_iv": float(self.spread_iv),
            "weight": float(self.weight),
            "source": str(self.source),
            "contract_symbol": (
                None if self.contract_symbol is None else str(self.contract_symbol)
            ),
        }


@dataclass(frozen=True)
class QuoteRejection:
    """One excluded quote, with the reason in both English and French."""

    T: float
    K: float
    reason: str
    detail: str = ""

    @property
    def reason_fr(self) -> str:
        return QUOTE_REASON_LABELS_FR.get(self.reason, self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "K": float(self.K),
            "reason": str(self.reason),
            "reason_fr": self.reason_fr,
            "detail": str(self.detail),
        }


@dataclass(frozen=True, eq=False)
class QuoteSet:
    """
    The cleaned, weighted calibration quote set.

    Attributes
    ----------
    quotes:
        Retained quotes, sorted by ``(T, K)``.
    maturities:
        Distinct maturities, sorted, in year fractions.
    weights:
        ``(n_quotes,)`` normalised weights, summing to ``1``.
    rejections:
        Every excluded quote with its reason.
    flags:
        Set-level flags (e.g. :data:`FLAG_SINGLE_MATURITY`).
    """

    quotes: tuple[CalibrationQuote, ...]
    maturities: tuple[float, ...]
    weights: np.ndarray
    forwards: tuple[float, ...]
    discounts: tuple[float, ...]
    rejections: tuple[QuoteRejection, ...]
    config: WeightConfig
    S0: float
    flags: tuple[str, ...] = ()
    S0_inferred: bool = False

    @property
    def n_quotes(self) -> int:
        return len(self.quotes)

    @property
    def n_maturities(self) -> int:
        return len(self.maturities)

    def maturity_index(self) -> dict[float, int]:
        """``{T: row}`` for the pricing grid."""
        return {float(T): i for i, T in enumerate(self.maturities)}

    def forward_map(self) -> dict[float, float]:
        return {float(T): float(F) for T, F in zip(self.maturities, self.forwards)}

    def discount_map(self) -> dict[float, float]:
        return {float(T): float(D) for T, D in zip(self.maturities, self.discounts)}

    def array(self, attribute: str) -> np.ndarray:
        """A read-only ``(n_quotes,)`` array of one scalar attribute."""
        out = np.asarray([float(getattr(q, attribute)) for q in self.quotes], dtype=float)
        out.setflags(write=False)
        return out

    def diagnostics(self) -> dict[str, Any]:
        by_maturity: dict[str, int] = {}
        for q in self.quotes:
            key = f"{float(q.T):.10g}"
            by_maturity[key] = by_maturity.get(key, 0) + 1
        weights = np.asarray(self.weights, dtype=float)
        return {
            "n_quotes": int(self.n_quotes),
            "n_maturities": int(self.n_maturities),
            "maturities": [float(x) for x in self.maturities],
            "maturities_days": [float(x) * 365.0 for x in self.maturities],
            "n_quotes_by_maturity": by_maturity,
            "n_rejected": int(len(self.rejections)),
            "rejections_by_reason": _count_reasons(self.rejections),
            "weight_sum": float(weights.sum()) if weights.size else 0.0,
            "weight_min": float(weights.min()) if weights.size else float("nan"),
            "weight_max": float(weights.max()) if weights.size else float("nan"),
            "k_min": float(min((q.k for q in self.quotes), default=float("nan"))),
            "k_max": float(max((q.k for q in self.quotes), default=float("nan"))),
            "objective": str(self.config.objective),
            "flags": [str(f) for f in self.flags],
            "S0": float(self.S0),
            "S0_inferred": bool(self.S0_inferred),
        }


def _count_reasons(rejections: Sequence[QuoteRejection]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rejection in rejections:
        out[rejection.reason] = out.get(rejection.reason, 0) + 1
    return out


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _iter_surface_points(option_surface: Any) -> list[Any]:
    """Flatten any of the shapes Phase 1/3 hand around into a list of points."""
    if option_surface is None:
        return []
    if isinstance(option_surface, Mapping):
        collected: list[Any] = []
        for value in option_surface.values():
            collected.extend(_iter_surface_points(value))
        return collected
    if isinstance(option_surface, (str, bytes)):
        raise TypeError("option_surface cannot be a string.")
    if hasattr(option_surface, "iv") and hasattr(option_surface, "K"):
        return [option_surface]
    try:
        items = list(option_surface)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"option_surface is not iterable: {type(option_surface).__name__}."
        ) from exc
    collected = []
    for item in items:
        if isinstance(item, Mapping) or (
            hasattr(item, "iv") and hasattr(item, "K")
        ):
            collected.append(item)
        else:
            collected.extend(_iter_surface_points(item))
    return collected


def _raw_quote_fields(point: Any, *, S0: float | None) -> dict[str, Any]:
    """Read one surface point (object or mapping) into plain floats."""
    if isinstance(point, Mapping):
        get = point.get
    else:
        get = lambda name, default=None: getattr(point, name, default)  # noqa: E731
    T = _finite(get("T"))
    K = _finite(get("K", get("strike")))
    F = _finite(get("F"))
    D = _finite(get("D"))
    iv = _finite(get("iv"))
    k = _finite(get("k"))
    if not math.isfinite(k) and math.isfinite(K) and math.isfinite(F) and F > 0.0 and K > 0.0:
        k = math.log(K / F)
    return {
        "T": T,
        "K": K,
        "F": F,
        "D": D,
        "iv": iv,
        "k": k,
        "mid": _finite(get("mid")),
        "option_type": str(get("option_type", "") or ""),
        "contract_symbol": get("contract_symbol", None),
        "spread_abs": _finite(get("spread_abs")),
        "spread_rel": _finite(get("spread_rel")),
        "S0": S0,
    }


def _quotes_from_surface_grid(surface: SurfaceGrid) -> list[dict[str, Any]]:
    """Grid mode: the repo's ``(t_grid x m_grid)`` IV surface as raw quote rows."""
    S0 = float(surface.S0)
    r = float(surface.r)
    q = float(surface.q)
    m_grid = np.asarray(surface.m_grid, dtype=float)
    t_grid = np.asarray(surface.t_grid, dtype=float)
    iv_market = np.asarray(surface.iv_market, dtype=float)
    mask = effective_mask(iv_market, surface.mask, fit_to_observed_only=True)
    rows: list[dict[str, Any]] = []
    for i_t, T in enumerate(t_grid):
        T_val = float(T)
        if not (math.isfinite(T_val) and T_val > 0.0):
            continue
        F = S0 * math.exp((r - q) * T_val)
        D = math.exp(-r * T_val)
        for j_m, m in enumerate(m_grid):
            if not bool(mask[i_t, j_m]):
                continue
            K = float(m) * S0
            rows.append(
                {
                    "T": T_val,
                    "K": K,
                    "F": F,
                    "D": D,
                    "iv": float(iv_market[i_t, j_m]),
                    "k": math.log(K / F) if (K > 0.0 and F > 0.0) else float("nan"),
                    "mid": float("nan"),
                    "option_type": "",
                    "contract_symbol": None,
                    "spread_abs": float("nan"),
                    "spread_rel": float("nan"),
                    "S0": S0,
                }
            )
    return rows


def _spread_for(row: Mapping[str, Any], lookup: Mapping[Any, float] | None) -> float:
    """Absolute price spread of a quote, from the row or from the Phase-1 chains."""
    direct = _finite(row.get("spread_abs"))
    if math.isfinite(direct) and direct >= 0.0:
        return direct
    if not lookup:
        return float("nan")
    symbol = row.get("contract_symbol")
    if symbol:
        found = lookup.get(("symbol", str(symbol)))
        if found is not None:
            return float(found)
    option_type = str(row.get("option_type") or "")
    if option_type:
        found = lookup.get(
            (
                "strike",
                option_type,
                round(float(row.get("K", float("nan"))), 10),
                round(float(row.get("T", float("nan"))), 10),
            )
        )
        if found is not None:
            return float(found)
    for candidate in ("call", "put"):
        found = lookup.get(
            (
                "strike",
                candidate,
                round(float(row.get("K", float("nan"))), 10),
                round(float(row.get("T", float("nan"))), 10),
            )
        )
        if found is not None:
            return float(found)
    return float("nan")


def build_calibration_quotes(
    option_surface: Any,
    *,
    weights_cfg: WeightConfig | None = None,
    clean_chains: Sequence[CleanChain] | None = None,
    S0: float | None = None,
) -> QuoteSet:
    """
    Clean, re-express out of the money, and weight the market quotes.

    Parameters
    ----------
    option_surface:
        A flat sequence of Phase-1 ``SurfacePoint`` objects, a mapping keyed by
        maturity, a nested sequence, a sequence of mappings carrying the same
        field names, or a repo ``SurfaceGrid`` (the 9x6 grid mode).
    weights_cfg:
        :class:`WeightConfig`; the spec-4.10 defaults when omitted.
    clean_chains:
        The Phase-1 cleaned chains, used **only** to recover the bid-ask spread
        that ``SurfacePoint`` does not carry (via the estimator's own
        ``build_spread_lookup``).
    S0:
        Spot, for reporting only. Read from the ``SurfaceGrid`` in grid mode.

    Returns
    -------
    QuoteSet

    Raises
    ------
    RBergomiCalibrationError
        When fewer than ``weights_cfg.min_quotes`` quotes survive: there is then
        nothing to calibrate against and an empty fit would be a lie.
    """
    cfg = weights_cfg or WeightConfig()
    if isinstance(option_surface, SurfaceGrid):
        rows = _quotes_from_surface_grid(option_surface)
        spot = float(option_surface.S0)
        source = "surface_grid"
    else:
        points = _iter_surface_points(option_surface)
        rows = [_raw_quote_fields(p, S0=S0) for p in points]
        spot = float(S0) if S0 is not None else float("nan")
        source = "surface_point"

    lookup = build_spread_lookup(clean_chains) if clean_chains else {}
    rejections: list[QuoteRejection] = []

    # -- pass 1: validity, moneyness cuts, OTM re-expression ----------------
    kept: list[dict[str, Any]] = []
    for row in rows:
        T = float(row["T"])
        K = float(row["K"])
        F = float(row["F"])
        D = float(row["D"])
        iv = float(row["iv"])
        if not (math.isfinite(T) and T > 0.0):
            rejections.append(QuoteRejection(T, K, REASON_INVALID_MATURITY))
            continue
        if not (
            math.isfinite(K)
            and K > 0.0
            and math.isfinite(F)
            and F > 0.0
            and math.isfinite(D)
            and D > 0.0
        ):
            rejections.append(QuoteRejection(T, K, REASON_INVALID_STRIKE))
            continue
        if not (math.isfinite(iv) and iv > 0.0):
            rejections.append(QuoteRejection(T, K, REASON_NON_FINITE_IV))
            continue
        k = float(row["k"])
        if not math.isfinite(k):
            k = math.log(K / F)
        if abs(k) > float(cfg.k_max):
            rejections.append(
                QuoteRejection(T, K, REASON_K_TOO_FAR, f"|k|={abs(k):.4f}")
            )
            continue
        spread_abs = _spread_for(row, lookup)
        mid = float(row["mid"])
        spread_rel = _finite(row.get("spread_rel"))
        if not math.isfinite(spread_rel) and math.isfinite(spread_abs) and math.isfinite(mid) and mid > 0.0:
            spread_rel = spread_abs / mid
        if math.isfinite(spread_rel) and spread_rel > float(cfg.spread_rel_max):
            rejections.append(
                QuoteRejection(T, K, REASON_SPREAD_TOO_WIDE, f"spread_rel={spread_rel:.4f}")
            )
            continue

        option_type = "call" if K >= F else "put"
        if option_type == "call":
            price = black76_call_price(F=F, K=K, T=T, D=D, vol=iv)
        else:
            price = black76_put_price(F=F, K=K, T=T, D=D, vol=iv)
        vega = black76_vega(F=F, K=K, T=T, D=D, vol=iv)
        if not (math.isfinite(vega) and vega > 0.0):
            rejections.append(QuoteRejection(T, K, REASON_ZERO_VEGA))
            continue
        kept.append(
            {
                "T": T,
                "K": K,
                "k": k,
                "F": F,
                "D": D,
                "iv": iv,
                "option_type": option_type,
                "price": float(price),
                "vega": float(vega),
                "spread_abs": spread_abs,
                "spread_rel": spread_rel,
                "contract_symbol": row.get("contract_symbol"),
                "source": source,
            }
        )

    # -- pass 2: standardised moneyness, against a per-maturity ATM vol -----
    # The reference volatility is the one of the quote closest to k = 0 at that
    # maturity, NOT the quote's own implied volatility: with a skewed smile the
    # latter would prune the (low-vol) call wing far harder than the (high-vol)
    # put wing and silently make the fitted quote set asymmetric.
    if cfg.standardised_k_max is not None and kept:
        atm_vol: dict[float, float] = {}
        closest_k: dict[float, float] = {}
        for item in kept:
            T_key = float(item["T"])
            distance = abs(float(item["k"]))
            if T_key not in closest_k or distance < closest_k[T_key]:
                closest_k[T_key] = distance
                atm_vol[T_key] = float(item["iv"])
        survivors: list[dict[str, Any]] = []
        for item in kept:
            T_key = float(item["T"])
            scale = atm_vol[T_key] * math.sqrt(T_key)
            standardised = abs(float(item["k"])) / scale if scale > 0.0 else float("inf")
            if standardised > float(cfg.standardised_k_max):
                rejections.append(
                    QuoteRejection(
                        T_key,
                        float(item["K"]),
                        REASON_K_STANDARDISED,
                        f"|k|/(sigma_ATM sqrt(T))={standardised:.3f}",
                    )
                )
                continue
            survivors.append(item)
        kept = survivors

    # -- pass 3: per-maturity viability ------------------------------------
    counts: dict[float, int] = {}
    for item in kept:
        counts[float(item["T"])] = counts.get(float(item["T"]), 0) + 1
    viable = []
    for item in kept:
        if counts[float(item["T"])] < int(cfg.min_quotes_per_maturity):
            rejections.append(
                QuoteRejection(
                    float(item["T"]),
                    float(item["K"]),
                    REASON_TOO_FEW_PER_MATURITY,
                    f"n={counts[float(item['T'])]}",
                )
            )
            continue
        viable.append(item)

    if len(viable) < int(cfg.min_quotes):
        raise RBergomiCalibrationError(
            "Jeu de calibration insuffisant : "
            f"{len(viable)} cotations retenues pour un minimum de {int(cfg.min_quotes)} "
            f"({len(rejections)} exclusions)."
        )

    viable.sort(key=lambda item: (float(item["T"]), float(item["K"])))

    # -- weights ------------------------------------------------------------
    # w_i ~ 1 / s_iv_i^2 with s_iv_i the half bid-ask in VOLATILITY units, i.e.
    # a plain inverse variance in the units the residual is measured in. With a
    # strike-independent absolute spread this is w ~ vega^2 / s^2 - the
    # documented intent. It is NOT multiplied by a further vega factor: doing so
    # made the effective exponent 3 and stripped the wings (see WeightConfig).
    spread_iv = np.empty(len(viable), dtype=float)
    known = np.zeros(len(viable), dtype=bool)
    for i, item in enumerate(viable):
        raw = float(item["spread_abs"])
        if math.isfinite(raw) and raw > 0.0:
            known[i] = True
            value = 0.5 * raw / float(item["vega"])
        else:
            value = float(cfg.default_spread_iv)
        spread_iv[i] = min(
            max(value, float(cfg.spread_iv_floor)), float(cfg.spread_iv_cap)
        )
    # A quote whose spread is unknown must be NEUTRAL against the ones whose
    # spread is known - which means the median of the known ones AFTER the clip,
    # not a constant that happens to sit two decades away from it.
    if known.any() and not known.all():
        spread_iv[~known] = float(np.median(spread_iv[known]))
    reference = float(np.median(spread_iv))
    spread_factor = (reference / spread_iv) ** 2

    maturity = np.asarray([float(item["T"]) for item in viable], dtype=float)
    weights = np.array(spread_factor, dtype=float)
    maturities = tuple(float(x) for x in np.unique(maturity))
    if cfg.per_maturity_normalisation:
        for T in maturities:
            rows_at_T = maturity == T
            total = float(np.sum(weights[rows_at_T]))
            if total > 0.0:
                weights[rows_at_T] = weights[rows_at_T] / total / len(maturities)
    total = float(np.sum(weights))
    if not (math.isfinite(total) and total > 0.0):  # pragma: no cover - defensive
        raise RBergomiCalibrationError(
            "Pondérations de calibration dégénérées (somme nulle ou non finie)."
        )
    weights = weights / total
    weights.setflags(write=False)

    quotes = tuple(
        CalibrationQuote(
            T=float(item["T"]),
            K=float(item["K"]),
            k=float(item["k"]),
            F=float(item["F"]),
            D=float(item["D"]),
            iv=float(item["iv"]),
            option_type=str(item["option_type"]),
            price=float(item["price"]),
            vega=float(item["vega"]),
            spread_abs=float(item["spread_abs"]),
            spread_rel=float(item["spread_rel"]),
            spread_iv=float(spread_iv[i]),
            weight=float(weights[i]),
            source=str(item["source"]),
            contract_symbol=item.get("contract_symbol"),
        )
        for i, item in enumerate(viable)
    )

    # One forward and one discount per expiry. Spec 4.2 stamps exactly one
    # (F, D) pair on every SurfacePoint of an expiry, so a disagreement here
    # means the caller mixed two forward curves - silently keeping the first
    # would put a pure forward mismatch straight into (H, eta, rho).
    forwards = []
    discounts = []
    for T in maturities:
        at_T = [q for q in quotes if q.T == T]
        F_values = np.asarray([q.F for q in at_T], dtype=float)
        D_values = np.asarray([q.D for q in at_T], dtype=float)
        for label, values in (("F", F_values), ("D", D_values)):
            reference = float(values[0])
            if np.max(np.abs(values - reference)) > 1e-9 * abs(reference):
                raise RBergomiCalibrationError(
                    f"Cotations incohérentes à T = {T:.6g} : plusieurs valeurs de "
                    f"{label} ({float(values.min())!r} … {float(values.max())!r}). "
                    "Une échéance porte un seul forward et un seul facteur "
                    "d'actualisation (spec 4.2)."
                )
        forwards.append(float(F_values[0]))
        discounts.append(float(D_values[0]))

    flags: list[str] = []
    if rejections:
        flags.append(FLAG_QUOTES_DROPPED)
    if len(maturities) < 2:
        flags.append(FLAG_SINGLE_MATURITY)

    if not (math.isfinite(spot) and spot > 0.0):
        # `SurfacePoint` carries (F, D) but not the spot. The dividend-free
        # implied spot F(T_min) * D(T_min) is the only thing the quote set
        # determines; it is recorded as inferred rather than silently assumed,
        # and it only anchors the t=0 end of the log-forward interpolation - the
        # model forward is pinned to F(T) at every quoted maturity regardless.
        spot = float(forwards[0]) * float(discounts[0])
        s0_inferred = True
    else:
        s0_inferred = False

    return QuoteSet(
        quotes=quotes,
        maturities=maturities,
        weights=weights,
        forwards=tuple(forwards),
        discounts=tuple(discounts),
        rejections=tuple(rejections),
        config=cfg,
        S0=spot,
        flags=tuple(flags),
        S0_inferred=bool(s0_inferred),
    )


# ---------------------------------------------------------------------------
# Drift built from the market forward curve
# ---------------------------------------------------------------------------
def forward_step_rates(
    *, grid: SimulationGrid, maturities: Sequence[float], forwards: Sequence[float], S0: float
) -> np.ndarray:
    """
    Per-step rates that make the model forward match the market forward exactly.

    ``ln F`` is interpolated linearly in ``T`` through the anchor ``(0, ln S0)``
    and differentiated cell by cell, so with ``q = 0`` the simulated
    ``S0 * exp(sum r_i dt_i)`` reproduces ``F(T)`` to machine precision at every
    quoted maturity (they are grid nodes by construction) and interpolates
    log-linearly in between.

    BEYOND THE LAST QUOTED MATURITY
    -------------------------------
    The grid is built from the union of the quoted **and** the reported
    maturities (``JointObjective._all_maturities``), so it does extend past the
    last quote whenever the caller asks for a reporting maturity there - and
    ``RBergomiJointHCalibrator.calibrate`` always passes the UI's ``t_grid`` as
    ``report_grid``.  ``numpy.interp`` *clamps* outside its knots, which would
    make every step rate beyond the last quote exactly ``0.0``: measured with
    quotes to 2 y and a reporting grid to 3 y, the model forward came out at
    ``F(3 y) = 102.020134`` against a market ``103.045453`` (relative error
    ``-9.95e-03``), about 0.15 volatility point of implied-volatility error at
    every strike of that maturity, contaminating ``iv_model``, ``iv_error`` and
    both metric blocks.

    The tail is therefore **extrapolated at the last known forward rate**
    ``(ln F_n - ln F_{n-1}) / (T_n - T_{n-1})`` - the continuation of the curve
    the quotes themselves determine - and the caller is told about it with
    :data:`FLAG_REPORT_BEYOND_QUOTES`, because no quote constrains the model
    there whatever the drift does.
    """
    spot = float(S0)
    if not (math.isfinite(spot) and spot > 0.0):
        raise ValueError(f"S0 must be finite and strictly positive; got {S0!r}.")
    knots = np.concatenate([[0.0], np.asarray(maturities, dtype=float)])
    log_forwards = np.concatenate(
        [[math.log(spot)], np.log(np.asarray(forwards, dtype=float))]
    )
    if not np.all(np.isfinite(log_forwards)):
        raise ValueError("Market forwards must all be finite and strictly positive.")
    t = np.asarray(grid.t, dtype=float)
    nodes = np.interp(t, knots, log_forwards)
    beyond = t > float(knots[-1])
    if np.any(beyond) and knots.size >= 2:
        width = float(knots[-1]) - float(knots[-2])
        tail_rate = (
            (float(log_forwards[-1]) - float(log_forwards[-2])) / width
            if width > 0.0
            else 0.0
        )
        nodes[beyond] = float(log_forwards[-1]) + tail_rate * (
            t[beyond] - float(knots[-1])
        )
    return np.diff(nodes) / np.asarray(grid.dt, dtype=float)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ObjectiveEvaluation:
    """One evaluation of the joint objective at one ``theta``."""

    theta: np.ndarray
    loss: float
    rmse: float
    residuals: np.ndarray
    iv_model: np.ndarray
    iv_error: np.ndarray
    price_model: np.ndarray
    price_stderr: np.ndarray
    iv_stderr: np.ndarray
    loss_noise_std: float
    n_paths: int
    n_batches: int
    seed: int | None
    grid_hash: str
    grid_n: int
    n_uninvertible: int
    elapsed_s: float
    report_iv: np.ndarray | None = None

    @property
    def params(self) -> RBergomiParams:
        return RBergomiParams(
            H=float(self.theta[0]), eta=float(self.theta[1]), rho=float(self.theta[2])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": [float(x) for x in self.theta],
            "params": self.params.to_dict(),
            "loss": float(self.loss),
            "rmse": float(self.rmse),
            "loss_noise_std": float(self.loss_noise_std),
            "n_paths": int(self.n_paths),
            "n_batches": int(self.n_batches),
            "seed": None if self.seed is None else int(self.seed),
            "grid_hash": str(self.grid_hash),
            "grid_n": int(self.grid_n),
            "n_uninvertible": int(self.n_uninvertible),
            "elapsed_s": float(self.elapsed_s),
        }


def _pool_price_results(results: Sequence[PriceResult]) -> PriceResult:
    """
    Pool independent equal-size batches into one :class:`PriceResult`.

    The batches are independent samples of the same estimator, so the pooled mean
    is the mean of the batch means and the pooled variance is
    ``sum_b SE_b^2 / B^2``.  Everything deterministic (strikes, discounts,
    forward factors) is carried over from the first batch, which is identical
    across batches by construction.
    """
    if not results:  # pragma: no cover - defensive
        raise ValueError("At least one PriceResult is required.")
    if len(results) == 1:
        return results[0]
    head = results[0]
    count = len(results)
    price = np.mean([np.asarray(r.price, dtype=float) for r in results], axis=0)
    stderr = np.sqrt(
        np.sum([np.asarray(r.stderr, dtype=float) ** 2 for r in results], axis=0)
    ) / count
    sample_forward = np.mean(
        [np.asarray(r.sample_forward, dtype=float) for r in results], axis=0
    )
    sample_forward_stderr = np.sqrt(
        np.sum(
            [np.asarray(r.sample_forward_stderr, dtype=float) ** 2 for r in results],
            axis=0,
        )
    ) / count
    return PriceResult(
        option_type=head.option_type,
        estimator=head.estimator,
        maturities=head.maturities,
        strikes=head.strikes,
        price=price,
        stderr=stderr,
        discount_factors=head.discount_factors,
        forward_factors=head.forward_factors,
        model_forwards=head.model_forwards,
        sample_forward=sample_forward,
        sample_forward_stderr=sample_forward_stderr,
        S0=head.S0,
        q=head.q,
        n_paths=int(sum(r.n_paths for r in results)),
        n_base_paths=int(sum(r.n_base_paths for r in results)),
        antithetic=head.antithetic,
        stderr_is_conservative=head.stderr_is_conservative,
    )


class JointObjective:
    """
    The spec-4.10 objective: deterministic in ``theta`` under common random numbers.

    The instance owns everything that is *not* optimised - the frozen ``xi0``
    holder, the quote set, the simulation grid, the market-forward drift - and
    exposes a callable whose only argument is the length-3 vector
    ``(H, eta, rho)``.  There is no code path from that argument to the
    forward-variance curve.
    """

    def __init__(
        self,
        *,
        quotes: QuoteSet,
        xi0: FrozenXi0,
        config: JointMCConfig,
        report_maturities: Sequence[float] | None = None,
        report_strikes: np.ndarray | None = None,
    ) -> None:
        if not isinstance(quotes, QuoteSet):
            raise TypeError(f"quotes must be a QuoteSet; got {type(quotes).__name__}.")
        if not isinstance(xi0, FrozenXi0):
            raise TypeError(f"xi0 must be a FrozenXi0; got {type(xi0).__name__}.")
        self.quotes = quotes
        self._xi0 = xi0
        self.config = config
        self.n_eval = 0
        self.total_eval_seconds = 0.0
        self._cache: "OrderedDict[tuple[Any, ...], ObjectiveEvaluation]" = OrderedDict()
        self._cache_maxsize = 64

        self._report_maturities = (
            tuple(float(x) for x in report_maturities) if report_maturities else ()
        )
        self._report_strikes = (
            np.asarray(report_strikes, dtype=float) if report_strikes is not None else None
        )
        if self._report_strikes is not None and self._report_strikes.ndim != 2:
            raise ValueError("report_strikes must be a (n_T, n_K) array.")

        all_maturities = sorted(
            set(float(x) for x in quotes.maturities) | set(self._report_maturities)
        )
        self._all_maturities = tuple(all_maturities)
        self._grids: dict[int, SimulationGrid] = {}
        self._rates: dict[str, np.ndarray] = {}

        # Per-maturity strike blocks, split into the OTM put side and the OTM
        # call side so every quote is priced on its low-variance leg.
        index_of = {float(T): i for i, T in enumerate(quotes.maturities)}
        put_rows: list[list[float]] = [[] for _ in quotes.maturities]
        call_rows: list[list[float]] = [[] for _ in quotes.maturities]
        self._slot: list[tuple[str, int, int]] = []
        for quote in quotes.quotes:
            row = index_of[float(quote.T)]
            if quote.option_type == "put":
                self._slot.append(("put", row, len(put_rows[row])))
                put_rows[row].append(float(quote.K))
            else:
                self._slot.append(("call", row, len(call_rows[row])))
                call_rows[row].append(float(quote.K))
        self._put_strikes, self._put_width = _pad_strike_rows(put_rows, quotes.forwards)
        self._call_strikes, self._call_width = _pad_strike_rows(call_rows, quotes.forwards)

        self._weights = np.asarray(quotes.weights, dtype=float)
        self._iv_market = quotes.array("iv")
        self._price_market = quotes.array("price")
        self._vega_market = quotes.array("vega")
        self._quote_maturity = quotes.array("T")
        self._quote_discount = quotes.array("D")
        self._quote_rows = np.asarray(
            [index_of[float(q.T)] for q in quotes.quotes], dtype=int
        )

    # -- immutable accessors ------------------------------------------------
    @property
    def xi0(self) -> FrozenXi0:
        """The sealed forward-variance holder (read-only; never handed to theta)."""
        return self._xi0

    @property
    def n_quotes(self) -> int:
        return self.quotes.n_quotes

    def grid(self, *, n_max: int | None = None) -> SimulationGrid:
        """The (memoised) simulation grid for a given ``n_max``."""
        key = int(self.config.grid_n_max if n_max is None else n_max)
        grid = self._grids.get(key)
        if grid is None:
            grid = build_simulation_grid(
                maturities=self._all_maturities, config=self.config.grid_config(n_max=key)
            )
            self._grids[key] = grid
        return grid

    def step_rates(self, grid: SimulationGrid) -> np.ndarray:
        """Per-step drift rates on ``grid`` (memoised by grid hash)."""
        rates = self._rates.get(grid.grid_hash)
        if rates is None:
            rates = forward_step_rates(
                grid=grid,
                maturities=self.quotes.maturities,
                forwards=self.quotes.forwards,
                S0=float(self.quotes.S0),
            )
            rates.setflags(write=False)
            self._rates[grid.grid_hash] = rates
        return rates

    # -- evaluation ---------------------------------------------------------
    def evaluate(
        self,
        theta: Sequence[float],
        *,
        n_paths: int,
        seed: int | None,
        n_max: int | None = None,
        batch_paths: int | None = None,
        with_report: bool = False,
        use_cache: bool = True,
    ) -> ObjectiveEvaluation:
        """
        Price the whole quote set at ``theta`` and return the weighted loss.

        ``seed`` is the common-random-number seed of the calling stage: two calls
        with the same ``(seed, n_paths, n_max)`` and the same ``theta`` return
        **bit-identical** numbers, and two calls with the same
        ``(seed, n_paths, n_max)`` at different ``theta`` share the same
        underlying normals.
        """
        vector = np.asarray(theta, dtype=float).ravel()
        if vector.size != 3:
            raise ValueError(
                "The joint objective takes exactly three parameters "
                f"(H, eta, rho); got {vector.size}. xi0 is frozen data and is "
                "not part of theta."
            )
        key = (
            vector.tobytes(),
            int(n_paths),
            None if seed is None else int(seed),
            int(self.config.grid_n_max if n_max is None else n_max),
            bool(with_report),
        )
        if use_cache and seed is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        started = time.perf_counter()
        params = RBergomiParams(
            H=float(vector[0]), eta=float(vector[1]), rho=float(vector[2])
        )
        grid = self.grid(n_max=n_max)
        rates = self.step_rates(grid)
        batch = int(batch_paths or self.config.batch_paths)
        sizes = _batch_sizes(int(n_paths), batch, antithetic=bool(self.config.antithetic))
        batch_seeds = _batch_seeds(seed, len(sizes))

        call_batches: list[PriceResult] = []
        put_batches: list[PriceResult] = []
        report_batches: list[tuple[PriceResult, PriceResult]] = []
        maturities = [float(x) for x in self.quotes.maturities]
        for size, batch_seed in zip(sizes, batch_seeds):
            sim_config = SimulationConfig(
                n_paths=int(size),
                antithetic=bool(self.config.antithetic),
                seed=batch_seed,
                grid_config=self.config.grid_config(n_max=n_max),
            )
            paths = simulate_rbergomi_xi_curve(
                S0=float(self.quotes.S0),
                xi_curve=self._xi0.curve,
                params=params,
                maturities=self._all_maturities,
                grid=grid,
                r=rates,
                q=0.0,
                config=sim_config,
            )
            if self._call_width:
                call_batches.append(
                    price_call(
                        paths,
                        strikes=self._call_strikes,
                        maturities=maturities,
                        estimator=self.config.estimator,
                    )
                )
            if self._put_width:
                put_batches.append(
                    price_put(
                        paths,
                        strikes=self._put_strikes,
                        maturities=maturities,
                        estimator=self.config.estimator,
                    )
                )
            if with_report and self._report_strikes is not None:
                report_maturities = [float(x) for x in self._report_maturities]
                report_batches.append(
                    (
                        price_call(
                            paths,
                            strikes=self._report_strikes,
                            maturities=report_maturities,
                            estimator=self.config.estimator,
                        ),
                        price_put(
                            paths,
                            strikes=self._report_strikes,
                            maturities=report_maturities,
                            estimator=self.config.estimator,
                        ),
                    )
                )

        iv_model = np.full(self.n_quotes, np.nan, dtype=float)
        price_model = np.full(self.n_quotes, np.nan, dtype=float)
        price_stderr = np.full(self.n_quotes, np.nan, dtype=float)
        blocks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if call_batches:
            pooled = _pool_price_results(call_batches)
            blocks["call"] = (
                implied_vol_surface(pooled),
                np.asarray(pooled.price, dtype=float),
                np.asarray(pooled.stderr, dtype=float),
            )
            call_discounts = np.asarray(pooled.discount_factors, dtype=float)
        else:  # pragma: no cover - a surface with no OTM call at all
            call_discounts = np.asarray([], dtype=float)
        if put_batches:
            pooled = _pool_price_results(put_batches)
            blocks["put"] = (
                implied_vol_surface(pooled),
                np.asarray(pooled.price, dtype=float),
                np.asarray(pooled.stderr, dtype=float),
            )
            put_discounts = np.asarray(pooled.discount_factors, dtype=float)
        else:  # pragma: no cover - a surface with no OTM put at all
            put_discounts = np.asarray([], dtype=float)
        path_discounts = call_discounts if call_discounts.size else put_discounts

        for i, (side, row, col) in enumerate(self._slot):
            iv_block, price_block, stderr_block = blocks[side]
            iv_model[i] = float(iv_block[row, col])
            rescale = float(self._quote_discount[i]) / float(path_discounts[row])
            price_model[i] = float(price_block[row, col]) * rescale
            price_stderr[i] = float(stderr_block[row, col]) * rescale

        # Residuals. An uninvertible model price is NOT silently dropped: it is
        # linearised in vol units through the market vega, which is exact to
        # first order and keeps the objective continuous in theta.
        uninvertible = ~np.isfinite(iv_model)
        n_uninvertible = int(np.count_nonzero(uninvertible))
        if self.quotes.config.objective == OBJECTIVE_IV:
            errors = iv_model - self._iv_market
            if n_uninvertible:
                linearised = (price_model - self._price_market) / np.maximum(
                    self._vega_market, 1e-12
                )
                fallback = np.where(
                    np.isfinite(linearised),
                    np.clip(linearised, -UNINVERTIBLE_PENALTY_VOL, UNINVERTIBLE_PENALTY_VOL),
                    UNINVERTIBLE_PENALTY_VOL,
                )
                errors = np.where(uninvertible, fallback, errors)
            iv_stderr = price_stderr / np.maximum(self._vega_market, 1e-12)
            sigma = iv_stderr
        else:
            denominator = np.maximum(
                float(self.quotes.config.price_floor), self._price_market
            )
            errors = (price_model - self._price_market) / denominator
            iv_stderr = price_stderr / np.maximum(self._vega_market, 1e-12)
            sigma = price_stderr / denominator

        # Last-resort guard: a residual that is still not finite would turn the
        # whole loss into NaN and make the optimizer wander silently. Charge the
        # documented worst case instead, and count it - never return NaN and
        # never drop the quote (dropping would make the loss theta-dependent in
        # its very definition, which destroys the smoothness CRN buys).
        if not np.all(np.isfinite(errors)):
            broken = ~np.isfinite(errors)
            n_uninvertible = max(n_uninvertible, int(np.count_nonzero(broken)))
            worst = (
                UNINVERTIBLE_PENALTY_VOL
                if self.quotes.config.objective == OBJECTIVE_IV
                else 1.0
            )
            errors = np.where(broken, worst, errors)

        iv_error = iv_model - self._iv_market
        loss = float(np.sum(self._weights * errors * errors))
        residuals = np.sqrt(self._weights) * errors

        # Delta-method propagation of the Monte-Carlo error into the loss:
        # dL/de_i = 2 w_i e_i, so Var(L) ~ sum_i 4 w_i^2 e_i^2 sigma_i^2 with
        # sigma_i the Monte-Carlo standard error of residual i.
        variance = float(
            np.nansum(4.0 * (self._weights * errors) ** 2 * np.square(sigma))
        )
        noise_std = math.sqrt(variance) if math.isfinite(variance) and variance > 0.0 else 0.0

        report_iv = None
        if with_report and report_batches:
            # Every reporting cell is inverted from its OUT-OF-THE-MONEY leg,
            # exactly like the fitted quotes: put-call parity is exact on the
            # shared sample, so the two agree in expectation, but the OTM leg
            # carries far less Monte-Carlo variance deep in either wing.
            pooled_call = _pool_price_results([pair[0] for pair in report_batches])
            pooled_put = _pool_price_results([pair[1] for pair in report_batches])
            iv_call = implied_vol_surface(pooled_call)
            iv_put = implied_vol_surface(pooled_put)
            forwards = np.asarray(pooled_call.model_forwards, dtype=float)[:, None]
            report_iv = np.where(
                np.asarray(pooled_call.strikes, dtype=float) >= forwards, iv_call, iv_put
            )

        elapsed = time.perf_counter() - started
        self.n_eval += 1
        self.total_eval_seconds += elapsed
        evaluation = ObjectiveEvaluation(
            theta=np.array(vector, dtype=float),
            loss=loss,
            rmse=math.sqrt(loss) if loss >= 0.0 else float("nan"),
            residuals=residuals,
            iv_model=iv_model,
            iv_error=iv_error,
            price_model=price_model,
            price_stderr=price_stderr,
            iv_stderr=iv_stderr,
            loss_noise_std=float(noise_std),
            n_paths=int(sum(sizes)),
            n_batches=int(len(sizes)),
            seed=None if seed is None else int(seed),
            grid_hash=grid.grid_hash,
            grid_n=int(grid.n),
            n_uninvertible=n_uninvertible,
            elapsed_s=float(elapsed),
            report_iv=report_iv,
        )
        if use_cache and seed is not None:
            self._cache[key] = evaluation
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)
        return evaluation

    def loss_function(
        self, *, n_paths: int, seed: int | None, n_max: int | None = None
    ) -> Callable[[Sequence[float]], float]:
        """A scalar ``theta -> loss`` closure bound to one common-random-number draw."""

        def loss(theta: Sequence[float]) -> float:
            return float(self.evaluate(theta, n_paths=n_paths, seed=seed, n_max=n_max).loss)

        return loss

    def residual_function(
        self, *, n_paths: int, seed: int | None, n_max: int | None = None
    ) -> Callable[[Sequence[float]], np.ndarray]:
        """A ``theta -> sqrt(w) * e`` closure; its least-squares cost is ``Loss / 2``."""

        def residuals(theta: Sequence[float]) -> np.ndarray:
            return np.asarray(
                self.evaluate(theta, n_paths=n_paths, seed=seed, n_max=n_max).residuals,
                dtype=float,
            )

        return residuals


def _pad_strike_rows(
    rows: Sequence[Sequence[float]], forwards: Sequence[float]
) -> tuple[np.ndarray, int]:
    """
    Rectangularise a ragged per-maturity strike list.

    ``price_call`` / ``price_put`` want a ``(n_T, n_K)`` grid, but a real surface
    has a different number of quotes per expiry.  Short rows are padded with the
    maturity's own forward: the padded columns are priced and then never read,
    and a strike equal to ``F`` is always invertible, so the padding can never
    contaminate a diagnostic.
    """
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return np.zeros((len(rows), 0), dtype=float), 0
    padded = np.empty((len(rows), width), dtype=float)
    for i, row in enumerate(rows):
        filler = float(forwards[i])
        values = list(row) + [filler] * (width - len(row))
        padded[i, :] = values
    return padded, width


def _batch_sizes(n_paths: int, batch_paths: int, *, antithetic: bool) -> list[int]:
    """Split ``n_paths`` into equal batches of at most ``batch_paths`` paths."""
    total = int(n_paths)
    if total < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths!r}.")
    cap = max(int(batch_paths), 2 if antithetic else 1)
    n_batches = max(1, int(math.ceil(total / cap)))
    per_batch = int(math.ceil(total / n_batches))
    if antithetic and per_batch % 2 != 0:
        per_batch += 1
    return [per_batch] * n_batches


def _batch_seeds(seed: int | None, n_batches: int) -> list[int | None]:
    """
    Deterministic sub-seeds for the batches of one evaluation.

    They depend only on ``(seed, batch index)``, never on ``theta``, so the
    common-random-number property survives batching.
    """
    if seed is None:
        return [None] * n_batches
    if n_batches == 1:
        return [int(seed)]
    generator = np.random.default_rng(int(seed))
    return [int(generator.integers(0, 2**31 - 1)) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Identifiability diagnostics (spec 4.11)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseFloor:
    """
    The Monte-Carlo noise floor every "is this variation real?" test is judged against.

    THE YARDSTICK HAS TO MATCH THE QUANTITY
    ---------------------------------------
    Everything this module tests is a **difference of two losses computed on ONE
    shared CRN draw**: :attr:`ProfileSlice.span`, :attr:`ValleyProfile.span`, the
    optimum-versus-profile-minimum gap, and ``improvement = loss_initial -
    loss_crn``.  The module's own docstring says CRN cancels the sampling noise
    to first order in exactly those differences - so judging them against the
    scatter of the loss *level* across independent seeds is judging them against
    a noise they do not contain.  Measured at ``n_max = 128``, 3 000 paths,
    ``theta = [0.15, 1.1, -0.55]``, 12 seeds:

    ============================================ ==========
    quantity                                     std
    ============================================ ==========
    ``L`` across independent seeds               3.837e-06
    CRN one-step difference in ``H``             1.512e-07
    CRN one-step difference in ``eta``           7.980e-07
    CRN one-step difference in ``rho``           5.808e-07
    ============================================ ==========

    up to **25x** smaller.  The level-based floor was 1.157e-05 where the honest
    2-sigma floor for the ``H`` span was 5.376e-06, i.e. 2.2x too large, and a
    reproducible ``H`` gradient worth 11.4x its own noise sat *below* it.

    WHAT IS REPORTED
    ----------------
    * ``difference_std`` / ``difference_value`` - **the floor that is used.**
      The whole CRN set is re-drawn ``n_replicates`` times; for each parameter
      the *same* one-step difference ``L(theta* + step e_p) - L(theta*)`` is
      recomputed inside each draw, and ``difference_std`` is the largest
      run-to-run standard deviation over the parameters.  ``difference_value``
      is ``multiplier`` times it.  With fewer than two replicates, or with no
      free parameter, it falls back to ``value`` rather than to a meaningless
      zero, and ``difference_is_fallback`` says so.
    * ``replicated_std`` / ``value`` - the old ACROSS-draw quantity, kept
      because it is the right yardstick for the one comparison that *is* made
      across draws (a fresh-seed loss against an in-sample one) and because it
      feeds :attr:`ProfileSlice.standard_error`.
    * ``delta_std`` - the delta-method propagation of the per-quote Monte-Carlo
      standard errors, ``Var(L) ~ sum_i 4 w_i^2 e_i^2 sigma_i^2``.  Free, and a
      useful cross-check: it collapses to zero at a perfect fit, where the
      replicated estimate does not.  **It is an approximation**: it sums the
      per-quote variances with no covariance terms, although every quote at one
      maturity is priced on the SAME paths and their errors are therefore
      strongly positively correlated, so it under-states the true variance of
      ``L`` - which is precisely why it is only ever used as the *larger of two*
      and never on its own.

    ``value`` is ``multiplier * max(replicated_std, delta_std)``.
    """

    value: float
    replicated_std: float
    delta_std: float
    multiplier: float
    n_replicates: int
    losses: tuple[float, ...]
    seeds: tuple[int | None, ...]
    difference_value: float = float("nan")
    difference_std: float = float("nan")
    difference_by_parameter: tuple[float, ...] = ()
    difference_parameters: tuple[str, ...] = ()
    difference_is_fallback: bool = True

    @property
    def sigma_level(self) -> float:
        """The loss-LEVEL noise sigma (no multiplier), for the profile standard errors."""
        return float(max(float(self.replicated_std), float(self.delta_std)))

    def difference_for(self, parameter: str) -> float:
        """
        The CRN-difference floor of ONE parameter, or the aggregate when unmeasured.

        Per-parameter is the right resolution: the ``eta`` step is 2 % of a box
        4.95 wide, so its one-step difference is far noisier than ``H``'s over a
        box 0.48 wide, and judging the ``H`` span against ``eta``'s noise throws
        away most of the resolution CRN bought.
        """
        for name, value in zip(self.difference_parameters, self.difference_by_parameter):
            if name == parameter and math.isfinite(value) and value > 0.0:
                return float(self.multiplier) * float(value)
        return float(self.difference_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "replicated_std": float(self.replicated_std),
            "delta_std": float(self.delta_std),
            "multiplier": float(self.multiplier),
            "n_replicates": int(self.n_replicates),
            "losses": [float(x) for x in self.losses],
            "seeds": [None if s is None else int(s) for s in self.seeds],
            "difference_value": float(self.difference_value),
            "difference_std": float(self.difference_std),
            "difference_by_parameter": dict(
                zip(
                    [str(p) for p in self.difference_parameters],
                    [float(x) for x in self.difference_by_parameter],
                )
            ),
            "difference_is_fallback": bool(self.difference_is_fallback),
            "definition_fr": (
                "Plancher de bruit Monte-Carlo utilisé pour juger les écarts à "
                f"tirage commun : {float(self.multiplier):.1f} x l'écart-type "
                "d'une MÊME différence de coûts recalculée sur plusieurs tirages "
                f"complets ({float(self.difference_value):.3e}). Le plancher "
                "historique, mesuré sur le NIVEAU du coût à graines "
                f"indépendantes ({float(self.value):.3e}), reste reporté : il "
                "vaut pour les comparaisons faites d'un tirage à l'autre."
            ),
            "delta_std_caveat_fr": (
                "delta_std somme les variances par cotation sans terme de "
                "covariance alors que toutes les cotations d'une échéance sont "
                "valorisées sur les MÊMES trajectoires : c'est une approximation "
                "qui sous-estime la variance de L."
            ),
        }


@dataclass(frozen=True)
class ProfileSlice:
    """
    A 1-D slice of the loss along one parameter, the other two held at the optimum.

    The nodes span the parameter's **effective bounds** (spec 4.11 asks for the
    bound-to-bound variation), with the optimum and the two finite-difference
    neighbours ``theta* +- step`` inserted so the local gradient and curvature
    come from the same evaluations.

    THE NUMBER THAT ACTUALLY ANSWERS "IS THIS PARAMETER IDENTIFIED?"
    ---------------------------------------------------------------
    ``span`` - the bound-to-bound loss variation - is dominated by the far arms
    of the slice and is very nearly independent of how sharply the optimum is
    resolved.  Measured on a quote set restricted to 1 y and 2 y (truth
    ``H = 0.10``), three seeds returned ``H = 0.1260 / 0.0447 / 0.1780`` - a
    spread of 0.133, i.e. 2.7x the suite's own ``TOL_H`` - while ``span`` sat
    250-320x above the noise floor in all three, so a span-based flatness test
    could never fire.

    :attr:`standard_error` is the honest number, and it costs nothing extra
    because both ingredients are already measured here::

        SE(p) = sqrt( 2 * sigma_L / (d2L/dp2) )

    the half-width of the interval over which the loss rises by less than one
    Monte-Carlo sigma above its minimum, i.e. the set of parameter values this
    surface cannot tell apart from the optimum.  On the reference surface,
    ``d2L/dH2 = 1.226e-02`` (stable across steps 0.005-0.02) and
    ``sigma_L = 1.375e-06`` give ``SE(H) = 0.015``, directly comparable to the
    spec-4.5 ``H0`` standard error of 0.0061.  A non-positive curvature means the
    optimum is not a minimum along this axis at all and yields ``SE = inf``.

    :attr:`stationary` is the module's own invariant, checked rather than
    assumed: ``optimum_loss`` must not exceed the smallest loss on its own slice
    by more than the (CRN-difference) noise floor.  Every one of those losses is
    computed on the SAME draw, so the comparison is exact up to that floor.
    """

    parameter: str
    values: np.ndarray
    losses: np.ndarray
    optimum_value: float
    optimum_loss: float
    bounds: tuple[float, float]
    step: float
    gradient: float
    curvature: float
    span: float
    noise_floor: float
    flat: bool
    n_paths: int
    seed: int | None
    sigma_level: float = float("nan")
    standard_error: float = float("nan")
    se_threshold: float = float("nan")
    weakly_identified: bool = False
    stationarity_gap: float = float("nan")
    stationarity_floor: float = float("nan")
    stationary: bool = True

    @property
    def message_fr(self) -> str:
        head = (
            f"Profil de {self.parameter} : erreur type "
            f"{self.standard_error:.4g} (seuil {self.se_threshold:.4g}), "
            f"variation bord à bord {self.span:.3e} contre un plancher de bruit "
            f"{self.noise_floor:.3e}"
        )
        if self.flat:
            return (
                head
                + f" sur [{self.bounds[0]:.4g}, {self.bounds[1]:.4g}] — profil "
                "PLAT, ce paramètre n'est pas identifié par cette surface."
            )
        # Ordered by severity: a non-stationary optimum blocks `success`, a weak
        # identification only warns, so the blocking verdict is named first.
        if not self.stationary:
            suffix = (
                f" — optimum NON STATIONNAIRE : le coût rendu dépasse le minimum "
                f"du profil de {self.stationarity_gap:.3e} (tolérance "
                f"{self.stationarity_floor:.3e})."
            )
            if self.weakly_identified:
                suffix += " Paramètre également FAIBLEMENT identifié."
            return head + suffix
        if self.weakly_identified:
            return head + " — paramètre FAIBLEMENT identifié."
        return head + " — profil informatif."

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": str(self.parameter),
            "values": [float(x) for x in self.values],
            "losses": [float(x) for x in self.losses],
            "optimum_value": float(self.optimum_value),
            "optimum_loss": float(self.optimum_loss),
            "bounds": [float(self.bounds[0]), float(self.bounds[1])],
            "step": float(self.step),
            "gradient": float(self.gradient),
            "curvature": float(self.curvature),
            "span": float(self.span),
            "noise_floor": float(self.noise_floor),
            "flat": bool(self.flat),
            "sigma_level": float(self.sigma_level),
            "standard_error": float(self.standard_error),
            "se_threshold": float(self.se_threshold),
            "weakly_identified": bool(self.weakly_identified),
            "stationarity_gap": float(self.stationarity_gap),
            "stationarity_floor": float(self.stationarity_floor),
            "stationary": bool(self.stationary),
            "n_paths": int(self.n_paths),
            "seed": None if self.seed is None else int(self.seed),
            "message_fr": self.message_fr,
        }


@dataclass(frozen=True)
class ValleyProfile:
    """
    The loss along the ``rho * eta = const`` hyperbola through the optimum.

    ``MATH_ORACLE`` section 8 measured that the short-dated ATM skew constant
    ``c_hat`` is independent of ``rho`` to three decimals, i.e. the leading-order
    skew identifies only the **product** ``rho * eta``.  Breaking that degeneracy
    is the stated job of the joint fit; this slice is the evidence that it did
    (a curved valley) or did not (a flat one).
    """

    product: float
    eta_values: np.ndarray
    rho_values: np.ndarray
    losses: np.ndarray
    optimum_loss: float
    span: float
    noise_floor: float
    flat: bool
    n_points: int
    n_paths: int
    seed: int | None

    @property
    def message_fr(self) -> str:
        if self.flat:
            return (
                "Vallée (eta, rho) à rho*eta = "
                f"{self.product:+.4f} PLATE (variation {self.span:.3e} sous le "
                f"plancher {self.noise_floor:.3e}) : seul le produit rho*eta est "
                "identifié, pas ses deux facteurs."
            )
        return (
            "Vallée (eta, rho) à rho*eta = "
            f"{self.product:+.4f} incurvée (variation {self.span:.3e} contre un "
            f"plancher {self.noise_floor:.3e}) : la surface complète sépare eta "
            "et rho, la dégénérescence du skew court terme est levée."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_rho_eta": float(self.product),
            "eta_values": [float(x) for x in self.eta_values],
            "rho_values": [float(x) for x in self.rho_values],
            "losses": [float(x) for x in self.losses],
            "optimum_loss": float(self.optimum_loss),
            "span": float(self.span),
            "noise_floor": float(self.noise_floor),
            "flat": bool(self.flat),
            "n_points": int(self.n_points),
            "n_paths": int(self.n_paths),
            "seed": None if self.seed is None else int(self.seed),
            "message_fr": self.message_fr,
        }


@dataclass(frozen=True)
class GridBiasReport:
    """
    First-order estimate of the residual log-Euler discretisation bias, in
    parameter units.

    The loss gradient at the optimum is measured **twice on one fresh draw** -
    once on the calibration grid, once on a grid ``factor`` times finer.  The
    calibration-grid gradient is the residual tilt that has nothing to do with
    discretisation (``theta*`` minimises the loss on the local stage's draw, not
    on this one); the difference is therefore the grid effect alone.  Dividing by
    the curvature read off the profile slices turns it into the Newton step
    ``theta`` would take on the finer grid:

    .. code-block:: text

        shift_p = - ( dL/dp |_refined - dL/dp |_calibration ) / ( d2L/dp2 |_calibration )

    It is an **estimate of the residual bias, never a correction**: nothing is
    subtracted from the calibrated parameters.

    "NOT MEASURED" IS NaN, NEVER ZERO
    ---------------------------------
    Two situations make a parameter's shift unmeasurable: the finite-difference
    step does not fit inside its bounds (``step_H = 0.02 * (0.49 - 0.01) =
    0.0096``, so **any** calibrated ``H < 0.0196`` - the small-``H`` rough regime
    that is the entire point of this pipeline), or the profile curvature is
    non-positive.  The first path used to report ``theta_shift = 0.0`` and
    ``theta_shift_relative = 0.0``, which the ``material`` verdict then read as
    "no bias": a live run that landed at ``H = 0.01832`` reported
    ``theta_shift = {'H': 0.0, ...}, material = False``, i.e. it fabricated the
    one number the caller asked for.  The second path already reported ``NaN``,
    so the module also contradicted itself.  Both now report ``NaN``,
    :attr:`unmeasured` names the parameters concerned, and ``material`` is
    computed over the measured ones only.

    ``theta_shift_relative`` is the shift as a fraction of :data:`PARAM_SCALE` -
    the parameter's own recovery tolerance - not of the bound-to-bound width.
    """

    n_calibration: int
    n_refined: int
    factor: int
    loss_calibration: float
    loss_refined: float
    gradient_calibration: tuple[float, ...]
    gradient_refined: tuple[float, ...]
    curvature: tuple[float, ...]
    theta_shift: tuple[float, ...]
    theta_shift_relative: tuple[float, ...]
    material: bool
    threshold: float
    n_paths: int
    seed: int | None
    unmeasured: tuple[str, ...] = ()
    unmeasured_reasons: tuple[str, ...] = ()

    @property
    def message_fr(self) -> str:
        parts = ", ".join(
            f"{name} " + ("NON MESURÉ" if not math.isfinite(shift) else f"{shift:+.4f}")
            for name, shift in zip(PARAM_ORDER, self.theta_shift)
        )
        if not any(math.isfinite(x) for x in self.theta_shift):
            verdict = "INDÉTERMINÉ"
        else:
            verdict = "NON NÉGLIGEABLE" if self.material else "négligeable"
        message = (
            f"Biais de discrétisation résiduel estimé ({verdict}) en passant de "
            f"n={self.n_calibration} à n={self.n_refined} nœuds : {parts}. "
            "Estimation reportée, jamais soustraite du résultat."
        )
        if self.unmeasured:
            details = "; ".join(
                f"{name} ({reason})"
                for name, reason in zip(self.unmeasured, self.unmeasured_reasons)
            )
            message += (
                " Paramètres non mesurables (valeur rendue NaN, pas 0) : "
                f"{details}."
            )
        return message

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calibration": int(self.n_calibration),
            "n_refined": int(self.n_refined),
            "factor": int(self.factor),
            "loss_calibration": float(self.loss_calibration),
            "loss_refined": float(self.loss_refined),
            "gradient_calibration": dict(
                zip(PARAM_ORDER, [float(x) for x in self.gradient_calibration])
            ),
            "gradient_refined": dict(
                zip(PARAM_ORDER, [float(x) for x in self.gradient_refined])
            ),
            "curvature": dict(zip(PARAM_ORDER, [float(x) for x in self.curvature])),
            "theta_shift": dict(zip(PARAM_ORDER, [float(x) for x in self.theta_shift])),
            "theta_shift_relative": dict(
                zip(PARAM_ORDER, [float(x) for x in self.theta_shift_relative])
            ),
            "material": bool(self.material),
            "threshold": float(self.threshold),
            "threshold_basis_fr": (
                "Seuil exprimé en fraction de l'échelle propre du paramètre "
                f"(tolérances de recouvrement {PARAM_SCALE}), pas de la largeur "
                "des bornes."
            ),
            "unmeasured": [str(x) for x in self.unmeasured],
            "unmeasured_reasons": [str(x) for x in self.unmeasured_reasons],
            "n_paths": int(self.n_paths),
            "seed": None if self.seed is None else int(self.seed),
            "message_fr": self.message_fr,
            "known_issue_fr": (
                "Schéma log-Euler à variance gelée au point gauche : le skew ATM "
                "du modèle est sous-estimé, davantage à l'échéance courte qu'à "
                "l'échéance longue (Phase 3 : ~9,7 % à 7 j contre ~3,0 % à 2 ans "
                "sur la grille par défaut n_max=256). La calibration utilise "
                "délibérément une grille plus fine (n_max=384 par défaut) et "
                "mesure ici ce qu'il en reste."
            ),
        }


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Everything spec 4.11 asks to be reported alongside the calibrated theta."""

    noise_floor: NoiseFloor
    profiles: tuple[ProfileSlice, ...]
    valley: ValleyProfile | None
    H0: float
    H0_se: float
    H0_ci95: tuple[float, float]
    H0_is_fallback: bool
    H_calibrated: float
    H_in_ci: bool
    loss_initial: float
    loss_optimum: float
    improvement: float
    improvement_significant: bool
    flags: tuple[str, ...]

    def profile(self, parameter: str) -> ProfileSlice | None:
        for slice_ in self.profiles:
            if slice_.parameter == parameter:
                return slice_
        return None

    @property
    def standard_errors(self) -> dict[str, float]:
        """``SE(p) = sqrt(2 sigma_L / curvature)`` per profiled parameter (spec 4.11)."""
        return {p.parameter: float(p.standard_error) for p in self.profiles}

    @property
    def H_standard_error(self) -> float:
        """The one directly comparable to the spec-4.5 ``H0`` standard error."""
        slice_ = self.profile("H")
        return float("nan") if slice_ is None else float(slice_.standard_error)

    @property
    def identification_fr(self) -> str:
        se_H = self.H_standard_error
        head = (
            f"Identification de H : erreur type issue de la courbure du profil "
            f"SE(H) = {se_H:.4g}"
        )
        if math.isfinite(self.H0_se) and self.H0_se > 0.0:
            head += (
                f", contre SE(H0) = {self.H0_se:.4g} pour la régression de skew "
                f"(spec 4.5), rapport {se_H / self.H0_se:.2f}"
            )
        return head + "."

    @property
    def warnings_fr(self) -> tuple[str, ...]:
        return tuple(
            JOINT_CALIBRATION_LABELS_FR.get(flag, flag) for flag in self.flags
        )

    @property
    def has_H0_ci(self) -> bool:
        """Whether an ``H0`` confidence interval was actually supplied."""
        return bool(all(math.isfinite(x) for x in self.H0_ci95))

    @property
    def h_comparison_fr(self) -> str:
        if not self.has_H0_ci:
            return (
                f"H calibré = {self.H_calibrated:.4f} ; H0 = {self.H0:.4f} sans "
                "intervalle de confiance fourni (initialisation transmise sans "
                "l'erreur standard de la régression spec 4.5) — aucune "
                "comparaison statistique possible."
            )
        if self.H0_is_fallback:
            return (
                f"H0 = {self.H0:.4f} est une valeur de repli (estimation de Hurst "
                f"instable) ; H calibré = {self.H_calibrated:.4f}. La comparaison "
                "n'a pas de valeur probante."
            )
        verdict = "dans" if self.H_in_ci else "HORS DE"
        return (
            f"H calibré = {self.H_calibrated:.4f}, {verdict} l'IC95 de "
            f"H0 = {self.H0:.4f} [{self.H0_ci95[0]:.4f}, {self.H0_ci95[1]:.4f}] "
            f"(SE = {self.H0_se:.4f})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_floor": self.noise_floor.to_dict(),
            "profiles": [p.to_dict() for p in self.profiles],
            "valley": None if self.valley is None else self.valley.to_dict(),
            "H0": float(self.H0),
            "H0_se": float(self.H0_se),
            "H0_ci95": [float(self.H0_ci95[0]), float(self.H0_ci95[1])],
            "H0_is_fallback": bool(self.H0_is_fallback),
            "has_H0_ci": bool(self.has_H0_ci),
            "H_calibrated": float(self.H_calibrated),
            "H_in_ci": bool(self.H_in_ci) if self.has_H0_ci else None,
            "h_comparison_fr": self.h_comparison_fr,
            "standard_errors": {k: float(v) for k, v in self.standard_errors.items()},
            "H_standard_error": float(self.H_standard_error),
            "identification_fr": self.identification_fr,
            "loss_initial": float(self.loss_initial),
            "loss_optimum": float(self.loss_optimum),
            "improvement": float(self.improvement),
            "improvement_significant": bool(self.improvement_significant),
            "flags": [str(f) for f in self.flags],
            "warnings_fr": [str(w) for w in self.warnings_fr],
        }


def measure_noise_floor(
    objective: JointObjective,
    theta: Sequence[float],
    *,
    config: JointMCConfig,
    seeds: Sequence[int | None],
    n_paths: int,
    delta_std: float,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    free: Sequence[int] | None = None,
) -> NoiseFloor:
    """
    Measure the Monte-Carlo noise floor at ``theta``, on levels *and* on differences.

    Two measurements are made on the same ``len(seeds)`` independent CRN draws:

    * the scatter of the loss **level** across draws (``replicated_std``) - the
      right yardstick for a comparison made *between* draws;
    * the scatter, across draws, of the **same one-step difference** taken
      *inside* each draw (``difference_std``) - the right yardstick for every
      span, gradient and improvement this module reports, all of which are CRN
      differences.  See :class:`NoiseFloor`.

    The step is the profile step, ``2 %`` of the bound-to-bound width, mirrored
    inwards when ``theta + step`` would leave the box.  Cost:
    ``len(seeds) * (1 + n_free)`` evaluations.

    ``len(seeds) < 2`` cannot produce a standard deviation; the replicated
    estimate is then ``0.0``, the difference floor falls back to the level-based
    ``value`` and says so in ``difference_is_fallback``, rather than collapsing
    to a zero floor that would make every flatness test unfireable.
    """
    base = np.asarray(theta, dtype=float).ravel()
    indices = [int(i) for i in (free if free is not None else range(len(PARAM_ORDER)))]
    steps: dict[int, float] = {}
    if bounds is not None:
        for index in indices:
            lower, upper = (float(x) for x in bounds[PARAM_ORDER[index]])
            if upper <= lower:
                continue
            step = 0.02 * (upper - lower)
            if base[index] + step > upper:
                step = -step
            if not (lower <= base[index] + step <= upper):
                continue
            steps[index] = float(step)

    losses: list[float] = []
    differences: dict[int, list[float]] = {index: [] for index in steps}
    for seed in seeds:
        centre = float(objective.evaluate(base, n_paths=int(n_paths), seed=seed).loss)
        losses.append(centre)
        for index, step in steps.items():
            shifted = base.copy()
            shifted[index] = float(base[index] + step)
            moved = float(
                objective.evaluate(shifted, n_paths=int(n_paths), seed=seed).loss
            )
            differences[index].append(moved - centre)

    if len(losses) >= 2:
        replicated = float(np.std(np.asarray(losses, dtype=float), ddof=1))
    else:
        replicated = 0.0
    value = float(config.noise_sigma_multiplier) * max(replicated, float(delta_std))

    per_parameter: list[float] = []
    names: list[str] = []
    for index in sorted(differences):
        series = np.asarray(differences[index], dtype=float)
        names.append(PARAM_ORDER[index])
        per_parameter.append(
            float(np.std(series, ddof=1)) if series.size >= 2 else float("nan")
        )
    finite = [x for x in per_parameter if math.isfinite(x)]
    if finite and max(finite) > 0.0:
        difference_std = float(max(finite))
        difference_value = float(config.noise_sigma_multiplier) * difference_std
        fallback = False
    else:
        difference_std = float("nan")
        difference_value = float(value)
        fallback = True

    return NoiseFloor(
        value=float(value),
        replicated_std=float(replicated),
        delta_std=float(delta_std),
        multiplier=float(config.noise_sigma_multiplier),
        n_replicates=len(losses),
        losses=tuple(losses),
        seeds=tuple(seeds),
        difference_value=float(difference_value),
        difference_std=float(difference_std),
        difference_by_parameter=tuple(per_parameter),
        difference_parameters=tuple(names),
        difference_is_fallback=bool(fallback),
    )


def profile_slice(
    objective: JointObjective,
    theta: Sequence[float],
    *,
    index: int,
    bounds: tuple[float, float],
    n_points: int,
    n_paths: int,
    seed: int | None,
    noise_floor: float,
    sigma_level: float = float("nan"),
    se_threshold: float = float("nan"),
    stationarity_floor: float | None = None,
) -> ProfileSlice:
    """
    One bound-to-bound 1-D profile of the loss, with the local gradient and curvature.

    The finite-difference step is 2 % of the bound-to-bound span, large enough
    that the smooth CRN objective dominates the residual Monte-Carlo wiggle and
    small enough to stay local.

    ``noise_floor`` must be the **CRN-difference** floor
    (:attr:`NoiseFloor.difference_value`): the span and the optimum-versus-slice
    gap are both differences taken inside this one draw.  ``sigma_level`` is the
    loss-LEVEL noise sigma (:attr:`NoiseFloor.sigma_level`), which is what the
    ``SE = sqrt(2 sigma_L / curvature)`` half-width is defined against, and
    ``se_threshold`` the value past which the parameter is called weakly
    identified.

    ``stationarity_floor`` defaults to ``noise_floor``.  The driver passes the
    **larger** of the CRN-difference floor and the loss-level floor there,
    deliberately: the optimum was located on the *restart's* draw and the profile
    is walked on the *selection* draw, so even a perfectly converged run leaves a
    small positive gap whose size is governed by the draw-to-draw scatter of the
    argmin - a between-draw effect.  This tolerance flips ``success``, so it is
    the conservative one; the D1 measurement (a truncated ``max_nfev = 80`` run
    at ``theta* = (0.0757, 1.4942, -0.7908)`` with ``optimum_loss = 6.349e-06``
    against ``losses.min() = 3.583e-06`` at ``rho = -0.7508``) clears it at 1.64x
    even so.
    """
    base = np.asarray(theta, dtype=float).ravel()
    lower, upper = float(bounds[0]), float(bounds[1])
    centre = float(base[index])
    step = 0.02 * (upper - lower)
    nodes = list(np.linspace(lower, upper, int(n_points)))
    nodes.append(centre)
    for candidate in (centre - step, centre + step):
        if lower <= candidate <= upper:
            nodes.append(candidate)
    values = np.asarray(sorted(set(round(float(x), 12) for x in nodes)), dtype=float)

    losses = np.empty(values.size, dtype=float)
    for i, value in enumerate(values):
        point = base.copy()
        point[index] = float(value)
        losses[i] = float(
            objective.evaluate(point, n_paths=int(n_paths), seed=seed).loss
        )

    def _loss_at(value: float) -> float:
        position = int(np.argmin(np.abs(values - value)))
        return float(losses[position])

    centre_loss = _loss_at(centre)
    gradient = float("nan")
    curvature = float("nan")
    if lower <= centre - step and centre + step <= upper:
        forward = _loss_at(centre + step)
        backward = _loss_at(centre - step)
        gradient = (forward - backward) / (2.0 * step)
        curvature = (forward - 2.0 * centre_loss + backward) / (step * step)

    span = float(np.max(losses) - np.min(losses)) if losses.size else float("nan")

    # SE(p) = sqrt(2 sigma_L / curvature): the half-width over which the loss
    # rises by less than one Monte-Carlo sigma. A non-positive curvature is not
    # a small standard error, it is an infinite one.
    sigma = float(sigma_level)
    standard_error = float("nan")
    if math.isfinite(sigma) and sigma >= 0.0:
        if math.isfinite(curvature) and curvature > 0.0:
            standard_error = math.sqrt(2.0 * sigma / curvature)
        else:
            standard_error = float("inf")
    weak = bool(
        math.isfinite(se_threshold)
        and se_threshold > 0.0
        and not (standard_error <= se_threshold)
        and not math.isnan(standard_error)
    )

    gap = (
        float(centre_loss - float(np.min(losses)))
        if losses.size
        else float("nan")
    )
    stationarity_tolerance = float(
        noise_floor if stationarity_floor is None else stationarity_floor
    )
    stationary = bool(math.isfinite(gap) and gap <= stationarity_tolerance)

    return ProfileSlice(
        parameter=PARAM_ORDER[index],
        values=values,
        losses=losses,
        optimum_value=centre,
        optimum_loss=centre_loss,
        bounds=(lower, upper),
        step=float(step),
        gradient=float(gradient),
        curvature=float(curvature),
        span=span,
        noise_floor=float(noise_floor),
        flat=bool(math.isfinite(span) and span < float(noise_floor)),
        n_paths=int(n_paths),
        seed=None if seed is None else int(seed),
        sigma_level=float(sigma),
        standard_error=float(standard_error),
        se_threshold=float(se_threshold),
        weakly_identified=weak,
        stationarity_gap=float(gap),
        stationarity_floor=float(stationarity_tolerance),
        stationary=stationary,
    )


def eta_rho_valley(
    objective: JointObjective,
    theta: Sequence[float],
    *,
    bounds: Mapping[str, tuple[float, float]],
    n_points: int,
    n_paths: int,
    seed: int | None,
    noise_floor: float,
    span_factor: float = 2.0,
) -> ValleyProfile | None:
    """
    Walk the ``rho * eta = const`` hyperbola through the optimum, ``H`` held fixed.

    Returns ``None`` when fewer than three admissible points exist (a pinned
    ``eta`` or ``rho``, or a product that leaves the box immediately) - an
    honest absence rather than a two-point "valley".
    """
    base = np.asarray(theta, dtype=float).ravel()
    eta_star = float(base[1])
    rho_star = float(base[2])
    product = eta_star * rho_star
    eta_lo, eta_hi = (float(x) for x in bounds["eta"])
    rho_lo, rho_hi = (float(x) for x in bounds["rho"])
    if eta_star <= 0.0 or rho_star == 0.0 or eta_lo >= eta_hi or rho_lo >= rho_hi:
        return None

    low = max(eta_lo, eta_star / float(span_factor))
    high = min(eta_hi, eta_star * float(span_factor))
    if not (high > low):
        return None
    grid = np.unique(
        np.concatenate(
            [np.geomspace(low, high, int(n_points)), np.asarray([eta_star])]
        )
    )
    etas: list[float] = []
    rhos: list[float] = []
    losses: list[float] = []
    for eta in grid:
        rho = product / float(eta)
        if not (rho_lo <= rho <= rho_hi):
            continue
        point = base.copy()
        point[1] = float(eta)
        point[2] = float(rho)
        etas.append(float(eta))
        rhos.append(float(rho))
        losses.append(
            float(objective.evaluate(point, n_paths=int(n_paths), seed=seed).loss)
        )
    if len(losses) < 3:
        return None

    loss_array = np.asarray(losses, dtype=float)
    span = float(np.max(loss_array) - np.min(loss_array))
    position = int(np.argmin(np.abs(np.asarray(etas, dtype=float) - eta_star)))
    return ValleyProfile(
        product=float(product),
        eta_values=np.asarray(etas, dtype=float),
        rho_values=np.asarray(rhos, dtype=float),
        losses=loss_array,
        optimum_loss=float(loss_array[position]),
        span=span,
        noise_floor=float(noise_floor),
        flat=bool(span < float(noise_floor)),
        n_points=len(losses),
        n_paths=int(n_paths),
        seed=None if seed is None else int(seed),
    )


def grid_refinement_bias(
    objective: JointObjective,
    theta: Sequence[float],
    *,
    profiles: Sequence[ProfileSlice],
    config: JointMCConfig,
    bounds: Mapping[str, tuple[float, float]],
    n_paths: int,
    seed: int | None,
) -> GridBiasReport | None:
    """
    Estimate the residual discretisation bias by refining the simulation grid.

    Both gradients are re-measured **on this function's own draw**, one on the
    calibration grid and one on the refined grid, rather than reading the
    calibration-grid one off the profile slices: ``theta*`` minimises the loss on
    the *local stage's* draw, so on any other draw its gradient is not zero, and
    subtracting a gradient measured on a third draw would mix that residual tilt
    into the reported bias.  The two grids cannot share their normals (the draw
    has ``3n`` dimensions and ``n`` differs), so no further cancellation is
    available; what is achieved is that the difference contains only the grid
    effect plus symmetric sampling noise.  The curvature is still read off the
    profiles - a second difference is far more stable than a first one.

    Costs ``2 + 4 * n_free`` evaluations.  Returns ``None`` when the refined grid
    cannot be built (the quoted maturities alone already exceeding the refined
    budget, which cannot happen for ``factor >= 1`` but is checked rather than
    assumed).
    """
    base = np.asarray(theta, dtype=float).ravel()
    factor = int(config.refinement_factor)
    refined_n_max = int(config.grid_n_max) * factor
    try:
        refined = objective.grid(n_max=refined_n_max)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("rBergomi joint: refined grid unavailable (%s).", exc)
        return None
    calibration = objective.grid()

    loss_refined = float(
        objective.evaluate(base, n_paths=int(n_paths), seed=seed, n_max=refined_n_max).loss
    )
    loss_calibration = float(
        objective.evaluate(base, n_paths=int(n_paths), seed=seed).loss
    )

    by_name = {p.parameter: p for p in profiles}
    gradient_refined: list[float] = []
    gradient_calibration: list[float] = []
    curvature: list[float] = []
    shift: list[float] = []
    relative: list[float] = []
    unmeasured: list[str] = []
    unmeasured_reasons: list[str] = []
    for index, name in enumerate(PARAM_ORDER):
        slice_ = by_name.get(name)
        lower, upper = (float(x) for x in bounds[name])
        if slice_ is None or not math.isfinite(slice_.step) or upper <= lower:
            gradient_refined.append(float("nan"))
            gradient_calibration.append(float("nan"))
            curvature.append(float("nan"))
            shift.append(float("nan"))
            relative.append(float("nan"))
            unmeasured.append(name)
            unmeasured_reasons.append("paramètre figé ou profil absent")
            continue
        step = float(slice_.step)
        if not (lower <= base[index] - step and base[index] + step <= upper):
            # NOT MEASURED. Reporting 0.0 here made "no measurement" read as "no
            # bias" - and it is exactly the small-H rough regime that lands here.
            gradient_refined.append(float("nan"))
            gradient_calibration.append(float(slice_.gradient))
            curvature.append(float(slice_.curvature))
            shift.append(float("nan"))
            relative.append(float("nan"))
            unmeasured.append(name)
            unmeasured_reasons.append(
                f"pas de différence finie {step:.4g} hors des bornes "
                f"[{lower:.4g}, {upper:.4g}] autour de {float(base[index]):.4g}"
            )
            continue
        up = base.copy()
        up[index] = float(base[index] + step)
        down = base.copy()
        down[index] = float(base[index] - step)

        def _central(n_max: int | None) -> float:
            forward = float(
                objective.evaluate(up, n_paths=int(n_paths), seed=seed, n_max=n_max).loss
            )
            backward = float(
                objective.evaluate(down, n_paths=int(n_paths), seed=seed, n_max=n_max).loss
            )
            return (forward - backward) / (2.0 * step)

        g_refined = _central(refined_n_max)
        g_calibration = _central(None)
        h = float(slice_.curvature)
        gradient_refined.append(float(g_refined))
        gradient_calibration.append(g_calibration)
        curvature.append(h)
        if math.isfinite(h) and h > 0.0 and math.isfinite(g_refined) and math.isfinite(g_calibration):
            value = -(g_refined - g_calibration) / h
        else:
            value = float("nan")
            unmeasured.append(name)
            unmeasured_reasons.append(
                f"courbure du profil non exploitable (d2L/d{name}2 = {h:.3e})"
            )
        shift.append(float(value))
        # Relative to the parameter's OWN scale (its recovery tolerance), never
        # to the bound-to-bound width - see PARAM_SCALE.
        scale = float(PARAM_SCALE.get(name, upper - lower))
        relative.append(
            float(abs(value) / scale)
            if (math.isfinite(value) and scale > 0.0)
            else float("nan")
        )

    finite_relative = [x for x in relative if math.isfinite(x)]
    material = bool(
        finite_relative and max(finite_relative) > float(config.grid_bias_material)
    )
    return GridBiasReport(
        n_calibration=int(calibration.n),
        n_refined=int(refined.n),
        factor=factor,
        loss_calibration=loss_calibration,
        loss_refined=loss_refined,
        gradient_calibration=tuple(gradient_calibration),
        gradient_refined=tuple(gradient_refined),
        curvature=tuple(curvature),
        theta_shift=tuple(shift),
        theta_shift_relative=tuple(relative),
        material=material,
        threshold=float(config.grid_bias_material),
        n_paths=int(n_paths),
        seed=None if seed is None else int(seed),
        unmeasured=tuple(unmeasured),
        unmeasured_reasons=tuple(unmeasured_reasons),
    )


# ---------------------------------------------------------------------------
# Stage bookkeeping and the result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageOneReport:
    """The coarse Latin-hypercube stage."""

    design: tuple[tuple[float, ...], ...]
    losses: tuple[float, ...]
    top_indices: tuple[int, ...]
    anchor_index: int
    n_paths: int
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_design": int(len(self.design)),
            "design": [[float(v) for v in row] for row in self.design],
            "losses": [float(x) for x in self.losses],
            "top_indices": [int(i) for i in self.top_indices],
            "anchor_index": int(self.anchor_index),
            "best_loss": float(min(self.losses)) if self.losses else float("nan"),
            "n_paths": int(self.n_paths),
            "seed": None if self.seed is None else int(self.seed),
        }


@dataclass(frozen=True)
class LocalRun:
    """One Stage-2 restart."""

    index: int
    x0: tuple[float, ...]
    x: tuple[float, ...]
    loss_own_draw: float
    loss_selection_draw: float
    nfev: int
    converged: bool
    message: str
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "x0": [float(v) for v in self.x0],
            "x": [float(v) for v in self.x],
            "loss_own_draw": float(self.loss_own_draw),
            "loss_selection_draw": float(self.loss_selection_draw),
            "nfev": int(self.nfev),
            "converged": bool(self.converged),
            "message": str(self.message),
            "seed": None if self.seed is None else int(self.seed),
        }


@dataclass(frozen=True)
class StageTwoReport:
    """
    The local stage: every restart, and which one won on the selection draw.

    ``n_starts_requested`` is what the caller asked for and
    ``n_starts_effective`` is what actually ran - **they are not always equal**.
    Stage 2 can only start from the points Stage 1 handed it, so with Stage 1
    disabled (``n_design = 0``) there is exactly one start whatever
    ``settings.n_starts`` says.  When Stage 1 does run, ``top_k`` is now widened
    to ``max(top_k, n_starts)`` so the request is honoured; when it cannot be,
    :data:`FLAG_RESTARTS_TRUNCATED` is raised.  Measured before the fix:
    ``n_starts=4 / n_design=6 / top_k=2`` performed 2 local runs while reporting
    4 restarts and 4 restart seeds, and ``n_starts=4 / n_design=0`` performed 1.

    ``max_nfev`` is the **effective** per-run budget (see
    :meth:`JointMCConfig.local_max_nfev`), not the raw ``settings.max_nfev``.
    """

    method: str
    runs: tuple[LocalRun, ...]
    best_index: int
    n_paths: int
    max_nfev: int
    selection_seed: int | None
    n_starts_requested: int = 1
    max_nfev_source: str = "config"

    @property
    def n_starts_effective(self) -> int:
        return int(len(self.runs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "n_runs": int(len(self.runs)),
            "runs": [run.to_dict() for run in self.runs],
            "best_index": int(self.best_index),
            "n_paths": int(self.n_paths),
            "max_nfev": int(self.max_nfev),
            "max_nfev_source": str(self.max_nfev_source),
            "n_starts_requested": int(self.n_starts_requested),
            "n_starts_effective": int(self.n_starts_effective),
            "selection_seed": (
                None if self.selection_seed is None else int(self.selection_seed)
            ),
            "total_nfev": int(sum(run.nfev for run in self.runs)),
        }


@dataclass(frozen=True, eq=False)
class JointCalibrationResult:
    """
    The spec-4.10 / 4.11 result.

    ``xi0_curve`` is **the caller's own object**, returned by identity: the
    calibration read it and could not have written it.
    """

    success: bool
    message_fr: str
    params: RBergomiParams
    theta: np.ndarray
    initial_params: RBergomiParams
    bounds: dict[str, tuple[float, float]]
    pinned: tuple[str, ...]
    xi0_curve: ForwardVarianceCurve
    xi0: FrozenXi0
    quotes: QuoteSet
    loss_crn: float
    loss_fresh: float
    loss_initial: float
    rmse_fresh: float
    #: In-sample and out-of-sample loss at the SAME path count - the only
    #: comparison that can detect over-fitting of the optimizer's own draw.
    loss_crn_matched: float
    loss_fresh_matched: float
    matched_paths: int
    iv_model: np.ndarray
    iv_error: np.ndarray
    price_model: np.ndarray
    weights: np.ndarray
    metrics: dict[str, float]
    metrics_vw: dict[str, float]
    stage1: StageOneReport | None
    stage2: StageTwoReport | None
    identifiability: IdentifiabilityReport | None
    grid_bias: GridBiasReport | None
    final_evaluation: ObjectiveEvaluation
    config: JointMCConfig
    seed: int | None
    n_objective_evaluations: int
    mean_evaluation_seconds: float
    elapsed_s: float
    flags: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def warnings_fr(self) -> tuple[str, ...]:
        return tuple(JOINT_CALIBRATION_LABELS_FR.get(f, f) for f in self.flags)

    @property
    def params_dict(self) -> dict[str, float]:
        return self.params.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload; survives the controller's ``_json_safe``."""
        return {
            "success": bool(self.success),
            "message_fr": str(self.message_fr),
            "model": "rbergomi",
            "method": "joint_h_mc",
            "params": self.params.to_dict(),
            "initial_params": self.initial_params.to_dict(),
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in self.bounds.items()},
            "pinned": [str(p) for p in self.pinned],
            "xi0": self.xi0.to_dict(),
            "quotes": self.quotes.diagnostics(),
            "loss_crn": float(self.loss_crn),
            "loss_fresh": float(self.loss_fresh),
            "loss_initial": float(self.loss_initial),
            "rmse_fresh": float(self.rmse_fresh),
            "fresh_seed_gap": {
                "loss_crn_matched": float(self.loss_crn_matched),
                "loss_fresh_matched": float(self.loss_fresh_matched),
                "n_paths_crn": int(self.matched_paths),
                "n_paths_fresh": int(self.matched_paths),
                "ratio": (
                    float(self.loss_fresh_matched / self.loss_crn_matched)
                    if self.loss_crn_matched > 0.0
                    else float("nan")
                ),
                "threshold": float(self.config.fresh_seed_gap_ratio),
                "definition_fr": (
                    "Coût dans l'échantillon (tirage effectivement ajusté par "
                    "l'étape locale) contre coût hors échantillon (graine "
                    "fraîche), au MÊME nombre de trajectoires."
                ),
            },
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "metrics_vw": {k: float(v) for k, v in self.metrics_vw.items()},
            "stage1": None if self.stage1 is None else self.stage1.to_dict(),
            "stage2": None if self.stage2 is None else self.stage2.to_dict(),
            "identifiability": (
                None if self.identifiability is None else self.identifiability.to_dict()
            ),
            "grid_bias": None if self.grid_bias is None else self.grid_bias.to_dict(),
            "final_evaluation": self.final_evaluation.to_dict(),
            "config": self.config.to_dict(),
            "weights_config": self.quotes.config.to_dict(),
            "seed": None if self.seed is None else int(self.seed),
            "n_objective_evaluations": int(self.n_objective_evaluations),
            "mean_evaluation_seconds": float(self.mean_evaluation_seconds),
            "elapsed_s": float(self.elapsed_s),
            "flags": [str(f) for f in self.flags],
            "warnings_fr": [str(w) for w in self.warnings_fr],
            "details": dict(self.details),
        }


def calibration_report(result: JointCalibrationResult) -> dict[str, Any]:
    """Compact French-facing summary for the Phase-5 report."""
    identifiability = result.identifiability
    grid_bias = result.grid_bias
    return {
        "success": bool(result.success),
        "H": float(result.params.H),
        "eta": float(result.params.eta),
        "rho": float(result.params.rho),
        "H0": None if identifiability is None else float(identifiability.H0),
        "H0_ci95": (
            None
            if identifiability is None or not identifiability.has_H0_ci
            else [float(identifiability.H0_ci95[0]), float(identifiability.H0_ci95[1])]
        ),
        "H_in_H0_ci95": (
            None
            if identifiability is None or not identifiability.has_H0_ci
            else bool(identifiability.H_in_ci)
        ),
        "h_comparison_fr": (
            None if identifiability is None else identifiability.h_comparison_fr
        ),
        "H_standard_error": (
            None if identifiability is None else float(identifiability.H_standard_error)
        ),
        "identification_fr": (
            None if identifiability is None else identifiability.identification_fr
        ),
        "loss_crn": float(result.loss_crn),
        "loss_fresh": float(result.loss_fresh),
        "loss_crn_matched": float(result.loss_crn_matched),
        "loss_fresh_matched": float(result.loss_fresh_matched),
        "matched_paths": int(result.matched_paths),
        "loss_initial": float(result.loss_initial),
        "rmse_fresh_vol_points": float(result.rmse_fresh) * 100.0,
        "improvement_significant": (
            None
            if identifiability is None
            else bool(identifiability.improvement_significant)
        ),
        "noise_floor": (
            None
            if identifiability is None
            else float(identifiability.noise_floor.difference_value)
        ),
        "noise_floor_level": (
            None if identifiability is None else float(identifiability.noise_floor.value)
        ),
        "n_starts_effective": (
            None if result.stage2 is None else int(result.stage2.n_starts_effective)
        ),
        "max_nfev_effective": (
            None if result.stage2 is None else int(result.stage2.max_nfev)
        ),
        "h_profile_flat": (
            None
            if identifiability is None
            else bool(FLAG_H_PROFILE_FLAT in identifiability.flags)
        ),
        "eta_rho_valley_flat": (
            None
            if identifiability is None or identifiability.valley is None
            else bool(identifiability.valley.flat)
        ),
        "grid_bias_theta_shift": (
            None
            if grid_bias is None
            else dict(zip(PARAM_ORDER, [float(x) for x in grid_bias.theta_shift]))
        ),
        "grid_bias_material": None if grid_bias is None else bool(grid_bias.material),
        "grid_bias_message_fr": None if grid_bias is None else grid_bias.message_fr,
        "n_quotes": int(result.quotes.n_quotes),
        "n_maturities": int(result.quotes.n_maturities),
        "n_objective_evaluations": int(result.n_objective_evaluations),
        "mean_evaluation_seconds": float(result.mean_evaluation_seconds),
        "seed": None if result.seed is None else int(result.seed),
        "xi0_frozen": True,
        "xi0_fingerprint": str(result.xi0.fingerprint),
        "flags": [str(f) for f in result.flags],
        "warnings_fr": [str(w) for w in result.warnings_fr],
        "message_fr": str(result.message_fr),
    }


# ---------------------------------------------------------------------------
# Initial point and bounds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InitialContext:
    """The spec-4.9 seed, plus whatever spec-4.5 evidence came with it."""

    params: RBergomiParams
    H0: float
    H0_se: float
    H0_ci95: tuple[float, float]
    H0_is_fallback: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.to_dict(),
            "H0": float(self.H0),
            "H0_se": float(self.H0_se),
            "H0_ci95": [float(self.H0_ci95[0]), float(self.H0_ci95[1])],
            "H0_is_fallback": bool(self.H0_is_fallback),
            "source": str(self.source),
        }


def _read_initial_params(initial_params: Any) -> InitialContext:
    """
    Normalise the many shapes a caller may legitimately hand in.

    Accepted: an ``RBergomiParams``; the ``(RBergomiParams, diagnostics)`` pair
    that ``initializer.initial_rbergomi_params`` returns; a mapping with
    ``H``/``eta``/``rho`` (optionally alongside ``hurst_se``, ``H0_ci95``,
    ``H0_is_fallback`` or a nested ``diagnostics``); or a length-3 sequence.
    """
    diagnostics: Mapping[str, Any] | None = None
    params: RBergomiParams | None = None
    source = "params"

    candidate = initial_params
    if (
        isinstance(candidate, tuple)
        and len(candidate) == 2
        and isinstance(candidate[1], Mapping)
    ):
        diagnostics = candidate[1]
        candidate = candidate[0]
        source = "initializer_pair"

    if isinstance(candidate, RBergomiParams):
        params = candidate
    elif isinstance(candidate, Mapping):
        nested = candidate.get("diagnostics")
        if diagnostics is None and isinstance(nested, Mapping):
            diagnostics = nested
        if diagnostics is None:
            diagnostics = candidate
        source = "mapping"
        try:
            params = RBergomiParams(
                H=float(candidate["H"]),
                eta=float(candidate["eta"]),
                rho=float(candidate["rho"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RBergomiCalibrationError(
                "initial_params doit contenir H, eta et rho ; "
                f"reçu {sorted(candidate)!r} ({exc})."
            ) from exc
    else:
        try:
            values = [float(x) for x in candidate]
        except (TypeError, ValueError) as exc:
            raise RBergomiCalibrationError(
                "initial_params doit être un RBergomiParams, un couple "
                "(RBergomiParams, diagnostics), un mapping H/eta/rho ou une "
                f"séquence de trois réels ; reçu {type(initial_params).__name__}."
            ) from exc
        if len(values) != 3:
            raise RBergomiCalibrationError(
                f"initial_params doit contenir exactement (H, eta, rho) ; reçu {len(values)} valeurs."
            )
        params = RBergomiParams(H=values[0], eta=values[1], rho=values[2])
        source = "sequence"

    H0 = float(params.H)
    H0_se = float("nan")
    is_fallback = False
    ci = (float("nan"), float("nan"))
    if diagnostics is not None:
        raw_H0 = diagnostics.get("H0_input", diagnostics.get("H0"))
        if isinstance(raw_H0, (int, float)) and math.isfinite(float(raw_H0)):
            H0 = float(raw_H0)
        raw_se = diagnostics.get("hurst_se", diagnostics.get("se"))
        if isinstance(raw_se, (int, float)) and math.isfinite(float(raw_se)):
            H0_se = float(raw_se)
        is_fallback = bool(
            diagnostics.get("H0_is_fallback", diagnostics.get("hurst_unstable", False))
        )
        raw_ci = diagnostics.get("H0_ci95", diagnostics.get("ci95"))
        if isinstance(raw_ci, Sequence) and not isinstance(raw_ci, (str, bytes)):
            try:
                ci = (float(raw_ci[0]), float(raw_ci[1]))
            except (IndexError, TypeError, ValueError):
                ci = (float("nan"), float("nan"))
    if not all(math.isfinite(x) for x in ci) and math.isfinite(H0_se):
        ci = (H0 - Z95 * H0_se, H0 + Z95 * H0_se)

    return InitialContext(
        params=params,
        H0=float(H0),
        H0_se=float(H0_se),
        H0_ci95=ci,
        H0_is_fallback=bool(is_fallback),
        source=source,
    )


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_param_constraint(val: Any) -> tuple[float | None, float | None, float | None]:
    """
    The repo constraints protocol, copied verbatim from the surrogate calibrator.

    Returns ``(min, max, fixed)``: a scalar or ``{"value": v}`` pins the
    parameter, ``[min, max]`` or ``{"min": .., "max": ..}`` tightens its bounds.
    """
    if isinstance(val, (int, float)):
        v = float(val)
        return v, v, v
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return _coerce_float(val[0]), _coerce_float(val[1]), None
    if isinstance(val, dict):
        if "value" in val:
            v = _coerce_float(val.get("value"))
            if v is None:
                return None, None, None
            return v, v, v
        return _coerce_float(val.get("min")), _coerce_float(val.get("max")), None
    return None, None, None


def resolve_bounds(
    constraints: Mapping[str, Any] | None,
) -> tuple[dict[str, tuple[float, float]], tuple[str, ...]]:
    """
    Apply the constraints protocol to the spec-4.10 box.

    Returns ``(bounds, pinned)``.  A **range** only ever tightens: it is
    intersected with the hard spec-4.6 box and can never widen it.  A **pin**
    outside that box is a different statement - the caller is asking for a model
    that does not exist - and raises.  ``"xi0"`` is **rejected** outright: it is
    frozen data, and silently ignoring an attempt to constrain it would hide the
    fact that the caller believed it was calibratable.
    """
    lower = {name: float(DEFAULT_BOUNDS[name][0]) for name in PARAM_ORDER}
    upper = {name: float(DEFAULT_BOUNDS[name][1]) for name in PARAM_ORDER}
    pinned: list[str] = []
    if isinstance(constraints, Mapping):
        if "xi0" in constraints or "xi0_scalar" in constraints:
            raise RBergomiCalibrationError(
                "ξ₀ n'est pas un paramètre calibrable : il provient des swaps de "
                "variance (spec 4.3/4.4) et reste figé pendant la calibration "
                "jointe (spec 4.10, séparation A). Retirez la contrainte 'xi0' ; "
                "la courbe se passe via constraints['xi0_curve']."
            )
        for name in PARAM_ORDER:
            if name not in constraints:
                continue
            mn, mx, fixed = _parse_param_constraint(constraints.get(name))
            if fixed is not None:
                lower[name] = float(fixed)
                upper[name] = float(fixed)
                pinned.append(name)
                continue
            if mn is not None:
                lower[name] = max(lower[name], float(mn))
            if mx is not None:
                upper[name] = min(upper[name], float(mx))
    bounds: dict[str, tuple[float, float]] = {}
    for name in PARAM_ORDER:
        lo, hi = lower[name], upper[name]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RBergomiCalibrationError(f"Bornes non finies pour {name}.")
        if lo > hi:
            raise RBergomiCalibrationError(
                f"Bornes invalides pour {name} : min={lo!r} > max={hi!r}."
            )
        hard_lo, hard_hi = DEFAULT_BOUNDS[name]
        if lo < hard_lo - 1e-12 or hi > hard_hi + 1e-12:
            raise RBergomiCalibrationError(
                f"Bornes hors du domaine du modèle pour {name} : "
                f"[{lo}, {hi}] déborde de [{hard_lo}, {hard_hi}] (spec 4.6/4.10)."
            )
        bounds[name] = (float(lo), float(hi))
    return bounds, tuple(pinned)


def _clip_to_bounds(
    theta: Sequence[float], bounds: Mapping[str, tuple[float, float]]
) -> np.ndarray:
    values = np.asarray(theta, dtype=float).ravel().copy()
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = bounds[name]
        values[i] = min(max(float(values[i]), float(lo)), float(hi))
    return values


# ---------------------------------------------------------------------------
# Local stage
# ---------------------------------------------------------------------------
def _free_indices(bounds: Mapping[str, tuple[float, float]]) -> list[int]:
    """Indices of the parameters the optimizer may actually move."""
    return [
        i
        for i, name in enumerate(PARAM_ORDER)
        if float(bounds[name][1]) > float(bounds[name][0])
    ]


def _embed(base: np.ndarray, free: Sequence[int]) -> Callable[[Sequence[float]], np.ndarray]:
    """Turn a free sub-vector into the full ``(H, eta, rho)`` vector."""

    def expand(reduced: Sequence[float]) -> np.ndarray:
        full = np.asarray(base, dtype=float).copy()
        for slot, index in enumerate(free):
            full[index] = float(reduced[slot])
        return full

    return expand


def _run_local(
    objective: JointObjective,
    x0: np.ndarray,
    *,
    bounds: Mapping[str, tuple[float, float]],
    config: JointMCConfig,
    n_paths: int,
    seed: int | None,
    max_nfev: int,
) -> tuple[np.ndarray, int, bool, str]:
    """
    One local search on ONE common-random-number draw.

    Returns ``(x, nfev, converged, message)``.  Pinned parameters are removed
    from the search space rather than handed to the optimizer as a degenerate
    box - SciPy's simplex and trust-region methods both misbehave on ``lb == ub``.
    """
    free = _free_indices(bounds)
    base = np.asarray(x0, dtype=float).copy()
    if not free:
        return base, 0, True, "Tous les paramètres sont figés : aucune optimisation."
    expand = _embed(base, free)
    lower = np.asarray([float(bounds[PARAM_ORDER[i]][0]) for i in free], dtype=float)
    upper = np.asarray([float(bounds[PARAM_ORDER[i]][1]) for i in free], dtype=float)
    start = np.asarray([float(base[i]) for i in free], dtype=float)
    start = np.minimum(np.maximum(start, lower), upper)

    if config.local_method == LOCAL_LEAST_SQUARES:
        residual_fn = objective.residual_function(n_paths=n_paths, seed=seed)

        def residuals(reduced: np.ndarray) -> np.ndarray:
            return residual_fn(expand(reduced))

        best, summary = multi_start_least_squares(
            residuals,
            x0_candidates=[start],
            bounds=(lower, upper),
            max_nfev=int(max_nfev),
        )
        if best is None:
            return (
                base,
                0,
                False,
                str(summary.get("message", "least_squares indisponible")),
            )
        return expand(best.x), int(best.nfev), bool(best.converged), str(best.message)

    if minimize is None:  # pragma: no cover - SciPy is a hard dependency
        return base, 0, False, "SciPy indisponible : scipy.optimize.minimize requis."

    loss_fn = objective.loss_function(n_paths=n_paths, seed=seed)

    def scalar(reduced: np.ndarray) -> float:
        return float(loss_fn(expand(reduced)))

    if config.local_method == LOCAL_NELDER_MEAD:
        options = {
            "maxfev": int(max_nfev),
            "xatol": float(config.xatol),
            "fatol": float(config.fatol),
        }
        method = "Nelder-Mead"
    else:
        options = {
            "maxfev": int(max_nfev),
            "xtol": float(config.xatol),
            "ftol": float(config.fatol),
        }
        method = "Powell"

    outcome = minimize(
        scalar,
        start,
        method=method,
        bounds=list(zip(lower, upper)),
        options=options,
    )
    return (
        expand(np.asarray(outcome.x, dtype=float)),
        int(getattr(outcome, "nfev", 0) or 0),
        bool(getattr(outcome, "success", False)),
        str(getattr(outcome, "message", "")),
    )


# ---------------------------------------------------------------------------
# The public driver
# ---------------------------------------------------------------------------
def calibrate_rbergomi(
    option_surface: Any,
    forward_variance: Any,
    initial_params: Any,
    *,
    weights_cfg: WeightConfig | None = None,
    mc_cfg: JointMCConfig | None = None,
    settings: CalibratorSettings | None = None,
    constraints: Mapping[str, Any] | None = None,
    clean_chains: Sequence[CleanChain] | None = None,
    S0: float | None = None,
    report_grid: SurfaceGrid | None = None,
) -> JointCalibrationResult:
    """
    Calibrate ``(H, eta, rho)`` jointly with ``xi0`` frozen (spec 4.10 + 4.11).

    Parameters
    ----------
    option_surface:
        The cleaned market quotes: a Phase-1 ``SurfacePoint`` sequence (the
        reference form), any mapping/nesting of them, or a repo ``SurfaceGrid``
        for the 9x6 grid mode.
    forward_variance:
        The spec-4.4 ``ForwardVarianceCurve``. **Frozen data**: it is sealed in a
        :class:`FrozenXi0`, read-only, re-hashed at the end, and returned by
        identity.
    initial_params:
        The spec-4.9 seed - see :func:`_read_initial_params` for the accepted
        shapes. Passing the ``(params, diagnostics)`` pair that
        ``initializer.initial_rbergomi_params`` returns also brings the ``H0``
        standard error along, which is what spec 4.11 compares against.
    weights_cfg, mc_cfg, settings:
        :class:`WeightConfig`, :class:`JointMCConfig` and the repo's
        ``CalibratorSettings`` (``max_nfev``, ``n_starts``, ``seed``).
        ``settings.max_nfev`` is used **only when it differs from the
        ``CalibratorSettings`` class default**; otherwise the per-run budget is
        ``mc_cfg.local_max_nfev(n_free)``, because the shared default of 80
        truncates Nelder-Mead on this problem - see
        :attr:`JointMCConfig.local_nfev_per_param`.
    constraints:
        The repo constraints protocol, restricted to ``H``/``eta``/``rho``.
        ``"xi0"`` raises.
    clean_chains:
        Phase-1 cleaned chains, used only to recover bid-ask spreads.
    S0:
        Spot. Inferred as ``F(T_min) D(T_min)`` when the quotes do not carry it.
    report_grid:
        An optional ``SurfaceGrid`` on which the final model IV surface is
        reported (the UI's 9x6 grid), independently of the quotes actually fitted.

    Returns
    -------
    JointCalibrationResult

    Raises
    ------
    RBergomiCalibrationError
        On unusable inputs, an empty quote set, an attempt to constrain ``xi0``,
        or if the frozen curve's fingerprint ever changes.
    """
    started = time.perf_counter()
    config = mc_cfg or JointMCConfig()
    weights = weights_cfg or WeightConfig()
    options = settings or CalibratorSettings()

    xi0 = FrozenXi0.freeze(forward_variance)
    initial = _read_initial_params(initial_params)
    bounds, pinned = resolve_bounds(constraints)
    quotes = build_calibration_quotes(
        option_surface, weights_cfg=weights, clean_chains=clean_chains, S0=S0
    )

    report_maturities: tuple[float, ...] = ()
    report_strikes: np.ndarray | None = None
    beyond_quotes: tuple[float, ...] = ()
    if report_grid is not None:
        t_grid = np.asarray(report_grid.t_grid, dtype=float)
        m_grid = np.asarray(report_grid.m_grid, dtype=float)
        usable = np.isfinite(t_grid) & (t_grid > 0.0)
        report_maturities = tuple(float(x) for x in t_grid[usable])
        if report_maturities:
            strikes = float(report_grid.S0) * m_grid[None, :]
            report_strikes = np.repeat(strikes, len(report_maturities), axis=0)
        # Past the last quoted maturity the forward is EXTRAPOLATED (see
        # forward_step_rates) and no quote constrains the model: say so.
        last_quoted = float(max(quotes.maturities))
        beyond_quotes = tuple(
            float(T) for T in report_maturities if float(T) > last_quoted
        )

    objective = JointObjective(
        quotes=quotes,
        xi0=xi0,
        config=config,
        report_maturities=report_maturities,
        report_strikes=report_strikes,
    )

    # -- seeds: always integers, so common random numbers actually hold -----
    master = np.random.default_rng(options.seed)

    def next_seed() -> int:
        return int(master.integers(0, 2**31 - 1))

    stage1_seed = next_seed()
    selection_seed = next_seed()
    n_starts = max(1, int(options.n_starts))
    restart_seeds = [next_seed() for _ in range(n_starts)]
    final_seed = next_seed()
    noise_seeds = [next_seed() for _ in range(max(1, int(config.noise_replicates)))]
    refinement_seed = next_seed()

    theta0 = _clip_to_bounds(
        [initial.params.H, initial.params.eta, initial.params.rho], bounds
    )
    free = _free_indices(bounds)
    profile_paths = int(config.effective_profile_paths)
    flags: list[str] = list(quotes.flags)
    if pinned:
        flags.append(FLAG_PARAMETER_PINNED)
    if beyond_quotes:
        flags.append(FLAG_REPORT_BEYOND_QUOTES)

    # -- Stage-2 budget: this module's own, unless the caller set max_nfev --
    # CalibratorSettings is a shared repo dataclass whose default (80) truncates
    # Nelder-Mead here; it is read, never modified.
    if int(options.max_nfev) != int(_SETTINGS_DEFAULT_MAX_NFEV):
        max_nfev = int(options.max_nfev)
        max_nfev_source = "settings"
    else:
        max_nfev = int(config.local_max_nfev(len(free)))
        max_nfev_source = "config"

    # -- Stage 1: coarse Latin-hypercube design ----------------------------
    stage1: StageOneReport | None = None
    starts: list[np.ndarray] = [theta0]
    if free and int(config.n_design) > 0:
        design = latin_hypercube_samples(
            n=int(config.n_design),
            bounds=[
                (float(bounds[name][0]), float(bounds[name][1])) for name in PARAM_ORDER
            ],
            rng=np.random.default_rng(stage1_seed),
        )
        candidates = np.vstack([theta0[None, :], np.asarray(design, dtype=float)])
        losses = np.asarray(
            [
                float(
                    objective.evaluate(
                        row, n_paths=int(config.stage1_paths), seed=stage1_seed
                    ).loss
                )
                for row in candidates
            ],
            dtype=float,
        )
        order = [int(i) for i in np.argsort(np.where(np.isfinite(losses), losses, np.inf))]
        # top_k is widened to cover n_starts: handing Stage 2 fewer starts than
        # the caller asked for and then reporting n_starts would be a lie.
        top = order[: max(1, int(config.top_k), n_starts)]
        stage1 = StageOneReport(
            design=tuple(tuple(float(v) for v in row) for row in candidates),
            losses=tuple(float(x) for x in losses),
            top_indices=tuple(top),
            anchor_index=0,
            n_paths=int(config.stage1_paths),
            seed=int(stage1_seed),
        )
        starts = [np.asarray(candidates[i], dtype=float) for i in top]

    # -- Stage 2: derivative-free local search, one CRN draw per restart ----
    runs: list[LocalRun] = []
    for index in range(n_starts):
        if index >= len(starts):
            break
        seed = restart_seeds[index] if config.crn_per_restart else selection_seed
        x_opt, nfev, converged, message = _run_local(
            objective,
            starts[index],
            bounds=bounds,
            config=config,
            n_paths=int(config.stage2_paths),
            seed=int(seed),
            max_nfev=int(max_nfev),
        )
        x_opt = _clip_to_bounds(x_opt, bounds)
        own = float(
            objective.evaluate(x_opt, n_paths=int(config.stage2_paths), seed=int(seed)).loss
        )
        # Re-score on ONE common draw so restarts are compared like for like.
        selection = float(
            objective.evaluate(x_opt, n_paths=profile_paths, seed=selection_seed).loss
        )
        runs.append(
            LocalRun(
                index=index,
                x0=tuple(float(v) for v in starts[index]),
                x=tuple(float(v) for v in x_opt),
                loss_own_draw=own,
                loss_selection_draw=selection,
                nfev=int(nfev),
                converged=bool(converged),
                message=str(message),
                seed=int(seed),
            )
        )

    if runs:
        best_index = int(
            np.argmin([run.loss_selection_draw for run in runs])
        )
        theta_star = _clip_to_bounds(runs[best_index].x, bounds)
        stage2 = StageTwoReport(
            method=str(config.local_method),
            runs=tuple(runs),
            best_index=best_index,
            n_paths=int(config.stage2_paths),
            max_nfev=int(max_nfev),
            selection_seed=int(selection_seed),
            n_starts_requested=int(n_starts),
            max_nfev_source=str(max_nfev_source),
        )
        if not any(run.converged for run in runs):
            flags.append(FLAG_LOCAL_NOT_CONVERGED)
        if len(runs) < n_starts:
            flags.append(FLAG_RESTARTS_TRUNCATED)
    else:  # every parameter pinned: nothing to optimise, evaluate and report
        theta_star = theta0.copy()
        stage2 = None

    # -- losses on ONE common selection draw, so they are comparable -------
    at_optimum = objective.evaluate(theta_star, n_paths=profile_paths, seed=selection_seed)
    at_initial = objective.evaluate(theta0, n_paths=profile_paths, seed=selection_seed)
    loss_crn = float(at_optimum.loss)
    loss_initial = float(at_initial.loss)

    # -- final high-accuracy repricing, FRESH seed -------------------------
    final = objective.evaluate(
        theta_star,
        n_paths=int(config.final_paths),
        seed=int(final_seed),
        with_report=report_strikes is not None,
        use_cache=False,
    )
    loss_fresh = float(final.loss)
    if final.n_uninvertible:
        flags.append(FLAG_UNINVERTIBLE_MODEL_PRICE)

    # -- over-fitting of the optimizer's own draw, at MATCHED path counts ---
    # `loss_fresh` is measured on `final_paths` (100 000 by default) and is
    # therefore structurally smaller than a `stage2_paths` loss, whatever the
    # over-fit: comparing the two compares two estimators, not two draws. The
    # honest comparison is the loss at theta* on the draw the local stage
    # actually fitted against the loss at theta* on a fresh draw of the SAME
    # size.
    if runs:
        matched_paths = int(config.stage2_paths)
        loss_crn_matched = float(runs[best_index].loss_own_draw)
    else:  # everything pinned: the "in-sample" draw is the selection draw
        matched_paths = int(profile_paths)
        loss_crn_matched = float(loss_crn)
    loss_fresh_matched = float(
        objective.evaluate(
            theta_star, n_paths=matched_paths, seed=int(final_seed)
        ).loss
    )
    if (
        math.isfinite(loss_fresh_matched)
        and math.isfinite(loss_crn_matched)
        and loss_crn_matched > 0.0
        and loss_fresh_matched > float(config.fresh_seed_gap_ratio) * loss_crn_matched
    ):
        flags.append(FLAG_FRESH_SEED_LOSS_GAP)

    # -- spec 4.11 diagnostics ---------------------------------------------
    noise = measure_noise_floor(
        objective,
        theta_star,
        config=config,
        seeds=[int(s) for s in noise_seeds],
        n_paths=profile_paths,
        delta_std=float(at_optimum.loss_noise_std),
        bounds=bounds,
        free=free,
    )
    # Every span / gradient / improvement below is a difference taken inside ONE
    # draw, so the CRN-difference floor is the yardstick, not the across-draw
    # scatter of the loss level (see NoiseFloor).
    difference_floor = float(noise.difference_value)
    sigma_level = float(noise.sigma_level)
    profiles: list[ProfileSlice] = []
    for index in free:
        name = PARAM_ORDER[index]
        own_floor = float(noise.difference_for(name))
        profiles.append(
            profile_slice(
                objective,
                theta_star,
                index=index,
                bounds=bounds[name],
                n_points=int(config.profile_points),
                n_paths=profile_paths,
                seed=int(selection_seed),
                noise_floor=own_floor,
                sigma_level=sigma_level,
                se_threshold=float(config.se_material_ratio)
                * float(PARAM_SCALE.get(name, float("nan"))),
                stationarity_floor=max(own_floor, float(noise.value)),
            )
        )
    valley = None
    if 1 in free and 2 in free:
        valley = eta_rho_valley(
            objective,
            theta_star,
            bounds=bounds,
            n_points=int(config.valley_points),
            n_paths=profile_paths,
            seed=int(selection_seed),
            noise_floor=difference_floor,
        )

    identifiability_flags: list[str] = []
    weak_flag_of = {
        "H": FLAG_H_WEAKLY_IDENTIFIED,
        "eta": FLAG_ETA_WEAKLY_IDENTIFIED,
        "rho": FLAG_RHO_WEAKLY_IDENTIFIED,
    }
    for slice_ in profiles:
        if slice_.flat:
            identifiability_flags.append(
                {
                    "H": FLAG_H_PROFILE_FLAT,
                    "eta": FLAG_ETA_PROFILE_FLAT,
                    "rho": FLAG_RHO_PROFILE_FLAT,
                }[slice_.parameter]
            )
        if slice_.weakly_identified:
            identifiability_flags.append(weak_flag_of[slice_.parameter])
        # The module's OWN invariant, asserted rather than assumed: theta* must
        # be the cheapest point of its own profile, up to the noise floor.
        if not slice_.stationary:
            identifiability_flags.append(FLAG_PROFILE_NOT_STATIONARY)
    if valley is not None and valley.flat:
        identifiability_flags.append(FLAG_ETA_RHO_VALLEY_FLAT)

    # SE(H) against the spec-4.5 H0 standard error: an H whose profile-based
    # standard error dwarfs the skew regression's own is not a refinement of H0.
    h_profile = next((p for p in profiles if p.parameter == "H"), None)
    if (
        h_profile is not None
        and math.isfinite(initial.H0_se)
        and initial.H0_se > 0.0
        and not initial.H0_is_fallback
        and not (
            h_profile.standard_error
            <= float(config.se_vs_h0_factor) * float(initial.H0_se)
        )
    ):
        identifiability_flags.append(FLAG_H_WEAKLY_IDENTIFIED)

    improvement = loss_initial - loss_crn
    improvement_significant = bool(
        math.isfinite(improvement) and improvement > difference_floor
    )
    if not improvement_significant:
        identifiability_flags.append(FLAG_NO_IMPROVEMENT)
    if initial.H0_is_fallback:
        identifiability_flags.append(FLAG_H0_FALLBACK)

    ci_low, ci_high = initial.H0_ci95
    H_in_ci = bool(
        math.isfinite(ci_low)
        and math.isfinite(ci_high)
        and ci_low <= float(theta_star[0]) <= ci_high
    )
    if (
        math.isfinite(ci_low)
        and math.isfinite(ci_high)
        and not H_in_ci
        and not initial.H0_is_fallback
    ):
        identifiability_flags.append(FLAG_H_OUTSIDE_H0_CI)

    identifiability = IdentifiabilityReport(
        noise_floor=noise,
        profiles=tuple(profiles),
        valley=valley,
        H0=float(initial.H0),
        H0_se=float(initial.H0_se),
        H0_ci95=(float(ci_low), float(ci_high)),
        H0_is_fallback=bool(initial.H0_is_fallback),
        H_calibrated=float(theta_star[0]),
        H_in_ci=H_in_ci,
        loss_initial=loss_initial,
        loss_optimum=loss_crn,
        improvement=float(improvement),
        improvement_significant=improvement_significant,
        flags=tuple(dict.fromkeys(identifiability_flags)),
    )
    flags.extend(identifiability_flags)

    grid_bias = None
    if config.refinement_check and profiles:
        grid_bias = grid_refinement_bias(
            objective,
            theta_star,
            profiles=profiles,
            config=config,
            bounds=bounds,
            n_paths=profile_paths,
            seed=int(refinement_seed),
        )
        if grid_bias is not None and grid_bias.material:
            flags.append(FLAG_GRID_BIAS_MATERIAL)
        if grid_bias is not None and grid_bias.unmeasured:
            flags.append(FLAG_GRID_BIAS_NOT_MEASURED)

    # -- bounds check -------------------------------------------------------
    for index, name in enumerate(PARAM_ORDER):
        lo, hi = bounds[name]
        if hi <= lo:
            continue
        tolerance = float(config.bound_atol_rel) * (hi - lo)
        if theta_star[index] <= lo + tolerance or theta_star[index] >= hi - tolerance:
            flags.append(FLAG_PARAMETER_AT_BOUND)
            break

    # -- metrics on the fitted quote set -----------------------------------
    finite = np.isfinite(final.iv_error)
    metrics = iv_error_metrics(final.iv_error, finite)
    metrics_vw = iv_error_metrics_weighted(
        final.iv_error, finite, quotes.array("vega")
    )

    xi0.verify()

    params = RBergomiParams(
        H=float(theta_star[0]), eta=float(theta_star[1]), rho=float(theta_star[2])
    )
    elapsed = time.perf_counter() - started
    ordered_flags = tuple(dict.fromkeys(flags))

    # `success` is a VERDICT, not a constant. It is False whenever the result
    # carries no information about H: a flat H profile, no improvement over the
    # initial point beyond the measured noise, or an optimum that is not
    # stationary on its own profile. Anything else (a pinned parameter, an
    # unconverged-but-stationary local stage, a material grid bias) is a warning
    # carried in `flags`, not a failure. This matters beyond this module:
    # `apply_degeneracy_guard` only trips on an all-NaN surface, so without this
    # a meaningless H would reach the Phase-5 controller as a calibrated result.
    blocking = tuple(f for f in ordered_flags if f in BLOCKING_FLAGS)
    success = not blocking

    message_fr = (
        f"Calibration jointe rBergomi : H = {params.H:.4f}, eta = {params.eta:.3f}, "
        f"rho = {params.rho:+.3f} sur {quotes.n_quotes} cotations et "
        f"{quotes.n_maturities} échéances (ξ₀ figé). "
        f"RMSE hors échantillon {math.sqrt(max(loss_fresh, 0.0)) * 100.0:.2f} pt de vol, "
        f"coût CRN {loss_crn:.3e} contre {loss_initial:.3e} au point initial."
    )
    if not success:
        message_fr = (
            "Calibration jointe rBergomi NON CONCLUANTE — le résultat ne mesure "
            "rien d'exploitable : "
            + " ".join(
                JOINT_CALIBRATION_LABELS_FR.get(f, f) for f in blocking
            )
            + " Valeurs rendues à titre diagnostique uniquement : "
            + message_fr
        )

    result = JointCalibrationResult(
        success=bool(success),
        message_fr=message_fr,
        params=params,
        theta=np.asarray(theta_star, dtype=float),
        initial_params=initial.params,
        bounds=bounds,
        pinned=pinned,
        xi0_curve=xi0.curve,
        xi0=xi0,
        quotes=quotes,
        loss_crn=loss_crn,
        loss_fresh=loss_fresh,
        loss_initial=loss_initial,
        rmse_fresh=float(math.sqrt(max(loss_fresh, 0.0))),
        loss_crn_matched=float(loss_crn_matched),
        loss_fresh_matched=float(loss_fresh_matched),
        matched_paths=int(matched_paths),
        iv_model=final.iv_model,
        iv_error=final.iv_error,
        price_model=final.price_model,
        weights=np.asarray(quotes.weights, dtype=float),
        metrics={k: float(v) for k, v in metrics.items()},
        metrics_vw={k: float(v) for k, v in metrics_vw.items()},
        stage1=stage1,
        stage2=stage2,
        identifiability=identifiability,
        grid_bias=grid_bias,
        final_evaluation=final,
        config=config,
        seed=None if options.seed is None else int(options.seed),
        n_objective_evaluations=int(objective.n_eval),
        mean_evaluation_seconds=(
            float(objective.total_eval_seconds / objective.n_eval)
            if objective.n_eval
            else float("nan")
        ),
        elapsed_s=float(elapsed),
        flags=ordered_flags,
        details={
            "initial": initial.to_dict(),
            "seeds": {
                "stage1": int(stage1_seed),
                "selection": int(selection_seed),
                # Only the restart seeds that were actually consumed: reporting
                # n_starts seeds when fewer runs happened was the bug D5 names.
                "restarts": [int(s) for s in restart_seeds[: len(runs)]],
                "restarts_unused": [int(s) for s in restart_seeds[len(runs) :]],
                "final": int(final_seed),
                "noise": [int(s) for s in noise_seeds],
                "refinement": int(refinement_seed),
            },
            "settings": {
                "max_nfev_requested": int(options.max_nfev),
                "max_nfev_effective": int(max_nfev),
                "max_nfev_source": str(max_nfev_source),
                "n_starts_requested": int(options.n_starts),
                "n_starts_effective": int(len(runs)),
                "n_free_parameters": int(len(free)),
                "seed": None if options.seed is None else int(options.seed),
                "fit_to_observed_only": bool(options.fit_to_observed_only),
            },
            "report_maturities_beyond_quotes": [float(x) for x in beyond_quotes],
            "grid": objective.grid().diagnostics(),
            "quote_rejections": [r.to_dict() for r in quotes.rejections[:200]],
            "objective_definition_fr": (
                "Somme pondérée des carrés d'erreur de volatilité implicite sur "
                "les cotations nettoyées (poids = 1/spread² exprimé en unités de "
                "vol, soit véga²/spread_prix², normalisés par échéance), nombres "
                "aléatoires communs."
            ),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Repo calibrator
# ---------------------------------------------------------------------------
def _config_from_mapping(base: Any, overrides: Any) -> Any:
    """Rebuild a frozen config dataclass with the mapping's overrides applied."""
    if not isinstance(overrides, Mapping):
        return base
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    for key, value in overrides.items():
        if key in fields:
            fields[key] = value
    return type(base)(**fields)


class RBergomiJointHCalibrator(BaseSurfaceCalibrator):
    """
    ``BaseSurfaceCalibrator`` front end of the spec-4.10 joint calibration.

    ``model = "rbergomi"``, ``method = "joint_h_mc"``, ``PARAM_ORDER =
    ("H", "eta", "rho")``.  ``xi0`` is deliberately absent from ``PARAM_ORDER``
    and from ``DEFAULT_BOUNDS``, and constraining it raises.

    ``constraints`` keys, all optional except the first:

    ``xi0_curve``
        The spec-4.4 ``ForwardVarianceCurve``. **Required**: an IV grid alone
        does not determine a variance-swap curve, so its absence is an explicit
        failure rather than a silent flat-``xi0`` fallback.
    ``option_surface``
        Phase-1 ``SurfacePoint`` quotes. When present the fit runs on the real
        quotes and the ``SurfaceGrid`` is used only to report the model surface.
    ``clean_chains``
        Phase-1 cleaned chains, for the bid-ask spreads.
    ``initial_params``
        The spec-4.9 seed, in any shape :func:`_read_initial_params` accepts.
        Without it the anchor is the centre of the box, recorded as such - no
        parameter value is ever smuggled in as a default.
    ``mc_cfg`` / ``weights_cfg``
        Mappings overriding :class:`JointMCConfig` / :class:`WeightConfig`.
    ``H`` / ``eta`` / ``rho``
        The repo constraints protocol (scalar or ``{"value"}`` pins,
        ``[min, max]`` or ``{"min","max"}`` tightens).
    """

    model = "rbergomi"
    method = "joint_h_mc"

    PARAM_ORDER: tuple[str, ...] = PARAM_ORDER
    DEFAULT_BOUNDS: dict[str, tuple[float, float]] = dict(DEFAULT_BOUNDS)

    def _default_initial(
        self, bounds: Mapping[str, tuple[float, float]]
    ) -> tuple[RBergomiParams, str]:
        """The neutral centre of the (possibly tightened) box."""
        H_lo, H_hi = bounds["H"]
        eta_lo, eta_hi = bounds["eta"]
        rho_lo, rho_hi = bounds["rho"]
        return (
            RBergomiParams(
                H=0.5 * (H_lo + H_hi),
                eta=float(math.sqrt(max(eta_lo, 1e-12) * eta_hi)),
                rho=0.5 * (rho_lo + rho_hi),
            ),
            "box_centre",
        )

    def calibrate(
        self,
        surface: SurfaceGrid,
        *,
        constraints: dict[str, Any] | None = None,
        settings: CalibratorSettings | None = None,
    ) -> SurfaceCalibrationResult:
        options = settings or CalibratorSettings()
        payload = constraints if isinstance(constraints, Mapping) else {}

        curve = payload.get("xi0_curve")
        if curve is None:
            return SurfaceCalibrationResult(
                success=False,
                message=(
                    "Courbe de variance forward absente : la calibration jointe "
                    "rBergomi exige constraints['xi0_curve'] (spec 4.4). Une "
                    "grille de volatilités implicites ne détermine pas une "
                    "courbe de swaps de variance ; aucun repli n'est fabriqué."
                ),
                model=self.model,
                method=self.method,
                params={},
            )

        try:
            bounds, _pinned = resolve_bounds(payload)
        except RBergomiCalibrationError as exc:
            return SurfaceCalibrationResult(
                success=False,
                message=str(exc),
                model=self.model,
                method=self.method,
                params={},
            )

        mc_config = _config_from_mapping(JointMCConfig(), payload.get("mc_cfg"))
        weight_config = _config_from_mapping(WeightConfig(), payload.get("weights_cfg"))
        initial = payload.get("initial_params")
        initial_source = "constraints"
        if initial is None:
            initial, initial_source = self._default_initial(bounds)

        quote_source = payload.get("option_surface")
        fit_surface = surface if quote_source is None else quote_source

        try:
            result = calibrate_rbergomi(
                fit_surface,
                curve,
                initial,
                weights_cfg=weight_config,
                mc_cfg=mc_config,
                settings=options,
                constraints=payload,
                clean_chains=payload.get("clean_chains"),
                S0=float(surface.S0),
                report_grid=surface,
            )
        except (
            RBergomiCalibrationError,
            RBergomiSimulationError,
            VolterraGaussianError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:
            return SurfaceCalibrationResult(
                success=False,
                message=f"Calibration jointe rBergomi impossible : {exc}",
                model=self.model,
                method=self.method,
                params={},
            )

        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)
        iv_market = np.asarray(surface.iv_market, dtype=float)
        mask = effective_mask(
            iv_market, surface.mask, fit_to_observed_only=options.fit_to_observed_only
        )
        iv_model = np.full_like(iv_market, np.nan, dtype=float)
        report = result.final_evaluation.report_iv
        if report is not None:
            usable = np.isfinite(t_grid) & (t_grid > 0.0)
            iv_model[usable, :] = np.asarray(report, dtype=float)
        iv_error = np.where(mask, iv_model - iv_market, np.nan)
        grid_metrics = iv_error_metrics(iv_error, mask)
        vega_weights = compute_bs_vega_grid(
            float(surface.S0),
            m_grid,
            t_grid,
            float(surface.r),
            float(surface.q),
            iv_market,
        )
        grid_metrics_vw = iv_error_metrics_weighted(iv_error, mask, vega_weights)

        details = result.to_dict()
        details["report"] = calibration_report(result)
        details["initial_params_source"] = str(initial_source)
        details["quote_source"] = (
            "surface_grid" if quote_source is None else "option_surface"
        )
        details["quote_set_metrics"] = {k: float(v) for k, v in result.metrics.items()}
        details["quote_set_metrics_vw"] = {
            k: float(v) for k, v in result.metrics_vw.items()
        }

        outcome = SurfaceCalibrationResult(
            success=bool(result.success),
            message=str(result.message_fr),
            model=self.model,
            method=self.method,
            params=result.params.to_dict(),
            metrics={k: float(v) for k, v in grid_metrics.items()},
            metrics_vw={k: float(v) for k, v in grid_metrics_vw.items()},
            iv_model=iv_model,
            iv_error=iv_error,
            vega_weights=vega_weights,
            details=details,
        )
        return apply_degeneracy_guard(outcome)


__all__ = [
    "BLOCKING_FLAGS",
    "DEFAULT_BOUNDS",
    "FLAG_ETA_PROFILE_FLAT",
    "FLAG_ETA_RHO_VALLEY_FLAT",
    "FLAG_ETA_WEAKLY_IDENTIFIED",
    "FLAG_FRESH_SEED_LOSS_GAP",
    "FLAG_GRID_BIAS_MATERIAL",
    "FLAG_GRID_BIAS_NOT_MEASURED",
    "FLAG_H0_FALLBACK",
    "FLAG_H_OUTSIDE_H0_CI",
    "FLAG_H_PROFILE_FLAT",
    "FLAG_H_WEAKLY_IDENTIFIED",
    "FLAG_LOCAL_NOT_CONVERGED",
    "FLAG_NO_IMPROVEMENT",
    "FLAG_PARAMETER_AT_BOUND",
    "FLAG_PARAMETER_PINNED",
    "FLAG_PROFILE_NOT_STATIONARY",
    "FLAG_QUOTES_DROPPED",
    "FLAG_REPORT_BEYOND_QUOTES",
    "FLAG_RESTARTS_TRUNCATED",
    "FLAG_RHO_PROFILE_FLAT",
    "FLAG_RHO_WEAKLY_IDENTIFIED",
    "FLAG_SINGLE_MATURITY",
    "FLAG_UNINVERTIBLE_MODEL_PRICE",
    "JOINT_CALIBRATION_LABELS_FR",
    "PARAM_SCALE",
    "LOCAL_LEAST_SQUARES",
    "LOCAL_METHODS",
    "LOCAL_NELDER_MEAD",
    "LOCAL_POWELL",
    "OBJECTIVES",
    "OBJECTIVE_IV",
    "OBJECTIVE_PRICE_RELATIVE",
    "PARAM_ORDER",
    "QUOTE_REASON_LABELS_FR",
    "REASON_INVALID_MATURITY",
    "REASON_INVALID_STRIKE",
    "REASON_K_STANDARDISED",
    "REASON_K_TOO_FAR",
    "REASON_NON_FINITE_IV",
    "REASON_SPREAD_TOO_WIDE",
    "REASON_TOO_FEW_PER_MATURITY",
    "REASON_ZERO_VEGA",
    "UNINVERTIBLE_PENALTY_VOL",
    "Z95",
    "CalibrationQuote",
    "FrozenXi0",
    "GridBiasReport",
    "IdentifiabilityReport",
    "InitialContext",
    "JointCalibrationResult",
    "JointMCConfig",
    "JointObjective",
    "LocalRun",
    "NoiseFloor",
    "ObjectiveEvaluation",
    "ProfileSlice",
    "QuoteRejection",
    "QuoteSet",
    "RBergomiCalibrationError",
    "RBergomiJointHCalibrator",
    "StageOneReport",
    "StageTwoReport",
    "ValleyProfile",
    "WeightConfig",
    "build_calibration_quotes",
    "calibrate_rbergomi",
    "calibration_report",
    "eta_rho_valley",
    "forward_step_rates",
    "grid_refinement_bias",
    "measure_noise_floor",
    "profile_slice",
    "resolve_bounds",
]
