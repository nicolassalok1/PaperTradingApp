"""
Rough Bergomi Monte-Carlo simulator driven by a **forward-variance CURVE** (spec 4.8).

This module is the term-structure-aware successor of
``app/model/volatility_models/rbergomi/simulator.py``.  The legacy simulator
takes a *scalar* ``xi0``, builds the Volterra driver with a naive ``O(n^2)``
left-point Riemann sum and **snaps** every quoted maturity onto a uniform grid
with ``round(T / dt)``.  All three are defects for a Hurst-calibration
pipeline.  Here instead:

* ``xi0`` is the :class:`~app.model.calibration.rough_vol.forward_variance.ForwardVarianceCurve`
  bootstrapped from the variance-swap curve of spec 4.3/4.4 - it is **data**,
  frozen, and structurally impossible for this module to modify;
* the driver pair ``(W~, dB)`` comes from the **exact** joint Gaussian
  construction of :mod:`app.model.volatility_models.rbergomi.volterra_gaussian`
  (a single Cholesky, cached per ``(H, grid)``), so there is no discretisation
  error in the covariance structure at all;
* the grid comes from ``build_simulation_grid``: every quoted maturity sits on
  it with **exact float equality**, so all maturities are priced from ONE path
  set with no snapping bias.

MODEL - SPEC 4.6 CONVENTIONS (locked, reproduced verbatim)
==========================================================
.. code-block:: text

    W~_t = sqrt(2H) * int_0^t (t-s)^{H-1/2} dB_s          (Var W~_t = t^{2H})
    V_t  = xi0(t) * exp( eta*W~_t - 0.5*eta^2*t^{2H} )    (=> E[V_t] = xi0(t))
    dS_t = (r(t)-q)*S_t*dt + S_t*sqrt(V_t)*dW^S_t
    W^S  = rho*B + sqrt(1-rho^2)*B_perp

Parameter domain: ``0.01 <= H <= 0.49``, ``eta > 0``, ``-0.999 <= rho <= 0.999``.
``(0.05, 5.0]`` is the *calibration working range* for ``eta`` (see
:data:`ETA_WORKING_MIN` / :data:`ETA_WORKING_MAX`); it is documented, not
enforced, because the ``eta -> 0`` degeneracy to Black-Scholes is a required
validation of the model itself.

WHY THE CORRELATION IS REALISED THROUGH THE SHARED DRIVER ``B``
===============================================================
``W~`` is **not a semimartingale** (its quadratic variation is infinite for
``H < 1/2``), so the usual shorthand ``d<W^S, W~>_t = rho dt`` has no literal
meaning.  What *is* well defined - and what this module implements - is that the
spot noise and the Volterra driver are built from **the same Brownian motion**
``B``:

.. code-block:: text

    W~_t   = sqrt(2H) * int_0^t (t-s)^{H-1/2} dB_s        <- driven by B
    dW^S_i = rho * dB_i + sqrt(1-rho^2) * dB_perp_i       <- driven by B and B_perp

``dB_i`` in the second line is *the very same increment* that appears inside the
Volterra integral of the first line: ``volterra_gaussian`` returns
``(W~_{t_1..t_n}, dB_1..dB_n)`` as one exactly-correlated Gaussian vector
produced by a single Cholesky factor.  The consequence is the exact cross
moment

.. code-block:: text

    E[W^S_t * W~_t] = rho * sqrt(2H)/(H + 1/2) * t^{H+1/2},

which holds **by construction, with no discretisation error**, and which
``tests/quant/test_rv_rbergomi_mc.py`` checks statistically for several ``rho``
including a negative one.

**NEVER correlate fBm (or ``W~``) VALUES with the spot noise directly.**  Doing
so (e.g. drawing a standard fBm from Davies-Harte and correlating its marginals
against a Brownian motion) imposes a correlation on the marginal laws rather
than on the driving noise: the resulting joint law is *not* the rough Bergomi
law, the cross moment above is violated, and the smile skew it produces is
wrong.  :mod:`app.model.volatility_models.rbergomi.fbm` exists only for the
roughness *diagnostics* of spec 4.7c and must never be fed to this pricer.

DISCRETISATION - EXACTLY WHAT IS APPROXIMATED AND WHAT IS NOT
==============================================================
*Exact* (no discretisation error whatsoever):

* the joint law of ``(W~_{t_0..t_n}, dB_1..dB_n, dB_perp_1..dB_perp_n)`` on the
  grid - it is sampled from its exact covariance;
* therefore ``V_{t_i}`` at every grid node, and the spot/vol correlation.

*Approximated* - the **log-Euler scheme with left-point (adapted) variance**:

.. code-block:: text

    ln S_{i+1} = ln S_i + (r_i - q)*dt_i - 0.5*V_{t_i}*dt_i
                        + sqrt(V_{t_i}) * ( rho*dB_i + sqrt(1-rho^2)*dB_perp_i )

The variance is frozen at the **left** endpoint of each cell.  That choice is
not cosmetic: ``V_{t_i}`` is measurable with respect to ``F_{t_i}`` (the
Volterra kernel is causal, so ``Cov(W~_{t_i}, dB_j) = 0`` for every increment
``dB_j`` living strictly after ``t_i``), while ``dB_i`` and ``dB_perp_i`` span
``[t_i, t_{i+1}]`` and are independent of ``F_{t_i}``.  Hence

.. code-block:: text

    E[ exp( sqrt(V_{t_i}) dW^S_i - 0.5 V_{t_i} dt_i ) | F_{t_i} ] = 1   exactly,

so the **discounted, dividend-adjusted spot is an exact martingale of the
discrete scheme**: ``E[D(T) S_T e^{qT}] = S_0`` holds with *no* discretisation
bias, only Monte-Carlo error.  A midpoint or right-point variance would destroy
that.  What the scheme does approximate is the time integral of the variance
inside one cell (``int_{t_i}^{t_{i+1}} V_u du ~ V_{t_i} dt_i``); the error is
``O(dt)`` in the usual weak sense and shrinks with ``GridConfig.n_max``.

NON-UNIFORM ``dt`` - EVERY OCCURRENCE USES THE REAL STEP
=========================================================
The grid is deliberately **log-spaced** (dense at the short end, where the
Volterra kernel is most singular and where the variance curve moves fastest),
so ``dt_i`` varies by orders of magnitude across the grid.  Every single place a
step length appears - the drift ``(r_i - q) dt_i``, the Ito compensator
``-0.5 V_{t_i} dt_i``, the integrated variance ``sum V_{t_i} dt_i``, the
discount factors, the variance of the Brownian increments - uses the **actual**
``grid.dt[i]``.  Using a constant ``T/n`` anywhere is the classic silent bug of
this family of simulators; ``tests/quant/test_rv_rbergomi_mc.py`` pins it with a
strongly non-uniform grid, a per-step rate array and an ``eta -> 0``
degeneracy that would move by percent if a constant ``dt`` crept in.

xi0 ON THE GRID: NODE EVALUATION vs CELL AVERAGE
=================================================
Spec 4.8 states ``V_{t_i} = xi0(t_i) * exp(...)`` with the *node* value of
``xi0``.  That is the default here (:data:`XI0_NODE`) and it is exactly right
for the variance process itself.  It does, however, interact with the default
**piecewise-constant** forward-variance curve: ``xi0`` is left-continuous with
its jumps sitting exactly on the quoted maturities, which are exactly grid
nodes.  At such a node ``xi0(t_k)`` returns the level of the cell that *ends*
at ``t_k``, while the Euler cell ``[t_k, t_{k+1}]`` is governed by the *next*
level.  The left-point sum ``sum_i xi0(t_i) dt_i`` therefore differs from
``int_0^T xi0`` by

.. code-block:: text

    sum over knots T_j on (0, T) of  ( xi0(T_j) - xi0(T_j^+) ) * dt_{k_j} .

This is **never silent**: the exact, deterministic gap is computed and exposed
as :attr:`RBergomiPathSet.xi0_quadrature_error` (and in ``.diagnostics()``) at
every grid node.  :data:`XI0_CELL_AVERAGE` replaces the node value by the exact
cell average ``(1/dt_i) int_{t_i}^{t_{i+1}} xi0(u) du``, which makes that gap
identically zero and reproduces the variance-swap input exactly.

**CELL AVERAGE IS THE DEFAULT — a recorded deviation from the letter of spec
4.8, taken deliberately.**  Spec 4.8 writes ``V_{t_i} = xi0(t_i) exp(...)``, but
that is the *discretisation* of a continuous model whose variance is
``xi0(t) exp(...)``; the cell average is a strictly better quadrature of the SAME
continuous model, not a different model.  The node value is not:

* spec 4.4 requires ``int_0^{T_j} xi0 = T_j K_var(T_j)`` **exactly** at every
  quoted maturity, and spec 3 requires ``xi0`` to stay frozen DATA all the way
  through the calibration.  Under node evaluation the simulated variance-swap
  strike does **not** reproduce the ``K_var`` that ``xi0`` was bootstrapped from,
  so that guarantee is broken at the source;
* the resulting error is maturity-dependent and **changes sign** across the term
  structure (measured positive at 1-4 months, negative past 6 months), which is
  the worst possible shape: it tilts ``log|psi(T)| vs log T`` rather than shifting
  it, and therefore lands directly on ``H``;
* it also fails the ``eta -> 0`` Black-Scholes degeneracy by a few hundredths of
  a vol point, maturity by maturity.

:data:`XI0_NODE` remains available and fully tested for anyone who wants spec 4.8
verbatim; the choice is a single documented flag,
:attr:`SimulationConfig.xi0_evaluation`.

VARIANCE REDUCTION AND SAMPLING
================================
* **Antithetic pairs are built in** (``SimulationConfig.antithetic``, default
  ``True``): ``n_paths // 2`` base paths are drawn and mirrored by exact
  negation of *all* the underlying normals, so rows ``k`` and
  ``k + n_paths//2`` are bit-for-bit opposite in ``W~``, ``dB`` and
  ``dB_perp``.  Because the pairs are **not independent**, every standard error
  produced by this package is computed from the ``n_paths // 2`` *pair means* -
  see :func:`sample_mean_stderr`.  Reporting ``std / sqrt(n_paths)`` instead
  would be an outright lie about the accuracy.
* **Scrambled-Sobol QMC** (``SimulationConfig.qmc``, default ``False``) replaces
  the ``3n`` i.i.d. normals by an Owen-scrambled Sobol' sequence mapped through
  the inverse normal CDF.  Its sample standard deviation is **not** a valid
  error estimate for a randomised-QMC mean; the reported stderr is flagged as a
  conservative pseudo-MC proxy (``stderr_is_conservative``) rather than
  silently passed off as the QMC error.
* The **conditional (mixed) estimator** of Romano-Touzi / McCrickerd-Pakkanen
  lives in :mod:`app.model.volatility_models.rbergomi.pricing`; this module
  exposes the two statistics it needs, ``int_0^T V du`` and ``int_0^T sqrt(V)
  dB``, both accumulated with the real ``dt_i``.

CACHING AND REPRODUCIBILITY
============================
The Cholesky factor is memoised by :func:`volterra_gaussian.cholesky_factor`
under ``(H, grid hash, jitter policy)`` and the ``xi0`` evaluation is memoised
here under ``(curve fingerprint, grid hash, mode)``.  Both survive across
strikes, maturities and repeated calls, which is what makes a common-random-
numbers calibration loop affordable.  With ``SimulationConfig.seed`` fixed the
whole path set is **bit-identical** between runs.

MEMORY
======
The path set retains five ``(n_paths, n+1)``-shaped float64 arrays plus the
three ``(n_paths, n)`` arrays of the draw: roughly ``64 * n_paths * (n+1)``
bytes.  10 000 paths on a 64-step grid is about 33 MB.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from app.model.calibration.rough_vol.forward_variance import ForwardVarianceCurve
from app.model.volatility_models.rbergomi.volterra_gaussian import (
    DEFAULT_JITTER_REL,
    DEFAULT_MAX_JITTER_ATTEMPTS,
    GridConfig,
    JointGaussianDraw,
    SimulationGrid,
    VolterraFactor,
    build_simulation_grid,
    cholesky_factor,
    draw_joint_gaussian,
)

logger = logging.getLogger(__name__)


#: Hard lower bound on the Hurst index (spec 4.6).
H_MIN: float = 0.01
#: Hard upper bound on the Hurst index (spec 4.6).
H_MAX: float = 0.49
#: Hard upper bound on the vol-of-vol.
ETA_MAX: float = 5.0
#: Lower end of the *calibration working range* for ``eta`` - documented, not enforced.
ETA_WORKING_MIN: float = 0.05
#: Upper end of the calibration working range for ``eta``.
ETA_WORKING_MAX: float = 5.0
#: Hard bound on the spot/vol correlation (spec 4.6).
RHO_ABS_MAX: float = 0.999

#: ``xi0`` evaluated at the left node of each cell - spec 4.8 verbatim (default).
XI0_NODE: str = "node"
#: ``xi0`` replaced by its exact average over each cell - exact ``int xi0``.
XI0_CELL_AVERAGE: str = "cell_average"

#: How many ``xi0``-on-grid evaluations are kept in the module-level LRU.
DEFAULT_XI0_CACHE_MAXSIZE: int = 16
#: Clamp applied to Sobol' uniforms before the inverse normal CDF.
QMC_UNIFORM_EPS: float = 1e-12


class RBergomiSimulationError(RuntimeError):
    """Base class for failures of the xi0-curve rBergomi simulator."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _readonly(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as a float64 C-contiguous read-only view."""
    out = np.ascontiguousarray(array, dtype=float)
    out.setflags(write=False)
    return out


def sample_mean_stderr(
    values: np.ndarray, *, n_base_paths: int, antithetic: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo mean and **honest** standard error along axis 0.

    Under antithetic sampling the ``n_paths`` rows are *not* independent: rows
    ``k`` and ``k + n_base_paths`` are the two halves of one antithetic pair.
    The standard error is therefore estimated from the ``n_base_paths``
    independent **pair means**, not from the ``n_paths`` individual rows.  The
    point estimate is the ordinary mean over all rows, which is identical to the
    mean of the pair means in exact arithmetic and slightly more accurate in
    floating point.

    Parameters
    ----------
    values:
        ``(n_paths,)`` or ``(n_paths, ...)`` array of per-path estimator values.
    n_base_paths:
        Number of independently drawn paths (``n_paths // 2`` when antithetic).
    antithetic:
        Whether rows ``k`` and ``k + n_base_paths`` form antithetic pairs.

    Returns
    -------
    (mean, stderr):
        Arrays shaped like ``values.shape[1:]`` (0-d for a 1-d input).  The
        standard error is ``NaN`` when fewer than two independent samples exist.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("sample_mean_stderr needs at least one path along axis 0.")
    mean = array.mean(axis=0)
    if antithetic:
        base = int(n_base_paths)
        if 2 * base != array.shape[0]:
            raise ValueError(
                "Antithetic sampling expects n_paths == 2 * n_base_paths; got "
                f"n_paths={array.shape[0]}, n_base_paths={n_base_paths!r}."
            )
        reduced = 0.5 * (array[:base] + array[base:])
    else:
        reduced = array
    n_independent = int(reduced.shape[0])
    if n_independent < 2:
        stderr = np.full(mean.shape, np.nan, dtype=float)
    else:
        stderr = reduced.std(axis=0, ddof=1) / math.sqrt(n_independent)
    return np.asarray(mean, dtype=float), np.asarray(stderr, dtype=float)


# ---------------------------------------------------------------------------
# Parameters and configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RBergomiParams:
    """
    The three rough Bergomi parameters (spec 4.6 bounds enforced here).

    ``xi0`` is deliberately **absent**: it is the forward-variance curve, i.e.
    market data, and it is handed to the simulator separately so that a
    calibrator optimising over :class:`RBergomiParams` is *structurally* unable
    to move it.
    """

    H: float
    eta: float
    rho: float

    def __post_init__(self) -> None:
        H = float(self.H)
        eta = float(self.eta)
        rho = float(self.rho)
        if not np.isfinite(H) or not (H_MIN <= H <= H_MAX):
            raise ValueError(
                f"H must be finite and lie in [{H_MIN}, {H_MAX}]; got {self.H!r}."
            )
        if not np.isfinite(eta) or eta <= 0.0 or eta > ETA_MAX:
            raise ValueError(
                f"eta must be finite and lie in (0, {ETA_MAX}]; got {self.eta!r}."
            )
        if not np.isfinite(rho) or abs(rho) > RHO_ABS_MAX:
            raise ValueError(
                f"rho must be finite and lie in [-{RHO_ABS_MAX}, {RHO_ABS_MAX}]; "
                f"got {self.rho!r}."
            )
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "rho", rho)

    @property
    def rho_perp(self) -> float:
        """``sqrt(1 - rho^2)`` - the weight of the independent spot noise."""
        return float(math.sqrt(max(0.0, 1.0 - self.rho * self.rho)))

    @property
    def in_working_range(self) -> bool:
        """Whether ``eta`` sits in the documented calibration working range."""
        return ETA_WORKING_MIN < self.eta <= ETA_WORKING_MAX

    def to_dict(self) -> dict[str, Any]:
        return {"H": float(self.H), "eta": float(self.eta), "rho": float(self.rho)}


@dataclass(frozen=True)
class SimulationConfig:
    """
    Monte-Carlo knobs of :func:`simulate_rbergomi_xi_curve`.

    Attributes
    ----------
    n_paths:
        Total number of paths.  Must be even (and ``>= 2``) when ``antithetic``.
    antithetic:
        Draw ``n_paths // 2`` base paths and append their exact negation.
    seed:
        Seed of the pseudo-random (or Sobol' scrambling) stream.  ``None`` means
        non-reproducible; pass an int for bit-identical runs.
    qmc:
        Use a scrambled Sobol' sequence instead of pseudo-random normals.
    qmc_scramble:
        Owen scrambling of the Sobol' points.  ``False`` is available for
        debugging only: the un-scrambled sequence starts at the origin, which
        maps to an infinite normal quantile, and it carries no randomisation, so
        no error estimate of any kind can be formed from it.
    xi0_evaluation:
        :data:`XI0_CELL_AVERAGE` (default: exact ``int xi0``, keeps the spec-4.4
        reconstruction guarantee) or :data:`XI0_NODE` (spec 4.8 verbatim).
        See the module docstring for why the default deviates.
    martingale_correction:
        **Default OFF.**  When on, every node of every path is rescaled by the
        sample ratio ``mean(D(t) S_t e^{qt}) / S_0`` so that the discounted spot
        is a martingale *on the sample*.  The discrete scheme is already an
        exact martingale in expectation, so this removes only sampling noise -
        and it does so by dividing by an estimate built from the same sample,
        which introduces an ``O(1/n_paths)`` bias into every price and breaks
        the independence of the antithetic pairs that the standard errors rely
        on.  It trades one bias for another; enable it deliberately.
    use_factor_cache, use_xi0_cache:
        Whether to consult the module-level LRUs.
    jitter_rel, max_jitter_attempts:
        Passed through to :func:`volterra_gaussian.cholesky_factor`.
    grid_config:
        Grid policy used when :func:`simulate_rbergomi_xi_curve` has to build
        the grid itself.
    """

    n_paths: int = 20_000
    antithetic: bool = True
    seed: int | None = None
    qmc: bool = False
    qmc_scramble: bool = True
    xi0_evaluation: str = XI0_CELL_AVERAGE
    martingale_correction: bool = False
    use_factor_cache: bool = True
    use_xi0_cache: bool = True
    jitter_rel: float = DEFAULT_JITTER_REL
    max_jitter_attempts: int = DEFAULT_MAX_JITTER_ATTEMPTS
    grid_config: GridConfig | None = None

    def __post_init__(self) -> None:
        n_paths = int(self.n_paths)
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1; got {self.n_paths!r}.")
        if self.antithetic and (n_paths < 2 or n_paths % 2 != 0):
            raise ValueError(
                f"Antithetic sampling needs an even n_paths >= 2; got {self.n_paths!r}."
            )
        if self.xi0_evaluation not in (XI0_NODE, XI0_CELL_AVERAGE):
            raise ValueError(
                f"xi0_evaluation must be {XI0_NODE!r} or {XI0_CELL_AVERAGE!r}; "
                f"got {self.xi0_evaluation!r}."
            )
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "n_paths", n_paths)

    @property
    def n_base_paths(self) -> int:
        """Number of independently drawn paths."""
        return self.n_paths // 2 if self.antithetic else self.n_paths


# ---------------------------------------------------------------------------
# xi0 on the grid (cached)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Xi0OnGrid:
    """
    The forward-variance curve sampled on one simulation grid.

    Attributes
    ----------
    nodes:
        ``(n+1,)`` - ``xi0(t_i)`` at every grid node, ``i = 0..n``.  This is the
        spec-4.6 value of the *variance process* at that instant, whatever
        ``mode`` is.
    steps:
        ``(n,)`` - the level actually used by the Euler step over
        ``[t_i, t_{i+1}]``.  Equal to ``nodes[:-1]`` in :data:`XI0_NODE` mode and
        to the exact cell average in :data:`XI0_CELL_AVERAGE` mode.
    quadrature_error:
        ``(n+1,)`` - ``sum_{i<k} steps[i]*dt[i] - int_0^{t_k} xi0``, i.e. the
        exact, deterministic amount by which the scheme's integrated forward
        variance misses the curve's own integral.  Identically zero (to
        round-off) in cell-average mode.
    mode, grid_hash, curve_fingerprint:
        Provenance / cache key components.
    """

    nodes: np.ndarray
    steps: np.ndarray
    quadrature_error: np.ndarray
    mode: str
    grid_hash: str
    curve_fingerprint: str

    def diagnostics(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "mode": str(self.mode),
            "grid_hash": str(self.grid_hash),
            "curve_fingerprint": str(self.curve_fingerprint),
            "xi0_min": float(np.min(self.nodes)),
            "xi0_max": float(np.max(self.nodes)),
            "max_abs_quadrature_error": float(np.max(np.abs(self.quadrature_error))),
        }


_XI0_CACHE: "OrderedDict[tuple[str, str, str], Xi0OnGrid]" = OrderedDict()
_XI0_CACHE_MAXSIZE: int = DEFAULT_XI0_CACHE_MAXSIZE
_XI0_CACHE_HITS: int = 0
_XI0_CACHE_MISSES: int = 0


def clear_xi0_cache() -> None:
    """Empty the ``xi0``-on-grid cache and reset its counters."""
    global _XI0_CACHE_HITS, _XI0_CACHE_MISSES
    _XI0_CACHE.clear()
    _XI0_CACHE_HITS = 0
    _XI0_CACHE_MISSES = 0


def set_xi0_cache_maxsize(size: int) -> None:
    """Resize the ``xi0``-on-grid LRU, evicting least-recently-used entries."""
    global _XI0_CACHE_MAXSIZE
    value = int(size)
    if value < 1:
        raise ValueError(f"Cache size must be >= 1; got {size!r}.")
    _XI0_CACHE_MAXSIZE = value
    while len(_XI0_CACHE) > _XI0_CACHE_MAXSIZE:
        _XI0_CACHE.popitem(last=False)


def xi0_cache_info() -> dict[str, int]:
    """Hit / miss counters and occupancy of the ``xi0``-on-grid LRU."""
    return {
        "hits": int(_XI0_CACHE_HITS),
        "misses": int(_XI0_CACHE_MISSES),
        "size": len(_XI0_CACHE),
        "maxsize": int(_XI0_CACHE_MAXSIZE),
    }


def curve_fingerprint(curve: ForwardVarianceCurve) -> str:
    """
    Content hash of a :class:`ForwardVarianceCurve`.

    ``(method, T_knots, levels, V_knots)`` determines the curve completely for
    both build methods: the piecewise-constant curve is generated by
    ``(T_knots, levels)`` and the monotone-PCHIP variant by ``(T_knots,
    V_knots)`` through the fixed anchor ``(0, 0)``.  Hashing the content rather
    than the object identity keeps the cache correct across rebuilt-but-equal
    curves and safe against ``id`` reuse.
    """
    if not isinstance(curve, ForwardVarianceCurve):
        raise TypeError(
            f"xi_curve must be a ForwardVarianceCurve; got {type(curve).__name__}."
        )
    digest = hashlib.sha256()
    digest.update(b"rbergomi-xi0-curve-v1")
    digest.update(str(curve.method).encode("utf-8"))
    for block in (curve.T_knots, curve.levels, curve.V_knots):
        digest.update(np.ascontiguousarray(block, dtype="<f8").tobytes())
    return digest.hexdigest()


def evaluate_xi0_on_grid(
    *,
    curve: ForwardVarianceCurve,
    grid: SimulationGrid,
    mode: str = XI0_NODE,
    use_cache: bool = True,
) -> Xi0OnGrid:
    """
    Sample ``xi0`` on ``grid``, memoised by ``(curve, grid, mode)``.

    Raises
    ------
    RBergomiSimulationError
        When the curve returns a non-finite or non-positive forward variance on
        the grid - a variance path must stay strictly positive for
        ``sqrt(V)`` to exist, and the curve builder already floors ``xi0`` at
        ``eps_xi``, so this can only mean the curve was constructed by hand.
    """
    global _XI0_CACHE_HITS, _XI0_CACHE_MISSES

    if mode not in (XI0_NODE, XI0_CELL_AVERAGE):
        raise ValueError(
            f"mode must be {XI0_NODE!r} or {XI0_CELL_AVERAGE!r}; got {mode!r}."
        )
    if not isinstance(grid, SimulationGrid):
        raise TypeError(f"grid must be a SimulationGrid; got {type(grid).__name__}.")
    fingerprint = curve_fingerprint(curve)
    key = (fingerprint, grid.grid_hash, str(mode))
    if use_cache:
        cached = _XI0_CACHE.get(key)
        if cached is not None:
            _XI0_CACHE.move_to_end(key)
            _XI0_CACHE_HITS += 1
            return cached
        _XI0_CACHE_MISSES += 1

    t = np.asarray(grid.t, dtype=float)
    dt = np.asarray(grid.dt, dtype=float)
    nodes = np.asarray(curve.xi0(t), dtype=float)
    integrated = np.asarray(curve.integrated(t), dtype=float)
    if mode == XI0_NODE:
        steps = nodes[:-1].copy()
    else:
        # Exact cell average: (1/dt_i) * int_{t_i}^{t_{i+1}} xi0(u) du.
        steps = np.diff(integrated) / dt

    if not np.all(np.isfinite(nodes)) or np.any(nodes <= 0.0):
        raise RBergomiSimulationError(
            "La courbe de variance forward renvoie une valeur non finie ou "
            "negative sur la grille de simulation; V_t ne peut pas rester "
            "strictement positif."
        )
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise RBergomiSimulationError(
            "La variance forward moyennee par cellule est non finie ou negative "
            f"(mode={mode!r}); V_t ne peut pas rester strictement positif."
        )

    accumulated = np.empty(t.size, dtype=float)
    accumulated[0] = 0.0
    np.cumsum(steps * dt, out=accumulated[1:])
    quadrature_error = accumulated - integrated

    result = Xi0OnGrid(
        nodes=_readonly(nodes),
        steps=_readonly(steps),
        quadrature_error=_readonly(quadrature_error),
        mode=str(mode),
        grid_hash=grid.grid_hash,
        curve_fingerprint=fingerprint,
    )
    if use_cache:
        _XI0_CACHE[key] = result
        _XI0_CACHE.move_to_end(key)
        while len(_XI0_CACHE) > _XI0_CACHE_MAXSIZE:
            _XI0_CACHE.popitem(last=False)
    return result


# ---------------------------------------------------------------------------
# Short-rate resolution on a NON-UNIFORM grid
# ---------------------------------------------------------------------------
def resolve_step_rates(r: Any, grid: SimulationGrid) -> np.ndarray:
    """
    Per-step risk-free rates ``r_i`` on ``[t_i, t_{i+1}]``, ``i = 0..n-1``.

    Accepted forms
    --------------
    ``None``
        All-zero rates (undiscounted simulation).
    scalar
        A flat rate, broadcast to every step.
    array-like of length ``n``
        Per-step rates, used verbatim.  This is the form the drift needs on a
        non-uniform grid: ``exp(-sum r_i dt_i)`` is the discount factor, so the
        ``dt_i`` weighting is the caller's actual step lengths, never ``T/n``.
    yield-curve object
        Anything exposing ``discount_factor(T)`` (preferred) or ``zero_rate(T)``
        - e.g. the curve returned by
        :func:`app.model.yieldcurve.service.get_active_curve`.  The step rates
        are the **exact forward rates** implied by the curve's own discount
        factors, ``r_i = (ln D(t_i) - ln D(t_{i+1})) / dt_i``, so that
        ``exp(-sum_{i<k} r_i dt_i) == D(t_k)`` to machine precision at every
        node.  Reading ``zero_rate`` instead of ``discount_factor`` uses
        ``D(t) = exp(-z(t) t)``.

    An array of length ``n + 1`` is **rejected**: it is ambiguous (node zero
    rates? node instantaneous rates?) and guessing would silently misprice.
    """
    if not isinstance(grid, SimulationGrid):
        raise TypeError(f"grid must be a SimulationGrid; got {type(grid).__name__}.")
    n = grid.n
    dt = np.asarray(grid.dt, dtype=float)
    t = np.asarray(grid.t, dtype=float)

    if r is None:
        return np.zeros(n, dtype=float)

    if hasattr(r, "discount_factor") or hasattr(r, "zero_rate"):
        if hasattr(r, "discount_factor"):
            discounts = np.asarray(
                [float(r.discount_factor(float(x))) for x in t], dtype=float
            )
        else:
            zeros = np.asarray([float(r.zero_rate(float(x))) for x in t], dtype=float)
            discounts = np.exp(-zeros * t)
        if not np.all(np.isfinite(discounts)) or np.any(discounts <= 0.0):
            raise ValueError(
                "The yield curve returned a non-positive or non-finite discount "
                "factor on the simulation grid; the implied forward rates would "
                "be undefined."
            )
        return -np.diff(np.log(discounts)) / dt

    if np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0):
        value = float(r)
        if not np.isfinite(value):
            raise ValueError(f"r must be finite; got {r!r}.")
        return np.full(n, value, dtype=float)

    array = np.asarray(r, dtype=float).ravel()
    if array.size == n:
        if not np.all(np.isfinite(array)):
            raise ValueError("Per-step rates must all be finite.")
        return array.astype(float, copy=True)
    raise ValueError(
        f"r must be a scalar, a yield curve, or an array of exactly n={n} "
        f"per-step rates (one per grid cell); got an array of length "
        f"{array.size}. An array of n+1 node values is ambiguous and is "
        "rejected on purpose."
    )


# ---------------------------------------------------------------------------
# QMC draw
# ---------------------------------------------------------------------------
def _draw_joint_qmc(
    *,
    factor: VolterraFactor,
    n_paths: int,
    seed: int | None,
    antithetic: bool,
    scramble: bool,
) -> JointGaussianDraw:
    """
    Scrambled-Sobol' counterpart of :func:`volterra_gaussian.draw_joint_gaussian`.

    The ``3n`` dimensions are laid out exactly like the pseudo-random draw: the
    first ``2n`` feed the joint ``(W~, dB)`` block through ``L``, the last ``n``
    feed the independent ``dB_perp`` block.  Uniforms are clamped away from the
    open endpoints before :func:`scipy.special.ndtri`, because an
    un-scrambled Sobol' sequence starts exactly at the origin.
    """
    n = int(factor.n)
    count = int(n_paths)
    n_base = count // 2 if antithetic else count
    dimension = 3 * n

    if not scramble:
        logger.warning(
            "rBergomi QMC: scramble=False produces a deterministic Sobol' set "
            "with no randomisation; no error estimate of any kind is available "
            "and the first point is the (clamped) origin. Debugging only."
        )
    if n_base & (n_base - 1) != 0:
        logger.warning(
            "rBergomi QMC: n_base_paths=%d is not a power of two; the balance "
            "properties of the Sobol' sequence are lost.",
            n_base,
        )

    engine = qmc.Sobol(d=dimension, scramble=bool(scramble), seed=seed)
    uniforms = np.asarray(engine.random(n_base), dtype=float)
    np.clip(uniforms, QMC_UNIFORM_EPS, 1.0 - QMC_UNIFORM_EPS, out=uniforms)
    normals = ndtri(uniforms)

    joint = normals[:, : 2 * n] @ np.asarray(factor.L).T
    perp = normals[:, 2 * n :] * np.sqrt(np.asarray(factor.grid.dt, dtype=float))
    if antithetic:
        joint = np.concatenate([joint, -joint], axis=0)
        perp = np.concatenate([perp, -perp], axis=0)

    return JointGaussianDraw(
        W_tilde=np.ascontiguousarray(joint[:, :n]),
        dB=np.ascontiguousarray(joint[:, n:]),
        dB_perp=np.ascontiguousarray(perp),
        n_paths=count,
        n_base_paths=int(n_base),
        antithetic=bool(antithetic),
        H=float(factor.H),
        grid=factor.grid,
    )


# ---------------------------------------------------------------------------
# Path set
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RBergomiPathSet:
    """
    One rough Bergomi Monte-Carlo path set, priced at **every** quoted maturity.

    All the heavy arrays are indexed ``[path, grid node]`` with ``n + 1``
    columns, column ``0`` being ``t = 0``.

    Attributes
    ----------
    grid, params, S0, q:
        Provenance and the deterministic inputs.
    r_steps:
        ``(n,)`` per-cell risk-free rates actually used by the drift.
    discount_factors:
        ``(n+1,)`` ``D(t_k) = exp(-sum_{i<k} r_i dt_i)`` - built from the very
        same per-step rates and per-step ``dt`` as the drift, so put-call parity
        is exact on the sample.
    forward_factors:
        ``(n+1,)`` ``exp(sum_{i<k} (r_i - q) dt_i)``; ``S0 * forward_factors[k]``
        is the model forward at ``t_k``.
    xi0_on_grid:
        The :class:`Xi0OnGrid` used, including its exact quadrature error.
    variance:
        ``(n_paths, n+1)`` ``V_{t_i} = xi0(t_i) exp(eta W~_{t_i} - eta^2 t^{2H}/2)``
        - spec 4.6 verbatim at the nodes, whatever ``xi0_evaluation`` is.
    integrated_variance:
        ``(n_paths, n+1)`` ``sum_{i<k} V^step_i dt_i`` - the left-point
        approximation of ``int_0^{t_k} V du`` that the log-Euler scheme actually
        uses, and the effective variance of the conditional estimator.
    vol_dB_integral:
        ``(n_paths, n+1)`` ``sum_{i<k} sqrt(V^step_i) dB_i`` against the **same**
        ``dB`` that drives ``W~``.  This is what makes the Romano-Touzi
        conditional estimator exact.
    log_spot:
        ``(n_paths, n+1)`` ``ln S_{t_k}``.
    draw:
        The underlying :class:`JointGaussianDraw` (``W~``, ``dB``, ``dB_perp``),
        retained so that a caller can reuse the *same* random numbers across
        parameter sets (common random numbers) and so that the shared-driver
        construction can be validated statistically.
    """

    grid: SimulationGrid
    params: RBergomiParams
    S0: float
    q: float
    r_steps: np.ndarray
    discount_factors: np.ndarray
    forward_factors: np.ndarray
    xi0_on_grid: Xi0OnGrid
    variance: np.ndarray
    integrated_variance: np.ndarray
    vol_dB_integral: np.ndarray
    log_spot: np.ndarray
    draw: JointGaussianDraw
    config: SimulationConfig
    martingale_ratio: np.ndarray | None = field(default=None)

    # -- basic accessors ---------------------------------------------------
    @property
    def n_paths(self) -> int:
        return int(self.log_spot.shape[0])

    @property
    def n_base_paths(self) -> int:
        return int(self.draw.n_base_paths)

    @property
    def antithetic(self) -> bool:
        return bool(self.draw.antithetic)

    @property
    def maturities(self) -> tuple[float, ...]:
        """The quoted maturities carried by the grid, sorted."""
        return self.grid.quoted_maturities

    @property
    def xi0_quadrature_error(self) -> np.ndarray:
        """``(n+1,)`` deterministic gap between ``sum xi0 dt`` and ``int xi0``."""
        return self.xi0_on_grid.quadrature_error

    def index_of(self, T: float) -> int:
        """Grid index of a quoted maturity (exact float equality, no snapping)."""
        return self.grid.index_of(T)

    def resolve_maturities(
        self, maturities: Any = None
    ) -> tuple[tuple[float, ...], np.ndarray]:
        """Normalise a maturity selection to ``(values, grid indices)``."""
        if maturities is None:
            values = tuple(float(x) for x in self.grid.quoted_maturities)
        else:
            values = tuple(
                float(x) for x in np.atleast_1d(np.asarray(maturities, dtype=float))
            )
        indices = np.asarray([self.index_of(T) for T in values], dtype=int)
        return values, indices

    # -- per-maturity views ------------------------------------------------
    def spot(self, T: float) -> np.ndarray:
        """``S_T`` on every path (fresh array)."""
        return np.exp(self.log_spot[:, self.index_of(T)])

    def variance_at(self, T: float) -> np.ndarray:
        """``V_T`` on every path."""
        return np.array(self.variance[:, self.index_of(T)], dtype=float)

    def integrated_variance_at(self, T: float) -> np.ndarray:
        """``int_0^T V du`` (left-point) on every path."""
        return np.array(self.integrated_variance[:, self.index_of(T)], dtype=float)

    def realised_variance(self, T: float) -> np.ndarray:
        """``(1/T) int_0^T V du`` on every path - the variance-swap payoff."""
        return self.integrated_variance_at(T) / float(T)

    def vol_dB_integral_at(self, T: float) -> np.ndarray:
        """``int_0^T sqrt(V) dB`` on every path."""
        return np.array(self.vol_dB_integral[:, self.index_of(T)], dtype=float)

    def discount_factor(self, T: float) -> float:
        """``D(T)`` implied by the per-step rates."""
        return float(self.discount_factors[self.index_of(T)])

    def forward_factor(self, T: float) -> float:
        """``exp(int_0^T (r - q) du)``."""
        return float(self.forward_factors[self.index_of(T)])

    def model_forward(self, T: float) -> float:
        """``E[S_T] = S0 exp(int_0^T (r - q) du)`` - the theoretical forward."""
        return float(self.S0) * self.forward_factor(T)

    def volterra(self, T: float) -> np.ndarray:
        """``W~_T`` on every path (zeros at ``T = 0``)."""
        k = self.index_of(T)
        if k == 0:
            return np.zeros(self.n_paths, dtype=float)
        return np.array(self.draw.W_tilde[:, k - 1], dtype=float)

    def spot_brownian(self, T: float) -> np.ndarray:
        """
        ``W^S_T = rho B_T + sqrt(1-rho^2) B_perp_T`` on every path.

        Built by summing the *same* increments that drive ``W~`` - this is the
        quantity whose cross moment with ``W~_T`` validates the shared-driver
        construction.
        """
        k = self.index_of(T)
        if k == 0:
            return np.zeros(self.n_paths, dtype=float)
        b = self.draw.dB[:, :k].sum(axis=1)
        b_perp = self.draw.dB_perp[:, :k].sum(axis=1)
        return self.params.rho * b + self.params.rho_perp * b_perp

    def conditional_forward(self, T: float) -> np.ndarray:
        """
        ``E[S_T | (V_u), (B_u)] = S0 e^{int(r-q)} exp(rho M_T - rho^2 IV_T / 2)``.

        With ``M_T = int_0^T sqrt(V) dB`` and ``IV_T = int_0^T V du``.  This is
        the per-path conditional mean used by the mixed estimator; its own mean
        is the model forward, because ``exp(rho M - rho^2 IV / 2)`` is an exact
        discrete exponential martingale.
        """
        rho = self.params.rho
        m = self.vol_dB_integral_at(T)
        iv = self.integrated_variance_at(T)
        forward = self.model_forward(T)
        if self.martingale_ratio is not None:
            # The correction rescales `log_spot`, i.e. the PLAIN estimator's
            # paths. `integrated_variance` and `vol_dB_integral` are built before
            # it and are never touched, so without this the conditional estimator
            # would silently ignore the correction entirely and the two
            # estimators would price against two different forwards -- the very
            # disagreement the conditional-vs-plain test is supposed to detect.
            forward = forward / float(self.martingale_ratio[self.index_of(T)])
        return forward * np.exp(rho * m - 0.5 * rho * rho * iv)

    def conditional_total_variance(self, T: float) -> np.ndarray:
        """``(1 - rho^2) int_0^T V du`` - the conditional Black-Scholes variance."""
        rho = self.params.rho
        return (1.0 - rho * rho) * self.integrated_variance_at(T)

    # -- statistics --------------------------------------------------------
    def mean_stderr(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and antithetic-aware standard error along the path axis."""
        return sample_mean_stderr(
            values, n_base_paths=self.n_base_paths, antithetic=self.antithetic
        )

    def diagnostics(self) -> dict[str, Any]:
        """JSON-safe summary of the simulation."""
        return {
            "S0": float(self.S0),
            "q": float(self.q),
            "params": self.params.to_dict(),
            "n_paths": int(self.n_paths),
            "n_base_paths": int(self.n_base_paths),
            "antithetic": bool(self.antithetic),
            "qmc": bool(self.config.qmc),
            "seed": None if self.config.seed is None else int(self.config.seed),
            "martingale_correction": bool(self.config.martingale_correction),
            "xi0": self.xi0_on_grid.diagnostics(),
            "grid": self.grid.diagnostics(),
            "r_min": float(np.min(self.r_steps)),
            "r_max": float(np.max(self.r_steps)),
            "discount_factor_at_T_max": float(self.discount_factors[-1]),
            "max_abs_xi0_quadrature_error": float(
                np.max(np.abs(self.xi0_quadrature_error))
            ),
            "martingale_ratio_at_T_max": (
                None
                if self.martingale_ratio is None
                else float(self.martingale_ratio[-1])
            ),
        }


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------
def simulate_rbergomi_xi_curve(
    *,
    S0: float,
    xi_curve: ForwardVarianceCurve,
    params: RBergomiParams,
    maturities: Any = None,
    grid: SimulationGrid | None = None,
    r: Any = 0.0,
    q: float = 0.0,
    config: SimulationConfig | None = None,
) -> RBergomiPathSet:
    """
    Simulate rough Bergomi on a forward-variance curve (spec 4.8).

    Fully vectorised: there is **no Python loop over paths and none over grid
    steps** - the log-Euler recursion is closed in a single cumulative sum
    because the scheme is exact in ``S`` given ``V`` on the grid.

    Parameters
    ----------
    S0:
        Spot, strictly positive.
    xi_curve:
        The frozen :class:`ForwardVarianceCurve` of spec 4.4.  It is read, never
        written; a calibrator optimising ``(H, eta, rho)`` cannot touch it.
    params:
        :class:`RBergomiParams`.
    maturities:
        Quoted maturities to guarantee on the grid.  Defaults to
        ``grid.quoted_maturities`` when a grid is supplied, and to
        ``xi_curve.T_knots`` otherwise.
    grid:
        A pre-built :class:`SimulationGrid` (reuse it across calls so the
        Cholesky cache hits).  Built from ``maturities`` when omitted.
    r:
        Risk-free rate: scalar, per-step array of length ``n``, or a yield
        curve.  See :func:`resolve_step_rates`.
    q:
        Continuous dividend yield (scalar, per spec 4.6).
    config:
        :class:`SimulationConfig`.

    Returns
    -------
    RBergomiPathSet

    Raises
    ------
    ValueError
        On invalid inputs, or when a requested maturity is not on the grid.
    RBergomiSimulationError
        When the forward-variance curve is not strictly positive on the grid.
    """
    cfg = config if config is not None else SimulationConfig()
    if not isinstance(params, RBergomiParams):
        raise TypeError(
            f"params must be an RBergomiParams; got {type(params).__name__}."
        )
    if not isinstance(xi_curve, ForwardVarianceCurve):
        raise TypeError(
            f"xi_curve must be a ForwardVarianceCurve; got {type(xi_curve).__name__}."
        )
    spot0 = float(S0)
    if not np.isfinite(spot0) or spot0 <= 0.0:
        raise ValueError(f"S0 must be finite and strictly positive; got {S0!r}.")
    div = float(q)
    if not np.isfinite(div):
        raise ValueError(f"q must be finite; got {q!r}.")

    # -- grid ---------------------------------------------------------------
    if grid is None:
        wanted = xi_curve.T_knots if maturities is None else maturities
        grid = build_simulation_grid(maturities=wanted, config=cfg.grid_config)
    elif maturities is not None:
        for T in np.atleast_1d(np.asarray(maturities, dtype=float)):
            grid.index_of(float(T))  # raises KeyError if not exactly on the grid

    n = grid.n
    t = np.asarray(grid.t, dtype=float)
    dt = np.asarray(grid.dt, dtype=float)

    # -- deterministic term structure ---------------------------------------
    r_steps = resolve_step_rates(r, grid)
    minus_int_r = np.empty(n + 1, dtype=float)
    minus_int_r[0] = 0.0
    np.cumsum(-r_steps * dt, out=minus_int_r[1:])
    discount_factors = np.exp(minus_int_r)
    drift_cum = np.empty(n + 1, dtype=float)
    drift_cum[0] = 0.0
    np.cumsum((r_steps - div) * dt, out=drift_cum[1:])
    forward_factors = np.exp(drift_cum)

    xi0_grid = evaluate_xi0_on_grid(
        curve=xi_curve,
        grid=grid,
        mode=cfg.xi0_evaluation,
        use_cache=cfg.use_xi0_cache,
    )

    # -- exact joint draw ----------------------------------------------------
    factor = cholesky_factor(
        H=params.H,
        grid=grid,
        jitter_rel=cfg.jitter_rel,
        max_jitter_attempts=cfg.max_jitter_attempts,
        use_cache=cfg.use_factor_cache,
    )
    if cfg.qmc:
        draw = _draw_joint_qmc(
            factor=factor,
            n_paths=cfg.n_paths,
            seed=cfg.seed,
            antithetic=cfg.antithetic,
            scramble=cfg.qmc_scramble,
        )
    else:
        draw = draw_joint_gaussian(
            factor=factor,
            n_paths=cfg.n_paths,
            seed=cfg.seed,
            antithetic=cfg.antithetic,
        )
    n_paths = draw.n_paths

    # -- variance path (spec 4.6, exact at the nodes) -------------------------
    # W~_{t_0} = 0 by definition; volterra_gaussian returns t_1..t_n.
    volterra = np.zeros((n_paths, n + 1), dtype=float)
    volterra[:, 1:] = draw.W_tilde
    # The compensator uses t_i^{2H} -- the exact variance of W~_{t_i} -- so that
    # E[V_t] = xi0(t) EXACTLY, which is what pins the -0.5*eta^2*t^{2H} term.
    compensator = 0.5 * params.eta * params.eta * t ** (2.0 * params.H)
    exponential = np.exp(params.eta * volterra - compensator[None, :])
    variance = np.asarray(xi0_grid.nodes)[None, :] * exponential

    if cfg.xi0_evaluation == XI0_NODE:
        # Same xi0 value AND same exponential factor: the step variance is
        # literally the node variance, so share the memory instead of copying.
        variance_step = variance[:, :n]
    else:
        variance_step = np.asarray(xi0_grid.steps)[None, :] * exponential[:, :n]

    vol_step = np.sqrt(variance_step)

    # -- integrals with the REAL per-step dt ---------------------------------
    integrated_variance = np.zeros((n_paths, n + 1), dtype=float)
    np.cumsum(variance_step * dt[None, :], axis=1, out=integrated_variance[:, 1:])

    vol_dB_integral = np.zeros((n_paths, n + 1), dtype=float)
    np.cumsum(vol_step * draw.dB, axis=1, out=vol_dB_integral[:, 1:])

    vol_dB_perp_integral = np.zeros((n_paths, n + 1), dtype=float)
    np.cumsum(vol_step * draw.dB_perp, axis=1, out=vol_dB_perp_integral[:, 1:])

    # -- log-Euler, closed in one cumulative sum -----------------------------
    #   ln S_{i+1} = ln S_i + (r_i - q) dt_i - 0.5 V_i dt_i
    #                + sqrt(V_i) (rho dB_i + sqrt(1-rho^2) dB^perp_i)
    log_spot = (
        math.log(spot0)
        + drift_cum[None, :]
        - 0.5 * integrated_variance
        + params.rho * vol_dB_integral
        + params.rho_perp * vol_dB_perp_integral
    )

    martingale_ratio: np.ndarray | None = None
    if cfg.martingale_correction:
        # mean over paths of D(t) S_t e^{qt} / S0 = mean of exp(ln S - ln S0 - int(r-q)).
        deflated = np.exp(log_spot - math.log(spot0) - drift_cum[None, :])
        martingale_ratio = np.asarray(deflated.mean(axis=0), dtype=float)
        if np.any(martingale_ratio <= 0.0) or not np.all(np.isfinite(martingale_ratio)):
            raise RBergomiSimulationError(
                "La correction de martingale est impossible: le ratio "
                "E[D(t) S_t e^{qt}]/S0 estime sur l'echantillon n'est pas fini "
                "et strictement positif."
            )
        logger.warning(
            "rBergomi: martingale correction ENABLED. Every node is rescaled by "
            "the sample ratio (max deviation %.3e at T=%.6g). This removes the "
            "sample's martingale defect but injects an O(1/n_paths) bias into "
            "every price and correlates the antithetic pairs.",
            float(np.max(np.abs(martingale_ratio - 1.0))),
            float(t[int(np.argmax(np.abs(martingale_ratio - 1.0)))]),
        )
        log_spot = log_spot - np.log(martingale_ratio)[None, :]
        martingale_ratio = _readonly(martingale_ratio)

    # The path set is meant to be SHARED (common random numbers across an
    # optimizer's iterations, across strikes and across maturities), so the
    # heavy arrays are frozen: a consumer cannot corrupt someone else's sample.
    for block in (variance, integrated_variance, vol_dB_integral, log_spot):
        block.setflags(write=False)

    return RBergomiPathSet(
        grid=grid,
        params=params,
        S0=spot0,
        q=div,
        r_steps=_readonly(r_steps),
        discount_factors=_readonly(discount_factors),
        forward_factors=_readonly(forward_factors),
        xi0_on_grid=xi0_grid,
        variance=variance,
        integrated_variance=integrated_variance,
        vol_dB_integral=vol_dB_integral,
        log_spot=log_spot,
        draw=draw,
        config=cfg,
        martingale_ratio=martingale_ratio,
    )


__all__ = [
    "DEFAULT_XI0_CACHE_MAXSIZE",
    "ETA_MAX",
    "ETA_WORKING_MAX",
    "ETA_WORKING_MIN",
    "H_MAX",
    "H_MIN",
    "QMC_UNIFORM_EPS",
    "RHO_ABS_MAX",
    "XI0_CELL_AVERAGE",
    "XI0_NODE",
    "RBergomiParams",
    "RBergomiPathSet",
    "RBergomiSimulationError",
    "SimulationConfig",
    "Xi0OnGrid",
    "clear_xi0_cache",
    "curve_fingerprint",
    "evaluate_xi0_on_grid",
    "resolve_step_rates",
    "sample_mean_stderr",
    "set_xi0_cache_maxsize",
    "simulate_rbergomi_xi_curve",
    "xi0_cache_info",
]
