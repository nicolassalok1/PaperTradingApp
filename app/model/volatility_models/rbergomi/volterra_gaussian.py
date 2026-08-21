"""
Exact joint Gaussian construction of the rough Bergomi driver pair
``(W~_{t_1..t_n}, dB_1..dB_n)`` on a fixed simulation grid.

CONVENTION - WHICH FRACTIONAL PROCESS THIS MODULE PRODUCES (recorded decision)
=============================================================================
``W~`` is the **Riemann-Liouville Volterra driver** of rough Bergomi,

    W~_t = sqrt(2H) * int_0^t (t - s)^{H - 1/2} dB_s ,     Var(W~_t) = t^{2H},

driven by the SAME standard Brownian motion ``B`` whose increments this module
also returns.  It is **NOT** standard (Mandelbrot-Van Ness) fractional Brownian
motion.  ``app/model/volatility_models/rbergomi/fbm.py`` (Davies-Harte) provides
standard fBm for the roughness *diagnostics*; the two processes are different
Gaussian processes that share only the marginal variance ``t^{2H}``:

    standard fBm : Cov(B^H_u, B^H_v) = 0.5 * (u^{2H} + v^{2H} - |u-v|^{2H})
    RL Volterra  : Cov(W~_u, W~_v)   = 2H * int_0^{min(u,v)} (u-s)^{H-1/2}
                                                             (v-s)^{H-1/2} ds

The **pricing path uses THIS module**, precisely because the spot/vol
correlation is realised through the shared driver:

    dW^S = rho * dB + sqrt(1 - rho^2) * dB^perp ,

so ``E[W^S_t * W~_t] = rho * sqrt(2H)/(H+1/2) * t^{H+1/2}`` holds *by
construction*, path by path, with no discretisation error in the correlation.
**NEVER correlate fBm values with W~ values directly**: that would impose a
correlation on the marginals instead of on the driving noise, and the resulting
joint law is not the rough Bergomi law.

COVARIANCE - CLOSED FORMS IMPLEMENTED HERE
==========================================
On the grid ``0 = t_0 < t_1 < ... < t_n`` the vector
``(W~_{t_1}, ..., W~_{t_n}, dB_1, ..., dB_n)`` with ``dB_i = B_{t_i} - B_{t_{i-1}}``
is centred Gaussian with the 2n x 2n covariance ``Sigma = [[A, C], [C^T, D]]``:

1. ``A_ij = Cov(W~_{t_i}, W~_{t_j})``.  With ``u = min(t_i, t_j)``,
   ``v = max(t_i, t_j)``, the substitution ``s = u*sigma`` in the defining
   integral plus Euler's integral representation of the Gauss hypergeometric
   function give the exact form

       Cov(W~_u, W~_v) = 2H/(H+1/2) * u^{H+1/2} * v^{H-1/2}
                                    * 2F1(1/2 - H, 1; H + 3/2; u/v).

   This arrangement is numerically STABLE as ``u -> v`` (the argument ``u/v``
   tends to 1 from below, where ``2F1`` converges because
   ``c - a - b = 2H > 0``), unlike the naive ``(v-u)`` arrangement whose
   ``(v-u)^{H-1/2}`` factor blows up.  At ``u = v = t`` Gauss's theorem gives
   ``2F1(1/2-H, 1; H+3/2; 1) = (H+1/2)/(2H)``, so the diagonal collapses to
   ``t^{2H}`` exactly - verified to <= 1e-15 relative, asserted to 1e-10 by the
   unit tests.  ``A`` is symmetric *by construction*: ``min`` and ``max`` are
   symmetric functions of the index pair, so the elementwise expression is bit
   for bit symmetric.

2. ``C_ij = Cov(W~_{t_i}, dB_j)``.  Ito isometry against the deterministic
   integrand ``sqrt(2H) (t-s)^{H-1/2}`` over ``[t_{j-1}, min(t_j, t_i)]`` gives,
   for ``a = t_{j-1} < min(b, t) = min(t_j, t_i)``,

       Cov(W~_t, B_b - B_a) = sqrt(2H)/(H+1/2)
                              * [ (t-a)^{H+1/2} - (t-min(b,t))^{H+1/2} ],

   and 0 when ``a >= t``.  On an increasing grid this makes ``C`` lower
   triangular (``C_ij = 0`` for ``j > i``): the increment ``dB_j`` lives in the
   future of ``t_i`` and the Volterra kernel is causal.  The difference of two
   nearly equal powers is evaluated in the cancellation-free form

       (t-a)^{H+1/2} * ( -expm1( (H+1/2) * log1p( -(b-a)/(t-a) ) ) ),

   which is exact to full precision even when ``(b-a) << (t-a)`` (fine grids at
   long lags), where the naive subtraction loses digits.

3. ``D_ij = Cov(dB_i, dB_j) = delta_ij * dt_i`` - the Brownian increments are
   independent with variance equal to the step length.

FACTORISATION, JITTER AND CACHING
=================================
``Sigma`` is Cholesky-factored ONCE per ``(H, grid)`` and every path is then
produced as ``L @ Z`` with ``Z ~ N(0, I_{2n})``.  Complexity: ``O(n^3)`` once,
``O(n^2 * paths)`` per draw.  The factor is memoised in a bounded LRU keyed by
``(H, grid hash, jitter policy)`` so it is reused across paths, strikes,
maturities and the common-random-number draws of an optimizer.  The jitter
policy is part of the key because it changes ``L``; the spec's ``(H, grid-hash)``
key is a strict prefix of it.

``Sigma`` is positive semi-definite but can be *singular*: at ``H = 1/2`` the
kernel degenerates to 1, ``W~_t = B_t`` exactly, and ``W~`` is a deterministic
function of the increments - the joint law is genuinely rank ``n`` and
``numpy.linalg.cholesky`` rejects it.  Round-off can also push a legitimately
positive definite but badly conditioned ``Sigma`` (``H`` very close to 1/2, or a
grid with a very small first step) marginally negative.  In both cases a
*documented and logged* diagonal jitter ``jitter_rel * max(diag Sigma)`` is
added and the factorisation retried, escalating by a factor 10 per attempt.
Nothing is ever silently regularised: the amount actually applied is reported in
``VolterraFactor.jitter_applied`` / ``.diagnostics()`` and emitted through
``logging.getLogger(__name__).warning``.  Exhausting the attempts raises
:class:`CovarianceFactorizationError` rather than returning a wrong factor.

GRID POLICY
===========
The grid is the sorted union of

  * **every quoted maturity, verbatim** - no snapping, no rounding: the float
    handed in is the float stored in ``SimulationGrid.t`` and
    ``grid.index_of(T)`` recovers its exact position.  (The pre-existing
    ``simulator.py`` snaps maturities onto a uniform grid with
    ``round(T/dt)``; that maturity-dependent bias is exactly what this policy
    removes.)
  * a **log-spaced short-end block** of ``min_steps - 1`` points geometrically
    spaced on ``[short_end_ratio * T_min, T_min)``, so that the shortest quoted
    maturity is reached in at least ``min_steps`` steps - a 7-day expiry is not
    resolved by two points;
  * a **log-spaced bulk fill** of interior points on ``[T_min, T_max]``, dense
    near the short end where the variance curve moves fastest.

A geometric block cannot reach 0, so the first cell ``[0, short_end_ratio *
T_min]`` is the grading floor and is the one cell whose length is not part of
the geometric progression (it is larger than its immediate successor, by the
factor ``1/(q-1)`` of the block's common ratio ``q``).  That is inherent to any
finite grid on a rough process: ``Var(W~_t) = t^{2H}`` already accumulates most
of its short-end mass inside the first cell whatever the floor.  Lower
``short_end_ratio`` (or raise ``min_steps``) to push the floor down.

Fill points falling within ``dedup_rtol`` of a quoted maturity (or of each
other) are dropped - the quoted maturity always wins - because two grid nodes a
round-off apart make ``Sigma`` numerically singular for no modelling benefit.
The total node count obeys ``n <= n_max``; the budget is spent on quoted
maturities first, then the short-end block, then the bulk fill.  When
``n_max`` cannot even hold the quoted maturities plus the short-end block the
builder raises :class:`GridConfigurationError` instead of silently violating
either requirement.

REPRODUCIBILITY
===============
Pass ``rng=`` (a ``numpy.random.Generator``) or ``seed=`` (an int), never both.
For a fixed seed the draw is bit-identical: the ``2n`` normals of the joint
block are drawn first, then the ``n`` normals of the independent ``dB^perp``
block.  Antithetic draws consume normals for ``n_paths // 2`` base paths only
and mirror them by exact negation, so the second half of every returned array is
bit for bit ``-1`` times the first half.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import hyp2f1

logger = logging.getLogger(__name__)


#: Maximum number of grid steps (``n``) a simulation grid may carry.
DEFAULT_N_MAX: int = 256
#: Minimum number of grid points in ``(0, T_min]`` for the shortest maturity.
DEFAULT_MIN_STEPS: int = 16
#: Lower end of the short-end block, as a fraction of the shortest maturity.
DEFAULT_SHORT_END_RATIO: float = 1e-2
#: Relative tolerance under which two grid nodes are considered duplicates.
DEFAULT_DEDUP_RTOL: float = 1e-9
#: Diagonal jitter, relative to ``max(diag Sigma)``, tried when Cholesky fails.
DEFAULT_JITTER_REL: float = 1e-12
#: How many escalating jitter attempts (x10 each) before giving up.
DEFAULT_MAX_JITTER_ATTEMPTS: int = 6
#: Default number of Cholesky factors kept in the module-level LRU cache.
DEFAULT_FACTOR_CACHE_MAXSIZE: int = 8


class VolterraGaussianError(RuntimeError):
    """Base class for failures of the exact joint Gaussian construction."""


class GridConfigurationError(VolterraGaussianError):
    """The requested grid cannot satisfy the grid policy (budget conflict)."""


class CovarianceFactorizationError(VolterraGaussianError):
    """``Sigma`` could not be Cholesky-factored, even with the maximum jitter."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_hurst(H: float) -> float:
    """Validate and normalise the Hurst index; returns it as a plain float."""
    value = float(H)
    if not np.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"H must lie strictly inside (0, 1); got {H!r}.")
    return value


def _resolve_rng(
    rng: np.random.Generator | None, seed: int | None
) -> np.random.Generator:
    """Same convention as :mod:`fbm`: ``rng`` XOR ``seed``, or neither."""
    if rng is not None and seed is not None:
        raise ValueError("Provide either `rng` or `seed`, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as a float64 C-contiguous read-only view."""
    out = np.ascontiguousarray(array, dtype=float)
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridConfig:
    """
    Knobs of the simulation-grid policy.

    Attributes
    ----------
    n_max:
        Hard cap on the number of steps ``n`` (grid nodes excluding ``t_0 = 0``).
        ``Sigma`` is ``2n x 2n`` and the Cholesky is ``O(n^3)``, so this caps the
        one-off setup cost.
    min_steps:
        Minimum number of grid points in ``(0, T_min]``, i.e. the shortest quoted
        maturity must sit at index ``>= min_steps``.
    short_end_ratio:
        The short-end block is geometrically spaced on
        ``[short_end_ratio * T_min, T_min)``; must lie in ``(0, 1)``.
    dedup_rtol:
        Two nodes closer than ``dedup_rtol * max(|node|, 1)`` are duplicates.
        Fill points lose against quoted maturities.
    """

    n_max: int = DEFAULT_N_MAX
    min_steps: int = DEFAULT_MIN_STEPS
    short_end_ratio: float = DEFAULT_SHORT_END_RATIO
    dedup_rtol: float = DEFAULT_DEDUP_RTOL

    def __post_init__(self) -> None:
        if int(self.n_max) < 1:
            raise ValueError(f"n_max must be >= 1; got {self.n_max!r}.")
        if int(self.min_steps) < 1:
            raise ValueError(f"min_steps must be >= 1; got {self.min_steps!r}.")
        if not np.isfinite(self.short_end_ratio) or not (
            0.0 < float(self.short_end_ratio) < 1.0
        ):
            raise ValueError(
                "short_end_ratio must lie strictly inside (0, 1); "
                f"got {self.short_end_ratio!r}."
            )
        if not np.isfinite(self.dedup_rtol) or float(self.dedup_rtol) < 0.0:
            raise ValueError(
                f"dedup_rtol must be a finite, non-negative float; got {self.dedup_rtol!r}."
            )


@dataclass(frozen=True, eq=False)
class SimulationGrid:
    """
    A simulation grid ``0 = t_0 < t_1 < ... < t_n`` carrying its quoted maturities.

    Attributes
    ----------
    t:
        Read-only ``(n + 1,)`` array, ``t[0] == 0.0``, strictly increasing.
    dt:
        Read-only ``(n,)`` array of step lengths, ``dt[i] = t[i+1] - t[i]``.
    quoted_maturities:
        The maturities requested by the caller, sorted, stored verbatim - every
        one of them appears in ``t`` with exact float equality.
    maturity_index:
        ``{T: k}`` with ``t[k] == T`` exactly, for each quoted maturity.
    grid_hash:
        SHA-256 of the little-endian float64 bytes of ``t``; the cache key
        component that identifies this grid.
    config:
        The :class:`GridConfig` the grid was built with.
    n_short_end_fill, n_bulk_fill, n_dropped_fill:
        How many fill points the two log-spaced blocks contributed and how many
        were dropped as duplicates of a quoted maturity or of each other.
    """

    t: np.ndarray
    dt: np.ndarray
    quoted_maturities: tuple[float, ...]
    maturity_index: dict[float, int]
    grid_hash: str
    config: GridConfig
    n_short_end_fill: int
    n_bulk_fill: int
    n_dropped_fill: int

    @property
    def n(self) -> int:
        """Number of steps: ``len(t) - 1``."""
        return int(self.t.size - 1)

    @property
    def times(self) -> np.ndarray:
        """The ``n`` strictly positive nodes ``t_1..t_n`` (read-only view)."""
        return self.t[1:]

    def index_of(self, T: float) -> int:
        """
        Index ``k`` with ``t[k] == T`` for a quoted maturity ``T``.

        Raises ``KeyError`` for anything that is not a quoted maturity: this
        module never snaps a maturity onto a neighbouring node.
        """
        key = float(T)
        try:
            return self.maturity_index[key]
        except KeyError as exc:
            raise KeyError(
                f"{key!r} is not a quoted maturity of this grid "
                f"(quoted: {self.quoted_maturities!r})."
            ) from exc

    def diagnostics(self) -> dict[str, Any]:
        """JSON-safe summary of the grid."""
        return {
            "n": self.n,
            "t_min": float(self.t[1]) if self.n >= 1 else 0.0,
            "t_max": float(self.t[-1]),
            "n_quoted": len(self.quoted_maturities),
            "quoted_maturities": [float(x) for x in self.quoted_maturities],
            "maturity_indices": [
                int(self.maturity_index[float(x)]) for x in self.quoted_maturities
            ],
            "min_steps_realised": (
                int(self.maturity_index[self.quoted_maturities[0]])
                if self.quoted_maturities
                else 0
            ),
            "n_short_end_fill": int(self.n_short_end_fill),
            "n_bulk_fill": int(self.n_bulk_fill),
            "n_dropped_fill": int(self.n_dropped_fill),
            "n_max": int(self.config.n_max),
            "min_steps": int(self.config.min_steps),
            "grid_hash": self.grid_hash,
        }


def _grid_hash(t: np.ndarray) -> str:
    """Stable content hash of a grid (endianness- and layout-independent)."""
    digest = hashlib.sha256()
    digest.update(b"rbergomi-volterra-grid-v1")
    digest.update(np.ascontiguousarray(t, dtype="<f8").tobytes())
    return digest.hexdigest()


def _greedy_dedup(values: np.ndarray, *, rtol: float) -> np.ndarray:
    """Keep a strictly increasing subsequence with a relative separation ``rtol``."""
    kept: list[float] = []
    last = -np.inf
    for value in values:
        item = float(value)
        if item - last > rtol * max(abs(item), 1.0):
            kept.append(item)
            last = item
    return np.asarray(kept, dtype=float)


def build_simulation_grid(
    *,
    maturities: np.ndarray | list[float] | tuple[float, ...],
    config: GridConfig | None = None,
) -> SimulationGrid:
    """
    Build the simulation grid of the exact joint construction.

    Every quoted maturity lands on the grid with **exact float equality** (no
    snapping), the shortest one is reached in at least ``config.min_steps``
    steps, the remaining budget is spent on log-spaced fill points dense near
    zero, and the total step count never exceeds ``config.n_max``.

    Parameters
    ----------
    maturities:
        Quoted maturities in year fractions, strictly positive and finite.
        Exact duplicates are collapsed; two *distinct* maturities closer than
        ``config.dedup_rtol`` are rejected (they would make ``Sigma`` singular
        without adding information).
    config:
        Grid policy; the default :class:`GridConfig` is used when omitted.

    Returns
    -------
    SimulationGrid

    Raises
    ------
    GridConfigurationError
        When the budget cannot honour both ``n_max`` and ``min_steps``, or when
        two quoted maturities are indistinguishable at ``dedup_rtol``.
    ValueError
        On empty / non-finite / non-positive maturities.
    """
    cfg = config if config is not None else GridConfig()
    quoted_raw = np.asarray(maturities, dtype=float).ravel()
    if quoted_raw.size == 0:
        raise ValueError("At least one quoted maturity is required.")
    if not np.all(np.isfinite(quoted_raw)):
        raise ValueError(f"Maturities must all be finite; got {maturities!r}.")
    if np.any(quoted_raw <= 0.0):
        raise ValueError(
            f"Maturities must all be strictly positive; got {maturities!r}."
        )

    quoted = np.unique(quoted_raw)  # exact-equality dedup + sort
    n_quoted = int(quoted.size)
    if n_quoted > 1:
        gaps = np.diff(quoted)
        scale = np.maximum(np.abs(quoted[1:]), 1.0)
        too_close = gaps <= cfg.dedup_rtol * scale
        if np.any(too_close):
            bad = int(np.argmax(too_close))
            raise GridConfigurationError(
                "Quoted maturities "
                f"{quoted[bad]!r} and {quoted[bad + 1]!r} are closer than "
                f"dedup_rtol={cfg.dedup_rtol!r}; they cannot both sit on a "
                "well-conditioned grid."
            )

    n_max = int(cfg.n_max)
    min_steps = int(cfg.min_steps)
    n_short_target = min_steps - 1
    if n_quoted + n_short_target > n_max:
        raise GridConfigurationError(
            f"n_max={n_max} cannot hold {n_quoted} quoted maturities plus the "
            f"{n_short_target} short-end fill points required by "
            f"min_steps={min_steps}. Raise n_max or lower min_steps."
        )

    t_min = float(quoted[0])
    t_max = float(quoted[-1])

    # Split the fill budget between the short-end block [ratio*T_min, T_min) and
    # the bulk (T_min, T_max] IN PROPORTION TO THE LOG-SPAN each covers, with
    # `min_steps - 1` kept as a hard floor on the short end.
    #
    # A fixed `n_short_target = min_steps - 1` starves the shortest maturity: it
    # would receive exactly `min_steps` steps however large `n_max` is, because
    # every remaining fill point lands strictly above T_min. On a
    # [7d .. 730d] chain at n_max=256 that leaves the short-end log-step ~15x
    # coarser than the bulk -- precisely where the Volterra kernel
    # (t-s)^{H-1/2} is most singular for small H, and precisely the maturities
    # spec 4.5 regresses to obtain H.
    n_fill_budget = n_max - n_quoted
    log_span_short = -math.log(float(cfg.short_end_ratio))
    log_span_bulk = math.log(t_max / t_min) if t_max > t_min else 0.0
    log_span_total = log_span_short + log_span_bulk
    if n_fill_budget > 0 and log_span_total > 0.0:
        n_short_target = int(round(n_fill_budget * log_span_short / log_span_total))
        n_short_target = max(n_short_target, min_steps - 1)
        n_short_target = min(n_short_target, n_fill_budget)
    else:
        n_short_target = min(min_steps - 1, max(n_fill_budget, 0))

    if n_short_target > 0:
        short_end = np.geomspace(
            t_min * float(cfg.short_end_ratio), t_min, n_short_target + 1
        )[:-1]
    else:
        short_end = np.empty(0, dtype=float)

    n_bulk_target = n_fill_budget - n_short_target
    if n_bulk_target > 0 and t_max > t_min:
        bulk = np.geomspace(t_min, t_max, n_bulk_target + 2)[1:-1]
    else:
        bulk = np.empty(0, dtype=float)

    n_fill_requested = int(short_end.size + bulk.size)
    fill = np.concatenate([short_end, bulk]) if n_fill_requested else np.empty(0)
    if fill.size:
        fill = np.unique(fill)
        # Drop fill points that collide with a quoted maturity: the quoted
        # value always wins, so `t` keeps the caller's float bit for bit.
        slot = np.searchsorted(quoted, fill)
        left = quoted[np.clip(slot - 1, 0, n_quoted - 1)]
        right = quoted[np.clip(slot, 0, n_quoted - 1)]
        distance = np.minimum(np.abs(fill - left), np.abs(fill - right))
        scale = np.maximum(np.abs(fill), 1.0)
        fill = fill[distance > cfg.dedup_rtol * scale]
        fill = _greedy_dedup(fill, rtol=cfg.dedup_rtol)

    nodes = np.sort(np.concatenate([quoted, fill])) if fill.size else quoted.copy()
    t = np.concatenate([[0.0], nodes])
    if not np.all(np.diff(t) > 0.0):
        raise GridConfigurationError(
            "Internal invariant violated: the assembled grid is not strictly "
            "increasing. Raise dedup_rtol."
        )
    n_steps = int(t.size - 1)
    if n_steps > n_max:
        raise GridConfigurationError(
            f"Internal invariant violated: n={n_steps} exceeds n_max={n_max}."
        )

    slots = np.searchsorted(t, quoted)
    if not np.array_equal(t[slots], quoted):
        raise GridConfigurationError(
            "Internal invariant violated: a quoted maturity is not on the grid "
            "with exact float equality."
        )
    maturity_index = {float(T): int(k) for T, k in zip(quoted, slots)}
    if int(slots[0]) < min_steps:
        raise GridConfigurationError(
            f"Internal invariant violated: the shortest maturity {t_min!r} sits "
            f"at index {int(slots[0])} < min_steps={min_steps}."
        )

    t_ro = _readonly(t)
    return SimulationGrid(
        t=t_ro,
        dt=_readonly(np.diff(t)),
        quoted_maturities=tuple(float(x) for x in quoted),
        maturity_index=maturity_index,
        grid_hash=_grid_hash(t_ro),
        config=cfg,
        n_short_end_fill=int(short_end.size),
        n_bulk_fill=int(bulk.size),
        n_dropped_fill=int(n_fill_requested - fill.size),
    )


# ---------------------------------------------------------------------------
# Closed-form covariances
# ---------------------------------------------------------------------------
def volterra_autocovariance(u: Any, v: Any, *, H: float) -> np.ndarray:
    """
    ``Cov(W~_u, W~_v)`` of the Riemann-Liouville driver, exact 2F1 form.

    ``= 2H/(H+1/2) * min^{H+1/2} * max^{H-1/2} * 2F1(1/2-H, 1; H+3/2; min/max)``

    ``u`` and ``v`` broadcast against each other and may be arrays; both must be
    non-negative.  The result is symmetric in ``(u, v)`` bit for bit, equals
    ``t^{2H}`` on the diagonal (Gauss's theorem, exact to round-off) and is 0
    whenever either argument is 0.

    This arrangement stays accurate as ``u -> v``; the equivalent form built on
    ``(v - u)^{H-1/2}`` does not.
    """
    hurst = _validate_hurst(H)
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if np.any(u_arr < 0.0) or np.any(v_arr < 0.0):
        raise ValueError("Volterra times must be non-negative.")
    lo = np.minimum(u_arr, v_arr)
    hi = np.maximum(u_arr, v_arr)
    positive = hi > 0.0
    hi_safe = np.where(positive, hi, 1.0)
    z = np.where(positive, lo / hi_safe, 0.0)
    value = (
        2.0
        * hurst
        / (hurst + 0.5)
        * lo ** (hurst + 0.5)
        * hi_safe ** (hurst - 0.5)
        * hyp2f1(0.5 - hurst, 1.0, hurst + 1.5, z)
    )
    return np.where(positive, value, 0.0)


def volterra_brownian_cross_covariance(
    t: Any, a: Any, b: Any, *, H: float
) -> np.ndarray:
    """
    ``Cov(W~_t, B_b - B_a)`` for ``a <= b``, exact and cancellation-free.

    ``= sqrt(2H)/(H+1/2) * [ (t-a)^{H+1/2} - (t-min(b,t))^{H+1/2} ]`` when
    ``a < t``, and 0 otherwise (a Brownian increment strictly after ``t``
    carries no information about ``W~_t``: the Volterra kernel is causal).

    The bracket is evaluated as
    ``(t-a)^{H+1/2} * (-expm1((H+1/2) * log1p(-(min(b,t)-a)/(t-a))))``, which
    keeps full relative precision when the increment is tiny compared with the
    lag - the regime of a fine grid at a long horizon, where the direct
    subtraction of two near-equal powers loses digits.

    ``t``, ``a`` and ``b`` broadcast against each other.
    """
    hurst = _validate_hurst(H)
    t_arr = np.asarray(t, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if np.any(t_arr < 0.0) or np.any(a_arr < 0.0):
        raise ValueError("Volterra times must be non-negative.")
    if np.any(b_arr < a_arr):
        raise ValueError("The Brownian increment requires a <= b.")
    lag = t_arr - a_arr
    active = lag > 0.0
    lag_safe = np.where(active, lag, 1.0)
    upper = np.minimum(b_arr, t_arr)
    ratio = np.where(active, (upper - a_arr) / lag_safe, 0.0)
    with np.errstate(divide="ignore"):
        # ratio == 1 (b >= t) -> log1p(-1) = -inf -> expm1(-inf) = -1, which is
        # exactly the (t - t)^{H+1/2} = 0 limit. The warning is spurious.
        bracket = -np.expm1((hurst + 0.5) * np.log1p(-ratio))
    value = (
        np.sqrt(2.0 * hurst)
        / (hurst + 0.5)
        * lag_safe ** (hurst + 0.5)
        * bracket
    )
    return np.where(active, value, 0.0)


def build_joint_covariance(*, H: float, grid: SimulationGrid) -> np.ndarray:
    """
    Assemble the ``2n x 2n`` covariance of ``(W~_{t_1..t_n}, dB_1..dB_n)``.

    Layout ``Sigma = [[A, C], [C^T, D]]`` with ``A`` the Volterra
    autocovariance, ``C`` the (lower-triangular) driver/increment
    cross-covariance and ``D = diag(dt)``.  Fully vectorised: no Python loop
    runs over the grid.  ``Sigma`` is exactly symmetric by construction.
    """
    hurst = _validate_hurst(H)
    times = np.asarray(grid.times, dtype=float)
    n = int(times.size)
    if n < 1:
        raise ValueError("The simulation grid must carry at least one step.")

    block_a = volterra_autocovariance(times[:, None], times[None, :], H=hurst)
    block_c = volterra_brownian_cross_covariance(
        times[:, None], grid.t[:-1][None, :], grid.t[1:][None, :], H=hurst
    )
    block_d = np.diag(np.asarray(grid.dt, dtype=float))

    sigma = np.empty((2 * n, 2 * n), dtype=float)
    sigma[:n, :n] = block_a
    sigma[:n, n:] = block_c
    sigma[n:, :n] = block_c.T
    sigma[n:, n:] = block_d
    return sigma


# ---------------------------------------------------------------------------
# Cholesky factory + cache
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class VolterraFactor:
    """
    A cached Cholesky factor of the joint covariance for one ``(H, grid)``.

    Attributes
    ----------
    H, grid, n:
        Hurst index, the grid the factor belongs to, and its step count.
    L:
        Read-only ``(2n, 2n)`` lower-triangular factor, ``Sigma = L @ L.T``.
        Rows ``0..n-1`` generate ``W~_{t_1..t_n}``, rows ``n..2n-1`` generate
        ``dB_1..dB_n``.
    max_abs_diagonal:
        ``max(diag Sigma)``; the scale the jitter is expressed relative to.
    jitter_applied:
        Absolute amount added to the diagonal, ``0.0`` when the raw ``Sigma``
        factored cleanly.  Always non-zero when ``jitter_attempts > 0``.
    jitter_attempts:
        Number of rejected factorisations before success.
    min_pivot, max_pivot:
        Extremes of ``diag(L)``.  ``min_pivot`` is the actionable
        near-singularity signal: it is exactly what collapses to 0 when the
        factorisation is about to be rejected, and it is exact and free.
    pivot_ratio_squared:
        ``(max_pivot / min_pivot) ** 2``.  For a triangular ``L``,
        ``sigma_max(L) >= max_i |L_ii|`` and ``sigma_min(L) <= min_i |L_ii|``,
        so this is a rigorous LOWER BOUND on ``cond_2(Sigma) = cond_2(L)**2``.
        It is NOT the condition number and it is typically loose - measured up
        to ~5 orders of magnitude below the truth on a 256-step grid.  The exact
        ``cond_2`` needs an SVD of a ``2n x 2n`` matrix, which a calibration
        would pay on every optimizer iteration (``H`` changes each time), so it
        is deliberately not computed here; call ``numpy.linalg.cond`` on
        :func:`build_joint_covariance` when the exact number is actually wanted.
    """

    H: float
    grid: SimulationGrid
    n: int
    L: np.ndarray
    max_abs_diagonal: float
    jitter_applied: float
    jitter_attempts: int
    min_pivot: float
    max_pivot: float
    pivot_ratio_squared: float

    def diagnostics(self) -> dict[str, Any]:
        """JSON-safe summary of the factorisation."""
        return {
            "H": float(self.H),
            "n": int(self.n),
            "grid_hash": self.grid.grid_hash,
            "max_abs_diagonal": float(self.max_abs_diagonal),
            "jitter_applied": float(self.jitter_applied),
            "jitter_attempts": int(self.jitter_attempts),
            "min_pivot": float(self.min_pivot),
            "max_pivot": float(self.max_pivot),
            "pivot_ratio_squared": float(self.pivot_ratio_squared),
        }


_FACTOR_CACHE: "OrderedDict[tuple[float, str, float, int], VolterraFactor]" = (
    OrderedDict()
)
_FACTOR_CACHE_MAXSIZE: int = DEFAULT_FACTOR_CACHE_MAXSIZE
_FACTOR_CACHE_HITS: int = 0
_FACTOR_CACHE_MISSES: int = 0


def clear_factor_cache() -> None:
    """Empty the Cholesky cache and reset its hit / miss counters."""
    global _FACTOR_CACHE_HITS, _FACTOR_CACHE_MISSES
    _FACTOR_CACHE.clear()
    _FACTOR_CACHE_HITS = 0
    _FACTOR_CACHE_MISSES = 0


def set_factor_cache_maxsize(size: int) -> None:
    """Resize the Cholesky LRU, evicting the least recently used entries."""
    global _FACTOR_CACHE_MAXSIZE
    value = int(size)
    if value < 1:
        raise ValueError(f"Cache size must be >= 1; got {size!r}.")
    _FACTOR_CACHE_MAXSIZE = value
    while len(_FACTOR_CACHE) > _FACTOR_CACHE_MAXSIZE:
        _FACTOR_CACHE.popitem(last=False)


def factor_cache_info() -> dict[str, int]:
    """Hit / miss counters and occupancy of the Cholesky LRU."""
    return {
        "hits": int(_FACTOR_CACHE_HITS),
        "misses": int(_FACTOR_CACHE_MISSES),
        "size": len(_FACTOR_CACHE),
        "maxsize": int(_FACTOR_CACHE_MAXSIZE),
    }


def cholesky_factor(
    *,
    H: float,
    grid: SimulationGrid,
    jitter_rel: float = DEFAULT_JITTER_REL,
    max_jitter_attempts: int = DEFAULT_MAX_JITTER_ATTEMPTS,
    use_cache: bool = True,
) -> VolterraFactor:
    """
    Cholesky-factor the joint covariance once per ``(H, grid, jitter policy)``.

    The factor is memoised in a bounded LRU: repeated calls with the same key
    return **the same object** (identity holds), so the ``O(n^3)`` cost is paid
    once and amortised over every path, strike, maturity and optimizer
    common-random-number draw.

    Parameters
    ----------
    H:
        Hurst index in ``(0, 1)``.  ``H = 1/2`` makes ``Sigma`` exactly singular
        (``W~ == B``) and therefore always goes through the jitter path.
    grid:
        A :class:`SimulationGrid`.
    jitter_rel:
        Diagonal jitter relative to ``max(diag Sigma)``, tried only if the raw
        factorisation fails, escalating x10 per attempt.  ``0.0`` disables it.
    max_jitter_attempts:
        How many escalating attempts before raising.
    use_cache:
        Set ``False`` to bypass the LRU entirely (no lookup, no insertion).

    Returns
    -------
    VolterraFactor

    Raises
    ------
    CovarianceFactorizationError
        When every attempt failed; the raw ``LinAlgError`` is chained.
    """
    global _FACTOR_CACHE_HITS, _FACTOR_CACHE_MISSES

    hurst = _validate_hurst(H)
    if not isinstance(grid, SimulationGrid):
        raise TypeError(f"grid must be a SimulationGrid; got {type(grid)!r}.")
    if not np.isfinite(jitter_rel) or float(jitter_rel) < 0.0:
        raise ValueError(
            f"jitter_rel must be a finite, non-negative float; got {jitter_rel!r}."
        )
    if int(max_jitter_attempts) < 0:
        raise ValueError(
            f"max_jitter_attempts must be >= 0; got {max_jitter_attempts!r}."
        )

    key = (hurst, grid.grid_hash, float(jitter_rel), int(max_jitter_attempts))
    if use_cache:
        cached = _FACTOR_CACHE.get(key)
        if cached is not None:
            _FACTOR_CACHE.move_to_end(key)
            _FACTOR_CACHE_HITS += 1
            return cached
        _FACTOR_CACHE_MISSES += 1

    sigma = build_joint_covariance(H=hurst, grid=grid)
    size = sigma.shape[0]
    max_diag = float(np.max(np.diag(sigma)))
    diag_slice = (np.arange(size), np.arange(size))

    attempts = 0
    jitter = 0.0
    working = sigma
    while True:
        try:
            chol = np.linalg.cholesky(working)
            break
        except np.linalg.LinAlgError as exc:
            if attempts >= int(max_jitter_attempts) or float(jitter_rel) <= 0.0:
                raise CovarianceFactorizationError(
                    "The rBergomi joint covariance could not be Cholesky-factored "
                    f"for H={hurst!r}, n={grid.n} after {attempts} jitter "
                    f"attempt(s) (last jitter={jitter!r}, "
                    f"max(diag)={max_diag!r}). The grid is degenerate."
                ) from exc
            jitter = float(jitter_rel) * max_diag * (10.0 ** attempts)
            attempts += 1
            logger.warning(
                "rBergomi joint covariance: Cholesky rejected Sigma "
                "(H=%.6f, n=%d); retrying with diagonal jitter %.3e "
                "(%.1e x max(diag)=%.3e), attempt %d/%d.",
                hurst,
                grid.n,
                jitter,
                float(jitter_rel) * (10.0 ** (attempts - 1)),
                max_diag,
                attempts,
                int(max_jitter_attempts),
            )
            working = sigma.copy()
            working[diag_slice] += jitter

    if attempts > 0:
        logger.warning(
            "rBergomi joint covariance: factorisation succeeded after %d jitter "
            "attempt(s) with jitter=%.3e applied to the diagonal (H=%.6f, n=%d). "
            "The simulated covariance is Sigma + %.3e * I, not Sigma.",
            attempts,
            jitter,
            hurst,
            grid.n,
            jitter,
        )

    pivots = np.diag(chol)
    min_pivot = float(np.min(pivots))
    max_pivot = float(np.max(pivots))
    pivot_ratio_squared = (
        (max_pivot / min_pivot) ** 2 if min_pivot > 0.0 else float("inf")
    )

    factor = VolterraFactor(
        H=hurst,
        grid=grid,
        n=grid.n,
        L=_readonly(chol),
        max_abs_diagonal=max_diag,
        jitter_applied=float(jitter) if attempts > 0 else 0.0,
        jitter_attempts=int(attempts),
        min_pivot=min_pivot,
        max_pivot=max_pivot,
        pivot_ratio_squared=float(pivot_ratio_squared),
    )
    if use_cache:
        _FACTOR_CACHE[key] = factor
        _FACTOR_CACHE.move_to_end(key)
        while len(_FACTOR_CACHE) > _FACTOR_CACHE_MAXSIZE:
            _FACTOR_CACHE.popitem(last=False)
    return factor


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class JointGaussianDraw:
    """
    One sample of the exact joint Gaussian construction.

    Attributes
    ----------
    W_tilde:
        ``(n_paths, n)`` values of the Volterra driver at ``t_1..t_n``.
    dB:
        ``(n_paths, n)`` increments of the driving Brownian motion ``B``,
        ``dB[:, i] = B_{t_{i+1}} - B_{t_i}``, jointly exact with ``W_tilde``.
    dB_perp:
        ``(n_paths, n)`` increments of an INDEPENDENT Brownian motion, variance
        ``dt_i``.  The spot noise is ``rho * dB + sqrt(1 - rho^2) * dB_perp``,
        which is where the exact ``rho``-correlation with the variance path
        comes from.
    n_paths, n_base_paths:
        Total paths and, under antithetic sampling, the number of independently
        drawn ones (``n_paths // 2``; equal to ``n_paths`` otherwise).
    antithetic:
        Whether the second half mirrors the first by exact negation.
    H, grid:
        Provenance of the draw.
    """

    W_tilde: np.ndarray
    dB: np.ndarray
    dB_perp: np.ndarray
    n_paths: int
    n_base_paths: int
    antithetic: bool
    H: float
    grid: SimulationGrid


def draw_joint_gaussian(
    *,
    factor: VolterraFactor,
    n_paths: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    antithetic: bool = False,
) -> JointGaussianDraw:
    """
    Draw ``n_paths`` exact joint samples ``(W~, dB)`` plus independent ``dB^perp``.

    ``X = Z @ L.T`` with ``Z ~ N(0, I_{2n})`` reproduces ``Sigma`` exactly, so
    the correlation structure carries no discretisation error - only the ``dt``
    resolution of whatever integral the caller builds on top.

    Parameters
    ----------
    factor:
        A :class:`VolterraFactor` from :func:`cholesky_factor`.
    n_paths:
        Number of paths; must be even and ``>= 2`` when ``antithetic``.
    rng, seed:
        At most one of them.  The ``2n`` joint normals are consumed first, the
        ``n`` perpendicular ones second.
    antithetic:
        Draw ``n_paths // 2`` base paths and append their exact negation.  Rows
        ``k`` and ``k + n_paths // 2`` are bit for bit opposite in all three
        returned arrays, so any odd-order Monte-Carlo bias cancels identically.

    Returns
    -------
    JointGaussianDraw
    """
    if not isinstance(factor, VolterraFactor):
        raise TypeError(f"factor must be a VolterraFactor; got {type(factor)!r}.")
    count = int(n_paths)
    if count < 1:
        raise ValueError(f"n_paths must be >= 1; got {n_paths!r}.")
    if antithetic and count % 2 != 0:
        raise ValueError(
            f"Antithetic sampling needs an even n_paths; got {n_paths!r}."
        )
    generator = _resolve_rng(rng, seed)

    n = int(factor.n)
    n_base = count // 2 if antithetic else count
    joint = generator.standard_normal((n_base, 2 * n)) @ np.asarray(factor.L).T
    perp = generator.standard_normal((n_base, n)) * np.sqrt(
        np.asarray(factor.grid.dt, dtype=float)
    )
    if antithetic:
        # Exact negation, not a second matmul: guarantees bit-identical mirroring
        # and halves the flops.
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


__all__ = [
    "DEFAULT_DEDUP_RTOL",
    "DEFAULT_FACTOR_CACHE_MAXSIZE",
    "DEFAULT_JITTER_REL",
    "DEFAULT_MAX_JITTER_ATTEMPTS",
    "DEFAULT_MIN_STEPS",
    "DEFAULT_N_MAX",
    "DEFAULT_SHORT_END_RATIO",
    "CovarianceFactorizationError",
    "GridConfig",
    "GridConfigurationError",
    "JointGaussianDraw",
    "SimulationGrid",
    "VolterraFactor",
    "VolterraGaussianError",
    "build_joint_covariance",
    "build_simulation_grid",
    "cholesky_factor",
    "clear_factor_cache",
    "draw_joint_gaussian",
    "factor_cache_info",
    "set_factor_cache_maxsize",
    "volterra_autocovariance",
    "volterra_brownian_cross_covariance",
]
