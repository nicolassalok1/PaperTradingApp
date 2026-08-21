"""
Forward-variance curve ``xi0(T)`` (spec 4.4).

The variance-swap curve of spec 4.3 gives, at every market maturity ``T_j``, the
**total** variance

.. code-block:: text

    V(0, T_j) = T_j * K_var(T_j) = int_0^{T_j} xi0(u) du .

``xi0`` is therefore a *derivative* of market data, and market data is noisy:
differentiating ``(T_j, V_j)`` directly with finite differences amplifies quote
noise into a saw-toothed forward-variance curve that the rBergomi simulator would
faithfully reproduce as a saw-toothed term structure. This module never does
that. It proceeds in two steps.

1. Monotonicity repair (calendar-arbitrage)
-------------------------------------------
``V(0, T)`` must be **non-decreasing** in ``T``: a decreasing total variance is a
calendar-spread arbitrage, not information. It is repaired by **isotonic
regression** — the pool-adjacent-violators algorithm (PAVA, Ayer et al. 1955),
which returns the exact least-squares non-decreasing fit, implemented here
(``scipy.optimize.isotonic_regression`` does not exist in the pinned scipy
1.11.4 and scikit-learn is not a dependency of this repo). Every adjusted point
and the size of its adjustment are logged in
:class:`ForwardVarianceMetadata.isotonic_adjustments` — the repair is never
silent.

2. Piecewise-constant forward variance (the default)
----------------------------------------------------
.. code-block:: text

    xi0(t) = V_1 / T_1                              on [0, T_1]
    xi0(t) = (V_{j+1} - V_j) / (T_{j+1} - T_j)      on (T_j, T_{j+1}]
    xi0(t) = xi0(T_last)                            for t > T_last  (flat)

**Deliberate deviation from the source spec, recorded on purpose.** The source
spec's headline asks for a "smooth" ``xi0``. The pipeline default here is
piecewise-constant because it is exact by construction
(``int_0^{T_j} xi0 = V_j`` at every market maturity, to machine precision),
oscillation-free (no interpolant can invent a wiggle between two maturities), and
it is what the market itself does when bootstrapping forward variance from
variance-swap quotes. The smooth variant remains available behind
:attr:`ForwardVarianceConfig.method`.

Positivity is enforced by a floor at :attr:`ForwardVarianceConfig.eps_xi`.
Flooring a level **breaks** the exact reconstruction on that interval — by
construction, since the floored level no longer integrates to the market
increment — so it raises :data:`FLAG_XI0_FLOORED`, records the affected indices
and reports the resulting reconstruction gap. After the isotonic repair the
increments are non-negative, so the floor only ever bites on a genuinely flat or
negative first increment.

3. Optional smooth variant (``method="pchip_monotone"``)
--------------------------------------------------------
A monotone PCHIP interpolant is fitted to ``(0, 0), (T_1, V_1), ...`` and
differentiated in closed form; ``xi0 = V'``. It is accepted only if it passes
three validations, otherwise the builder **falls back to the piecewise-constant
curve** and records the rejection reason:

* **positivity** — ``xi0 > eps_xi`` on a fine grid over ``[0, T_last]``;
* **no oscillation** — the number of *material* turning points of ``xi0`` must
  not exceed the number of material turning points of the piecewise-constant
  level sequence, i.e. "sign changes of ``xi0'`` bounded by the data". "Material"
  means the swing across the turning point exceeds
  ``oscillation_rel_tol * (max(levels) - min(levels))``: a wiggle below that band
  is Hermite end-condition noise, far below the accuracy of the ``K_var`` inputs
  themselves, and counting it would reject every real term structure;
* **quadrature reconstruction** — ``|int_0^{T_j} xi0 - V_j| <
  quadrature_tol`` at every market maturity, evaluated by composite Simpson
  interval by interval (``xi0`` is piecewise quadratic, so Simpson is exact and
  the check really tests that the differentiated interpolant still integrates
  back to the data).

The curve object
----------------
:class:`ForwardVarianceCurve` exposes ``xi0(t)`` (**vectorised**: scalars in,
scalar out; arrays in, same-shape array out), ``integrated(T)``, the construction
metadata and the extrapolation policy. **It is what the rBergomi simulator
consumes and what the joint calibration freezes**, so it is a frozen dataclass
holding only tuples plus non-writeable numpy views: nothing a consumer can reach
lets it mutate the curve, and it is safe to share across threads and across
calibration restarts.

Layering: model layer only. No side effects, no network, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from app.model.calibration.rough_vol.variance_swap import (
    VarianceSwapCurve,
    VarianceSwapPoint,
)

#: Construction methods.
METHOD_PIECEWISE_CONSTANT = "piecewise_constant"
METHOD_PCHIP = "pchip_monotone"

#: The only extrapolation policy this module implements, exposed on the curve.
EXTRAPOLATION_FLAT = "flat_beyond_last_maturity"

# --- diagnostic flags -------------------------------------------------------
FLAG_ISOTONIC_REPAIR = "isotonic_repair_applied"
FLAG_XI0_FLOORED = "xi0_floored"
FLAG_SINGLE_MATURITY = "single_maturity"
FLAG_PCHIP_FALLBACK = "pchip_rejected_fallback_piecewise_constant"

# --- PCHIP rejection reasons ------------------------------------------------
REJECT_TOO_FEW_MATURITIES = "too_few_maturities"
REJECT_POSITIVITY = "xi0_not_strictly_positive"
REJECT_OSCILLATION = "xi0_oscillates_beyond_data"
REJECT_QUADRATURE = "quadrature_reconstruction_off"

#: French labels for the Phase-5 report.
FORWARD_VARIANCE_LABELS_FR: dict[str, str] = {
    REJECT_TOO_FEW_MATURITIES: "Trop peu de maturités pour l'interpolation lisse",
    REJECT_POSITIVITY: "Variance forward lissée non strictement positive",
    REJECT_OSCILLATION: "Variance forward lissée oscillante au-delà des données",
    REJECT_QUADRATURE: "Reconstruction par quadrature hors tolérance",
    FLAG_ISOTONIC_REPAIR: "Réparation isotone appliquée (arbitrage calendaire)",
    FLAG_XI0_FLOORED: "Variance forward plancherée à eps_xi",
    FLAG_SINGLE_MATURITY: "Une seule maturité: courbe plate",
    FLAG_PCHIP_FALLBACK: "Variante lisse rejetée: repli sur la variante constante par morceaux",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardVarianceConfig:
    """
    Thresholds of spec 4.4, in one frozen place.

    Attributes
    ----------
    method:
        :data:`METHOD_PIECEWISE_CONSTANT` (default, exact-by-construction) or
        :data:`METHOD_PCHIP` (smooth variant, subject to the three validations).
    eps_xi:
        Positivity floor on ``xi0``. Flooring breaks the exact reconstruction on
        the affected interval and is always flagged.
    isotonic_log_tol:
        Adjustments larger than this (in absolute total variance) are logged.
        ``0.0`` logs every point the repair actually moved.
    positivity_grid_points:
        Size of the fine grid on ``[0, T_last]`` used to validate the smooth
        variant's positivity and to measure ``min xi0``.
    oscillation_rel_tol:
        Dead band of the turning-point count, relative to the spread of the
        piecewise-constant levels. See the module docstring.
    quadrature_tol:
        Acceptance tolerance of the smooth variant's quadrature reconstruction
        (spec 4.4 asks for ``1e-8``).
    quadrature_nodes_per_interval:
        Simpson nodes used per ``[T_{j-1}, T_j]`` sub-interval; forced odd.
    """

    method: str = METHOD_PIECEWISE_CONSTANT
    eps_xi: float = 1e-6
    isotonic_log_tol: float = 0.0
    positivity_grid_points: int = 2001
    oscillation_rel_tol: float = 1e-2
    quadrature_tol: float = 1e-8
    quadrature_nodes_per_interval: int = 21


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA)
# ---------------------------------------------------------------------------


def pool_adjacent_violators(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> np.ndarray:
    """
    Exact least-squares **non-decreasing** fit of ``values`` (PAVA).

    Ayer, Brunk, Ewing, Reid & Silverman (1955). The algorithm walks left to
    right maintaining a stack of pooled blocks; whenever a new block violates
    monotonicity against the block below it, the two are merged into their
    weighted mean. The result minimises ``sum_i w_i (y_i - yhat_i)^2`` subject to
    ``yhat_1 <= ... <= yhat_n`` and is unique.

    Implemented here rather than imported: ``scipy.optimize.isotonic_regression``
    does not exist in the pinned scipy 1.11.4 and scikit-learn is not a
    dependency of this repository.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("pool_adjacent_violators: un vecteur 1-D est attendu")
    n = y.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("pool_adjacent_violators: poids et valeurs de tailles différentes")
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("pool_adjacent_violators: poids non finis ou non strictement positifs")
    if not np.all(np.isfinite(y)):
        raise ValueError("pool_adjacent_violators: valeurs non finies")

    block_value: list[float] = []
    block_weight: list[float] = []
    block_size: list[int] = []

    for i in range(n):
        block_value.append(float(y[i]))
        block_weight.append(float(w[i]))
        block_size.append(1)
        while len(block_value) > 1 and block_value[-2] > block_value[-1]:
            v2, w2, s2 = block_value.pop(), block_weight.pop(), block_size.pop()
            v1, w1, s1 = block_value.pop(), block_weight.pop(), block_size.pop()
            merged_w = w1 + w2
            block_value.append((v1 * w1 + v2 * w2) / merged_w)
            block_weight.append(merged_w)
            block_size.append(s1 + s2)

    out = np.empty(n, dtype=np.float64)
    pos = 0
    for value, size in zip(block_value, block_size):
        out[pos : pos + size] = value
        pos += size
    return out


@dataclass(frozen=True)
class IsotonicAdjustment:
    """One market maturity moved by the calendar-arbitrage repair."""

    index: int
    T: float
    v_raw: float
    v_fitted: float

    @property
    def delta(self) -> float:
        """``v_fitted - v_raw``: signed size of the adjustment."""
        return float(self.v_fitted) - float(self.v_raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "T": float(self.T),
            "v_raw": float(self.v_raw),
            "v_fitted": float(self.v_fitted),
            "delta": float(self.delta),
            "k_var_raw": float(self.v_raw / self.T) if self.T > 0.0 else float("nan"),
            "k_var_fitted": float(self.v_fitted / self.T) if self.T > 0.0 else float("nan"),
        }


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def material_turning_points(values: Sequence[float], band: float) -> int:
    """
    Count the interior extrema of ``values`` whose swing exceeds ``band``.

    A zig-zag filter: monotone runs are extended, and a reversal only opens a new
    extremum once it has moved further than ``band`` away from the running
    extreme. With ``band = 0`` this is the plain count of sign changes of the
    first difference; with ``band > 0`` sub-band wiggles are ignored. Used on
    both sides of the oscillation test so the model and the data are measured
    with exactly the same yardstick.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 3:
        return 0
    tol = float(band)
    extrema: list[float] = [float(v[0])]
    direction = 0
    for x in v[1:]:
        x = float(x)
        last = extrema[-1]
        if direction == 0:
            if abs(x - last) > tol:
                direction = 1 if x > last else -1
                extrema.append(x)
        elif (x - last) * direction > 0.0:
            extrema[-1] = x
        elif abs(x - last) > tol:
            direction = -direction
            extrema.append(x)
    return max(len(extrema) - 2, 0)


def _simpson_uniform(y: np.ndarray, h: float) -> float:
    """Composite Simpson on a uniform grid of ``2m+1`` nodes spaced by ``h``."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3 or n % 2 == 0:
        raise ValueError(f"_simpson_uniform: nombre de noeuds invalide n={n}")
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardVarianceMetadata:
    """Everything needed to audit a :class:`ForwardVarianceCurve` after the fact."""

    method: str = METHOD_PIECEWISE_CONSTANT
    requested_method: str = METHOD_PIECEWISE_CONSTANT
    n_maturities: int = 0
    eps_xi: float = 1e-6
    extrapolation_policy: str = EXTRAPOLATION_FLAT
    isotonic_adjustments: tuple[IsotonicAdjustment, ...] = ()
    isotonic_max_abs_adjustment: float = 0.0
    floored_indices: tuple[int, ...] = ()
    max_reconstruction_error: float = 0.0
    quadrature_max_error: float = float("nan")
    min_xi0: float = float("nan")
    max_xi0: float = float("nan")
    n_turning_points_data: int = 0
    n_turning_points_model: int = 0
    rejection_reasons: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    messages_fr: tuple[str, ...] = ()

    @property
    def n_floored(self) -> int:
        return len(self.floored_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "requested_method": str(self.requested_method),
            "n_maturities": int(self.n_maturities),
            "eps_xi": float(self.eps_xi),
            "extrapolation_policy": str(self.extrapolation_policy),
            "isotonic_adjustments": [a.to_dict() for a in self.isotonic_adjustments],
            "n_isotonic_adjustments": int(len(self.isotonic_adjustments)),
            "isotonic_max_abs_adjustment": float(self.isotonic_max_abs_adjustment),
            "floored_indices": [int(i) for i in self.floored_indices],
            "n_floored": int(self.n_floored),
            "max_reconstruction_error": float(self.max_reconstruction_error),
            "quadrature_max_error": float(self.quadrature_max_error),
            "min_xi0": float(self.min_xi0),
            "max_xi0": float(self.max_xi0),
            "n_turning_points_data": int(self.n_turning_points_data),
            "n_turning_points_model": int(self.n_turning_points_model),
            "rejection_reasons": [str(r) for r in self.rejection_reasons],
            "rejection_reasons_fr": [
                FORWARD_VARIANCE_LABELS_FR.get(r, r) for r in self.rejection_reasons
            ],
            "flags": [str(f) for f in self.flags],
            "messages_fr": [str(m) for m in self.messages_fr],
        }


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardVarianceCurve:
    """
    Forward-variance curve consumed by the simulator and frozen by the calibration.

    Attributes
    ----------
    T_knots:
        Market maturities, strictly increasing, in years (calendar days / 365).
    levels:
        Piecewise-constant forward variances; ``levels[0]`` applies on
        ``[0, T_knots[0]]`` and ``levels[j]`` on ``(T_knots[j-1], T_knots[j]]``.
        Always populated, whatever the method — the smooth variant keeps them as
        the data summary and as the fallback it was validated against.
    V_knots:
        ``integrated(T_j)`` — the curve's own total variance at each knot. Equal
        to :attr:`V_repaired` except where a level was floored.
    V_market / V_repaired:
        Total variance ``T_j K_var(T_j)`` before and after the isotonic repair.
    metadata:
        :class:`ForwardVarianceMetadata`.

    Immutability. The dataclass is frozen and every field is a tuple or another
    frozen dataclass; the cached numpy views built in ``__post_init__`` are
    non-writeable, and ``xi0`` / ``integrated`` always allocate fresh output. A
    consumer cannot mutate the curve, so it is safe to share between the
    simulator, the calibrator and any report.
    """

    T_knots: tuple[float, ...]
    levels: tuple[float, ...]
    V_knots: tuple[float, ...]
    V_market: tuple[float, ...]
    V_repaired: tuple[float, ...]
    metadata: ForwardVarianceMetadata = field(default_factory=ForwardVarianceMetadata)
    pchip: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        T_arr = np.asarray(self.T_knots, dtype=np.float64)
        levels_arr = np.asarray(self.levels, dtype=np.float64)
        V_arr = np.asarray(self.V_knots, dtype=np.float64)
        for arr in (T_arr, levels_arr, V_arr):
            arr.flags.writeable = False
        object.__setattr__(self, "_T_arr", T_arr)
        object.__setattr__(self, "_levels_arr", levels_arr)
        object.__setattr__(self, "_V_arr", V_arr)
        derivative = None
        tail_level = float(levels_arr[-1]) if levels_arr.size else float("nan")
        if self.metadata.method == METHOD_PCHIP and self.pchip is not None:
            derivative = self.pchip.derivative()
            tail_level = float(derivative(float(T_arr[-1])))
            # Harden the interpolants too: a consumer holding `curve.pchip` must
            # not be able to reshape the curve under the simulator's feet.
            for spline in (self.pchip, derivative):
                for name in ("c", "x"):
                    buffer = getattr(spline, name, None)
                    if isinstance(buffer, np.ndarray) and buffer.flags.owndata:
                        buffer.flags.writeable = False
        object.__setattr__(self, "_pchip_d1", derivative)
        object.__setattr__(self, "_tail_level", tail_level)

    # -- accessors ---------------------------------------------------------

    @property
    def method(self) -> str:
        return str(self.metadata.method)

    @property
    def extrapolation_policy(self) -> str:
        """``flat_beyond_last_maturity`` — ``xi0`` is held constant after ``T_last``."""
        return str(self.metadata.extrapolation_policy)

    @property
    def T_max(self) -> float:
        return float(self._T_arr[-1])

    def __len__(self) -> int:
        return int(self._T_arr.size)

    # -- evaluation --------------------------------------------------------

    def xi0(self, t: Any) -> Any:
        """
        Instantaneous forward variance at ``t``.

        Vectorised: a scalar returns a ``float``, an array-like returns a
        ``numpy`` array of the same shape. ``t < 0`` returns NaN (there is no
        forward variance before today); ``t > T_max`` returns the flat
        extrapolation.
        """
        query = np.asarray(t, dtype=np.float64)
        flat = np.atleast_1d(query)
        out = self._xi0_array(flat)
        if query.ndim == 0:
            return float(out[0])
        return out.reshape(query.shape)

    def integrated(self, T: Any) -> Any:
        """
        ``int_0^T xi0(u) du`` — the total variance implied by this curve.

        Exact at every market maturity for the piecewise-constant construction
        (up to floating-point accumulation), and equal to the PCHIP interpolant
        for the smooth variant. Same vectorisation contract as :meth:`xi0`.
        """
        query = np.asarray(T, dtype=np.float64)
        flat = np.atleast_1d(query)
        out = self._integrated_array(flat)
        if query.ndim == 0:
            return float(out[0])
        return out.reshape(query.shape)

    def reconstruction_errors(self) -> tuple[float, ...]:
        """``integrated(T_j) - V_repaired[j]`` at every market maturity."""
        got = self._integrated_array(np.asarray(self.T_knots, dtype=np.float64))
        want = np.asarray(self.V_repaired, dtype=np.float64)
        return tuple(float(x) for x in (got - want))

    # -- internals ---------------------------------------------------------

    def _xi0_array(self, t: np.ndarray) -> np.ndarray:
        T_arr = self._T_arr
        if self._pchip_d1 is not None:
            T_last = float(T_arr[-1])
            clipped = np.clip(t, 0.0, T_last)
            out = np.asarray(self._pchip_d1(clipped), dtype=np.float64).copy()
            out = np.where(t > T_last, float(self._tail_level), out)
        else:
            idx = np.clip(np.searchsorted(T_arr, t, side="left"), 0, T_arr.size - 1)
            out = self._levels_arr[idx].copy()
        # NaN in, NaN out. `searchsorted` places NaN at the end and `np.clip`
        # keeps it there, so without this a NaN query time would silently be
        # answered with the flat tail level -- a real forward variance for a
        # time that does not exist.
        out = np.where(t < 0.0, np.nan, out)
        out = np.where(np.isnan(t), np.nan, out)
        return np.asarray(out, dtype=np.float64)

    def _integrated_array(self, T: np.ndarray) -> np.ndarray:
        T_arr = self._T_arr
        T_last = float(T_arr[-1])
        if self._pchip_d1 is not None:
            clipped = np.clip(T, 0.0, T_last)
            out = np.asarray(self.pchip(clipped), dtype=np.float64).copy()
            beyond = float(self._V_arr[-1]) + float(self._tail_level) * (T - T_last)
            out = np.where(T > T_last, beyond, out)
        else:
            idx = np.clip(np.searchsorted(T_arr, T, side="left"), 0, T_arr.size - 1)
            base_V = np.where(idx == 0, 0.0, self._V_arr[np.maximum(idx - 1, 0)])
            base_T = np.where(idx == 0, 0.0, T_arr[np.maximum(idx - 1, 0)])
            out = base_V + self._levels_arr[idx] * (T - base_T)
        out = np.where(T < 0.0, np.nan, out)
        return np.asarray(out, dtype=np.float64)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_knots": [float(x) for x in self.T_knots],
            "levels": [float(x) for x in self.levels],
            "V_knots": [float(x) for x in self.V_knots],
            "V_market": [float(x) for x in self.V_market],
            "V_repaired": [float(x) for x in self.V_repaired],
            "reconstruction_errors": [float(x) for x in self.reconstruction_errors()],
            "metadata": self.metadata.to_dict(),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _extract_points(variance_curve: Any) -> list[VarianceSwapPoint]:
    """Accept a :class:`VarianceSwapCurve` or a bare sequence of points."""
    if isinstance(variance_curve, VarianceSwapCurve):
        points = list(variance_curve.points)
    else:
        points = list(variance_curve)
    for point in points:
        if not isinstance(point, VarianceSwapPoint):
            raise TypeError(
                "build_forward_variance_curve: variance_curve doit être un VarianceSwapCurve "
                f"ou une séquence de VarianceSwapPoint; reçu {type(point).__name__}"
            )
    return points


def _levels_from_totals(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Piecewise-constant forward variance from total variances (spec 4.4)."""
    levels = np.empty(T.size, dtype=np.float64)
    levels[0] = V[0] / T[0]
    if T.size > 1:
        levels[1:] = np.diff(V) / np.diff(T)
    return levels


def _try_pchip(
    T: np.ndarray,
    V: np.ndarray,
    levels: np.ndarray,
    config: ForwardVarianceConfig,
) -> tuple[PchipInterpolator | None, str | None, dict[str, float]]:
    """
    Fit and validate the smooth variant. Returns ``(interpolator, reason, stats)``.

    ``reason`` is ``None`` on acceptance and the rejection code otherwise; the
    caller then falls back to the piecewise-constant curve.
    """
    stats: dict[str, float] = {
        "min_xi0": float("nan"),
        "max_xi0": float("nan"),
        "quadrature_max_error": float("nan"),
        "n_turning_points_data": 0.0,
        "n_turning_points_model": 0.0,
    }
    if T.size < 2:
        return None, REJECT_TOO_FEW_MATURITIES, stats

    T_ext = np.concatenate(([0.0], T))
    V_ext = np.concatenate(([0.0], V))
    interpolator = PchipInterpolator(T_ext, V_ext)
    d1 = interpolator.derivative()

    n_grid = max(int(config.positivity_grid_points), 3)
    grid = np.unique(np.concatenate([np.linspace(0.0, float(T[-1]), n_grid), T_ext]))
    xi_grid = np.asarray(d1(grid), dtype=np.float64)
    stats["min_xi0"] = float(np.min(xi_grid))
    stats["max_xi0"] = float(np.max(xi_grid))

    spread = float(np.max(levels) - np.min(levels))
    scale = max(spread, 1e-3 * float(np.max(np.abs(levels))))
    band = float(config.oscillation_rel_tol) * scale
    n_data = material_turning_points(levels, band)
    n_model = material_turning_points(xi_grid, band)
    stats["n_turning_points_data"] = float(n_data)
    stats["n_turning_points_model"] = float(n_model)

    if stats["min_xi0"] <= float(config.eps_xi):
        return None, REJECT_POSITIVITY, stats
    if n_model > n_data:
        return None, REJECT_OSCILLATION, stats

    nodes = max(int(config.quadrature_nodes_per_interval), 3)
    if nodes % 2 == 0:
        nodes += 1
    total = 0.0
    max_err = 0.0
    for j in range(T.size):
        a, b = float(T_ext[j]), float(T_ext[j + 1])
        sub = np.linspace(a, b, nodes, dtype=np.float64)
        total += _simpson_uniform(np.asarray(d1(sub), dtype=np.float64), (b - a) / (nodes - 1))
        max_err = max(max_err, abs(total - float(V[j])))
    stats["quadrature_max_error"] = float(max_err)
    if max_err >= float(config.quadrature_tol):
        return None, REJECT_QUADRATURE, stats

    return interpolator, None, stats


def build_forward_variance_curve(
    variance_curve: Any,
    *,
    config: ForwardVarianceConfig | None = None,
) -> ForwardVarianceCurve:
    """
    Build the forward-variance curve ``xi0(T)`` from the spec-4.3 curve.

    Parameters
    ----------
    variance_curve:
        A :class:`~app.model.calibration.rough_vol.variance_swap.VarianceSwapCurve`
        (its refused expiries are ignored — they carry no number by design) or a
        bare sequence of
        :class:`~app.model.calibration.rough_vol.variance_swap.VarianceSwapPoint`.
    config:
        :class:`ForwardVarianceConfig`; the default is the exact-by-construction
        piecewise-constant build.

    Raises
    ------
    ValueError
        When no usable maturity is available, when a maturity or a ``K_var`` is
        not finite, when a maturity is not strictly positive, or when two points
        share the same maturity (which would make the forward-variance increment
        undefined). Refusing is deliberate: silently pooling or dropping
        duplicated maturities would hide a broken upstream chain grouping.
    """
    cfg = config or ForwardVarianceConfig()
    points = _extract_points(variance_curve)
    if not points:
        raise ValueError(
            "build_forward_variance_curve: aucune maturité exploitable dans la courbe de "
            "variance swap; impossible de construire xi0."
        )

    points = sorted(points, key=lambda p: float(p.T))
    T = np.asarray([float(p.T) for p in points], dtype=np.float64)
    k_var = np.asarray([float(p.k_var) for p in points], dtype=np.float64)
    if not np.all(np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError(
            f"build_forward_variance_curve: maturités invalides {T.tolist()!r} "
            "(elles doivent être finies et strictement positives)."
        )
    if np.any(np.diff(T) <= 0.0):
        raise ValueError(
            f"build_forward_variance_curve: maturités dupliquées ou non ordonnées {T.tolist()!r}; "
            "l'incrément de variance forward serait indéfini."
        )
    if not np.all(np.isfinite(k_var)):
        raise ValueError(
            f"build_forward_variance_curve: K_var non fini dans {k_var.tolist()!r}."
        )
    # A total variance V(0,T) = T*K_var must be strictly positive: it is the
    # integral of a positive instantaneous variance. The isotonic repair CANNOT
    # catch a negative V -- a negative-then-rising sequence is already
    # non-decreasing, so PAVA is a no-op -- and the piecewise-constant level on
    # the interval that follows, (V_1 - V_0)/(T_1 - T_0), then absorbs the whole
    # negative offset and comes out enormous (a measured case produced
    # xi0 = 12.58, i.e. 355 % instantaneous vol, with no flag at all). Refuse
    # here rather than propagate it into the calibration.
    if np.any(k_var < 0.0):
        bad = [
            f"T={float(T[j]):.6g}: K_var={float(k_var[j]):.6g}"
            for j in range(T.size)
            if float(k_var[j]) < 0.0
        ]
        raise ValueError(
            "build_forward_variance_curve: variance totale négative sur "
            f"{len(bad)} maturité(s) [{'; '.join(bad)}]; V(0,T) doit être "
            "strictement positive (intégrale d'une variance instantanée). "
            "Vérifier le forward de parité et les cotations en amont."
        )

    V_market = T * k_var

    # --- 1. calendar-arbitrage repair ------------------------------------
    V_repaired = pool_adjacent_violators(V_market)
    adjustments = tuple(
        IsotonicAdjustment(index=j, T=float(T[j]), v_raw=float(V_market[j]), v_fitted=float(V_repaired[j]))
        for j in range(T.size)
        # Scale-relative floor on top of the absolute knob: PAVA pooling a
        # genuinely FLAT segment rewrites it to its own mean, which differs from
        # the input by an ulp. Reporting that as a repaired calendar arbitrage
        # (with a French message saying so) is a false positive on pure
        # floating-point noise, so a move has to clear a few ulps of the local
        # variance scale to count as real.
        if abs(float(V_repaired[j]) - float(V_market[j]))
        > max(
            float(cfg.isotonic_log_tol),
            8.0 * np.finfo(np.float64).eps * max(abs(float(V_market[j])), 1e-12),
        )
    )

    flags: list[str] = []
    messages: list[str] = []
    if adjustments:
        flags.append(FLAG_ISOTONIC_REPAIR)
        worst = max(adjustments, key=lambda a: abs(a.delta))
        messages.append(
            f"Réparation isotone de la variance totale: {len(adjustments)} maturité(s) ajustée(s), "
            f"ajustement maximal {abs(worst.delta):.6g} en variance totale à T={worst.T:.6g} an(s) "
            f"(V {worst.v_raw:.6g} -> {worst.v_fitted:.6g}); la variance totale doit être "
            "non décroissante (arbitrage calendaire)."
        )
    if T.size == 1:
        flags.append(FLAG_SINGLE_MATURITY)
        messages.append(
            "Une seule maturité exploitable: xi0 est constant et l'extrapolation est plate "
            "sur tout l'horizon."
        )

    # --- 2. piecewise-constant levels + positivity floor -------------------
    levels = _levels_from_totals(T, V_repaired)
    floored = tuple(int(j) for j in np.nonzero(levels < float(cfg.eps_xi))[0])
    if floored:
        levels = np.maximum(levels, float(cfg.eps_xi))
        flags.append(FLAG_XI0_FLOORED)
        messages.append(
            f"{len(floored)} niveau(x) de variance forward plancheré(s) à eps_xi="
            f"{float(cfg.eps_xi):.3g}; la reconstruction int xi0 = V n'est plus exacte sur "
            "ces intervalles (l'écart est reporté dans reconstruction_errors)."
        )

    # --- 3. optional smooth variant ---------------------------------------
    requested = str(cfg.method)
    method = METHOD_PIECEWISE_CONSTANT
    interpolator: PchipInterpolator | None = None
    rejection_reasons: tuple[str, ...] = ()
    stats: dict[str, float] = {}

    if requested == METHOD_PCHIP:
        if floored:
            rejection_reasons = (REJECT_POSITIVITY,)
        else:
            interpolator, reason, stats = _try_pchip(T, V_repaired, levels, cfg)
            if reason is not None:
                rejection_reasons = (reason,)
        if interpolator is None:
            flags.append(FLAG_PCHIP_FALLBACK)
            messages.append(
                "Variante lisse (PCHIP monotone) rejetée ("
                + ", ".join(FORWARD_VARIANCE_LABELS_FR.get(r, r) for r in rejection_reasons)
                + "); repli sur la variante constante par morceaux, exacte par construction."
            )
        else:
            method = METHOD_PCHIP
    elif requested != METHOD_PIECEWISE_CONSTANT:
        raise ValueError(
            f"build_forward_variance_curve: méthode inconnue {requested!r} "
            f"(attendu {METHOD_PIECEWISE_CONSTANT!r} ou {METHOD_PCHIP!r})."
        )

    # --- 4. the curve's own total variance at the knots --------------------
    if method == METHOD_PCHIP:
        V_knots = V_repaired.copy()
    else:
        V_knots = np.empty(T.size, dtype=np.float64)
        previous_V = 0.0
        previous_T = 0.0
        for j in range(T.size):
            previous_V = previous_V + float(levels[j]) * (float(T[j]) - previous_T)
            previous_T = float(T[j])
            V_knots[j] = previous_V

    max_reconstruction_error = float(np.max(np.abs(V_knots - V_repaired))) if T.size else 0.0

    # `min_xi0` / `max_xi0` always describe the curve that was actually built,
    # never the rejected candidate; the rejected candidate survives through
    # `rejection_reasons` and `quadrature_max_error`.
    spread = float(np.max(levels) - np.min(levels))
    scale = max(spread, 1e-3 * float(np.max(np.abs(levels))))
    n_turning_data = material_turning_points(levels, float(cfg.oscillation_rel_tol) * scale)
    if method == METHOD_PCHIP:
        min_xi0 = float(stats.get("min_xi0", float("nan")))
        max_xi0 = float(stats.get("max_xi0", float("nan")))
        n_turning_model = int(stats.get("n_turning_points_model", 0))
    else:
        min_xi0 = float(np.min(levels))
        max_xi0 = float(np.max(levels))
        n_turning_model = n_turning_data

    metadata = ForwardVarianceMetadata(
        method=method,
        requested_method=requested,
        n_maturities=int(T.size),
        eps_xi=float(cfg.eps_xi),
        extrapolation_policy=EXTRAPOLATION_FLAT,
        isotonic_adjustments=adjustments,
        isotonic_max_abs_adjustment=float(
            max((abs(a.delta) for a in adjustments), default=0.0)
        ),
        floored_indices=floored,
        max_reconstruction_error=max_reconstruction_error,
        quadrature_max_error=float(stats.get("quadrature_max_error", float("nan"))),
        min_xi0=min_xi0,
        max_xi0=max_xi0,
        n_turning_points_data=int(n_turning_data),
        n_turning_points_model=int(n_turning_model),
        rejection_reasons=rejection_reasons,
        flags=tuple(flags),
        messages_fr=tuple(messages),
    )

    return ForwardVarianceCurve(
        T_knots=tuple(float(x) for x in T),
        levels=tuple(float(x) for x in levels),
        V_knots=tuple(float(x) for x in V_knots),
        V_market=tuple(float(x) for x in V_market),
        V_repaired=tuple(float(x) for x in V_repaired),
        metadata=metadata,
        pchip=interpolator,
    )


def forward_variance_report(curve: ForwardVarianceCurve) -> dict[str, Any]:
    """Plain-python summary for the Phase-5 report (survives ``_json_safe``)."""
    errors = curve.reconstruction_errors()
    return {
        "method": curve.method,
        "extrapolation_policy": curve.extrapolation_policy,
        "n_maturities": int(len(curve)),
        "T_knots": [float(x) for x in curve.T_knots],
        "levels": [float(x) for x in curve.levels],
        "xi0_vol": [float(math.sqrt(x)) if x > 0.0 else float("nan") for x in curve.levels],
        "V_market": [float(x) for x in curve.V_market],
        "V_repaired": [float(x) for x in curve.V_repaired],
        "max_abs_reconstruction_error": float(max((abs(e) for e in errors), default=0.0)),
        "metadata": curve.metadata.to_dict(),
    }


__all__ = [
    "EXTRAPOLATION_FLAT",
    "FLAG_ISOTONIC_REPAIR",
    "FLAG_PCHIP_FALLBACK",
    "FLAG_SINGLE_MATURITY",
    "FLAG_XI0_FLOORED",
    "FORWARD_VARIANCE_LABELS_FR",
    "METHOD_PCHIP",
    "METHOD_PIECEWISE_CONSTANT",
    "REJECT_OSCILLATION",
    "REJECT_POSITIVITY",
    "REJECT_QUADRATURE",
    "REJECT_TOO_FEW_MATURITIES",
    "ForwardVarianceConfig",
    "ForwardVarianceCurve",
    "ForwardVarianceMetadata",
    "IsotonicAdjustment",
    "build_forward_variance_curve",
    "forward_variance_report",
    "material_turning_points",
    "pool_adjacent_violators",
]
