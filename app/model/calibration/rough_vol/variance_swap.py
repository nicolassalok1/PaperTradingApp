"""
Variance-swap strike curve ``K_var(T)`` (spec 4.3).

Continuous reference formulation
--------------------------------
Demeterfi, Derman, Kamal & Zou (1999), *More Than You Ever Wanted To Know About
Volatility Swaps*. With a reference strike ``S*`` and **undiscounted** option
mids, the fair variance of a variance swap maturing at ``T`` is

.. code-block:: text

    K_var(T) = (2/T) * [ rT - (S0*e^{rT}/S* - 1) - ln(S*/S0) ]
             + (2*e^{rT}/T) * [ int_0^{S*} P(K,T)/K^2 dK
                              + int_{S*}^{inf} C(K,T)/K^2 dK ]

which for ``S* = F(T)`` (the put-call-parity forward of spec 4.2) collapses to

.. code-block:: text

    K_var(T) = (2*e^{rT}/T) * [ int_0^F P/K^2 dK + int_F^inf C/K^2 dK ].

Discrete implementation (CBOE-style)
------------------------------------
Quotes live on a finite, irregular strike ladder, so the pipeline evaluates the
CBOE VIX white-paper discretisation of the formula above on the **cleaned**
strike set of spec 4.1:

.. code-block:: text

    K_var(T) = (2/T) * sum_i (dK_i / K_i^2) * (1/D(T)) * Q(K_i, T)
             + (2/T) * [ ln(F/K_0) - (F/K_0 - 1) ]

* ``K_0`` is the **largest strike <= F** — not the nearest strike;
* ``Q(K_i)`` is the put mid for ``K_i < K_0``, the call mid for ``K_i > K_0``
  and the **average of the call and put mids at ``K_0``**;
* ``dK_i = (K_{i+1} - K_{i-1}) / 2``, one-sided at both edges
  (``dK_0 = K_1 - K_0``, ``dK_n = K_n - K_{n-1}``);
* ``1/D(T)`` is the reciprocal discount factor carried by the spec-4.2
  :class:`~app.model.calibration.rough_vol.forward_curve.ForwardPoint` — it comes
  from the repo yield curve, never from ``e^{rT}`` with a guessed flat ``r``;
* implied volatilities are **never averaged**: the sum runs on prices.

Why the correction term is *subtracted*, and why the EXACT form is used. Taking
``S* = K_0`` in the continuous formula and substituting ``F = S0 e^{rT}`` turns
the first bracket into ``(2/T) * [1 - F/K_0 + ln(F/K_0)]``. With
``x = F/K_0 - 1 >= 0`` (``K_0`` is the largest strike **at or below** ``F``, so
``x`` is never negative) this is ``(2/T) * [ln(1+x) - x]``, a negative quantity
that vanishes when ``K_0 == F``.

CBOE truncates it at second order as ``-(1/T) x^2``. This module uses the
**exact** closed form instead, at identical cost: expanding
``x - ln(1+x) = x^2/2 - x^3/3 + ...`` shows the CBOE term is *larger* than the
exact one, and since it is subtracted, CBOE biases ``K_var`` one-sidedly **low**
by ``(2/(3T)) x^3 + O(x^4)`` — a term that grows without bound as ``T -> 0``,
i.e. precisely at the short end that spec 4.5 regresses to obtain ``H``. The
CBOE value is still computed and reported as
:attr:`VarianceSwapDiagnostics.correction_term_cboe` for comparability.

Strike-grid discretisation bias (reported, not hidden)
------------------------------------------------------
The discrete replication portfolio interpolates the log payoff **piecewise
linearly** between listed strikes. For the convex payoff ``-2 ln(S/F)`` the
interpolant lies above the function, and integrating the interpolation error
``f''(K) (K - K_i)(K_{i+1} - K)/2`` against the risk-neutral density with
``f'' = 2/K^2`` gives ``f'' h^2 / 12`` per cell, i.e. a **total-variance excess**

.. code-block:: text

    V_bias = h^2 / (6 F^2)          (h = strike spacing at the money)

independent of ``T`` and of the volatility level. Measured against this law the
module reproduces it to a ratio of ``1.0000`` across ``h`` in ``[0.5, 4]``,
``F`` in ``[10, 5000]`` and ``sigma`` in ``[0.10, 0.80]``.

This matters here in a way it does not for CBOE. VIX is a *defined* index, so the
bias is part of its definition; this pipeline uses ``K_var`` as an *estimator* of
integrated variance feeding ``xi0(T)`` and ultimately ``H``. In ``K_var`` terms
the bias is ``h^2/(6 F^2 T)``, i.e. it **decays like 1/T** — on a $5 ladder over a
$100 underlying it is ``+54 %`` at 7 days and ``+0.5 %`` at 2 years. Being
constant in ``V``, it cancels in every forward-variance increment **except the
first**, so it lands entirely on ``xi0`` over ``[0, T_1]``.

The value returned stays the faithful CBOE-style sum (spec 4.3 mandates that
form). The bias is quantified in
:attr:`VarianceSwapDiagnostics.discretisation_bias`, expressed relative to
``K_var`` in ``discretisation_bias_rel``, and raises
:data:`FLAG_COARSE_STRIKE_LADDER` past
:attr:`VarianceSwapConfig.discretisation_warn_rel`. Downstream stages can then
subtract it or drop the flagged maturity — what they must not do is consume the
number unaware.

Finite strike truncation
------------------------
Two numbers are produced for every expiry:

``k_var_trunc``
    the sum above on the **quoted** strikes only;
``k_var``
    the same sum plus the two missing tails, completed by **flat-IV
    extrapolation** of the last reliable OTM quote on each side. The anchor IV is
    inverted with spec 4.2's :func:`implied_vol_from_forward`; Black-76 prices at
    that flat IV are then integrated in ``K`` outwards until the integrand
    ``Q/K^2`` falls below :attr:`VarianceSwapConfig.eps_tail` (or the hard
    ``tail_n_std`` cap binds, whichever comes first).

``k_var - k_var_trunc >= 0`` is the **truncation-error estimate** and is attached
to every point. The pipeline default is ``k_var`` (tail-completed); when no
reliable OTM anchor exists on a side, that side contributes nothing and the point
is flagged.

Quadrature of the tails. The tail integrand is a Black-76 price at a *constant*
volatility divided by ``K^2``: it is C-infinity on the tail interval, has no
kink and no singularity, and the interval is bounded by the ``eps_tail`` /
``tail_n_std`` stop. **Composite Simpson on a uniform grid in K** is therefore
used: its ``O(h^4)`` error sits many orders of magnitude below the tail
contribution itself, it needs no adaptive branching (so the result is
bit-reproducible), and it costs one vectorised Black-76 evaluation. The tail
interval starts at the *cell edge* of the discrete sum — ``K_min - dK_first/2``
below and ``K_max + dK_last/2`` above — because the CBOE weights already cover
``[K_min - dK_first/2, K_max + dK_last/2]``; starting at the quoted strikes
themselves would double-count the two edge half-cells.

Deep put tails are priced by exact put-call parity off
:func:`~app.model.calibration.rough_vol.forward_curve.black76_call_price`
(``P = C - D (F - K)``) rather than by a second pricer. That subtraction cancels
catastrophically once ``P`` is tiny, with an absolute floor of about
``eps_machine * D * F``; the ``eps_tail`` stop is reached long before that floor
matters (at the stop the relative error is ``~eps_machine*F/(eps_tail*K^2)``,
i.e. ``4e-8`` on a 100-priced underlying), and any residual negative noise is
clipped at zero and counted in the diagnostics.

Refusals (never a silent number)
--------------------------------
An expiry produces a :class:`VarianceSwapFailure` — with a French message — and
**no** number when:

* spec 4.1's minimum viability fails (``chain.viability.usable_for_kvar`` False);
* fewer than :attr:`VarianceSwapConfig.min_usable_strikes` strikes survive the
  ladder construction (the spec's "<= 2 usable strikes" degenerate case);
* no strike sits at or below ``F`` (``K_0`` undefined);
* the maturity, forward or discount factor is not finite and strictly positive;
* no spec-4.2 forward point matches the expiry;
* the supplied forward point carries a different maturity than the chain;
* ``K_var`` comes out negative or non-finite — a fair variance cannot be
  negative, and handing such a number to spec 4.4 would manufacture an absurd
  ``xi0`` on the following interval (a measured case produced 355 % vol).

Bid/ask filtering, stale quotes, zero or negative prices, the CBOE zero-bid wall
and the vertical / butterfly arbitrage repairs are all **inherited from spec
4.1** and are not repeated here.

Missing strikes and irregular spacing need no special code: they are absorbed by
``dK_i`` by construction, which is exactly why the CBOE weights are used.

Layering: model layer only, no side effects, no network. Discounting is either
taken from the injected forward points or resolved through spec 4.2's
:func:`resolve_discount`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from app.model.calibration.rough_vol.chain_cleaning import (
    CALL,
    PUT,
    CleanChain,
    CleanQuote,
    ViabilityReport,
    _expiry_to_str,
    evaluate_viability,
)
from app.model.calibration.rough_vol.forward_curve import (
    ForwardPoint,
    black76_call_price,
    implied_vol_from_forward,
    resolve_discount,
)

# --- refusal reason codes ---------------------------------------------------
REASON_VIABILITY_FAILED = "viability_failed"
REASON_TOO_FEW_STRIKES = "too_few_usable_strikes"
REASON_NO_STRIKE_BELOW_FORWARD = "no_strike_below_forward"
REASON_INVALID_MATURITY = "invalid_maturity"
REASON_INVALID_FORWARD = "invalid_forward"
REASON_INVALID_DISCOUNT = "invalid_discount"
REASON_NO_FORWARD_POINT = "no_forward_point"
REASON_NEGATIVE_K_VAR = "negative_k_var"
REASON_FORWARD_POINT_MISMATCH = "forward_point_mismatch"

#: French user-facing labels for the Phase-5 report.
VARIANCE_REASON_LABELS_FR: dict[str, str] = {
    REASON_VIABILITY_FAILED: "Échéance non exploitable pour K_var (viabilité 4.1 insuffisante)",
    REASON_TOO_FEW_STRIKES: "Trop peu de strikes exploitables pour intégrer K_var",
    REASON_NO_STRIKE_BELOW_FORWARD: "Aucun strike au niveau ou en dessous du forward",
    REASON_INVALID_MATURITY: "Maturité invalide",
    REASON_INVALID_FORWARD: "Forward invalide",
    REASON_INVALID_DISCOUNT: "Facteur d'actualisation invalide",
    REASON_NO_FORWARD_POINT: "Aucun point de courbe forward pour cette échéance",
    REASON_NEGATIVE_K_VAR: "K_var négatif (non physique): échéance refusée",
    REASON_FORWARD_POINT_MISMATCH: "Le point forward ne correspond pas à l'échéance de la chaîne",
}

# --- diagnostic flags -------------------------------------------------------
FLAG_NO_PUT_TAIL_ANCHOR = "no_put_tail_anchor"
FLAG_NO_CALL_TAIL_ANCHOR = "no_call_tail_anchor"
FLAG_TAIL_CAP_REACHED = "tail_extrapolation_cap_reached"
FLAG_K0_SINGLE_SIDED = "k0_single_sided"
FLAG_STRIKES_SKIPPED = "strikes_skipped"
FLAG_IRREGULAR_SPACING = "irregular_strike_spacing"
FLAG_LARGE_TRUNCATION = "large_truncation_error"
FLAG_COARSE_STRIKE_LADDER = "coarse_strike_ladder"
FLAG_NEGATIVE_K_VAR = "negative_k_var"
FLAG_VIABILITY_LOST_ON_FORWARD = "viability_lost_on_parity_forward"
FLAG_NEGATIVE_TAIL_INTEGRAND_CLIPPED = "negative_tail_integrand_clipped"

#: Why the outward tail walk stopped.
TAIL_STOP_INTEGRAND = "integrand_below_eps"
TAIL_STOP_CAP = "n_std_cap"
TAIL_STOP_MAX_STEPS = "max_steps"
TAIL_STOP_NO_ANCHOR = "no_anchor"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceSwapConfig:
    """
    Every numerical threshold of spec 4.3, in one frozen place.

    Attributes
    ----------
    min_usable_strikes:
        Minimum number of strikes surviving the ``Q`` ladder. The spec's
        degenerate case is "<= 2 usable strikes", hence the default ``3``. Below
        it, ``dK_i`` cannot even be formed (the interior formula needs a left and
        a right neighbour) and the expiry is refused instead of returning a
        number.
    eps_tail:
        Stop threshold on the tail integrand ``Q(K)/K^2``. This is the spec's
        literal criterion and is therefore an **absolute** threshold, so it is
        scale dependent: on a very high-priced underlying the ATM integrand is
        itself small and the threshold binds sooner. The stop reason actually
        used is always reported in the diagnostics, so the interplay with
        ``tail_n_std`` is never hidden.
    tail_n_std:
        Hard cap on the outward walk, expressed in standard deviations of the
        extrapolation log-normal: the walk never leaves
        ``[F e^{-n sigma sqrt(T)}, F e^{+n sigma sqrt(T)}]``. Scale free, unlike
        ``eps_tail``, and it guarantees termination.
    tail_step_log:
        Geometric step of the outward walk, in log-strike.
    tail_max_steps:
        Belt-and-braces bound on the walk (never reached with the cap above).
    tail_nodes:
        Number of Simpson nodes per tail. Forced to the next odd integer.
    tail_anchor_two_sided_only:
        Only use a genuinely two-way quote (``bid > 0``) as the flat-IV anchor.
        A one-sided quote has no reliable mid; using it would extrapolate the
        whole tail off a half-tick ask.
    min_tail_vol / max_tail_vol:
        Acceptance window for the anchor IV. Outside it the inversion is treated
        as failed and the next quote inward is tried. Mirrors the bracket of
        ``implied_vol_call``.
    truncation_warn_rel:
        Relative truncation error ``(k_var - k_var_trunc) / k_var`` above which
        :data:`FLAG_LARGE_TRUNCATION` is raised. Diagnostic only.
    discretisation_warn_rel:
        Flag :data:`FLAG_COARSE_STRIKE_LADDER` when the closed-form strike-grid
        discretisation bias ``h_atm^2/(6*F^2*T)`` exceeds this fraction of
        ``K_var``. Default 2 %. The bias decays like ``1/T``, so on a coarse
        ladder it fires at the short end only -- which is exactly the end that
        determines ``H`` (spec 4.5), hence a flag rather than silence.
    irregular_spacing_ratio:
        ``max(dK)/min(dK)`` above which :data:`FLAG_IRREGULAR_SPACING` is raised.
        Diagnostic only — irregular spacing is *handled* by ``dK_i``, it is not
        an error.
    """

    min_usable_strikes: int = 3
    eps_tail: float = 1e-10
    tail_n_std: float = 12.0
    tail_step_log: float = 0.02
    tail_max_steps: int = 4000
    tail_nodes: int = 2001
    tail_anchor_two_sided_only: bool = True
    min_tail_vol: float = 1e-4
    max_tail_vol: float = 5.0
    truncation_warn_rel: float = 0.05
    irregular_spacing_ratio: float = 3.0
    discretisation_warn_rel: float = 0.02


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailEstimate:
    """One completed tail: the flat-IV anchor, the integral and how it stopped."""

    side: str
    anchor_strike: float = float("nan")
    anchor_mid: float = float("nan")
    anchor_vol: float = float("nan")
    edge_strike: float = float("nan")
    stop_strike: float = float("nan")
    stop_reason: str = TAIL_STOP_NO_ANCHOR
    integral: float = 0.0
    n_nodes: int = 0
    n_clipped: int = 0

    @property
    def available(self) -> bool:
        return math.isfinite(self.anchor_vol) and self.anchor_vol > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": str(self.side),
            "anchor_strike": float(self.anchor_strike),
            "anchor_mid": float(self.anchor_mid),
            "anchor_vol": float(self.anchor_vol),
            "edge_strike": float(self.edge_strike),
            "stop_strike": float(self.stop_strike),
            "stop_reason": str(self.stop_reason),
            "integral": float(self.integral),
            "n_nodes": int(self.n_nodes),
            "n_clipped": int(self.n_clipped),
            "available": bool(self.available),
        }


@dataclass(frozen=True)
class VarianceSwapDiagnostics:
    """
    Everything needed to audit one ``K_var(T)`` without re-running the pipeline.

    ``correction_term`` is the quantity that was **subtracted**, i.e.
    ``(1/T) (F/K_0 - 1)^2``; ``raw_integral`` is ``sum_i dK_i Q_i / K_i^2``
    *before* the ``1/D`` and ``2/T`` factors.
    """

    n_strikes: int = 0
    K_0: float = float("nan")
    K_min: float = float("nan")
    K_max: float = float("nan")
    log_moneyness_min: float = float("nan")
    log_moneyness_max: float = float("nan")
    forward_over_k0: float = float("nan")
    correction_term: float = float("nan")
    correction_term_cboe: float = float("nan")
    discretisation_bias: float = float("nan")
    discretisation_bias_rel: float = float("nan")
    h_atm: float = float("nan")
    raw_integral: float = float("nan")
    k_var_trunc: float = float("nan")
    k_var_tail: float = float("nan")
    truncation_error: float = float("nan")
    truncation_error_rel: float = float("nan")
    dk_min: float = float("nan")
    dk_max: float = float("nan")
    dk_ratio: float = float("nan")
    n_strikes_skipped: int = 0
    skipped_strikes: tuple[float, ...] = ()
    put_tail: TailEstimate | None = None
    call_tail: TailEstimate | None = None
    viability_on_forward: ViabilityReport | None = None
    flags: tuple[str, ...] = ()
    messages_fr: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_strikes": int(self.n_strikes),
            "K_0": float(self.K_0),
            "K_min": float(self.K_min),
            "K_max": float(self.K_max),
            "log_moneyness_min": float(self.log_moneyness_min),
            "log_moneyness_max": float(self.log_moneyness_max),
            "forward_over_k0": float(self.forward_over_k0),
            "correction_term": float(self.correction_term),
            "correction_term_cboe": float(self.correction_term_cboe),
            "discretisation_bias": float(self.discretisation_bias),
            "discretisation_bias_rel": float(self.discretisation_bias_rel),
            "h_atm": float(self.h_atm),
            "raw_integral": float(self.raw_integral),
            "k_var_trunc": float(self.k_var_trunc),
            "k_var_tail": float(self.k_var_tail),
            "truncation_error": float(self.truncation_error),
            "truncation_error_rel": float(self.truncation_error_rel),
            "dk_min": float(self.dk_min),
            "dk_max": float(self.dk_max),
            "dk_ratio": float(self.dk_ratio),
            "n_strikes_skipped": int(self.n_strikes_skipped),
            "skipped_strikes": [float(k) for k in self.skipped_strikes],
            "put_tail": None if self.put_tail is None else self.put_tail.to_dict(),
            "call_tail": None if self.call_tail is None else self.call_tail.to_dict(),
            "viability_on_forward": (
                None if self.viability_on_forward is None else self.viability_on_forward.to_dict()
            ),
            "flags": [str(f) for f in self.flags],
            "messages_fr": [str(m) for m in self.messages_fr],
        }


@dataclass(frozen=True)
class VarianceSwapPoint:
    """
    One usable expiry of the variance-swap curve (spec 4.3 output contract).

    ``k_var`` is the **tail-completed** strike (the pipeline default);
    ``k_var_trunc`` is the quoted-strikes-only strike. Their difference is the
    truncation-error estimate, also mirrored in
    :attr:`VarianceSwapDiagnostics.truncation_error`.

    ``n_puts`` / ``n_calls`` count the quotes that actually entered the sum; the
    ``K_0`` strike contributes to both when it is quoted on both sides.
    """

    T: float
    k_var: float
    k_var_trunc: float
    n_puts: int
    n_calls: int
    F: float
    D: float
    diagnostics: VarianceSwapDiagnostics = field(default_factory=VarianceSwapDiagnostics)
    expiry: Any = None

    @property
    def truncation_error(self) -> float:
        """``k_var - k_var_trunc >= 0`` — the missing-tail estimate."""
        return float(self.k_var) - float(self.k_var_trunc)

    @property
    def total_variance(self) -> float:
        """``V(0,T) = T * K_var(T)``, the quantity spec 4.4 integrates."""
        return float(self.T) * float(self.k_var)

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(self.diagnostics.flags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "k_var": float(self.k_var),
            "k_var_trunc": float(self.k_var_trunc),
            "vol_var_swap": float(math.sqrt(self.k_var)) if self.k_var >= 0.0 else float("nan"),
            "total_variance": float(self.total_variance),
            "truncation_error": float(self.truncation_error),
            "n_puts": int(self.n_puts),
            "n_calls": int(self.n_calls),
            "F": float(self.F),
            "D": float(self.D),
            "expiry": _expiry_to_str(self.expiry),
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class VarianceSwapFailure:
    """A refused expiry: a reason code, a French message and what was known."""

    T: float
    reason: str
    message_fr: str
    expiry: Any = None
    detail: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "reason": str(self.reason),
            "reason_fr": VARIANCE_REASON_LABELS_FR.get(self.reason, self.reason),
            "message_fr": str(self.message_fr),
            "expiry": _expiry_to_str(self.expiry),
            "detail": {str(k): float(v) for k, v in dict(self.detail).items()},
        }


@dataclass(frozen=True)
class VarianceSwapCurve:
    """
    The whole ``K_var`` term structure: usable points plus refused expiries.

    Points are sorted by increasing maturity. Refusals are kept — an empty
    ``failures`` tuple is a *claim* that every expiry was usable, so it must not
    be reconstructible only by counting.
    """

    points: tuple[VarianceSwapPoint, ...] = ()
    failures: tuple[VarianceSwapFailure, ...] = ()

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, item: int) -> VarianceSwapPoint:
        return self.points[item]

    @property
    def maturities(self) -> tuple[float, ...]:
        return tuple(float(p.T) for p in self.points)

    @property
    def k_var(self) -> tuple[float, ...]:
        return tuple(float(p.k_var) for p in self.points)

    @property
    def total_variances(self) -> tuple[float, ...]:
        """``V(0,T_j) = T_j K_var(T_j)`` — the input of spec 4.4."""
        return tuple(float(p.total_variance) for p in self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "failures": [f.to_dict() for f in self.failures],
        }


# ---------------------------------------------------------------------------
# Quadrature and pricing helpers
# ---------------------------------------------------------------------------


def _simpson_uniform(y: np.ndarray, h: float) -> float:
    """
    Composite Simpson on a uniform grid of ``2m+1`` nodes spaced by ``h``.

    Written out rather than pulled from scipy so the quadrature used by the tail
    completion is visible next to the formula it serves and stays bit-stable
    across scipy releases.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3 or n % 2 == 0:
        raise ValueError(f"_simpson_uniform: nombre de noeuds invalide n={n}")
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def black76_put_price(*, F: float, K: float, T: float, D: float, vol: float) -> float:
    """
    Black-76 put through **exact** put-call parity on spec 4.2's call pricer.

    ``P = C - D (F - K)``. No second pricer is introduced; see the module
    docstring for the cancellation floor of this identity deep in the put wing.
    """
    call = black76_call_price(F=F, K=K, T=T, D=D, vol=vol)
    if not math.isfinite(call):
        return float("nan")
    return float(call - float(D) * (float(F) - float(K)))


def _tail_integrand(
    strikes: np.ndarray, *, F: float, T: float, D: float, vol: float, side: str
) -> tuple[np.ndarray, int]:
    """``Q(K)/K^2`` on ``strikes`` at the flat extrapolation vol, clipped at zero."""
    prices = np.empty(strikes.size, dtype=np.float64)
    if side == PUT:
        for i, K in enumerate(strikes):
            prices[i] = black76_put_price(F=F, K=float(K), T=T, D=D, vol=vol)
    else:
        for i, K in enumerate(strikes):
            prices[i] = black76_call_price(F=F, K=float(K), T=T, D=D, vol=vol)
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    n_clipped = int(np.count_nonzero(prices < 0.0))
    np.maximum(prices, 0.0, out=prices)
    return prices / (strikes * strikes), n_clipped


def _find_tail_anchor(
    quotes: Sequence[CleanQuote],
    *,
    side: str,
    F: float,
    T: float,
    D: float,
    config: VarianceSwapConfig,
) -> tuple[float, float, float]:
    """
    Last *reliable* OTM quote on one side, with its flat implied volatility.

    Walks inward from the extreme OTM strike and returns the first quote that is
    OTM, two-way (unless disabled) and whose Black-76 inversion lands strictly
    inside ``[min_tail_vol, max_tail_vol]``. Returns ``(nan, nan, nan)`` when no
    such quote exists — the caller then leaves that tail empty and flags it.
    """
    if side == PUT:
        candidates = [q for q in quotes if q.strike < F]
        candidates.sort(key=lambda q: q.strike)  # deepest OTM first
    else:
        candidates = [q for q in quotes if q.strike > F]
        candidates.sort(key=lambda q: -q.strike)  # deepest OTM first

    for quote in candidates:
        if config.tail_anchor_two_sided_only and quote.one_sided:
            continue
        if not (math.isfinite(quote.mid) and quote.mid > 0.0):
            continue
        vol = implied_vol_from_forward(
            quote.mid, K=quote.strike, T=T, F=F, D=D, option_type=side
        )
        if not math.isfinite(vol):
            continue
        if not (config.min_tail_vol < vol < config.max_tail_vol):
            continue
        return float(quote.strike), float(quote.mid), float(vol)
    return float("nan"), float("nan"), float("nan")


def _complete_tail(
    *,
    side: str,
    quotes: Sequence[CleanQuote],
    F: float,
    T: float,
    D: float,
    edge_strike: float,
    config: VarianceSwapConfig,
) -> TailEstimate:
    """
    ``int Q(K)/K^2 dK`` over the missing wing, at the anchor's flat IV.

    The interval starts at ``edge_strike`` (the outer edge of the discrete sum's
    edge cell) and walks outward geometrically until the integrand drops below
    ``eps_tail`` or the ``tail_n_std`` cap binds; the integral is then evaluated
    by composite Simpson on a uniform grid over that interval.
    """
    anchor_K, anchor_mid, vol = _find_tail_anchor(
        quotes, side=side, F=F, T=T, D=D, config=config
    )
    if not math.isfinite(vol):
        return TailEstimate(
            side=side,
            edge_strike=float(edge_strike),
            stop_reason=TAIL_STOP_NO_ANCHOR,
            integral=0.0,
        )

    sd = float(vol) * math.sqrt(float(T))
    sign = -1.0 if side == PUT else 1.0
    cap = float(F) * math.exp(sign * float(config.tail_n_std) * sd)
    step = math.exp(sign * float(config.tail_step_log))

    def integrand_at(K: float) -> float:
        values, _ = _tail_integrand(
            np.array([K], dtype=np.float64), F=F, T=T, D=D, vol=vol, side=side
        )
        return float(values[0])

    K = float(edge_strike)
    stop_reason = TAIL_STOP_MAX_STEPS
    if integrand_at(K) < float(config.eps_tail):
        # The wing is already numerically exhausted at the cell edge.
        return TailEstimate(
            side=side,
            anchor_strike=anchor_K,
            anchor_mid=anchor_mid,
            anchor_vol=vol,
            edge_strike=float(edge_strike),
            stop_strike=float(edge_strike),
            stop_reason=TAIL_STOP_INTEGRAND,
            integral=0.0,
        )

    for _ in range(int(config.tail_max_steps)):
        K_next = K * step
        if (side == PUT and K_next <= cap) or (side == CALL and K_next >= cap):
            K = cap
            stop_reason = TAIL_STOP_CAP
            break
        K = K_next
        if integrand_at(K) < float(config.eps_tail):
            stop_reason = TAIL_STOP_INTEGRAND
            break

    n_nodes = int(config.tail_nodes)
    if n_nodes % 2 == 0:
        n_nodes += 1
    n_nodes = max(n_nodes, 3)

    lo, hi = (K, float(edge_strike)) if side == PUT else (float(edge_strike), K)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo > 0.0):
        return TailEstimate(
            side=side,
            anchor_strike=anchor_K,
            anchor_mid=anchor_mid,
            anchor_vol=vol,
            edge_strike=float(edge_strike),
            stop_strike=float(K),
            stop_reason=stop_reason,
            integral=0.0,
        )

    grid = np.linspace(lo, hi, n_nodes, dtype=np.float64)
    values, n_clipped = _tail_integrand(grid, F=F, T=T, D=D, vol=vol, side=side)
    integral = _simpson_uniform(values, (hi - lo) / (n_nodes - 1))

    return TailEstimate(
        side=side,
        anchor_strike=anchor_K,
        anchor_mid=anchor_mid,
        anchor_vol=vol,
        edge_strike=float(edge_strike),
        stop_strike=float(K),
        stop_reason=stop_reason,
        integral=float(max(integral, 0.0)),
        n_nodes=n_nodes,
        n_clipped=n_clipped,
    )


# ---------------------------------------------------------------------------
# The Q ladder
# ---------------------------------------------------------------------------


def _delta_k(strikes: np.ndarray) -> np.ndarray:
    """CBOE ``dK_i = (K_{i+1} - K_{i-1})/2``, one-sided at both edges."""
    n = strikes.size
    if n < 3:
        raise ValueError(f"_delta_k: au moins 3 strikes requis, reçu {n}")
    dK = np.empty(n, dtype=np.float64)
    dK[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
    dK[0] = strikes[1] - strikes[0]
    dK[-1] = strikes[-1] - strikes[-2]
    return dK


def _quote_map(quotes: Sequence[CleanQuote]) -> dict[float, CleanQuote]:
    """Strike -> quote, keeping the tightest-spread quote on a duplicate strike."""
    out: dict[float, CleanQuote] = {}
    for quote in quotes:
        K = float(quote.strike)
        if not (math.isfinite(K) and K > 0.0):
            continue
        if not (math.isfinite(quote.mid) and quote.mid > 0.0):
            continue
        current = out.get(K)
        if current is None or float(quote.spread_rel) < float(current.spread_rel):
            out[K] = quote
    return out


def _build_q_ladder(
    chain: CleanChain, *, F: float, D: float
) -> tuple[np.ndarray, np.ndarray, float, int, int, list[float], bool]:
    """
    Build the sorted, de-duplicated ``(K_i, Q_i)`` ladder of the CBOE sum.

    Returns ``(strikes, Q, K_0, n_puts, n_calls, skipped, k0_single_sided)``.
    ``Q_i`` is the put mid below ``K_0``, the call mid above it, and the average
    of the two at ``K_0``. A strike whose required leg is missing is dropped and
    listed in ``skipped`` — that is precisely the "missing strikes" case, and the
    ``dK_i`` recomputed on the survivors absorbs it.

    When ``K_0`` is quoted on only **one** side the average is still recovered
    **exactly**, from put-call parity rather than from the lone mid: since
    ``C(K_0) - P(K_0) = D * (F - K_0)`` holds identically,

    .. code-block:: text

        (C + P)/2 = C - D*(F - K_0)/2      (call quoted alone)
        (C + P)/2 = P + D*(F - K_0)/2      (put quoted alone)

    Using the lone mid instead offsets ``Q(K_0)`` by ``+/- D*(F-K_0)/2``, which
    the ``2/(T*D)`` prefactor turns into a ``1/T``-decaying bias in ``K_var`` --
    measured at ``+12 %`` on a 7-day expiry with ``F - K_0 = 2.40`` on a $2.50
    ladder, i.e. the same maturity-dependent class of defect as the strike-grid
    discretisation bias. The point stays flagged :data:`FLAG_K0_SINGLE_SIDED`
    because a one-sided pivot remains a data-quality signal once corrected.
    """
    calls = _quote_map(chain.calls)
    puts = _quote_map(chain.puts)
    all_strikes = sorted(set(calls) | set(puts))

    below = [K for K in all_strikes if K <= F]
    if not below:
        raise _NoStrikeBelowForward(float(F))
    K0 = float(max(below))

    strikes: list[float] = []
    q_values: list[float] = []
    skipped: list[float] = []
    n_puts = 0
    n_calls = 0
    k0_single_sided = False

    for K in all_strikes:
        if K < K0:
            quote = puts.get(K)
            if quote is None:
                skipped.append(float(K))
                continue
            strikes.append(float(K))
            q_values.append(float(quote.mid))
            n_puts += 1
        elif K > K0:
            quote = calls.get(K)
            if quote is None:
                skipped.append(float(K))
                continue
            strikes.append(float(K))
            q_values.append(float(quote.mid))
            n_calls += 1
        else:
            call_q = calls.get(K)
            put_q = puts.get(K)
            if call_q is not None and put_q is not None:
                value = 0.5 * (float(call_q.mid) + float(put_q.mid))
                n_calls += 1
                n_puts += 1
            elif call_q is not None:
                # parity-implied put: P = C - D*(F-K0) => (C+P)/2 = C - D*(F-K0)/2
                value = float(call_q.mid) - 0.5 * float(D) * (float(F) - float(K))
                n_calls += 1
                k0_single_sided = True
            elif put_q is not None:
                # parity-implied call: C = P + D*(F-K0) => (C+P)/2 = P + D*(F-K0)/2
                value = float(put_q.mid) + 0.5 * float(D) * (float(F) - float(K))
                n_puts += 1
                k0_single_sided = True
            else:  # pragma: no cover - K0 comes from the union, so one leg exists
                skipped.append(float(K))
                continue
            strikes.append(float(K))
            q_values.append(value)

    return (
        np.asarray(strikes, dtype=np.float64),
        np.asarray(q_values, dtype=np.float64),
        K0,
        n_puts,
        n_calls,
        skipped,
        k0_single_sided,
    )


class _NoStrikeBelowForward(Exception):
    """Internal signal: ``K_0`` is undefined because every strike sits above F."""

    def __init__(self, F: float) -> None:
        super().__init__(f"aucun strike <= F={F!r}")
        self.F = float(F)


# ---------------------------------------------------------------------------
# Per-expiry builder
# ---------------------------------------------------------------------------


def build_variance_swap_point(
    chain: CleanChain,
    point: ForwardPoint,
    *,
    D: float | None = None,
    config: VarianceSwapConfig | None = None,
) -> VarianceSwapPoint | VarianceSwapFailure:
    """
    ``K_var(T)`` for a single cleaned expiry.

    Returns a :class:`VarianceSwapPoint` when the expiry is usable, and a
    :class:`VarianceSwapFailure` carrying a **French** message otherwise. It
    never returns a number for a refused expiry.

    ``D`` overrides the discount factor of ``point`` (see
    :func:`build_variance_swap_curve` for how ``rates`` feeds it).
    """
    cfg = config or VarianceSwapConfig()
    T = float(chain.T)

    # The maturity is taken from the chain but F and D come from the point, so a
    # mispaired (chain, point) -- which the documented pair-input path of
    # build_variance_swap_curve accepts verbatim -- would silently produce a
    # perfectly finite, completely wrong K_var. Same 1e-12 tolerance as
    # _match_forward_point uses when it does the pairing itself.
    T_point = float(getattr(point, "T", T))
    if math.isfinite(T_point) and abs(T_point - T) > 1e-12:
        return VarianceSwapFailure(
            T=T,
            reason=REASON_FORWARD_POINT_MISMATCH,
            message_fr=(
                f"Incohérence échéance/forward: la chaîne porte T={T:.6g} an(s) mais le point "
                f"forward porte T={T_point:.6g} an(s). K_var mélangerait deux échéances."
            ),
            detail={"T_chain": T, "T_point": T_point},
        )
    expiry = chain.expiry

    if not (math.isfinite(T) and T > 0.0):
        return VarianceSwapFailure(
            T=T,
            reason=REASON_INVALID_MATURITY,
            message_fr=f"K_var: maturité invalide (T={T!r}); échéance ignorée.",
            expiry=expiry,
        )

    F = float(point.F)
    if not (math.isfinite(F) and F > 0.0):
        return VarianceSwapFailure(
            T=T,
            reason=REASON_INVALID_FORWARD,
            message_fr=f"K_var: forward invalide (F={F!r}) à T={T:.6g} an(s); échéance ignorée.",
            expiry=expiry,
            detail={"F": F},
        )

    D_val = float(point.D) if D is None else float(D)
    if not (math.isfinite(D_val) and D_val > 0.0):
        return VarianceSwapFailure(
            T=T,
            reason=REASON_INVALID_DISCOUNT,
            message_fr=(
                f"K_var: facteur d'actualisation invalide (D={D_val!r}) à T={T:.6g} an(s); "
                "échéance ignorée."
            ),
            expiry=expiry,
            detail={"D": D_val},
        )

    if not bool(chain.viability.usable_for_kvar):
        reasons = ", ".join(chain.viability.reasons) or "critère de viabilité 4.1 non satisfait"
        return VarianceSwapFailure(
            T=T,
            reason=REASON_VIABILITY_FAILED,
            message_fr=(
                f"K_var refusé à T={T:.6g} an(s): l'échéance ne satisfait pas la viabilité "
                f"minimale du nettoyage 4.1 ({reasons}); "
                f"{chain.viability.n_otm_puts} put(s) OTM et "
                f"{chain.viability.n_otm_calls} call(s) OTM retenus."
            ),
            expiry=expiry,
            detail={
                "n_otm_calls": float(chain.viability.n_otm_calls),
                "n_otm_puts": float(chain.viability.n_otm_puts),
            },
        )

    try:
        strikes, q_values, K0, n_puts, n_calls, skipped, k0_single_sided = _build_q_ladder(
            chain, F=F, D=D_val
        )
    except _NoStrikeBelowForward:
        return VarianceSwapFailure(
            T=T,
            reason=REASON_NO_STRIKE_BELOW_FORWARD,
            message_fr=(
                f"K_var refusé à T={T:.6g} an(s): aucun strike coté au niveau ou en dessous du "
                f"forward F={F:.6g}; le strike pivot K_0 est indéfini."
            ),
            expiry=expiry,
            detail={"F": F},
        )

    n_strikes = int(strikes.size)
    if n_strikes < int(cfg.min_usable_strikes):
        return VarianceSwapFailure(
            T=T,
            reason=REASON_TOO_FEW_STRIKES,
            message_fr=(
                f"K_var refusé à T={T:.6g} an(s): seulement {n_strikes} strike(s) exploitable(s) "
                f"(minimum {int(cfg.min_usable_strikes)}); l'intégrale de réplication est "
                "dégénérée, aucune valeur n'est produite."
            ),
            expiry=expiry,
            detail={"n_strikes": float(n_strikes)},
        )

    # --- discrete CBOE sum, float64 --------------------------------------
    dK = _delta_k(strikes)
    summand = dK * q_values / (strikes * strikes)
    raw_integral = float(np.sum(summand, dtype=np.float64))
    # Exact Demeterfi-Derman-Kamal-Zou first bracket at S* = K0, rather than CBOE's
    # second-order truncation -(1/T)*x^2. Same cost, and the neglected O(x^3)/T term
    # blows up as T -> 0. With x = F/K0 - 1 >= 0 (K0 is the largest strike <= F) the
    # CBOE form is biased one-sidedly low; the exact form removes that entirely.
    x_k0 = F / K0 - 1.0
    correction_term_exact = -(2.0 / T) * (math.log(F / K0) - x_k0)
    correction_term_cboe = (1.0 / T) * x_k0 * x_k0
    correction_term = correction_term_exact
    k_var_trunc = (2.0 / T) * (raw_integral / D_val) - correction_term

    # --- strike-grid discretisation bias (closed form, reported not hidden) ---
    # The discrete replication portfolio interpolates the log payoff PIECEWISE
    # LINEARLY between listed strikes. For the convex payoff -2*ln(S/F) that
    # over-replicates by  int f'' * h^2/12  with f'' = 2/K^2, i.e. by
    #     V_bias = h^2 / (6 * F^2)          (total variance, per spec 4.4's V = T*K_var)
    # concentrated where the risk-neutral mass sits, hence h taken at the money.
    # Verified numerically: ratio to this law is 1.0000 across h in [0.5, 4],
    # F in [10, 5000] and sigma in [0.10, 0.80] -- it is exact to leading order,
    # independent of T and of the vol level.
    #
    # Why it matters here and not for CBOE: VIX is a *defined* index, so the bias is
    # part of its definition. This pipeline uses K_var as an *estimator* of integrated
    # variance feeding xi0(T) and ultimately H. Being constant in V, the bias cancels
    # in every forward-variance increment EXCEPT the first, so it lands entirely on
    # xi0 over [0, T_1] -- divided by T_1, i.e. largest exactly at the short end that
    # spec 4.5 regresses to obtain H. Left unflagged it is a silent bias on H.
    atm_idx = int(np.argmin(np.abs(strikes - F)))
    h_atm = float(dK[atm_idx])
    v_bias = (h_atm * h_atm) / (6.0 * F * F)
    k_var_discretisation_bias = v_bias / T

    # --- finite-strike truncation: complete both wings --------------------
    edge_low = float(strikes[0]) - 0.5 * float(dK[0])
    edge_high = float(strikes[-1]) + 0.5 * float(dK[-1])
    edge_low = max(edge_low, 1e-12 * F)

    put_tail = _complete_tail(
        side=PUT,
        quotes=chain.puts,
        F=F,
        T=T,
        D=D_val,
        edge_strike=edge_low,
        config=cfg,
    )
    call_tail = _complete_tail(
        side=CALL,
        quotes=chain.calls,
        F=F,
        T=T,
        D=D_val,
        edge_strike=edge_high,
        config=cfg,
    )
    tail_integral = float(put_tail.integral) + float(call_tail.integral)
    k_var_tail = k_var_trunc + (2.0 / (T * D_val)) * tail_integral

    truncation_error = k_var_tail - k_var_trunc
    truncation_error_rel = (
        truncation_error / k_var_tail if abs(k_var_tail) > 0.0 else float("nan")
    )

    # --- diagnostics -------------------------------------------------------
    flags: list[str] = []
    messages: list[str] = []

    if not put_tail.available:
        flags.append(FLAG_NO_PUT_TAIL_ANCHOR)
        messages.append(
            f"Aile put non extrapolée à T={T:.6g} an(s): aucune cotation put OTM fiable "
            "(bilatérale et inversible) pour ancrer la volatilité plate."
        )
    if not call_tail.available:
        flags.append(FLAG_NO_CALL_TAIL_ANCHOR)
        messages.append(
            f"Aile call non extrapolée à T={T:.6g} an(s): aucune cotation call OTM fiable "
            "(bilatérale et inversible) pour ancrer la volatilité plate."
        )
    if TAIL_STOP_CAP in (put_tail.stop_reason, call_tail.stop_reason):
        flags.append(FLAG_TAIL_CAP_REACHED)
        messages.append(
            "Extrapolation des ailes arrêtée par le plafond en écarts-types avant que "
            "l'intégrande ne passe sous eps_tail."
        )
    if put_tail.n_clipped or call_tail.n_clipped:
        flags.append(FLAG_NEGATIVE_TAIL_INTEGRAND_CLIPPED)
    if k0_single_sided:
        flags.append(FLAG_K0_SINGLE_SIDED)
        messages.append(
            f"Strike pivot K_0={K0:.6g} coté d'un seul côté: la jambe manquante est "
            "reconstruite par parité call-put (exacte); la moyenne reste correcte."
        )
    if skipped:
        flags.append(FLAG_STRIKES_SKIPPED)
        messages.append(
            f"{len(skipped)} strike(s) écarté(s) faute de la cotation OTM requise; "
            "l'espacement irrégulier est absorbé par dK_i."
        )

    dk_min = float(np.min(dK))
    dk_max = float(np.max(dK))
    dk_ratio = dk_max / dk_min if dk_min > 0.0 else float("inf")
    if dk_ratio > float(cfg.irregular_spacing_ratio):
        flags.append(FLAG_IRREGULAR_SPACING)

    discretisation_bias_rel = (
        k_var_discretisation_bias / k_var_tail if abs(k_var_tail) > 0.0 else float("nan")
    )
    if math.isfinite(discretisation_bias_rel) and discretisation_bias_rel > float(
        cfg.discretisation_warn_rel
    ):
        flags.append(FLAG_COARSE_STRIKE_LADDER)
        messages.append(
            f"Échelle de strikes grossière à T={T:.6g} an(s): le pas ATM ΔK={h_atm:.6g} induit "
            f"un biais de discrétisation estimé à {100.0 * discretisation_bias_rel:.2f} % de "
            "K_var (h²/(6F²) en variance totale). Ce biais décroît en 1/T et se reporte "
            "intégralement sur xi0 sur [0, T_1]."
        )

    if math.isfinite(truncation_error_rel) and truncation_error_rel > float(
        cfg.truncation_warn_rel
    ):
        flags.append(FLAG_LARGE_TRUNCATION)
        messages.append(
            f"Troncature des strikes importante à T={T:.6g} an(s): les ailes extrapolées "
            f"apportent {100.0 * truncation_error_rel:.2f} % de K_var."
        )

    viability_fwd = evaluate_viability(chain.calls, chain.puts, forward_ref=F)
    if not viability_fwd.usable_for_kvar:
        flags.append(FLAG_VIABILITY_LOST_ON_FORWARD)
        messages.append(
            f"Diagnostic à T={T:.6g} an(s): la viabilité recalculée sur le forward de parité "
            f"F={F:.6g} n'est plus satisfaite (elle l'était sur la référence spot du 4.1); "
            "K_var est produit mais doit être regardé avec prudence."
        )

    diagnostics = VarianceSwapDiagnostics(
        n_strikes=n_strikes,
        K_0=K0,
        K_min=float(strikes[0]),
        K_max=float(strikes[-1]),
        log_moneyness_min=math.log(float(strikes[0]) / F),
        log_moneyness_max=math.log(float(strikes[-1]) / F),
        forward_over_k0=F / K0,
        correction_term=float(correction_term),
        correction_term_cboe=float(correction_term_cboe),
        discretisation_bias=float(k_var_discretisation_bias),
        discretisation_bias_rel=float(discretisation_bias_rel),
        h_atm=float(h_atm),
        raw_integral=float(raw_integral),
        k_var_trunc=float(k_var_trunc),
        k_var_tail=float(k_var_tail),
        truncation_error=float(truncation_error),
        truncation_error_rel=float(truncation_error_rel),
        dk_min=dk_min,
        dk_max=dk_max,
        dk_ratio=float(dk_ratio),
        n_strikes_skipped=len(skipped),
        skipped_strikes=tuple(skipped),
        put_tail=put_tail,
        call_tail=call_tail,
        viability_on_forward=viability_fwd,
        flags=tuple(flags),
        messages_fr=tuple(messages),
    )

    # A variance-swap strike is a fair variance: no-arbitrage forbids K_var < 0.
    # A negative value means an upstream input is wrong (typically a spec-4.2 forward
    # that missed, e.g. a hard-to-borrow name or a parity-regression regression), and
    # returning it as a plain number would let spec 4.4 manufacture an absurd xi0 on
    # the following interval. Refuse instead, with the French message the spec wants.
    if not math.isfinite(k_var_tail) or k_var_tail < 0.0:
        return VarianceSwapFailure(
            T=T,
            reason=REASON_NEGATIVE_K_VAR,
            message_fr=(
                f"K_var négatif ou non fini à T={T:.6g} an(s) (K_var={k_var_tail:.6g}): "
                "résultat non physique, l'échéance est refusée. Vérifier le forward de "
                f"parité (F={F:.6g}) et les cotations de l'échéance."
            ),
            detail={"k_var": float(k_var_tail), "F": float(F), "D": float(D_val)},
        )

    return VarianceSwapPoint(
        T=T,
        k_var=float(k_var_tail),
        k_var_trunc=float(k_var_trunc),
        n_puts=int(n_puts),
        n_calls=int(n_calls),
        F=F,
        D=D_val,
        diagnostics=diagnostics,
        expiry=expiry,
    )


# ---------------------------------------------------------------------------
# Curve builder — the spec 5 public entry point
# ---------------------------------------------------------------------------


def _split_surface_entry(entry: Any) -> tuple[CleanChain, ForwardPoint | None]:
    """Accept either a bare ``CleanChain`` or a ``(CleanChain, ForwardPoint)`` pair."""
    if isinstance(entry, CleanChain):
        return entry, None
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        chain, point = entry
        if isinstance(chain, CleanChain) and isinstance(point, ForwardPoint):
            return chain, point
    raise TypeError(
        "build_variance_swap_curve: chaque élément de option_surface doit être un CleanChain "
        f"ou un couple (CleanChain, ForwardPoint); reçu {type(entry).__name__}"
    )


def _match_forward_point(
    T: float, forward_curve: Sequence[ForwardPoint]
) -> ForwardPoint | None:
    """Exact-``T`` match first, then the machine-precision neighbourhood of ``T``."""
    for candidate in forward_curve:
        if float(candidate.T) == float(T):
            return candidate
    tol = 1e-12 * max(1.0, abs(float(T)))
    for candidate in forward_curve:
        if abs(float(candidate.T) - float(T)) <= tol:
            return candidate
    return None


def _discount_for(
    T: float,
    point: ForwardPoint,
    rates: Any | None,
    currency: str | None,
) -> float:
    """
    Discount factor used by the ``1/D(T)`` factor of the CBOE sum.

    ``rates is None`` keeps the forward point's own ``D``, which spec 4.2 already
    resolved from the repo yield curve — that is the recommended path. A mapping
    ``{T: r}`` or a curve object override it through spec 4.2's
    :func:`resolve_discount`, so ``r = -ln(D)/T`` stays the single rate
    convention. A flat ``r`` is never guessed here.
    """
    if rates is None:
        return float(point.D)
    if isinstance(rates, Mapping):
        r_val = rates.get(float(T))
        if r_val is None:
            return float(point.D)
        return float(resolve_discount(T, r=float(r_val)).D)
    return float(resolve_discount(T, curve=rates, currency=currency).D)


def build_variance_swap_curve(
    option_surface: Sequence[Any],
    forward_curve: Sequence[ForwardPoint] | None = None,
    rates: Any | None = None,
    *,
    config: VarianceSwapConfig | None = None,
    currency: str | None = None,
) -> VarianceSwapCurve:
    """
    Build the variance-swap strike curve (spec 4.3 / spec 5 public contract).

    Parameters
    ----------
    option_surface:
        The spec-4.1 output: a sequence of :class:`CleanChain` (one per expiry).
        A sequence of ``(CleanChain, ForwardPoint)`` pairs is also accepted, in
        which case the pairing is taken as given and ``forward_curve`` is not
        consulted for those entries.
    forward_curve:
        The spec-4.2 output, ``list[ForwardPoint]``. Chains are matched to
        forward points on the maturity ``T`` (exact float first, then within
        ``1e-12 * max(1, T)``). Required unless every entry of ``option_surface``
        is a pair.
    rates:
        Optional discount override. ``None`` (default) uses ``ForwardPoint.D``,
        already resolved from the repo yield curve by spec 4.2. A
        ``Mapping[T, r]`` pins the continuously-compounded zero rate per
        maturity; anything else is treated as a curve object exposing
        ``discount_factor(T)``. Both go through spec 4.2's
        :func:`resolve_discount`.
    config, currency:
        Thresholds and the currency handed to the yield-curve resolution.

    Returns
    -------
    VarianceSwapCurve
        Usable points sorted by maturity, plus one
        :class:`VarianceSwapFailure` per refused expiry.
    """
    cfg = config or VarianceSwapConfig()
    curve_points = list(forward_curve or ())

    points: list[VarianceSwapPoint] = []
    failures: list[VarianceSwapFailure] = []

    for entry in option_surface:
        chain, paired_point = _split_surface_entry(entry)
        point = paired_point if paired_point is not None else _match_forward_point(
            chain.T, curve_points
        )
        if point is None:
            failures.append(
                VarianceSwapFailure(
                    T=float(chain.T),
                    reason=REASON_NO_FORWARD_POINT,
                    message_fr=(
                        f"K_var refusé à T={float(chain.T):.6g} an(s): aucun point de courbe "
                        "forward (4.2) ne correspond à cette échéance."
                    ),
                    expiry=chain.expiry,
                )
            )
            continue

        D_val = _discount_for(chain.T, point, rates, currency)
        outcome = build_variance_swap_point(chain, point, D=D_val, config=cfg)
        if isinstance(outcome, VarianceSwapFailure):
            failures.append(outcome)
        else:
            points.append(outcome)

    points.sort(key=lambda p: p.T)
    failures.sort(key=lambda f: f.T)
    return VarianceSwapCurve(points=tuple(points), failures=tuple(failures))


def variance_swap_report(curve: VarianceSwapCurve) -> dict[str, Any]:
    """Plain-python summary for the Phase-5 report (survives ``_json_safe``)."""
    truncations = [p.truncation_error for p in curve.points]
    return {
        "n_points": int(len(curve.points)),
        "n_failures": int(len(curve.failures)),
        "maturities": [float(T) for T in curve.maturities],
        "k_var": [float(v) for v in curve.k_var],
        "k_var_trunc": [float(p.k_var_trunc) for p in curve.points],
        "total_variances": [float(v) for v in curve.total_variances],
        "max_truncation_error": float(max(truncations)) if truncations else float("nan"),
        "failures": [f.to_dict() for f in curve.failures],
        "flags": sorted({str(flag) for p in curve.points for flag in p.flags}),
    }


__all__ = [
    "FLAG_IRREGULAR_SPACING",
    "FLAG_K0_SINGLE_SIDED",
    "FLAG_LARGE_TRUNCATION",
    "FLAG_NEGATIVE_TAIL_INTEGRAND_CLIPPED",
    "FLAG_NO_CALL_TAIL_ANCHOR",
    "FLAG_NO_PUT_TAIL_ANCHOR",
    "FLAG_STRIKES_SKIPPED",
    "FLAG_TAIL_CAP_REACHED",
    "FLAG_VIABILITY_LOST_ON_FORWARD",
    "REASON_INVALID_DISCOUNT",
    "REASON_INVALID_FORWARD",
    "REASON_INVALID_MATURITY",
    "REASON_NO_FORWARD_POINT",
    "REASON_NO_STRIKE_BELOW_FORWARD",
    "REASON_TOO_FEW_STRIKES",
    "REASON_VIABILITY_FAILED",
    "TAIL_STOP_CAP",
    "TAIL_STOP_INTEGRAND",
    "TAIL_STOP_MAX_STEPS",
    "TAIL_STOP_NO_ANCHOR",
    "VARIANCE_REASON_LABELS_FR",
    "TailEstimate",
    "VarianceSwapConfig",
    "VarianceSwapCurve",
    "VarianceSwapDiagnostics",
    "VarianceSwapFailure",
    "VarianceSwapPoint",
    "black76_put_price",
    "build_variance_swap_curve",
    "build_variance_swap_point",
    "variance_swap_report",
]
