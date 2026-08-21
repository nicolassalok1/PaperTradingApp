"""
Forward curve, discounting and market-IV recomputation (spec 4.2).

For each cleaned expiry ``T`` this module produces:

* ``D(T)`` and ``r(T)`` taken from the **repo yield curve**
  (``app.model.yieldcurve.service.get_active_curve(...).discount_factor(T)``) —
  never from the chain fetch, whose ``rf`` is hardcoded ``0.0``;
* ``F(T)`` from **put-call parity on cleaned mids**:
  ``C(K) - P(K) = D(T) * (F - K)``. The forward is the weighted least-squares
  estimate under the *known* slope ``-D(T)``, with a free-slope regression run
  alongside purely as a data-quality diagnostic (the fitted slope must be close
  to ``-D``; the gap is reported, never swallowed);
* ``q_implied(T) = r(T) - ln(F / S0) / T``, a diagnostic implied dividend yield;
* the OTM market surface in **log-forward-moneyness** ``k = ln(K / F(T))``, with
  IVs **recomputed** from cleaned mids — vendor IVs are never pipeline input.

Black-76 inversion
------------------
``implied_vol_call`` prices with a spot/carry parametrisation. To invert a price
quoted on forward ``F`` with discount ``D`` at maturity ``T`` we use the
documented substitution ``S0 = F * D``, ``r = -ln(D) / T``, ``q = 0``:

    bs_call_price(F*D, K, T, -ln(D)/T, 0, v)
        = F*D*N(d1) - K*exp(ln(D))*N(d2)
        = D * (F*N(d1) - K*N(d2))                       (Black-76)
    with d1 = (ln(F/K) + v^2 T / 2) / (v sqrt(T)).

That is exact, not an approximation; ``test_rv_forward_curve`` pins it with a
round-trip.

OTM convention
--------------
``k < 0`` (``K < F``) → puts, ``k >= 0`` → calls. Put quotes are converted to
their parity-equivalent call price ``C = P + D * (F - K)`` before inversion, so
call-side and put-side IVs are produced by exactly the same inverter.

Layering / purity: model layer only. The module imports with **no side effects
and no network**; ``get_active_curve`` is resolved lazily inside
:func:`resolve_discount`, and callers may inject either a curve object or
explicit ``(r, D)`` so tests never touch the curve loader.

Exercise style: see ``chain_cleaning`` — Yahoo single-name/ETF chains are
American while every formula here assumes European exercise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.model.calibration.implied_vol import bs_call_price, implied_vol_call
from app.model.calibration.rough_vol.chain_cleaning import (
    CALL,
    PUT,
    CleanChain,
    CleanQuote,
)

# --- surface-point rejection reason codes ----------------------------------
REASON_NO_OTM_QUOTE = "no_otm_quote"
REASON_ONE_SIDED_EXCLUDED = "one_sided_excluded"
REASON_OUTSIDE_MONEYNESS_WINDOW = "outside_moneyness_window"
REASON_NONPOSITIVE_CALL_EQUIVALENT = "nonpositive_call_equivalent"
REASON_IV_INVERSION_FAILED = "iv_inversion_failed"

#: French labels for the Phase-5 report.
SURFACE_REASON_LABELS_FR: dict[str, str] = {
    REASON_NO_OTM_QUOTE: "Aucune cotation OTM disponible",
    REASON_ONE_SIDED_EXCLUDED: "Cotation unilatérale (bid nul) exclue",
    REASON_OUTSIDE_MONEYNESS_WINDOW: "Hors fenêtre de moneyness",
    REASON_NONPOSITIVE_CALL_EQUIVALENT: "Prix call équivalent nul ou négatif",
    REASON_IV_INVERSION_FAILED: "Inversion de la volatilité implicite échouée",
}

#: Source of the (D, r) pair carried by a :class:`DiscountPoint`.
SOURCE_EXPLICIT_D = "explicit_D"
SOURCE_EXPLICIT_R = "explicit_r"
SOURCE_CURVE_OBJECT = "curve_object"
SOURCE_ACTIVE_CURVE = "active_curve"

#: How :attr:`ForwardPoint.F` was estimated.
METHOD_PARITY_REGRESSION = "parity_weighted_regression"
METHOD_PARITY_SINGLE_STRIKE = "parity_single_strike_fallback"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardConfig:
    """
    Thresholds for the put-call-parity forward estimation.

    Attributes
    ----------
    max_pairs:
        Number of (call, put) strike pairs retained for the regression, taken
        nearest the money — spec 4.2 asks explicitly for the near-the-money
        strikes, where parity is tightest and both legs are liquid. "Nearest the
        money" is measured against the single-strike forward seed, not against
        the anchor strike (see :func:`estimate_forward`).
    min_pairs_for_regression:
        Below this many pairs the regression is abandoned and the single-strike
        fallback ``F = K* + (C(K*) - P(K*)) / D`` is used at the nearest-ATM
        strike ``K*``.
    min_quote_sigma:
        Floor on the per-pair quote-noise scale used as a regression weight
        (``w_i = 1 / sigma_i^2`` with ``sigma_i`` the average half-spread of the
        two legs). The floor keeps zero-spread synthetic chains well-posed and
        prevents a single suspiciously tight quote from dominating the fit.
    slope_tolerance_rel:
        Relative gap ``|slope_free + D| / D`` above which the expiry is flagged
        :data:`FLAG_PARITY_SLOPE_OFF`. Diagnostic only — nothing is dropped.
    exclude_one_sided_from_parity:
        Ignore quotes with a zero bid when forming parity pairs: a one-sided
        quote has no reliable mid and would bias the forward.
    """

    max_pairs: int = 8
    min_pairs_for_regression: int = 2
    min_quote_sigma: float = 1e-4
    slope_tolerance_rel: float = 0.05
    exclude_one_sided_from_parity: bool = True


@dataclass(frozen=True)
class SurfaceConfig:
    """
    Thresholds for the OTM market-surface build.

    Attributes
    ----------
    include_one_sided:
        Keep zero-bid quotes (flagged upstream) in the far tails. They carry the
        wing information the rough-vol fit needs, at the cost of a noisy mid.
    max_abs_log_moneyness:
        Optional hard window on ``|k|``. ``None`` keeps every OTM strike.
    """

    include_one_sided: bool = True
    max_abs_log_moneyness: float | None = None


#: Chain-level diagnostic flags.
FLAG_PARITY_SLOPE_OFF = "parity_slope_off"
FLAG_PARITY_FALLBACK = "parity_single_strike_fallback"


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscountPoint:
    """
    Discount factor and the continuously-compounded zero rate consistent with it.

    ``r`` is always ``-ln(D) / T`` so that ``exp(-r T) == D`` exactly: everything
    downstream (Black-76 inversion, implied dividend) must see one single rate
    convention. ``r_curve`` carries the curve's own ``zero_rate(T)`` when one is
    available, so any interpolation inconsistency between the curve's discount
    factors and its zero rates is visible instead of silently averaged away.
    """

    T: float
    D: float
    r: float
    source: str
    r_curve: float | None = None

    @property
    def zero_rate_gap(self) -> float:
        """``r_curve - r``; NaN when the curve exposes no zero rate."""
        if self.r_curve is None:
            return float("nan")
        return float(self.r_curve) - float(self.r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "D": float(self.D),
            "r": float(self.r),
            "source": str(self.source),
            "r_curve": None if self.r_curve is None else float(self.r_curve),
            "zero_rate_gap": float(self.zero_rate_gap),
        }


def resolve_discount(
    T: float,
    *,
    curve: Any | None = None,
    r: float | None = None,
    D: float | None = None,
    currency: str | None = None,
) -> DiscountPoint:
    """
    Resolve ``(D, r)`` for maturity ``T`` (year fraction, calendar days / 365).

    Precedence, most explicit first:

    1. ``D`` given          → ``r = -ln(D) / T``          (``explicit_D``)
    2. ``r`` given          → ``D = exp(-r T)``           (``explicit_r``)
    3. ``curve`` injected   → ``D = curve.discount_factor(T)``  (``curve_object``)
    4. nothing given        → lazy ``get_active_curve(currency)`` (``active_curve``)

    Cases 1-3 never touch the yield-curve service, which is what keeps this
    module importable and testable offline. The import in case 4 is deliberately
    performed **inside** the function.
    """
    T_val = float(T)
    if not math.isfinite(T_val) or T_val <= 0.0:
        raise ValueError(f"resolve_discount: maturité invalide T={T!r}")

    if D is not None:
        D_val = float(D)
        if not math.isfinite(D_val) or D_val <= 0.0:
            raise ValueError(f"resolve_discount: facteur d'actualisation invalide D={D!r}")
        return DiscountPoint(
            T=T_val, D=D_val, r=-math.log(D_val) / T_val, source=SOURCE_EXPLICIT_D
        )

    if r is not None:
        r_val = float(r)
        if not math.isfinite(r_val):
            raise ValueError(f"resolve_discount: taux invalide r={r!r}")
        D_val = math.exp(-r_val * T_val)
        return DiscountPoint(T=T_val, D=D_val, r=r_val, source=SOURCE_EXPLICIT_R)

    active = curve
    source = SOURCE_CURVE_OBJECT
    if active is None:
        # Lazy import: the yield-curve service touches the filesystem at import
        # time, so it must never be pulled in when a curve or (r, D) is injected.
        from app.model.yieldcurve.service import get_active_curve

        active = get_active_curve(currency)[0]
        source = SOURCE_ACTIVE_CURVE

    D_val = float(active.discount_factor(T_val))
    if not math.isfinite(D_val) or D_val <= 0.0:
        raise ValueError(
            f"resolve_discount: la courbe a renvoyé un facteur d'actualisation invalide D={D_val!r}"
        )

    r_curve: float | None
    try:
        r_curve = float(active.zero_rate(T_val))
    except Exception:
        r_curve = None
    if r_curve is not None and not math.isfinite(r_curve):
        r_curve = None

    return DiscountPoint(
        T=T_val, D=D_val, r=-math.log(D_val) / T_val, source=source, r_curve=r_curve
    )


# ---------------------------------------------------------------------------
# Put-call parity forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityPair:
    """A strike quoted on both sides, with its parity difference and noise scale."""

    strike: float
    call_mid: float
    put_mid: float
    diff: float
    sigma: float
    one_sided: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strike": float(self.strike),
            "call_mid": float(self.call_mid),
            "put_mid": float(self.put_mid),
            "diff": float(self.diff),
            "sigma": float(self.sigma),
            "one_sided": bool(self.one_sided),
        }


@dataclass(frozen=True)
class ParitySlopeDiag:
    """
    Data-quality diagnostic on the ``C - P`` vs ``K`` regression.

    ``slope`` is the *free* weighted least-squares slope. Put-call parity forces
    it to equal ``-D(T)``; ``slope_expected`` records that target and
    ``slope_rel_error`` the relative gap. A large gap means the mids are not
    parity-consistent (stale legs, wrong discounting, hard-to-borrow underlying,
    or American early-exercise premium) — the forward is still reported, but the
    expiry is flagged.
    """

    method: str
    n_pairs: int
    slope: float
    slope_expected: float
    slope_abs_error: float
    slope_rel_error: float
    implied_discount: float
    intercept: float
    forward_free_slope: float
    forward_minus_free: float
    r_squared: float
    residual_rmse: float
    strikes: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "n_pairs": int(self.n_pairs),
            "slope": float(self.slope),
            "slope_expected": float(self.slope_expected),
            "slope_abs_error": float(self.slope_abs_error),
            "slope_rel_error": float(self.slope_rel_error),
            "implied_discount": float(self.implied_discount),
            "intercept": float(self.intercept),
            "forward_free_slope": float(self.forward_free_slope),
            "forward_minus_free": float(self.forward_minus_free),
            "r_squared": float(self.r_squared),
            "residual_rmse": float(self.residual_rmse),
            "strikes": [float(k) for k in self.strikes],
        }


@dataclass(frozen=True)
class ForwardPoint:
    """
    One point of the forward curve (spec 6 data contract).

    ``F`` is the put-call-parity forward, ``D`` / ``r`` the discounting from the
    repo yield curve, ``q_implied = r - ln(F / S0) / T`` the diagnostic implied
    dividend yield, and ``parity_slope_diag`` the regression health report.
    """

    T: float
    F: float
    D: float
    r: float
    q_implied: float
    parity_slope_diag: ParitySlopeDiag
    S0: float = float("nan")
    discount_source: str = SOURCE_ACTIVE_CURVE
    flags: tuple[str, ...] = ()

    def log_moneyness(self, strike: float) -> float:
        """``k = ln(K / F(T))``, the only moneyness used downstream."""
        K = float(strike)
        if not (math.isfinite(K) and K > 0.0 and math.isfinite(self.F) and self.F > 0.0):
            return float("nan")
        return math.log(K / self.F)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "F": float(self.F),
            "D": float(self.D),
            "r": float(self.r),
            "q_implied": float(self.q_implied),
            "S0": float(self.S0),
            "discount_source": str(self.discount_source),
            "parity_slope_diag": self.parity_slope_diag.to_dict(),
            "flags": [str(f) for f in self.flags],
        }


def collect_parity_pairs(
    chain: CleanChain,
    *,
    config: ForwardConfig | None = None,
) -> list[ParityPair]:
    """
    Pair up strikes quoted on both sides of a cleaned chain.

    ``sigma`` is the average half-spread of the two legs, floored at
    ``ForwardConfig.min_quote_sigma``: it is the natural noise scale of the
    parity residual ``C - P``, hence the natural inverse-variance weight.
    """
    cfg = config or ForwardConfig()
    calls: dict[float, CleanQuote] = {float(q.strike): q for q in chain.calls}
    puts: dict[float, CleanQuote] = {float(q.strike): q for q in chain.puts}

    pairs: list[ParityPair] = []
    for strike in sorted(set(calls) & set(puts)):
        call = calls[strike]
        put = puts[strike]
        one_sided = bool(call.one_sided or put.one_sided)
        if one_sided and cfg.exclude_one_sided_from_parity:
            continue
        sigma = max(
            0.5 * (0.5 * call.spread_abs + 0.5 * put.spread_abs),
            float(cfg.min_quote_sigma),
        )
        pairs.append(
            ParityPair(
                strike=float(strike),
                call_mid=float(call.mid),
                put_mid=float(put.mid),
                diff=float(call.mid - put.mid),
                sigma=float(sigma),
                one_sided=one_sided,
            )
        )
    return pairs


def _atm_anchor(pairs: Sequence[ParityPair]) -> ParityPair:
    """
    Nearest-ATM pair: the strike where ``|C - P|`` is smallest.

    Under parity ``C - P = D (F - K)``, so ``argmin |C - P| = argmin |F - K|``.
    This locates the money without needing ``F`` first. Ties break on the lower
    strike so the choice is deterministic.
    """
    return min(pairs, key=lambda p: (abs(p.diff), p.strike))


def _weighted_linear_fit(
    x: Sequence[float],
    y: Sequence[float],
    w: Sequence[float],
) -> tuple[float, float, float, float]:
    """
    Weighted least squares ``y = a + b x``.

    Returns ``(a, b, r_squared, residual_rmse)``. ``b`` is NaN when the design is
    degenerate (all strikes equal, or a single point).
    """
    sw = sum(w)
    if sw <= 0.0 or len(x) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    x_bar = sum(wi * xi for wi, xi in zip(w, x)) / sw
    y_bar = sum(wi * yi for wi, yi in zip(w, y)) / sw
    sxx = sum(wi * (xi - x_bar) ** 2 for wi, xi in zip(w, x))
    sxy = sum(wi * (xi - x_bar) * (yi - y_bar) for wi, xi, yi in zip(w, x, y))
    syy = sum(wi * (yi - y_bar) ** 2 for wi, yi in zip(w, y))
    if sxx <= 0.0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    b = sxy / sxx
    a = y_bar - b * x_bar
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    sse = sum(wi * ri * ri for wi, ri in zip(w, resid))
    r2 = 1.0 - sse / syy if syy > 0.0 else 1.0
    rmse = math.sqrt(sse / sw)
    return (float(a), float(b), float(r2), float(rmse))


def estimate_forward(
    pairs: Sequence[ParityPair],
    *,
    D: float,
    config: ForwardConfig | None = None,
) -> tuple[float, ParitySlopeDiag] | None:
    """
    Estimate ``F`` from put-call parity on mids.

    Parity states ``C(K) - P(K) = D (F - K)``, i.e. a straight line in ``K`` with
    intercept ``D F`` and slope ``-D``. ``D`` comes from the yield curve and is
    therefore *known*, so the forward is estimated with the slope pinned:

        F_hat = sum_i w_i (K_i + (C_i - P_i) / D) / sum_i w_i,   w_i = 1 / sigma_i^2

    which is the weighted least-squares solution of the one-parameter model and
    is strictly better conditioned than reading an extrapolated intercept off a
    two-parameter fit. The free two-parameter regression is run alongside so the
    fitted slope can be compared with ``-D`` — spec 4.2 asks for that discrepancy
    as a data-quality diagnostic, not as an input to ``F``.

    Returns ``None`` when no usable pair exists.
    """
    cfg = config or ForwardConfig()
    D_val = float(D)
    if not math.isfinite(D_val) or D_val <= 0.0:
        raise ValueError(f"estimate_forward: facteur d'actualisation invalide D={D!r}")

    usable = [
        p
        for p in pairs
        if math.isfinite(p.strike)
        and p.strike > 0.0
        and math.isfinite(p.diff)
        and math.isfinite(p.sigma)
        and p.sigma > 0.0
    ]
    if not usable:
        return None

    # Two-stage selection of the near-the-money strikes:
    #   1. the anchor (argmin |C - P|) gives a first, single-strike forward F0,
    #   2. the max_pairs strikes nearest F0 are retained for the fit.
    # Ranking against F0 rather than against the anchor *strike* matters as soon
    # as the forward sits between two listed strikes.
    anchor = _atm_anchor(usable)
    forward_seed = float(anchor.strike + anchor.diff / D_val)
    if not (math.isfinite(forward_seed) and forward_seed > 0.0):
        forward_seed = float(anchor.strike)
    selected = sorted(usable, key=lambda p: (abs(p.strike - forward_seed), p.strike))
    selected = selected[: max(1, int(cfg.max_pairs))]
    selected.sort(key=lambda p: p.strike)

    strikes = tuple(float(p.strike) for p in selected)
    weights = [1.0 / (p.sigma * p.sigma) for p in selected]

    if len(selected) < max(1, int(cfg.min_pairs_for_regression)):
        # Fallback: single nearest-ATM strike, F = K* + (C(K*) - P(K*)) / D.
        F = float(anchor.strike + anchor.diff / D_val)
        diag = ParitySlopeDiag(
            method=METHOD_PARITY_SINGLE_STRIKE,
            n_pairs=len(selected),
            slope=float("nan"),
            slope_expected=-D_val,
            slope_abs_error=float("nan"),
            slope_rel_error=float("nan"),
            implied_discount=float("nan"),
            intercept=float("nan"),
            forward_free_slope=float("nan"),
            forward_minus_free=float("nan"),
            r_squared=float("nan"),
            residual_rmse=float("nan"),
            strikes=(float(anchor.strike),),
        )
        return (F, diag)

    # --- constrained fit (slope pinned to -D): the pipeline's forward --------
    w_sum = sum(weights)
    F = sum(w * (p.strike + p.diff / D_val) for w, p in zip(weights, selected)) / w_sum

    # --- free fit: diagnostic only ------------------------------------------
    a, b, r2, rmse = _weighted_linear_fit(
        [p.strike for p in selected], [p.diff for p in selected], weights
    )
    if math.isfinite(b) and b != 0.0:
        forward_free = -a / b
        slope_abs_error = abs(b - (-D_val))
        slope_rel_error = slope_abs_error / D_val
        implied_discount = -b
    else:
        forward_free = float("nan")
        slope_abs_error = float("nan")
        slope_rel_error = float("nan")
        implied_discount = float("nan")

    diag = ParitySlopeDiag(
        method=METHOD_PARITY_REGRESSION,
        n_pairs=len(selected),
        slope=float(b),
        slope_expected=-D_val,
        slope_abs_error=float(slope_abs_error),
        slope_rel_error=float(slope_rel_error),
        implied_discount=float(implied_discount),
        intercept=float(a),
        forward_free_slope=float(forward_free),
        forward_minus_free=float(F - forward_free) if math.isfinite(forward_free) else float("nan"),
        r_squared=float(r2),
        residual_rmse=float(rmse),
        strikes=strikes,
    )
    return (float(F), diag)


def implied_dividend_yield(*, F: float, S0: float, T: float, r: float) -> float:
    """``q(T) = r(T) - ln(F / S0) / T``. NaN when ``S0`` or ``F`` is unusable."""
    F_val, S0_val, T_val = float(F), float(S0), float(T)
    if not (math.isfinite(F_val) and F_val > 0.0):
        return float("nan")
    if not (math.isfinite(S0_val) and S0_val > 0.0):
        return float("nan")
    if not (math.isfinite(T_val) and T_val > 0.0):
        return float("nan")
    return float(r) - math.log(F_val / S0_val) / T_val


def build_forward_point(
    chain: CleanChain,
    *,
    curve: Any | None = None,
    r: float | None = None,
    D: float | None = None,
    currency: str | None = None,
    S0: float | None = None,
    config: ForwardConfig | None = None,
) -> ForwardPoint | None:
    """
    Build the :class:`ForwardPoint` of one cleaned expiry.

    Returns ``None`` when the expiry carries no strike quoted on both sides —
    parity cannot be run, and inventing a forward from the spot would be a
    silent approximation.
    """
    cfg = config or ForwardConfig()
    if not (math.isfinite(chain.T) and chain.T > 0.0):
        return None

    discount = resolve_discount(chain.T, curve=curve, r=r, D=D, currency=currency)
    pairs = collect_parity_pairs(chain, config=cfg)
    estimate = estimate_forward(pairs, D=discount.D, config=cfg)
    if estimate is None:
        return None
    F, diag = estimate
    if not (math.isfinite(F) and F > 0.0):
        return None

    spot = float(S0) if S0 is not None else float(chain.S0)

    flags: list[str] = []
    if diag.method == METHOD_PARITY_SINGLE_STRIKE:
        flags.append(FLAG_PARITY_FALLBACK)
    if math.isfinite(diag.slope_rel_error) and diag.slope_rel_error > float(cfg.slope_tolerance_rel):
        flags.append(FLAG_PARITY_SLOPE_OFF)

    return ForwardPoint(
        T=float(chain.T),
        F=float(F),
        D=float(discount.D),
        r=float(discount.r),
        q_implied=implied_dividend_yield(F=F, S0=spot, T=chain.T, r=discount.r),
        parity_slope_diag=diag,
        S0=float(spot),
        discount_source=str(discount.source),
        flags=tuple(flags),
    )


def build_forward_curve(
    chains: Sequence[CleanChain],
    *,
    curve: Any | None = None,
    rates: Mapping[float, float] | None = None,
    discounts: Mapping[float, float] | None = None,
    currency: str | None = None,
    S0: float | None = None,
    config: ForwardConfig | None = None,
) -> list[ForwardPoint]:
    """
    Build the forward curve across expiries, sorted by increasing maturity.

    ``rates`` / ``discounts`` optionally pin ``r(T)`` / ``D(T)`` per maturity
    (keys are the chain ``T`` values); otherwise the injected ``curve`` — or,
    last, the repo's active curve — is used.
    """
    points: list[ForwardPoint] = []
    for chain in chains:
        T_key = float(chain.T)
        D_val = None if discounts is None else discounts.get(T_key)
        r_val = None if rates is None else rates.get(T_key)
        point = build_forward_point(
            chain,
            curve=curve,
            r=None if r_val is None else float(r_val),
            D=None if D_val is None else float(D_val),
            currency=currency,
            S0=S0,
            config=config,
        )
        if point is not None:
            points.append(point)
    points.sort(key=lambda p: p.T)
    return points


# ---------------------------------------------------------------------------
# IV recomputation on (F, D) — the pipeline's market surface
# ---------------------------------------------------------------------------


def black76_call_price(*, F: float, K: float, T: float, D: float, vol: float) -> float:
    """
    Black-76 call price ``D * (F N(d1) - K N(d2))``, expressed through the repo's
    ``bs_call_price`` with the documented substitution — no second pricer.
    """
    D_val = float(D)
    T_val = float(T)
    if not (math.isfinite(D_val) and D_val > 0.0 and math.isfinite(T_val) and T_val > 0.0):
        return float("nan")
    return float(
        bs_call_price(float(F) * D_val, float(K), T_val, -math.log(D_val) / T_val, 0.0, float(vol))
    )


def implied_vol_from_forward(
    price: float,
    *,
    K: float,
    T: float,
    F: float,
    D: float,
    option_type: str = CALL,
) -> float:
    """
    Invert a Black-76 price quoted on ``(F, D)``.

    Put prices are first converted to their parity-equivalent call price
    ``C = P + D (F - K)`` so a single inverter serves both sides. Returns NaN
    when the price is outside the no-arbitrage bounds or Brent fails to bracket
    a root (``implied_vol_call`` already returns NaN there).
    """
    D_val = float(D)
    T_val = float(T)
    F_val = float(F)
    K_val = float(K)
    px = float(price)
    if not (math.isfinite(D_val) and D_val > 0.0):
        return float("nan")
    if not (math.isfinite(T_val) and T_val > 0.0):
        return float("nan")
    if not (math.isfinite(F_val) and F_val > 0.0 and math.isfinite(K_val) and K_val > 0.0):
        return float("nan")
    if not math.isfinite(px):
        return float("nan")

    kind = str(option_type).strip().lower()
    if kind == PUT:
        px = px + D_val * (F_val - K_val)
    elif kind != CALL:
        raise ValueError(f"implied_vol_from_forward: type d'option inconnu {option_type!r}")

    if px <= 0.0:
        return float("nan")

    return float(
        implied_vol_call(px, F_val * D_val, K_val, T_val, -math.log(D_val) / T_val, 0.0)
    )


@dataclass(frozen=True)
class SurfacePoint:
    """
    One recomputed market-IV observation, in log-forward moneyness.

    ``iv`` is inverted from ``call_equivalent_price`` (the raw mid for calls, the
    parity-converted mid for puts) on ``(F, D)``. ``vendor_iv`` rides along for
    cross-checking only.
    """

    T: float
    K: float
    k: float
    F: float
    D: float
    iv: float
    option_type: str
    mid: float
    call_equivalent_price: float
    vendor_iv: float
    one_sided: bool
    contract_symbol: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def vendor_iv_gap(self) -> float:
        """``iv - vendor_iv``; NaN when the vendor IV is missing/sentinel."""
        if not math.isfinite(self.vendor_iv) or self.vendor_iv <= 0.0:
            return float("nan")
        return float(self.iv) - float(self.vendor_iv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": float(self.T),
            "K": float(self.K),
            "k": float(self.k),
            "F": float(self.F),
            "D": float(self.D),
            "iv": float(self.iv),
            "option_type": str(self.option_type),
            "mid": float(self.mid),
            "call_equivalent_price": float(self.call_equivalent_price),
            "vendor_iv": float(self.vendor_iv),
            "vendor_iv_gap": float(self.vendor_iv_gap),
            "one_sided": bool(self.one_sided),
            "contract_symbol": None if self.contract_symbol is None else str(self.contract_symbol),
            "flags": [str(f) for f in self.flags],
        }


@dataclass(frozen=True)
class SurfaceRejection:
    """A quote that could not become a :class:`SurfacePoint`, with its reason."""

    reason: str
    option_type: str
    strike: float
    T: float
    detail: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        detail = dict(self.detail or {})
        return {
            "reason": str(self.reason),
            "reason_fr": SURFACE_REASON_LABELS_FR.get(self.reason, self.reason),
            "option_type": str(self.option_type),
            "strike": float(self.strike),
            "T": float(self.T),
            "detail": {str(k): float(v) for k, v in detail.items()},
        }


def build_otm_surface(
    chain: CleanChain,
    point: ForwardPoint,
    *,
    config: SurfaceConfig | None = None,
) -> tuple[list[SurfacePoint], list[SurfaceRejection]]:
    """
    Recompute the OTM market surface of one expiry from cleaned mids.

    Convention (spec 4.2): ``k = ln(K / F)``; ``k < 0`` uses the put, ``k >= 0``
    uses the call. When the required side is missing at a strike, the strike is
    rejected with :data:`REASON_NO_OTM_QUOTE` rather than silently substituted by
    the ITM leg — an ITM American quote is exactly where the European assumption
    is worst.

    Returns ``(points sorted by k, rejections)``.
    """
    cfg = config or SurfaceConfig()
    calls: dict[float, CleanQuote] = {float(q.strike): q for q in chain.calls}
    puts: dict[float, CleanQuote] = {float(q.strike): q for q in chain.puts}

    points: list[SurfacePoint] = []
    rejections: list[SurfaceRejection] = []

    for strike in sorted(set(calls) | set(puts)):
        k = point.log_moneyness(strike)
        if not math.isfinite(k):
            rejections.append(
                SurfaceRejection(
                    reason=REASON_NO_OTM_QUOTE,
                    option_type="unknown",
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float("nan")},
                )
            )
            continue

        wanted = PUT if k < 0.0 else CALL
        quote = puts.get(strike) if wanted == PUT else calls.get(strike)
        if quote is None:
            rejections.append(
                SurfaceRejection(
                    reason=REASON_NO_OTM_QUOTE,
                    option_type=wanted,
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float(k)},
                )
            )
            continue

        if cfg.max_abs_log_moneyness is not None and abs(k) > float(cfg.max_abs_log_moneyness):
            rejections.append(
                SurfaceRejection(
                    reason=REASON_OUTSIDE_MONEYNESS_WINDOW,
                    option_type=wanted,
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float(k), "max_abs_k": float(cfg.max_abs_log_moneyness)},
                )
            )
            continue

        if quote.one_sided and not cfg.include_one_sided:
            rejections.append(
                SurfaceRejection(
                    reason=REASON_ONE_SIDED_EXCLUDED,
                    option_type=wanted,
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float(k), "bid": float(quote.bid)},
                )
            )
            continue

        call_equivalent = (
            float(quote.mid)
            if wanted == CALL
            else float(quote.mid) + point.D * (point.F - float(strike))
        )
        if not math.isfinite(call_equivalent) or call_equivalent <= 0.0:
            rejections.append(
                SurfaceRejection(
                    reason=REASON_NONPOSITIVE_CALL_EQUIVALENT,
                    option_type=wanted,
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float(k), "call_equivalent": float(call_equivalent)},
                )
            )
            continue

        iv = implied_vol_from_forward(
            quote.mid, K=strike, T=point.T, F=point.F, D=point.D, option_type=wanted
        )
        if not math.isfinite(iv) or iv <= 0.0:
            rejections.append(
                SurfaceRejection(
                    reason=REASON_IV_INVERSION_FAILED,
                    option_type=wanted,
                    strike=float(strike),
                    T=float(point.T),
                    detail={"k": float(k), "call_equivalent": float(call_equivalent)},
                )
            )
            continue

        points.append(
            SurfacePoint(
                T=float(point.T),
                K=float(strike),
                k=float(k),
                F=float(point.F),
                D=float(point.D),
                iv=float(iv),
                option_type=wanted,
                mid=float(quote.mid),
                call_equivalent_price=float(call_equivalent),
                vendor_iv=float(quote.vendor_iv),
                one_sided=bool(quote.one_sided),
                contract_symbol=quote.contract_symbol,
                flags=tuple(quote.flags),
            )
        )

    points.sort(key=lambda p: p.k)
    return (points, rejections)


def forward_curve_report(points: Sequence[ForwardPoint]) -> dict[str, Any]:
    """Plain-python summary of the forward curve for the Phase-5 report."""
    return {
        "n_expiries": int(len(points)),
        "n_parity_fallbacks": int(sum(1 for p in points if FLAG_PARITY_FALLBACK in p.flags)),
        "n_slope_off": int(sum(1 for p in points if FLAG_PARITY_SLOPE_OFF in p.flags)),
        "points": [p.to_dict() for p in points],
    }


__all__ = [
    "DiscountPoint",
    "FLAG_PARITY_FALLBACK",
    "FLAG_PARITY_SLOPE_OFF",
    "ForwardConfig",
    "ForwardPoint",
    "METHOD_PARITY_REGRESSION",
    "METHOD_PARITY_SINGLE_STRIKE",
    "ParityPair",
    "ParitySlopeDiag",
    "REASON_IV_INVERSION_FAILED",
    "REASON_NONPOSITIVE_CALL_EQUIVALENT",
    "REASON_NO_OTM_QUOTE",
    "REASON_ONE_SIDED_EXCLUDED",
    "REASON_OUTSIDE_MONEYNESS_WINDOW",
    "SOURCE_ACTIVE_CURVE",
    "SOURCE_CURVE_OBJECT",
    "SOURCE_EXPLICIT_D",
    "SOURCE_EXPLICIT_R",
    "SURFACE_REASON_LABELS_FR",
    "SurfaceConfig",
    "SurfacePoint",
    "SurfaceRejection",
    "black76_call_price",
    "build_forward_curve",
    "build_forward_point",
    "build_otm_surface",
    "collect_parity_pairs",
    "estimate_forward",
    "forward_curve_report",
    "implied_dividend_yield",
    "implied_vol_from_forward",
    "resolve_discount",
]
