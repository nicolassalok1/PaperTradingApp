"""
Pricing layer of the xi0-curve rough Bergomi simulator (spec 4.8).

This module is **independent of any calibration**: it takes a
:class:`~app.model.volatility_models.rbergomi.simulator_xi_curve.RBergomiPathSet`
and turns it into European call / put prices, standard errors, implied
volatilities and the diagnostics that prove the path set is sound.  A
calibrator is expected to sit *on top* of it, never inside it.

ONE PATH SET, EVERY MATURITY
============================
The simulation grid carries every quoted maturity with exact float equality, so
the whole surface - all strikes at all maturities - is priced from a **single**
path set, with a single Cholesky factorisation and a single set of random
numbers.  That is what makes common-random-number calibration cheap and what
removes the maturity-snapping bias of the legacy ``pricing_mc.py``.

THE TWO ESTIMATORS
==================
``estimator="plain"`` (:data:`ESTIMATOR_PLAIN`)
    The textbook discounted payoff, ``D(T) * mean( (S_T - K)^+ )``.

``estimator="conditional"`` (:data:`ESTIMATOR_CONDITIONAL`)
    The **mixed / conditional** estimator of Romano-Touzi, in the rough Bergomi
    form of McCrickerd-Pakkanen.  Conditionally on the whole variance path and
    on the driving Brownian path ``B``, the remaining randomness of ``ln S_T``
    is the single Gaussian ``sqrt(1-rho^2) * int_0^T sqrt(V) dB_perp``, so
    ``S_T`` is **conditionally lognormal** and the inner expectation is
    Black-Scholes in closed form:

    .. code-block:: text

        E[ (S_T - K)^+ | V, B ]
            = Black( F_cond , K , w_cond ) ,
        F_cond = S0 * exp(int_0^T (r-q) du) * exp( rho*int_0^T sqrt(V) dB
                                                   - 0.5*rho^2*int_0^T V du ) ,
        w_cond = (1 - rho^2) * int_0^T V du .

    The ``int sqrt(V) dB`` uses the **same** ``dB`` that drives ``W~`` - it is
    available exactly in the Cholesky scheme, which is precisely why the
    estimator is exact here and not an approximation.  Integrating out the
    perpendicular noise analytically removes the dominant source of Monte-Carlo
    variance, so this is the recommended estimator for calibration.  It is
    validated against the plain estimator (agreement within the standard error)
    by ``tests/quant/test_rv_rbergomi_mc.py``.

    Note that ``dB_perp`` is still *drawn* by the simulator (it is needed for
    the plain estimator and for the pathwise spot), so switching estimators
    costs nothing and both can be computed on the identical sample.

STANDARD ERRORS ARE HONEST
==========================
Antithetic sampling makes rows ``k`` and ``k + n_paths//2`` perfectly dependent,
so every standard error here is computed from the ``n_paths // 2`` independent
**pair means** (see
:func:`~app.model.volatility_models.rbergomi.simulator_xi_curve.sample_mean_stderr`).
Under scrambled-Sobol' QMC the sample standard deviation is *not* an estimate of
the randomised-QMC error at all; the result is then flagged
``stderr_is_conservative`` and the number must be read as a pseudo-Monte-Carlo
upper bound, never as the QMC accuracy.

EXACT SAMPLE-WISE INVARIANTS
============================
Because calls and puts are priced from the *same* sample, put-call parity is an
**algebraic identity**, not a statistical statement:

.. code-block:: text

    C(K,T) - P(K,T) = D(T) * ( sample_forward(T) - K )     to round-off,

with ``sample_forward`` the mean of ``S_T`` for the plain estimator and the mean
of the conditional forward ``F_cond`` for the conditional one (both are unbiased
estimates of the model forward ``S0 e^{int(r-q)}``).  It holds strike by strike
because ``(x)^+ - (-x)^+ = x`` pathwise, and because
``Black_call(F,K,w) - Black_put(F,K,w) = F - K`` identically.

IMPLIED VOLATILITIES
====================
:func:`implied_vol_surface` inverts the model prices with the repo's own Brent
inverter, :func:`app.model.calibration.implied_vol.implied_vol_call`, through
the **Black-76 substitution** documented in the repo brief: a price quoted on
forward ``F`` with discount ``D`` at maturity ``T`` is inverted by calling the
spot inverter with ``S0 = F*D``, ``r = -ln(D)/T`` and ``q = 0``, which
reproduces ``C = D*(F N(d1) - K N(d2))`` exactly.  Prices outside the no-arbitrage
band return ``NaN`` rather than a fabricated volatility, and puts are converted
to calls through the *sample* parity above so the inverted number corresponds to
the price the Monte Carlo actually produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr

from app.model.calibration.implied_vol import implied_vol_call
from app.model.volatility_models.rbergomi.simulator_xi_curve import RBergomiPathSet

#: Plain discounted-payoff Monte-Carlo estimator.
ESTIMATOR_PLAIN: str = "plain"
#: Conditional (mixed) Romano-Touzi / McCrickerd-Pakkanen estimator.
ESTIMATOR_CONDITIONAL: str = "conditional"
#: Both estimators, in the order they are documented.
ESTIMATORS: tuple[str, ...] = (ESTIMATOR_PLAIN, ESTIMATOR_CONDITIONAL)

#: Option types handled by this module.
OPTION_CALL: str = "call"
OPTION_PUT: str = "put"


# ---------------------------------------------------------------------------
# Black-76 core (forward measure), fully vectorised
# ---------------------------------------------------------------------------
def black_call_from_total_variance(
    F: np.ndarray, K: np.ndarray, total_variance: np.ndarray
) -> np.ndarray:
    """
    Undiscounted Black-76 call ``F N(d1) - K N(d2)`` from a **total** variance.

    ``total_variance`` is ``sigma^2 T`` (not an annualised volatility), which is
    what the conditional estimator produces naturally.  A zero (or negative,
    which cannot happen but is guarded) total variance collapses to the forward
    intrinsic ``(F - K)^+``, the exact zero-volatility limit.  All three
    arguments broadcast.
    """
    f = np.asarray(F, dtype=float)
    k = np.asarray(K, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    intrinsic = np.maximum(f - k, 0.0)
    positive = (w > 0.0) & (f > 0.0) & (k > 0.0)
    w_safe = np.where(positive, w, 1.0)
    k_safe = np.where(k > 0.0, k, 1.0)
    f_safe = np.where(f > 0.0, f, 1.0)
    sqrt_w = np.sqrt(w_safe)
    d1 = (np.log(f_safe / k_safe) + 0.5 * w_safe) / sqrt_w
    d2 = d1 - sqrt_w
    value = f_safe * ndtr(d1) - k_safe * ndtr(d2)
    return np.where(positive, value, intrinsic)


def black_put_from_total_variance(
    F: np.ndarray, K: np.ndarray, total_variance: np.ndarray
) -> np.ndarray:
    """
    Undiscounted Black-76 put ``K N(-d2) - F N(-d1)`` from a **total** variance.

    Written directly (rather than through parity) so that the pair
    ``call - put = F - K`` holds to round-off from the symmetry of
    :func:`scipy.special.ndtr`, which is what makes the sample put-call parity
    exact for the conditional estimator too.
    """
    f = np.asarray(F, dtype=float)
    k = np.asarray(K, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    intrinsic = np.maximum(k - f, 0.0)
    positive = (w > 0.0) & (f > 0.0) & (k > 0.0)
    w_safe = np.where(positive, w, 1.0)
    k_safe = np.where(k > 0.0, k, 1.0)
    f_safe = np.where(f > 0.0, f, 1.0)
    sqrt_w = np.sqrt(w_safe)
    d1 = (np.log(f_safe / k_safe) + 0.5 * w_safe) / sqrt_w
    d2 = d1 - sqrt_w
    value = k_safe * ndtr(-d2) - f_safe * ndtr(-d1)
    return np.where(positive, value, intrinsic)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PriceResult:
    """
    Prices and Monte-Carlo standard errors on a ``(maturity, strike)`` grid.

    Attributes
    ----------
    option_type:
        :data:`OPTION_CALL` or :data:`OPTION_PUT`.
    estimator:
        :data:`ESTIMATOR_PLAIN` or :data:`ESTIMATOR_CONDITIONAL`.
    maturities:
        The quoted maturities priced, in the order of the rows.
    strikes:
        ``(n_T, n_K)`` strikes - a 1-d strike grid handed in by the caller is
        broadcast to every maturity, a 2-d one is used verbatim.
    price, stderr:
        ``(n_T, n_K)`` discounted prices and their standard errors.
    discount_factors, forward_factors:
        ``(n_T,)`` ``D(T)`` and ``exp(int_0^T (r-q))`` used for the row.
    model_forwards:
        ``(n_T,)`` ``S0 * forward_factors`` - the theoretical forward.
    sample_forward, sample_forward_stderr:
        ``(n_T,)`` the Monte-Carlo forward the row was priced against
        (``mean(S_T)`` for the plain estimator, ``mean(F_cond)`` for the
        conditional one) and its standard error.  Put-call parity is exact
        against **this** number, not against ``model_forwards``.
    stderr_is_conservative:
        ``True`` when the path set was drawn with QMC, where the sample standard
        deviation over-states the true randomised-QMC error.
    """

    option_type: str
    estimator: str
    maturities: tuple[float, ...]
    strikes: np.ndarray
    price: np.ndarray
    stderr: np.ndarray
    discount_factors: np.ndarray
    forward_factors: np.ndarray
    model_forwards: np.ndarray
    sample_forward: np.ndarray
    sample_forward_stderr: np.ndarray
    S0: float
    q: float
    n_paths: int
    n_base_paths: int
    antithetic: bool
    stderr_is_conservative: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.maturities), int(self.strikes.shape[1]))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (survives ``_json_safe`` in the controller layer)."""
        return {
            "option_type": str(self.option_type),
            "estimator": str(self.estimator),
            "maturities": [float(x) for x in self.maturities],
            "strikes": [[float(k) for k in row] for row in self.strikes],
            "price": [[float(p) for p in row] for row in self.price],
            "stderr": [[float(s) for s in row] for row in self.stderr],
            "discount_factors": [float(x) for x in self.discount_factors],
            "model_forwards": [float(x) for x in self.model_forwards],
            "sample_forward": [float(x) for x in self.sample_forward],
            "sample_forward_stderr": [float(x) for x in self.sample_forward_stderr],
            "S0": float(self.S0),
            "q": float(self.q),
            "n_paths": int(self.n_paths),
            "n_base_paths": int(self.n_base_paths),
            "antithetic": bool(self.antithetic),
            "stderr_is_conservative": bool(self.stderr_is_conservative),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _broadcast_strikes(strikes: Any, n_maturities: int) -> np.ndarray:
    """Normalise a strike specification to a ``(n_T, n_K)`` float array."""
    array = np.asarray(strikes, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = np.broadcast_to(array[None, :], (n_maturities, array.size))
    elif array.ndim != 2:
        raise ValueError(
            f"strikes must be 0-, 1- or 2-dimensional; got ndim={array.ndim}."
        )
    if array.shape[0] != n_maturities:
        raise ValueError(
            f"A 2-d strike grid must have one row per maturity: expected "
            f"{n_maturities} rows, got {array.shape[0]}."
        )
    if array.size == 0:
        raise ValueError("At least one strike is required.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("Strikes must all be finite and strictly positive.")
    return np.ascontiguousarray(array, dtype=float)


def _validate_estimator(estimator: str) -> str:
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {ESTIMATORS!r}; got {estimator!r}."
        )
    return str(estimator)


def _price_one_type(
    paths: RBergomiPathSet,
    *,
    option_type: str,
    strikes: Any,
    maturities: Any,
    estimator: str,
) -> PriceResult:
    """Shared engine of :func:`price_call` and :func:`price_put`."""
    if not isinstance(paths, RBergomiPathSet):
        raise TypeError(
            f"paths must be an RBergomiPathSet; got {type(paths).__name__}."
        )
    kind = str(option_type)
    if kind not in (OPTION_CALL, OPTION_PUT):
        raise ValueError(
            f"option_type must be {OPTION_CALL!r} or {OPTION_PUT!r}; got {option_type!r}."
        )
    method = _validate_estimator(estimator)

    values, _indices = paths.resolve_maturities(maturities)
    strike_grid = _broadcast_strikes(strikes, len(values))
    n_T, n_K = strike_grid.shape

    price = np.empty((n_T, n_K), dtype=float)
    stderr = np.empty((n_T, n_K), dtype=float)
    discounts = np.empty(n_T, dtype=float)
    forward_factors = np.empty(n_T, dtype=float)
    model_forwards = np.empty(n_T, dtype=float)
    sample_forward = np.empty(n_T, dtype=float)
    sample_forward_stderr = np.empty(n_T, dtype=float)

    for row, T in enumerate(values):
        discount = paths.discount_factor(T)
        discounts[row] = discount
        forward_factors[row] = paths.forward_factor(T)
        model_forwards[row] = paths.model_forward(T)
        row_strikes = strike_grid[row][None, :]

        if method == ESTIMATOR_PLAIN:
            terminal = paths.spot(T)[:, None]
            if kind == OPTION_CALL:
                payoff = np.maximum(terminal - row_strikes, 0.0)
            else:
                payoff = np.maximum(row_strikes - terminal, 0.0)
            forward_sample = terminal[:, 0]
        else:
            conditional_forward = paths.conditional_forward(T)[:, None]
            conditional_variance = paths.conditional_total_variance(T)[:, None]
            if kind == OPTION_CALL:
                payoff = black_call_from_total_variance(
                    conditional_forward, row_strikes, conditional_variance
                )
            else:
                payoff = black_put_from_total_variance(
                    conditional_forward, row_strikes, conditional_variance
                )
            forward_sample = conditional_forward[:, 0]

        mean, error = paths.mean_stderr(payoff)
        price[row, :] = discount * mean
        stderr[row, :] = discount * error
        f_mean, f_error = paths.mean_stderr(forward_sample)
        sample_forward[row] = float(f_mean)
        sample_forward_stderr[row] = float(f_error)

    return PriceResult(
        option_type=kind,
        estimator=method,
        maturities=tuple(float(x) for x in values),
        strikes=strike_grid,
        price=price,
        stderr=stderr,
        discount_factors=discounts,
        forward_factors=forward_factors,
        model_forwards=model_forwards,
        sample_forward=sample_forward,
        sample_forward_stderr=sample_forward_stderr,
        S0=float(paths.S0),
        q=float(paths.q),
        n_paths=int(paths.n_paths),
        n_base_paths=int(paths.n_base_paths),
        antithetic=bool(paths.antithetic),
        stderr_is_conservative=bool(paths.config.qmc),
    )


# ---------------------------------------------------------------------------
# Public pricing API
# ---------------------------------------------------------------------------
def price_call(
    paths: RBergomiPathSet,
    *,
    strikes: Any,
    maturities: Any = None,
    estimator: str = ESTIMATOR_PLAIN,
) -> PriceResult:
    """
    European call prices and standard errors on a strike x maturity grid.

    Parameters
    ----------
    paths:
        The path set to price on.  Every maturity must sit on its grid.
    strikes:
        Scalar, ``(n_K,)`` (broadcast to every maturity) or ``(n_T, n_K)``.
    maturities:
        Maturities to price; defaults to **all** the quoted maturities of the
        path set, which is the point of the one-path-set design.
    estimator:
        :data:`ESTIMATOR_PLAIN` (default) or :data:`ESTIMATOR_CONDITIONAL`.

    Returns
    -------
    PriceResult
        ``price`` and ``stderr`` are ``(n_T, n_K)``.
    """
    return _price_one_type(
        paths,
        option_type=OPTION_CALL,
        strikes=strikes,
        maturities=maturities,
        estimator=estimator,
    )


def price_put(
    paths: RBergomiPathSet,
    *,
    strikes: Any,
    maturities: Any = None,
    estimator: str = ESTIMATOR_PLAIN,
) -> PriceResult:
    """European put prices and standard errors - see :func:`price_call`."""
    return _price_one_type(
        paths,
        option_type=OPTION_PUT,
        strikes=strikes,
        maturities=maturities,
        estimator=estimator,
    )


def price_calls_and_puts(
    paths: RBergomiPathSet,
    *,
    strikes: Any,
    maturities: Any = None,
    estimator: str = ESTIMATOR_PLAIN,
) -> tuple[PriceResult, PriceResult]:
    """
    Calls and puts on the **same** sample, so put-call parity is exact.

    Returns ``(calls, puts)``.
    """
    calls = price_call(
        paths, strikes=strikes, maturities=maturities, estimator=estimator
    )
    puts = price_put(
        paths, strikes=strikes, maturities=maturities, estimator=estimator
    )
    return calls, puts


def put_call_parity_residual(calls: PriceResult, puts: PriceResult) -> np.ndarray:
    """
    ``C - P - D(T) (sample_forward(T) - K)`` on the shared sample.

    An algebraic identity: the residual is round-off (``~1e-10`` absolute on
    equity-sized prices), *not* a statistical quantity.  Both results must come
    from the same path set, the same estimator and the same strike grid.
    """
    if calls.estimator != puts.estimator:
        raise ValueError(
            "Put-call parity is exact only across the same estimator; got "
            f"{calls.estimator!r} vs {puts.estimator!r}."
        )
    if calls.maturities != puts.maturities:
        raise ValueError("Calls and puts must share the same maturities.")
    if calls.strikes.shape != puts.strikes.shape or not np.array_equal(
        calls.strikes, puts.strikes
    ):
        raise ValueError("Calls and puts must share the same strike grid.")
    expected = calls.discount_factors[:, None] * (
        calls.sample_forward[:, None] - calls.strikes
    )
    return calls.price - puts.price - expected


# ---------------------------------------------------------------------------
# Implied volatilities
# ---------------------------------------------------------------------------
def implied_vol_surface(result: PriceResult) -> np.ndarray:
    """
    Invert a :class:`PriceResult` into Black implied volatilities.

    Uses the repo's Brent inverter through the Black-76 substitution
    (``S0 -> F*D``, ``r -> -ln(D)/T``, ``q -> 0``), which reproduces
    ``C = D (F N(d1) - K N(d2))`` exactly.  Put prices are first converted to
    calls with the **sample** put-call parity of this path set, so the inverted
    volatility is the one implied by the price the Monte Carlo actually
    produced.

    Returns
    -------
    ``(n_T, n_K)`` array of implied volatilities, ``NaN`` wherever the price
    falls outside the no-arbitrage band or Brent cannot bracket a root.  Nothing
    is extrapolated or clamped: an un-invertible price stays ``NaN``.
    """
    if not isinstance(result, PriceResult):
        raise TypeError(
            f"result must be a PriceResult; got {type(result).__name__}."
        )
    n_T, n_K = result.strikes.shape
    out = np.full((n_T, n_K), np.nan, dtype=float)
    for row, T in enumerate(result.maturities):
        maturity = float(T)
        discount = float(result.discount_factors[row])
        # Invert against the SAMPLE forward, not the model forward.
        #
        # The Monte-Carlo price is consistent with the forward the sample
        # actually realises, so the no-arbitrage floor that `implied_vol_call`
        # enforces must be built from that same forward. Using `model_forwards`
        # here made the floor `D*(F_model - K)` while the price carried
        # `D*(F_sample - K)`: whenever `F_sample < F_model` by more than the ITM
        # intrinsic slack, deep-ITM calls fell below their own bound and came
        # back NaN. Because the sampling error in the forward is largest in
        # relative terms at the SHORT end, that wiped out precisely the
        # maturities spec 4.5 regresses to obtain H.
        #
        # It also makes the two branches consistent: the put conversion below
        # already used the sample forward, so calls and puts now invert against
        # one forward and put-call parity holds exactly on the shared sample.
        # The F_sample vs F_model gap is not hidden -- it is reported by the
        # martingale diagnostics.
        forward = float(result.sample_forward[row])
        if maturity <= 0.0 or discount <= 0.0 or forward <= 0.0:
            continue
        spot_equivalent = forward * discount
        rate_equivalent = -math.log(discount) / maturity
        for col in range(n_K):
            strike = float(result.strikes[row, col])
            price = float(result.price[row, col])
            if not np.isfinite(price):
                continue
            noise = float(result.stderr[row, col])
            if result.option_type == OPTION_PUT:
                # Sample parity: C = P + D (F_sample - K).
                #
                # For an ITM put (K > F) this converts a LARGE put into a SMALL
                # OTM call, and the subtraction cancels catastrophically: deep
                # enough in, the plain-estimator put price IS exactly
                # D*(K - F_sample) (no path finished above K), so the two terms
                # agree to every bit they have and the residue left over is pure
                # round-off. Inverting that residue produced non-monotone
                # garbage IVs and, when it landed just above zero, the inverter's
                # fabricated floor. Guard on the cancellation scale.
                parity_leg = discount * (forward - strike)
                cancellation_scale = max(abs(price), abs(parity_leg))
                price = price + parity_leg
            else:
                cancellation_scale = abs(price)

            # All the volatility information lives in the TIME VALUE. Guard on
            # that, not on the price: it treats both wings symmetrically.
            #   * deep OTM call  -> price ~ 0, intrinsic 0        -> tv ~ 0
            #   * deep OTM put   -> converts to a deep ITM call,
            #                       price ~ intrinsic             -> tv ~ 0
            # In both cases the sample resolved no time value at all, and
            # `implied_vol_call` would hand back its Brent floor `vol_min = 1e-4`
            # -- a finite, plausible-looking, fabricated volatility whose cell
            # count is monotone in maturity, i.e. one more short-end corruption.
            # A time value below the round-off scale of the parity subtraction,
            # or below one standard error of the estimate, is not a price.
            intrinsic = max(discount * (forward - strike), 0.0)
            time_value = price - intrinsic
            floor = 64.0 * float(np.finfo(np.float64).eps) * max(
                cancellation_scale, intrinsic
            )
            if math.isfinite(noise) and noise > 0.0:
                floor = max(floor, noise)
            if time_value <= floor:
                continue
            # A call-equivalent price of zero carries NO volatility information:
            # every path finished out of the money. `implied_vol_call` would
            # return its Brent floor `vol_min = 1e-4` there -- `bs_call_price`
            # underflows to exactly 0.0 at that vol for a far strike, so the
            # bracket has a root sitting on the endpoint -- handing back a
            # finite, plausible-looking, entirely fabricated number. The count of
            # such cells is monotone in maturity (worst at the short end), so
            # left alone it is one more maturity-dependent corruption of the IV
            # surface. The module contract is NaN, and NaN is what it returns.
            if price <= 0.0:
                continue
            out[row, col] = implied_vol_call(
                price, spot_equivalent, strike, maturity, rate_equivalent, 0.0
            )
    return out


# ---------------------------------------------------------------------------
# Model-level diagnostics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class MartingaleDiagnostics:
    """
    ``E[D(T) S_T e^{qT}] == S0`` check, maturity by maturity.

    The left-point log-Euler scheme is an **exact** martingale in expectation
    (see the simulator docstring), so any deviation here is pure Monte-Carlo
    noise: ``max_abs_z`` is the worst deviation measured in standard errors and
    should sit at a few units, not tens.
    """

    maturities: tuple[float, ...]
    deflated_mean: np.ndarray
    deflated_stderr: np.ndarray
    ratio: np.ndarray
    z_score: np.ndarray
    S0: float

    @property
    def max_abs_z(self) -> float:
        finite = self.z_score[np.isfinite(self.z_score)]
        return float(np.max(np.abs(finite))) if finite.size else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maturities": [float(x) for x in self.maturities],
            "deflated_mean": [float(x) for x in self.deflated_mean],
            "deflated_stderr": [float(x) for x in self.deflated_stderr],
            "ratio": [float(x) for x in self.ratio],
            "z_score": [float(x) for x in self.z_score],
            "S0": float(self.S0),
            "max_abs_z": float(self.max_abs_z),
        }


def martingale_diagnostics(
    paths: RBergomiPathSet, *, maturities: Any = None
) -> MartingaleDiagnostics:
    """
    Check ``E[D(T) S_T e^{qT}] = S0`` at every requested maturity.

    ``D(T) e^{qT} S_T`` is deflated with the path set's own discount factors, so
    the statistic isolates the model, not the rate conventions.
    """
    values, _ = paths.resolve_maturities(maturities)
    n = len(values)
    mean = np.empty(n, dtype=float)
    stderr = np.empty(n, dtype=float)
    for row, T in enumerate(values):
        deflated = (
            paths.spot(T) * paths.discount_factor(T) * math.exp(paths.q * float(T))
        )
        m, s = paths.mean_stderr(deflated)
        mean[row] = float(m)
        stderr[row] = float(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(stderr > 0.0, (mean - paths.S0) / stderr, np.nan)
    return MartingaleDiagnostics(
        maturities=tuple(float(x) for x in values),
        deflated_mean=mean,
        deflated_stderr=stderr,
        ratio=mean / float(paths.S0),
        z_score=np.asarray(z, dtype=float),
        S0=float(paths.S0),
    )


@dataclass(frozen=True, eq=False)
class VarianceSwapEstimate:
    """
    Monte-Carlo variance-swap strikes ``E[(1/T) int_0^T V du]``.

    ``xi0_reference`` is the same quantity computed from the input forward
    variance curve (``curve.integrated(T)/T``), i.e. the ``K_var`` the pipeline
    fed in.  ``quadrature_error`` is the exact, deterministic gap contributed by
    the ``xi0`` node-vs-cell discretisation (zero in
    :data:`~app.model.volatility_models.rbergomi.simulator_xi_curve.XI0_CELL_AVERAGE`
    mode); the Monte-Carlo estimate is expected to reproduce
    ``xi0_reference + quadrature_error / T``, never ``xi0_reference`` alone.
    """

    maturities: tuple[float, ...]
    k_var: np.ndarray
    k_var_stderr: np.ndarray
    xi0_reference: np.ndarray
    quadrature_error: np.ndarray

    @property
    def expected(self) -> np.ndarray:
        """``xi0_reference + quadrature_error / T`` - what the MC must match."""
        T = np.asarray(self.maturities, dtype=float)
        return self.xi0_reference + self.quadrature_error / T

    def to_dict(self) -> dict[str, Any]:
        return {
            "maturities": [float(x) for x in self.maturities],
            "k_var": [float(x) for x in self.k_var],
            "k_var_stderr": [float(x) for x in self.k_var_stderr],
            "xi0_reference": [float(x) for x in self.xi0_reference],
            "quadrature_error": [float(x) for x in self.quadrature_error],
            "expected": [float(x) for x in self.expected],
        }


def variance_swap_estimate(
    paths: RBergomiPathSet,
    *,
    maturities: Any = None,
    xi_curve: Any = None,
) -> VarianceSwapEstimate:
    """
    Monte-Carlo variance-swap strikes implied by the simulated variance paths.

    ``E[(1/T) int_0^T V du] = (1/T) int_0^T xi0(u) du = K_var(T)`` is the
    defining property of the forward-variance curve, so this is the sharpest
    check that the ``-0.5 eta^2 t^{2H}`` compensator and the ``dt`` weighting are
    both right.

    Parameters
    ----------
    xi_curve:
        Optional :class:`ForwardVarianceCurve` used to fill ``xi0_reference``.
        When omitted the reference is reconstructed from the path set's own
        ``xi0`` samples, ``(sum steps*dt - quadrature_error)/T``, which is the
        curve's exact integral by definition of the quadrature error.
    """
    values, indices = paths.resolve_maturities(maturities)
    n = len(values)
    k_var = np.empty(n, dtype=float)
    k_var_stderr = np.empty(n, dtype=float)
    for row, T in enumerate(values):
        m, s = paths.mean_stderr(paths.realised_variance(T))
        k_var[row] = float(m)
        k_var_stderr[row] = float(s)

    T_array = np.asarray(values, dtype=float)
    quadrature = np.asarray(paths.xi0_quadrature_error)[indices]
    if xi_curve is not None:
        reference = np.asarray(xi_curve.integrated(T_array), dtype=float) / T_array
    else:
        dt = np.asarray(paths.grid.dt, dtype=float)
        accumulated = np.concatenate(
            [[0.0], np.cumsum(np.asarray(paths.xi0_on_grid.steps) * dt)]
        )
        reference = (accumulated[indices] - quadrature) / T_array

    return VarianceSwapEstimate(
        maturities=tuple(float(x) for x in values),
        k_var=k_var,
        k_var_stderr=k_var_stderr,
        xi0_reference=np.asarray(reference, dtype=float),
        quadrature_error=np.asarray(quadrature, dtype=float),
    )


def forward_variance_expectation(
    paths: RBergomiPathSet, *, times: Any = None
) -> dict[str, np.ndarray]:
    """
    ``E[V_t]`` against ``xi0(t)`` at grid nodes - the compensator check.

    ``V_t = xi0(t) exp(eta W~_t - eta^2 t^{2H}/2)`` and ``Var(W~_t) = t^{2H}``,
    so ``E[V_t] = xi0(t)`` **exactly**; any systematic gap means the
    ``-0.5 eta^2 t^{2H}`` compensator or the driver variance is wrong.

    Parameters
    ----------
    times:
        Grid nodes to report on.  ``None`` means every node.  Values that are
        not grid nodes are rejected rather than snapped.

    Returns
    -------
    ``{"times", "mean", "stderr", "xi0", "z_score"}``.
    """
    node_times = np.asarray(paths.grid.t, dtype=float)
    if times is None:
        # Skip t = 0. There V_0 = xi0(0) is deterministic on every path, so both
        # the numerator (mean - xi0) and the denominator (stderr) are pure
        # floating-point round-off. The `stderr > 0.0` guard below does not catch
        # it -- round-off is rarely exactly zero -- so the node produces a
        # meaningless z of arbitrary magnitude that then dominates the
        # `nanmax(|z|)` this feeds into `pricing_report`, making a healthy run
        # look broken.
        indices = np.arange(1, node_times.size, dtype=int)
    else:
        wanted = np.atleast_1d(np.asarray(times, dtype=float))
        indices = np.empty(wanted.size, dtype=int)
        for j, value in enumerate(wanted):
            match = np.flatnonzero(node_times == float(value))
            if match.size != 1:
                raise KeyError(
                    f"{float(value)!r} is not a grid node of this path set; "
                    "forward_variance_expectation never snaps a time."
                )
            indices[j] = int(match[0])

    mean, stderr = paths.mean_stderr(paths.variance[:, indices])
    xi0 = np.asarray(paths.xi0_on_grid.nodes)[indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(stderr > 0.0, (mean - xi0) / stderr, np.nan)
    return {
        "times": node_times[indices],
        "mean": np.asarray(mean, dtype=float),
        "stderr": np.asarray(stderr, dtype=float),
        "xi0": np.asarray(xi0, dtype=float),
        "z_score": np.asarray(z, dtype=float),
    }


def spot_volterra_cross_moment(
    paths: RBergomiPathSet, *, maturities: Any = None
) -> dict[str, np.ndarray]:
    """
    ``E[W^S_t W~_t]`` against ``rho sqrt(2H)/(H+1/2) t^{H+1/2}``.

    This is *the* test that the spot/vol correlation is realised through the
    shared driver ``B`` rather than by correlating fractional-Brownian values.
    The closed form is the Ito isometry of the Volterra kernel over ``[0, t]``
    times ``rho`` (see the module docstring of ``volterra_gaussian``).
    """
    values, _ = paths.resolve_maturities(maturities)
    H = paths.params.H
    rho = paths.params.rho
    n = len(values)
    mean = np.empty(n, dtype=float)
    stderr = np.empty(n, dtype=float)
    expected = np.empty(n, dtype=float)
    for row, T in enumerate(values):
        product = paths.spot_brownian(T) * paths.volterra(T)
        m, s = paths.mean_stderr(product)
        mean[row] = float(m)
        stderr[row] = float(s)
        expected[row] = (
            rho * math.sqrt(2.0 * H) / (H + 0.5) * float(T) ** (H + 0.5)
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(stderr > 0.0, (mean - expected) / stderr, np.nan)
    return {
        "maturities": np.asarray(values, dtype=float),
        "mean": mean,
        "stderr": stderr,
        "expected": expected,
        "z_score": np.asarray(z, dtype=float),
    }


def pricing_report(
    paths: RBergomiPathSet,
    *,
    strikes: Any,
    maturities: Any = None,
    estimator: str = ESTIMATOR_CONDITIONAL,
) -> dict[str, Any]:
    """
    One JSON-safe bundle: prices, implied vols and every model-level diagnostic.

    Intended for the controller / reporting layer; it never raises on a
    non-invertible price (the implied-volatility grid simply carries ``NaN``).
    """
    calls, puts = price_calls_and_puts(
        paths, strikes=strikes, maturities=maturities, estimator=estimator
    )
    residual = put_call_parity_residual(calls, puts)
    expectation = forward_variance_expectation(paths)
    cross = spot_volterra_cross_moment(paths, maturities=maturities)
    return {
        "simulation": paths.diagnostics(),
        "calls": calls.to_dict(),
        "puts": puts.to_dict(),
        "implied_vol_call": [
            [float(x) for x in row] for row in implied_vol_surface(calls)
        ],
        "put_call_parity_max_abs_residual": float(np.max(np.abs(residual))),
        "martingale": martingale_diagnostics(
            paths, maturities=maturities
        ).to_dict(),
        "variance_swap": variance_swap_estimate(
            paths, maturities=maturities
        ).to_dict(),
        "forward_variance_expectation": {
            "times": [float(x) for x in expectation["times"]],
            "mean": [float(x) for x in expectation["mean"]],
            "stderr": [float(x) for x in expectation["stderr"]],
            "xi0": [float(x) for x in expectation["xi0"]],
            "max_abs_z": float(
                np.nanmax(np.abs(expectation["z_score"]))
                if np.any(np.isfinite(expectation["z_score"]))
                else np.nan
            ),
        },
        "spot_volterra_cross_moment": {
            "maturities": [float(x) for x in cross["maturities"]],
            "mean": [float(x) for x in cross["mean"]],
            "expected": [float(x) for x in cross["expected"]],
            "stderr": [float(x) for x in cross["stderr"]],
            "max_abs_z": float(
                np.nanmax(np.abs(cross["z_score"]))
                if np.any(np.isfinite(cross["z_score"]))
                else np.nan
            ),
        },
    }


__all__ = [
    "ESTIMATORS",
    "ESTIMATOR_CONDITIONAL",
    "ESTIMATOR_PLAIN",
    "OPTION_CALL",
    "OPTION_PUT",
    "MartingaleDiagnostics",
    "PriceResult",
    "VarianceSwapEstimate",
    "black_call_from_total_variance",
    "black_put_from_total_variance",
    "forward_variance_expectation",
    "implied_vol_surface",
    "martingale_diagnostics",
    "price_call",
    "price_calls_and_puts",
    "price_put",
    "pricing_report",
    "put_call_parity_residual",
    "spot_volterra_cross_moment",
    "variance_swap_estimate",
]
