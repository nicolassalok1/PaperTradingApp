"""
Validation report and diagnostics of the rough-Bergomi Hurst pipeline (spec 9).

Target of this module: turn a finished :class:`JointCalibrationResult`
(spec 4.10 / 4.11) plus the market-side artefacts that produced it into ONE
JSON-safe dictionary, and - optionally, off by default - into PNG figures.

WHAT SPEC 9 ASKS FOR, AND WHERE IT LANDS IN THE REPORT
------------------------------------------------------
============================================  =================================
spec 9 item                                    key in :func:`build_calibration_diagnostics`
============================================  =================================
calibrated ``H``, ``eta``, ``rho``             ``parameters.calibrated``
initial ``H0`` and its confidence interval     ``parameters.H0`` / ``H0_ci95``
calibration loss, CRN and fresh seed           ``loss.crn`` / ``loss.fresh``
RMSE / MAE / max IV error                      ``iv_error`` (+ ``iv_error_weighted``)
price RMSE                                     ``price_error.rmse``
ATM-skew error term structure                  ``atm_skew_term_structure``
per-maturity smile error                       ``per_maturity[*].iv``
ATM IV term-structure error                    ``atm_iv_term_structure``
variance term structure (CONSISTENCY CHECK)    ``variance_term_structure``
spec-4.11 profile slices                       ``identifiability.profiles``
============================================  =================================

THE TWO KINDS OF NUMBER IN HERE - DO NOT CONFLATE THEM
-------------------------------------------------------
*Fit metrics* answer "how far is the model from the market?": the IV errors, the
price errors, the ATM level and ATM skew term-structure errors, the per-maturity
smile errors. They are all computed on quantities the optimiser actually saw.

*Consistency checks* answer "did the pipeline stay internally coherent?" and are
**expected to be ~0 by construction**. ``variance_term_structure`` compares the
model's own variance-swap curve ``(1/T) * integral_0^T xi0`` against the market
``K_var(T)`` the spec-4.4 curve was built from - the two agree to the isotonic
repair and the level floor, and nothing else. Reporting that number as a *fit*
quality would be self-congratulation: it measures the arithmetic of spec 4.4 and,
usefully, that ``xi0`` was never modified during the calibration (spec 4.10's
frozen-``xi0`` invariant, made visible). The same holds for
``forward_variance_term_structure``. Both carry ``"role": "consistency_check"``
and a French explanation so a reader cannot mistake one for the other.

Both variance curves are *analytic* model expectations, not Monte-Carlo
estimates: under rough Bergomi ``E[(1/T) integral_0^T V_u du] = (1/T) integral_0^T
xi0(u) du`` exactly. A simulated variance swap would additionally carry the
discretisation and sampling error of the scheme; use
``rbergomi.pricing.variance_swap_estimate`` for that measurement, it is a
different question.

MODEL-SIDE SKEW AND ATM LEVEL
------------------------------
The model ATM skew ``psi_model(T)`` and ATM level ``sigma_ATM_model(T)`` are read
off the **model implied-volatility surface at the quoted strikes**, through the
very same spec-4.5 estimator (:func:`build_skew_curve`) that produced the market
side. Same local quadratic, same window rule, same weighting - so the comparison
is apples to apples and any bias of the estimator cancels to first order in the
*difference*. It also means these are not independent measurements of the model:
they inherit whatever the estimator does, and the Monte-Carlo noise of
``iv_model`` rides along in ``psi_model``.

LAYERING
--------
This module is model-layer only: no streamlit, no controller, no view. The
plotting helper imports ``matplotlib`` **lazily inside the function** and forces
the ``Agg`` backend, so importing this module offline and headless costs nothing
and the tests never touch matplotlib. PNG generation is opt-in
(:func:`save_diagnostic_figures`), never a side effect of building the report.

DUCK TYPING
-----------
:func:`build_calibration_diagnostics` reads its inputs by attribute, never by
``isinstance``. That is deliberate: it keeps this module out of the import cycle
of ``volatility_models.rbergomi`` and it lets the unit tests drive it with small
hand-built stand-ins instead of a real Monte-Carlo calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from app.model.calibration.loss_surface import (
    iv_error_metrics,
    iv_error_metrics_weighted,
)
from app.model.calibration.rough_vol.forward_curve import SurfacePoint
from app.model.calibration.rough_vol.hurst_estimator import SkewConfig, build_skew_curve

__all__ = [
    "CONSISTENCY_CHECK",
    "DEFAULT_XI0_CONSISTENCY_ATOL",
    "FIGURE_ATM_SKEW",
    "FIGURE_FORWARD_VARIANCE",
    "FIGURE_HURST",
    "FIGURE_IV_SURFACE",
    "FIGURE_PROFILES",
    "FIGURE_RESIDUAL_HEATMAP",
    "FIGURE_VARIANCE_SWAP",
    "FIT_METRIC",
    "KNOWN_OPEN_ITEMS_FR",
    "DiagnosticsConfig",
    "PipelineArtifacts",
    "atm_iv_term_structure",
    "atm_skew_term_structure",
    "attach_diagnostics",
    "build_calibration_diagnostics",
    "diagnostics_summary_fr",
    "forward_variance_term_structure",
    "market_surface_points",
    "model_surface_points",
    "per_maturity_errors",
    "price_error_metrics",
    "quote_table",
    "residual_grid",
    "save_diagnostic_figures",
    "variance_term_structure",
]


#: Role tags. A reader (or a UI) must be able to tell a fit metric from an
#: internal-coherence check without reading the prose.
FIT_METRIC = "fit_metric"
CONSISTENCY_CHECK = "consistency_check"

#: Absolute tolerance on ``integrated(T)/T - K_var(T)``. The spec-4.4
#: piecewise-constant construction is exact at every market maturity, so the only
#: legitimate departures are the isotonic repair and the ``eps_xi`` level floor -
#: both of which the curve reports itself.
DEFAULT_XI0_CONSISTENCY_ATOL = 1e-10

FIGURE_IV_SURFACE = "iv_surface"
FIGURE_ATM_SKEW = "atm_skew"
FIGURE_VARIANCE_SWAP = "variance_swap"
FIGURE_FORWARD_VARIANCE = "forward_variance"
FIGURE_HURST = "hurst"
FIGURE_RESIDUAL_HEATMAP = "residual_heatmap"
FIGURE_PROFILES = "profiles"

#: The open items Phases 2-4 measured and deliberately did NOT fix. Spec 11 wants
#: them carried into the report rather than rediscovered by the next reader.
KNOWN_OPEN_ITEMS_FR: tuple[str, ...] = (
    "K_var porte le biais de discrétisation de la grille de strikes (forme CBOE, "
    "spec 4.3) ; il est quantifié dans diagnostics.discretisation_bias et signalé "
    "par FLAG_COARSE_STRIKE_LADDER. Il n'est pas retranché : il se reporte "
    "intégralement sur xi0.",
    "Le schéma log-Euler sous-estime le skew ATM du modèle davantage à l'échéance "
    "courte qu'à l'échéance longue ; le décalage résiduel sur theta est MESURÉ par "
    "grid_refinement_bias et rapporté (grid_bias), jamais corrigé en silence.",
    "La régression de skew de la spec 4.5 est biaisée vers le bas sur la surface de "
    "référence de la phase 3 (H0 = 0,0827 ± 0,0061 pour une vérité de 0,10, donc la "
    "vérité tombe HORS de son propre IC95). H0 est un point de départ, pas une mesure.",
    "CalibratorSettings.max_nfev = 80 transmis explicitement est indiscernable de "
    "« non transmis » : le calibrateur applique alors son propre budget "
    "(local_nfev_per_param * n_free).",
)


# ---------------------------------------------------------------------------
# Configuration and inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Knobs of the spec-9 report.

    Attributes
    ----------
    skew_config:
        The :class:`SkewConfig` used to re-measure the ATM skew and the ATM level
        on BOTH sides. One object for both is the point: the market and the model
        must go through the same estimator.
    xi0_consistency_atol:
        Absolute tolerance of the ``xi0`` / ``K_var`` consistency check.
    max_residual_rows:
        Cap on the number of per-quote residual rows carried in the report. The
        residual heatmap needs them all; a UI payload does not. ``0`` disables
        the cap.
    figure_dpi, figure_size:
        Only used by :func:`save_diagnostic_figures`.
    """

    skew_config: SkewConfig | None = None
    xi0_consistency_atol: float = DEFAULT_XI0_CONSISTENCY_ATOL
    max_residual_rows: int = 0
    figure_dpi: int = 120
    figure_size: tuple[float, float] = (7.5, 4.5)


@dataclass(frozen=True)
class PipelineArtifacts:
    """
    The market-side objects the calibration was built from.

    Every field is optional: the report degrades section by section rather than
    refusing to be produced. A missing input yields ``{"available": False,
    "reason_fr": ...}`` for its section, never a fabricated number.
    """

    market_surface: Sequence[Any] = ()
    forward_curve: Sequence[Any] = ()
    clean_chains: Sequence[Any] = ()
    variance_curve: Any = None
    xi0_curve: Any = None
    hurst: Any = None
    skew_curve: Sequence[Any] = ()
    initializer_diagnostics: Mapping[str, Any] | None = None
    source: str = ""
    is_synthetic: bool = False
    notes_fr: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Small numeric helpers (plain-Python outputs only - the report must survive
# the controller's `_json_safe` without any numpy scalar surviving in it)
# ---------------------------------------------------------------------------
def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite(value: Any) -> bool:
    return math.isfinite(_f(value))


def _floats(values: Any) -> list[float]:
    return [_f(v) for v in np.asarray(values, dtype=float).ravel()]


def _error_stats(error: Sequence[float]) -> dict[str, float]:
    """``mae`` / ``rmse`` / ``max_abs`` / ``n`` over the finite entries."""
    arr = np.asarray(list(error), dtype=float)
    return {k: float(v) for k, v in iv_error_metrics(arr, np.isfinite(arr)).items()}


def _relative(error: Sequence[float], reference: Sequence[float]) -> list[float]:
    err = np.asarray(list(error), dtype=float)
    ref = np.asarray(list(reference), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(ref) > 0.0, err / ref, np.nan)
    return [float(x) for x in out]


def _sorted_unique(values: Sequence[float], *, rtol: float = 1e-9) -> list[float]:
    out: list[float] = []
    for v in sorted(_f(x) for x in values):
        if not math.isfinite(v):
            continue
        if out and abs(v - out[-1]) <= rtol * max(1.0, abs(v)):
            continue
        out.append(v)
    return out


def _match(value: float, candidates: Sequence[float], *, rtol: float = 1e-9) -> int:
    """Index of ``value`` among ``candidates``, or ``-1``."""
    target = _f(value)
    for i, c in enumerate(candidates):
        if abs(_f(c) - target) <= rtol * max(1.0, abs(target)):
            return i
    return -1


def _unavailable(reason_fr: str) -> dict[str, Any]:
    return {"available": False, "reason_fr": str(reason_fr)}


# ---------------------------------------------------------------------------
# Quote-level table
# ---------------------------------------------------------------------------
def quote_table(result: Any) -> dict[str, list[Any]]:
    """
    Flatten the calibration result into per-quote columns.

    Every column is a plain Python list of the same length. ``iv_error`` is the
    calibrator's own ``iv_model - iv_market`` (see ``JointObjective.evaluate``),
    recomputed here only when the result does not carry it.
    """
    quotes = list(getattr(getattr(result, "quotes", None), "quotes", ()) or ())
    n = len(quotes)
    columns: dict[str, list[Any]] = {
        "T": [_f(getattr(q, "T", float("nan"))) for q in quotes],
        "K": [_f(getattr(q, "K", float("nan"))) for q in quotes],
        "k": [_f(getattr(q, "k", float("nan"))) for q in quotes],
        "F": [_f(getattr(q, "F", float("nan"))) for q in quotes],
        "D": [_f(getattr(q, "D", float("nan"))) for q in quotes],
        "iv_market": [_f(getattr(q, "iv", float("nan"))) for q in quotes],
        "price_market": [_f(getattr(q, "price", float("nan"))) for q in quotes],
        "vega": [_f(getattr(q, "vega", float("nan"))) for q in quotes],
        "spread_iv": [_f(getattr(q, "spread_iv", float("nan"))) for q in quotes],
        "option_type": [str(getattr(q, "option_type", "")) for q in quotes],
    }

    def _column(name: str, fallback: float = float("nan")) -> list[float]:
        raw = getattr(result, name, None)
        if raw is None:
            return [fallback] * n
        arr = np.asarray(raw, dtype=float).ravel()
        if arr.size != n:
            return [fallback] * n
        return [float(x) for x in arr]

    columns["iv_model"] = _column("iv_model")
    columns["price_model"] = _column("price_model")
    columns["weight"] = _column("weights")
    if all(math.isnan(w) for w in columns["weight"]):
        columns["weight"] = [
            _f(getattr(q, "weight", float("nan"))) for q in quotes
        ]

    iv_error = _column("iv_error")
    if all(math.isnan(e) for e in iv_error):
        iv_error = [
            columns["iv_model"][i] - columns["iv_market"][i] for i in range(n)
        ]
    columns["iv_error"] = iv_error
    columns["price_error"] = [
        columns["price_model"][i] - columns["price_market"][i] for i in range(n)
    ]
    return columns


def _surface_points(
    table: Mapping[str, Sequence[Any]], *, iv_key: str
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    n = len(table["T"])
    for i in range(n):
        iv = _f(table[iv_key][i])
        if not math.isfinite(iv) or iv <= 0.0:
            continue
        points.append(
            SurfacePoint(
                T=_f(table["T"][i]),
                K=_f(table["K"][i]),
                k=_f(table["k"][i]),
                F=_f(table["F"][i]),
                D=_f(table["D"][i]),
                iv=iv,
                option_type=str(table["option_type"][i]) or "call",
                mid=float("nan"),
                call_equivalent_price=float("nan"),
                vendor_iv=float("nan"),
                one_sided=False,
            )
        )
    return points


def market_surface_points(result: Any) -> list[SurfacePoint]:
    """The market quotes the calibration was scored against, as surface points."""
    return _surface_points(quote_table(result), iv_key="iv_market")


def model_surface_points(result: Any) -> list[SurfacePoint]:
    """
    The calibrated model's implied volatilities on the SAME ``(T, K)`` grid.

    NaN entries (a Monte-Carlo sample that resolved no time value - see
    ``pricing.implied_vol_surface``) are dropped rather than replaced: a missing
    model IV is missing, not zero.
    """
    return _surface_points(quote_table(result), iv_key="iv_model")


# ---------------------------------------------------------------------------
# Term structures
# ---------------------------------------------------------------------------
def _skew_points(
    points: Sequence[SurfacePoint], *, config: SkewConfig | None
) -> tuple[list[float], list[Any]]:
    """``(maturities, skew_points)`` from the spec-4.5 estimator, sorted by ``T``."""
    if not points:
        return [], []
    skews, _failures = build_skew_curve(list(points), None, config=config)
    ordered = sorted(skews, key=lambda s: _f(s.T))
    return [_f(s.T) for s in ordered], list(ordered)


def _aligned_skews(
    result: Any, *, config: DiagnosticsConfig
) -> tuple[list[float], list[Any], list[Any]]:
    """
    Market and model skew points on ONE maturity axis.

    A maturity present on one side only keeps its slot with ``None`` on the
    other, so the caller reports a NaN rather than silently shortening the
    term structure.
    """
    table = quote_table(result)
    market_T, market = _skew_points(
        _surface_points(table, iv_key="iv_market"), config=config.skew_config
    )
    model_T, model = _skew_points(
        _surface_points(table, iv_key="iv_model"), config=config.skew_config
    )
    maturities = _sorted_unique(market_T + model_T)
    market_aligned: list[Any] = []
    model_aligned: list[Any] = []
    for T in maturities:
        i = _match(T, market_T)
        j = _match(T, model_T)
        market_aligned.append(market[i] if i >= 0 else None)
        model_aligned.append(model[j] if j >= 0 else None)
    return maturities, market_aligned, model_aligned


def atm_skew_term_structure(
    result: Any, *, config: DiagnosticsConfig | None = None
) -> dict[str, Any]:
    """
    Market vs model ATM skew ``psi(T) = d sigma / dk |_{k=0}``, per maturity.

    Both sides go through :func:`build_skew_curve`, so this is the *difference of
    two identically-estimated* quantities. A maturity whose model smile could not
    be fitted (too few invertible model IVs) is reported with ``psi_model =
    NaN`` and counted in ``n_model_failed``; it is never silently dropped.
    """
    cfg = config or DiagnosticsConfig()
    maturities, market, model = _aligned_skews(result, config=cfg)
    if not maturities:
        return _unavailable(
            "Aucune échéance n'a pu être ajustée par l'estimateur de skew (spec 4.5)."
        )

    psi_market = [_f(getattr(mk, "psi", float("nan"))) for mk in market]
    psi_model = [_f(getattr(md, "psi", float("nan"))) for md in model]
    se_market = [_f(getattr(mk, "se", float("nan"))) for mk in market]
    n_strikes = [int(getattr(mk, "n_strikes", 0) or 0) for mk in market]

    error = [psi_model[i] - psi_market[i] for i in range(len(maturities))]
    stats = _error_stats(error)
    relative = _relative(error, psi_market)
    return {
        "available": True,
        "role": FIT_METRIC,
        "definition_fr": (
            "Erreur sur la structure par terme du skew ATM : psi_modele(T) - "
            "psi_marche(T), les deux mesurés par le MÊME estimateur quadratique "
            "local de la spec 4.5."
        ),
        "maturities": [float(T) for T in maturities],
        "maturities_days": [float(T) * 365.0 for T in maturities],
        "psi_market": psi_market,
        "psi_model": psi_model,
        "se_market": se_market,
        "n_strikes_market": n_strikes,
        "error": error,
        "relative_error": relative,
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "max_abs": stats["max_abs"],
        "n": int(stats["n"]),
        "n_model_failed": int(sum(1 for x in psi_model if not math.isfinite(x))),
    }


def atm_iv_term_structure(
    result: Any, *, config: DiagnosticsConfig | None = None
) -> dict[str, Any]:
    """
    Market vs model ATM implied-volatility LEVEL per maturity.

    This is the spec-9 "term structure error": ``sigma_ATM`` is the constant of
    the same local quadratic that gives ``psi``, i.e. the fitted level at
    ``k = 0``, not the closest quoted strike. Reported in volatility units and,
    for readability, in volatility points (``x100``).
    """
    cfg = config or DiagnosticsConfig()
    maturities, market, model = _aligned_skews(result, config=cfg)
    if not maturities:
        return _unavailable(
            "Aucune échéance n'a pu être ajustée par l'estimateur de skew (spec 4.5)."
        )

    sigma_market = [_f(getattr(mk, "sigma_atm", float("nan"))) for mk in market]
    sigma_model = [_f(getattr(md, "sigma_atm", float("nan"))) for md in model]

    error = [sigma_model[i] - sigma_market[i] for i in range(len(maturities))]
    stats = _error_stats(error)
    return {
        "available": True,
        "role": FIT_METRIC,
        "definition_fr": (
            "Erreur sur la structure par terme du niveau de volatilité implicite "
            "ATM : sigma_ATM_modele(T) - sigma_ATM_marche(T), niveau ajusté en "
            "k = 0 par la quadratique locale de la spec 4.5."
        ),
        "maturities": [float(T) for T in maturities],
        "maturities_days": [float(T) * 365.0 for T in maturities],
        "sigma_atm_market": sigma_market,
        "sigma_atm_model": sigma_model,
        "error": error,
        "error_vol_points": [x * 100.0 for x in error],
        "relative_error": _relative(error, sigma_market),
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "max_abs": stats["max_abs"],
        "rmse_vol_points": stats["rmse"] * 100.0,
        "n": int(stats["n"]),
    }


def variance_term_structure(
    result: Any,
    *,
    variance_curve: Any = None,
    xi0_curve: Any = None,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """
    CONSISTENCY CHECK - model vs market variance-swap curve ``K_var(T)``.

    The model side is the exact rough-Bergomi expectation
    ``(1/T) * integral_0^T xi0(u) du``: under the model the expected realised
    variance is the integrated forward variance, with no dependence on ``H``,
    ``eta`` or ``rho`` whatsoever. The market side is the spec-4.3 replication
    the spec-4.4 curve was built from. They must agree to the isotonic repair and
    the level floor.

    **This is not a measure of fit quality.** A perfect agreement here says
    exactly two things: the spec-4.4 reconstruction is arithmetically sound, and
    ``xi0`` was not modified by the calibration (spec 4.10's frozen-``xi0``
    invariant). It says nothing at all about ``(H, eta, rho)``.
    """
    cfg = config or DiagnosticsConfig()
    curve = xi0_curve if xi0_curve is not None else getattr(result, "xi0_curve", None)
    if curve is None or not hasattr(curve, "integrated"):
        return _unavailable(
            "Courbe de variance forward (spec 4.4) absente : vérification de "
            "cohérence impossible."
        )

    points = list(getattr(variance_curve, "points", ()) or ())
    if not points:
        knots = [_f(t) for t in getattr(curve, "T_knots", ()) or ()]
        if not knots:
            return _unavailable(
                "Courbe K_var de marché (spec 4.3) absente : vérification de "
                "cohérence impossible."
            )
        market = [_f(curve.integrated(T)) / _f(T) for T in knots]
        source = "xi0_curve.V_repaired"
    else:
        knots = [_f(p.T) for p in points]
        market = [_f(p.k_var) for p in points]
        source = "variance_swap_curve"

    model = [_f(curve.integrated(T)) / _f(T) if _f(T) > 0.0 else float("nan") for T in knots]
    error = [model[i] - market[i] for i in range(len(knots))]
    stats = _error_stats(error)
    atol = float(cfg.xi0_consistency_atol)
    consistent = bool(
        stats["n"] > 0
        and math.isfinite(stats["max_abs"])
        and stats["max_abs"] <= atol
    )
    reconstruction = [
        _f(x) for x in (getattr(curve, "reconstruction_errors", lambda: ())() or ())
    ]
    xi0_holder = getattr(result, "xi0", None)
    fingerprint = getattr(xi0_holder, "fingerprint", None)
    message_fr = (
        "Cohérence xi0 / K_var vérifiée : écart maximal "
        f"{stats['max_abs']:.3e} <= tolérance {atol:.1e}. xi0 n'a pas été modifié "
        "par la calibration."
        if consistent
        else (
            "Écart xi0 / K_var de "
            f"{stats['max_abs']:.3e} au-delà de la tolérance {atol:.1e} — à "
            "expliquer par la réparation isotone ou le plancher de niveau de la "
            "spec 4.4, sinon la courbe n'est PAS celle qui a servi à la calibration."
        )
    )
    return {
        "available": True,
        "role": CONSISTENCY_CHECK,
        "definition_fr": (
            "Vérification de cohérence (et NON une mesure de qualité "
            "d'ajustement) : K_var_modele(T) = (1/T) * integrale_0^T xi0(u) du "
            "contre K_var_marche(T) issu de la réplication spec 4.3. L'écart est "
            "nul par construction puisque xi0 est reconstruit à partir de K_var "
            "et gelé pendant la calibration."
        ),
        "market_source": source,
        "maturities": [float(T) for T in knots],
        "maturities_days": [float(T) * 365.0 for T in knots],
        "k_var_market": market,
        "k_var_model": model,
        "error": error,
        "relative_error": _relative(error, market),
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "max_abs": stats["max_abs"],
        "n": int(stats["n"]),
        "tolerance": atol,
        "consistent": consistent,
        "reconstruction_errors": reconstruction,
        "xi0_fingerprint": None if fingerprint is None else str(fingerprint),
        "message_fr": message_fr,
    }


def forward_variance_term_structure(
    result: Any,
    *,
    variance_curve: Any = None,
    xi0_curve: Any = None,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """
    CONSISTENCY CHECK - model vs market forward-variance curve ``xi0(t)``.

    The market side is the finite difference of the total variance implied by the
    spec-4.3 curve, ``(V_j - V_{j-1}) / (T_j - T_{j-1})`` with ``V_j = T_j
    K_var(T_j)`` and ``V_0 = 0`` at ``T = 0`` - i.e. exactly what spec 4.4
    inverts. The model side is the curve's own level on that interval, read at
    the cell midpoint so the piecewise-constant and the monotone-PCHIP variants
    are both handled.

    Departures are *informative*: they are precisely the isotonic repairs and the
    floored levels, both of which the curve reports in its metadata.
    """
    del config  # signature kept parallel to the other term structures
    curve = xi0_curve if xi0_curve is not None else getattr(result, "xi0_curve", None)
    if curve is None or not hasattr(curve, "xi0"):
        return _unavailable(
            "Courbe de variance forward (spec 4.4) absente : vérification de "
            "cohérence impossible."
        )

    points = list(getattr(variance_curve, "points", ()) or ())
    knots = [_f(p.T) for p in points] if points else [
        _f(t) for t in getattr(curve, "T_knots", ()) or ()
    ]
    if not knots:
        return _unavailable("Aucune échéance disponible pour la variance forward.")
    if points:
        totals = [_f(p.T) * _f(p.k_var) for p in points]
        source = "variance_swap_curve"
    else:
        totals = [_f(v) for v in getattr(curve, "V_market", ()) or ()]
        source = "xi0_curve.V_market"
    if len(totals) != len(knots):
        return _unavailable(
            "Variance totale de marché incohérente avec les échéances : "
            "vérification impossible."
        )

    market: list[float] = []
    model: list[float] = []
    previous_T = 0.0
    previous_V = 0.0
    for T, V in zip(knots, totals):
        width = _f(T) - previous_T
        market.append((_f(V) - previous_V) / width if width > 0.0 else float("nan"))
        mid = 0.5 * (previous_T + _f(T))
        model.append(_f(curve.xi0(mid)))
        previous_T = _f(T)
        previous_V = _f(V)

    error = [model[i] - market[i] for i in range(len(knots))]
    stats = _error_stats(error)
    metadata = getattr(curve, "metadata", None)
    return {
        "available": True,
        "role": CONSISTENCY_CHECK,
        "definition_fr": (
            "Vérification de cohérence : xi0 du modèle sur chaque intervalle "
            "contre la variance forward implicite du marché "
            "(V_j - V_{j-1}) / (T_j - T_{j-1}). Les écarts non nuls sont les "
            "réparations isotones et les niveaux plafonnés au plancher de la "
            "spec 4.4."
        ),
        "market_source": source,
        "maturities": [float(T) for T in knots],
        "maturities_days": [float(T) * 365.0 for T in knots],
        "xi0_market": market,
        "xi0_model": model,
        "levels": [_f(x) for x in getattr(curve, "levels", ()) or ()],
        "error": error,
        "relative_error": _relative(error, market),
        "mae": stats["mae"],
        "rmse": stats["rmse"],
        "max_abs": stats["max_abs"],
        "n": int(stats["n"]),
        "method": str(getattr(curve, "method", "")),
        "extrapolation_policy": str(getattr(curve, "extrapolation_policy", "")),
        "metadata_flags": [str(f) for f in (getattr(metadata, "flags", ()) or ())],
    }


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def price_error_metrics(result: Any) -> dict[str, Any]:
    """
    Price-space error of the calibrated model, absolute and relative.

    Both legs are the OUT-OF-THE-MONEY option (that is how
    ``build_calibration_quotes`` normalises them), so the relative error is taken
    against a strictly positive reference and is meaningful - but it still blows
    up on the cheapest wings, which is why the absolute RMSE is the headline
    number spec 9 asks for.
    """
    table = quote_table(result)
    error = table["price_error"]
    stats = _error_stats(error)
    relative = _relative(error, table["price_market"])
    relative_stats = _error_stats(relative)
    return {
        "role": FIT_METRIC,
        "rmse": stats["rmse"],
        "mae": stats["mae"],
        "max_abs": stats["max_abs"],
        "n": int(stats["n"]),
        "rmse_relative": relative_stats["rmse"],
        "mae_relative": relative_stats["mae"],
        "max_abs_relative": relative_stats["max_abs"],
        "definition_fr": (
            "Erreur de prix modèle - marché sur la jambe hors de la monnaie, en "
            "unités de prix ; la version relative est rapportée au prix de marché."
        ),
    }


def per_maturity_errors(
    result: Any,
    *,
    config: DiagnosticsConfig | None = None,
    skew: Mapping[str, Any] | None = None,
    level: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Per-maturity smile error: IV metrics, price RMSE, ATM level and ATM skew.

    One row per quoted maturity, sorted. The IV metrics reuse
    :func:`iv_error_metrics` and :func:`iv_error_metrics_weighted` (vega weights)
    so the per-maturity numbers and the global ones are the same estimator.

    ``skew`` / ``level`` accept the already-built term structures so the report
    does not re-run the spec-4.5 estimator three times over the same quotes.
    """
    cfg = config or DiagnosticsConfig()
    table = quote_table(result)
    maturities = _sorted_unique(table["T"])
    skew = dict(skew) if skew is not None else atm_skew_term_structure(result, config=cfg)
    level = dict(level) if level is not None else atm_iv_term_structure(result, config=cfg)
    skew_T = skew.get("maturities", []) if skew.get("available") else []
    level_T = level.get("maturities", []) if level.get("available") else []

    rows: list[dict[str, Any]] = []
    for T in maturities:
        idx = [
            i
            for i, t in enumerate(table["T"])
            if abs(_f(t) - T) <= 1e-9 * max(1.0, abs(T))
        ]
        iv_error = np.asarray([table["iv_error"][i] for i in idx], dtype=float)
        vega = np.asarray([table["vega"][i] for i in idx], dtype=float)
        price_error = [table["price_error"][i] for i in idx]
        mask = np.isfinite(iv_error)
        metrics = iv_error_metrics(iv_error, mask)
        metrics_vw = iv_error_metrics_weighted(iv_error, mask, vega)
        j = _match(T, skew_T)
        m = _match(T, level_T)
        rows.append(
            {
                "T": float(T),
                "T_days": float(T) * 365.0,
                "n_quotes": int(len(idx)),
                "k_min": float(min((table["k"][i] for i in idx), default=float("nan"))),
                "k_max": float(max((table["k"][i] for i in idx), default=float("nan"))),
                "iv": {k: float(v) for k, v in metrics.items()},
                "iv_weighted": {k: float(v) for k, v in metrics_vw.items()},
                "iv_bias": float(np.nanmean(iv_error)) if mask.any() else float("nan"),
                "price_rmse": _error_stats(price_error)["rmse"],
                "sigma_atm_market": (
                    float(level["sigma_atm_market"][m]) if m >= 0 else float("nan")
                ),
                "sigma_atm_model": (
                    float(level["sigma_atm_model"][m]) if m >= 0 else float("nan")
                ),
                "sigma_atm_error": (
                    float(level["error"][m]) if m >= 0 else float("nan")
                ),
                "psi_market": float(skew["psi_market"][j]) if j >= 0 else float("nan"),
                "psi_model": float(skew["psi_model"][j]) if j >= 0 else float("nan"),
                "psi_error": float(skew["error"][j]) if j >= 0 else float("nan"),
            }
        )
    return rows


def residual_grid(
    result: Any, *, config: DiagnosticsConfig | None = None
) -> dict[str, Any]:
    """
    The per-quote residual cloud, as ``(K, T)`` columns - the heatmap's data.

    The quote set is NOT a rectangle (every maturity has its own strike ladder),
    so this is a scatter, not a matrix. :func:`save_diagnostic_figures` renders
    it with one row per maturity and the strikes on the ``x`` axis, which is the
    honest picture of a ragged grid.
    """
    cfg = config or DiagnosticsConfig()
    table = quote_table(result)
    n = len(table["T"])
    limit = int(cfg.max_residual_rows)
    keep = list(range(n)) if limit <= 0 or n <= limit else list(range(limit))
    truncated = len(keep) < n
    columns = {
        name: [table[name][i] for i in keep]
        for name in (
            "T",
            "K",
            "k",
            "iv_market",
            "iv_model",
            "iv_error",
            "price_market",
            "price_model",
            "price_error",
            "weight",
            "option_type",
        )
    }
    columns["T_days"] = [float(t) * 365.0 for t in columns["T"]]
    columns["iv_error_vol_points"] = [float(e) * 100.0 for e in columns["iv_error"]]
    return {
        "available": bool(n > 0),
        "role": FIT_METRIC,
        "n_quotes": int(n),
        "n_rows": int(len(keep)),
        "truncated": bool(truncated),
        "maturities": _sorted_unique(table["T"]),
        "columns": columns,
        "definition_fr": (
            "Nuage des résidus par cotation : erreur de volatilité implicite "
            "(modèle - marché) en fonction du strike et de l'échéance."
        ),
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def _parameters_section(result: Any) -> dict[str, Any]:
    params = getattr(result, "params", None)
    initial = getattr(result, "initial_params", None)
    identifiability = getattr(result, "identifiability", None)

    def _triple(obj: Any) -> dict[str, float]:
        return {
            "H": _f(getattr(obj, "H", float("nan"))),
            "eta": _f(getattr(obj, "eta", float("nan"))),
            "rho": _f(getattr(obj, "rho", float("nan"))),
        }

    ci = getattr(identifiability, "H0_ci95", None)
    has_ci = bool(getattr(identifiability, "has_H0_ci", False))
    section: dict[str, Any] = {
        "calibrated": _triple(params),
        "initial": _triple(initial),
        "bounds": {
            str(k): [_f(v[0]), _f(v[1])]
            for k, v in (getattr(result, "bounds", {}) or {}).items()
        },
        "pinned": [str(p) for p in (getattr(result, "pinned", ()) or ())],
        "H0": _f(getattr(identifiability, "H0", float("nan"))),
        "H0_se": _f(getattr(identifiability, "H0_se", float("nan"))),
        "H0_ci95": (
            [_f(ci[0]), _f(ci[1])] if has_ci and ci is not None else None
        ),
        "H0_is_fallback": bool(getattr(identifiability, "H0_is_fallback", False)),
        "H_in_H0_ci95": (
            bool(getattr(identifiability, "H_in_ci", False)) if has_ci else None
        ),
        "H_standard_error": _f(
            getattr(identifiability, "H_standard_error", float("nan"))
        ),
        "standard_errors": {
            str(k): _f(v)
            for k, v in (getattr(identifiability, "standard_errors", {}) or {}).items()
        },
        "h_comparison_fr": (
            None
            if identifiability is None
            else str(getattr(identifiability, "h_comparison_fr", ""))
        ),
        "identification_fr": (
            None
            if identifiability is None
            else str(getattr(identifiability, "identification_fr", ""))
        ),
        "xi0_frozen": True,
        "xi0_fingerprint": (
            None
            if getattr(result, "xi0", None) is None
            else str(getattr(getattr(result, "xi0"), "fingerprint", ""))
        ),
    }
    return section


def _loss_section(result: Any) -> dict[str, Any]:
    identifiability = getattr(result, "identifiability", None)
    config = getattr(result, "config", None)
    crn_matched = _f(getattr(result, "loss_crn_matched", float("nan")))
    fresh_matched = _f(getattr(result, "loss_fresh_matched", float("nan")))
    ratio = fresh_matched / crn_matched if crn_matched > 0.0 else float("nan")
    return {
        "role": FIT_METRIC,
        "crn": _f(getattr(result, "loss_crn", float("nan"))),
        "fresh": _f(getattr(result, "loss_fresh", float("nan"))),
        "initial": _f(getattr(result, "loss_initial", float("nan"))),
        "crn_matched": crn_matched,
        "fresh_matched": fresh_matched,
        "matched_paths": int(getattr(result, "matched_paths", 0) or 0),
        "fresh_seed_gap_ratio": float(ratio),
        "fresh_seed_gap_threshold": _f(
            getattr(config, "fresh_seed_gap_ratio", float("nan"))
        ),
        "rmse_fresh": _f(getattr(result, "rmse_fresh", float("nan"))),
        "rmse_fresh_vol_points": _f(getattr(result, "rmse_fresh", float("nan"))) * 100.0,
        "improvement": _f(getattr(identifiability, "improvement", float("nan"))),
        "improvement_significant": (
            None
            if identifiability is None
            else bool(getattr(identifiability, "improvement_significant", False))
        ),
        "noise_floor": _f(
            getattr(
                getattr(identifiability, "noise_floor", None),
                "difference_value",
                float("nan"),
            )
        ),
        "noise_floor_level": _f(
            getattr(getattr(identifiability, "noise_floor", None), "value", float("nan"))
        ),
        "definition_fr": (
            "crn : coût sur le tirage à nombres aléatoires communs effectivement "
            "ajusté ; fresh : coût hors échantillon sur une graine fraîche. La "
            "comparaison honnête est crn_matched contre fresh_matched, au MÊME "
            "nombre de trajectoires."
        ),
    }


def _hurst_section(artifacts: PipelineArtifacts) -> dict[str, Any]:
    hurst = artifacts.hurst
    if hurst is None:
        return _unavailable("Estimation de Hurst (spec 4.5) non fournie.")
    ci = getattr(hurst, "ci95", (float("nan"), float("nan")))
    return {
        "available": True,
        "role": FIT_METRIC,
        "H0": _f(getattr(hurst, "H0", float("nan"))),
        "se": _f(getattr(hurst, "se", float("nan"))),
        "ci95": [_f(ci[0]), _f(ci[1])],
        "r2": _f(getattr(hurst, "r2", float("nan"))),
        "n_expiries": int(getattr(hurst, "n_expiries", 0) or 0),
        "window": [
            _f(getattr(hurst, "window", (float("nan"), float("nan")))[0]),
            _f(getattr(hurst, "window", (float("nan"), float("nan")))[1]),
        ],
        "unstable": bool(getattr(hurst, "unstable", False)),
        "message_fr": str(getattr(hurst, "message_fr", "")),
    }


def build_calibration_diagnostics(
    result: Any,
    *,
    artifacts: PipelineArtifacts | None = None,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """
    Assemble the spec-9 validation report as one JSON-safe dictionary.

    Parameters
    ----------
    result:
        A finished ``JointCalibrationResult`` (spec 4.10 / 4.11), or anything
        exposing the same attributes.
    artifacts:
        The market-side objects of the run (:class:`PipelineArtifacts`). Every
        field is optional; a missing one degrades its own section only.
    config:
        :class:`DiagnosticsConfig`.

    Returns
    -------
    dict
        Plain Python types only (``float``/``int``/``bool``/``str``/``list``/
        ``dict``/``None``) - no numpy scalar, no dataclass, no tuple. The payload
        therefore survives the controller's ``_json_safe`` unchanged and is
        directly serialisable with ``json.dump``.

    Notes
    -----
    Nothing here re-runs the Monte Carlo. Every model quantity is read off the
    result's final evaluation, so the report is cheap and adds no variance of its
    own beyond the estimator applied to ``iv_model``.
    """
    cfg = config or DiagnosticsConfig()
    art = artifacts or PipelineArtifacts()

    table = quote_table(result)
    iv_error = np.asarray(table["iv_error"], dtype=float)
    mask = np.isfinite(iv_error)
    vega = np.asarray(table["vega"], dtype=float)

    metrics = getattr(result, "metrics", None) or iv_error_metrics(iv_error, mask)
    metrics_vw = getattr(result, "metrics_vw", None) or iv_error_metrics_weighted(
        iv_error, mask, vega
    )

    identifiability = getattr(result, "identifiability", None)
    grid_bias = getattr(result, "grid_bias", None)
    quotes = getattr(result, "quotes", None)

    report: dict[str, Any] = {
        "spec": "9",
        "model": "rbergomi",
        "method": "joint_h_mc",
        "success": bool(getattr(result, "success", False)),
        "message_fr": str(getattr(result, "message_fr", "")),
        "source": str(art.source),
        "is_synthetic": bool(art.is_synthetic),
        "notes_fr": [str(x) for x in art.notes_fr],
        "parameters": _parameters_section(result),
        "loss": _loss_section(result),
        "iv_error": {
            "role": FIT_METRIC,
            **{str(k): float(v) for k, v in dict(metrics).items()},
            "rmse_vol_points": float(dict(metrics).get("rmse", float("nan"))) * 100.0,
            "definition_fr": (
                "Erreur de volatilité implicite modèle - marché sur les cotations "
                "effectivement ajustées."
            ),
        },
        "iv_error_weighted": {
            "role": FIT_METRIC,
            **{str(k): float(v) for k, v in dict(metrics_vw).items()},
            "definition_fr": "Mêmes erreurs, pondérées par le vega Black-76.",
        },
        "price_error": price_error_metrics(result),
        "atm_iv_term_structure": atm_iv_term_structure(result, config=cfg),
        "atm_skew_term_structure": atm_skew_term_structure(result, config=cfg),
        "per_maturity": per_maturity_errors(result, config=cfg),
        "variance_term_structure": variance_term_structure(
            result,
            variance_curve=art.variance_curve,
            xi0_curve=art.xi0_curve,
            config=cfg,
        ),
        "forward_variance_term_structure": forward_variance_term_structure(
            result,
            variance_curve=art.variance_curve,
            xi0_curve=art.xi0_curve,
            config=cfg,
        ),
        "residuals": residual_grid(result, config=cfg),
        "hurst": _hurst_section(art),
        "identifiability": (
            None if identifiability is None else identifiability.to_dict()
        ),
        "grid_bias": None if grid_bias is None else grid_bias.to_dict(),
        "quotes": None if quotes is None else quotes.diagnostics(),
        "config": (
            None
            if getattr(result, "config", None) is None
            else getattr(result, "config").to_dict()
        ),
        "seed": (
            None if getattr(result, "seed", None) is None else int(getattr(result, "seed"))
        ),
        "n_objective_evaluations": int(
            getattr(result, "n_objective_evaluations", 0) or 0
        ),
        "elapsed_s": _f(getattr(result, "elapsed_s", float("nan"))),
        "flags": [str(f) for f in (getattr(result, "flags", ()) or ())],
        "warnings_fr": [str(w) for w in (getattr(result, "warnings_fr", ()) or ())],
        "open_items_fr": list(KNOWN_OPEN_ITEMS_FR),
        "figures": {},
    }
    report["summary_fr"] = diagnostics_summary_fr(report)
    return report


def attach_diagnostics(
    result: Any,
    *,
    artifacts: PipelineArtifacts | None = None,
    config: DiagnosticsConfig | None = None,
    key: str = "validation_report",
) -> dict[str, Any]:
    """
    Build the spec-9 report and store it in ``result.details[key]``.

    ``JointCalibrationResult`` is a frozen dataclass whose ``details`` field is a
    plain mutable ``dict`` declared with ``default_factory=dict`` - writing into
    it is the documented extension point and does not touch any frozen field.
    The report is also returned, so a caller that does not want the side effect
    can use :func:`build_calibration_diagnostics` directly.
    """
    report = build_calibration_diagnostics(result, artifacts=artifacts, config=config)
    details = getattr(result, "details", None)
    if isinstance(details, dict):
        details[str(key)] = report
    return report


def diagnostics_summary_fr(report: Mapping[str, Any]) -> str:
    """A compact French verdict, ready to print in a terminal or a log."""
    params = dict(report.get("parameters") or {})
    calibrated = dict(params.get("calibrated") or {})
    loss = dict(report.get("loss") or {})
    iv = dict(report.get("iv_error") or {})
    price = dict(report.get("price_error") or {})
    lines: list[str] = []
    verdict = "RÉUSSIE" if report.get("success") else "NON CONCLUANTE"
    lines.append(
        f"Calibration {verdict} — H = {_f(calibrated.get('H')):.4f}, "
        f"eta = {_f(calibrated.get('eta')):.4f}, rho = {_f(calibrated.get('rho')):+.4f}"
    )
    ci = params.get("H0_ci95")
    if ci:
        lines.append(
            f"H0 initial = {_f(params.get('H0')):.4f} "
            f"(IC95 [{_f(ci[0]):.4f}, {_f(ci[1]):.4f}], "
            f"SE = {_f(params.get('H0_se')):.4f})"
        )
    else:
        lines.append(f"H0 initial = {_f(params.get('H0')):.4f} (IC95 non fourni)")
    if params.get("h_comparison_fr"):
        lines.append(str(params["h_comparison_fr"]))
    lines.append(
        f"Coût CRN = {_f(loss.get('crn')):.6e} ; graine fraîche = "
        f"{_f(loss.get('fresh')):.6e} ; initial = {_f(loss.get('initial')):.6e}"
    )
    lines.append(
        f"Erreur IV : RMSE = {_f(iv.get('rmse')) * 100.0:.3f} pt de vol, "
        f"MAE = {_f(iv.get('mae')) * 100.0:.3f} pt, "
        f"max = {_f(iv.get('max_abs')) * 100.0:.3f} pt sur {int(_f(iv.get('n')) or 0)} cotations"
    )
    lines.append(
        f"Erreur de prix : RMSE = {_f(price.get('rmse')):.4f} "
        f"({_f(price.get('rmse_relative')) * 100.0:.2f} % en relatif)"
    )
    level = dict(report.get("atm_iv_term_structure") or {})
    if level.get("available"):
        lines.append(
            "Structure par terme ATM (niveau) : RMSE = "
            f"{_f(level.get('rmse')) * 100.0:.3f} pt de vol"
        )
    skew = dict(report.get("atm_skew_term_structure") or {})
    if skew.get("available"):
        lines.append(
            "Structure par terme du skew ATM : RMSE = "
            f"{_f(skew.get('rmse')):.4f} (max {_f(skew.get('max_abs')):.4f})"
        )
    variance = dict(report.get("variance_term_structure") or {})
    if variance.get("available"):
        lines.append("Contrôle de cohérence — " + str(variance.get("message_fr", "")))
    warnings = list(report.get("warnings_fr") or [])
    if warnings:
        lines.append("Avertissements : " + " | ".join(str(w) for w in warnings))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures (opt-in, lazy matplotlib, headless)
# ---------------------------------------------------------------------------
def save_diagnostic_figures(
    report: Mapping[str, Any],
    out_dir: Any,
    *,
    config: DiagnosticsConfig | None = None,
    figures: Sequence[str] | None = None,
) -> dict[str, str]:
    """
    Render the spec-9 diagnostic figures as PNG files under ``out_dir``.

    OPT-IN AND OFF BY DEFAULT. ``matplotlib`` is imported **inside** this
    function and switched to the ``Agg`` backend before ``pyplot`` is touched, so
    the module imports cleanly offline and headless and no test ever needs
    matplotlib installed.

    Figures, in spec-9 order: (1) market vs model IV surface (one panel of smiles
    per maturity), (2) market vs model ATM skew ``psi(T)``, (3) market vs model
    variance-swap curve, (4) market vs model forward-variance curve, (5) ``H0``
    with its confidence band against the calibrated ``H``, (6) the residual
    heatmap over ``(K, T)``, plus (7) the spec-4.11 profile slices.

    Returns ``{figure_name: absolute_path}`` for the figures actually produced;
    a section the report could not fill is skipped, not faked.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config or DiagnosticsConfig()
    wanted = (
        set(str(f) for f in figures)
        if figures is not None
        else {
            FIGURE_IV_SURFACE,
            FIGURE_ATM_SKEW,
            FIGURE_VARIANCE_SWAP,
            FIGURE_FORWARD_VARIANCE,
            FIGURE_HURST,
            FIGURE_RESIDUAL_HEATMAP,
            FIGURE_PROFILES,
        }
    )
    directory = Path(str(out_dir))
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _save(fig: Any, name: str) -> None:
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=int(cfg.figure_dpi), bbox_inches="tight")
        plt.close(fig)
        written[name] = str(path)

    residuals = dict(report.get("residuals") or {})
    columns = dict(residuals.get("columns") or {})

    # (1) market vs model IV surface -- smiles, one line per maturity.
    if FIGURE_IV_SURFACE in wanted and residuals.get("available"):
        maturities = list(residuals.get("maturities") or [])
        n = max(1, len(maturities))
        ncols = min(3, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(cfg.figure_size[0] * ncols / 2.0, cfg.figure_size[1] * nrows / 2.0),
            squeeze=False,
        )
        for idx, T in enumerate(maturities):
            ax = axes[idx // ncols][idx % ncols]
            sel = [
                i
                for i, t in enumerate(columns.get("T", []))
                if abs(_f(t) - _f(T)) <= 1e-9 * max(1.0, abs(_f(T)))
            ]
            order = sorted(sel, key=lambda i: _f(columns["k"][i]))
            ks = [_f(columns["k"][i]) for i in order]
            ax.plot(ks, [_f(columns["iv_market"][i]) for i in order], "o-", ms=3, label="marché")
            ax.plot(ks, [_f(columns["iv_model"][i]) for i in order], "s--", ms=3, label="modèle")
            ax.set_title(f"T = {_f(T) * 365.0:.0f} j")
            ax.set_xlabel("k = ln(K/F)")
            ax.set_ylabel("vol. implicite")
            ax.grid(alpha=0.3)
        for idx in range(len(maturities), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        axes[0][0].legend(loc="best", fontsize="small")
        fig.suptitle("Surface de volatilité implicite — marché contre modèle")
        _save(fig, FIGURE_IV_SURFACE)

    # (2) ATM skew psi(T).
    skew = dict(report.get("atm_skew_term_structure") or {})
    if FIGURE_ATM_SKEW in wanted and skew.get("available"):
        fig, ax = plt.subplots(figsize=cfg.figure_size)
        days = skew.get("maturities_days", [])
        ax.plot(days, skew.get("psi_market", []), "o-", label="marché")
        ax.plot(days, skew.get("psi_model", []), "s--", label="modèle")
        ax.set_xscale("log")
        ax.set_xlabel("échéance (jours, échelle log)")
        ax.set_ylabel(r"$\psi(T) = \partial\sigma/\partial k|_{k=0}$")
        ax.set_title("Skew ATM — marché contre modèle")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, FIGURE_ATM_SKEW)

    # (3) variance-swap curve (consistency check).
    variance = dict(report.get("variance_term_structure") or {})
    if FIGURE_VARIANCE_SWAP in wanted and variance.get("available"):
        fig, ax = plt.subplots(figsize=cfg.figure_size)
        days = variance.get("maturities_days", [])
        ax.plot(days, variance.get("k_var_market", []), "o-", label="marché (spec 4.3)")
        ax.plot(days, variance.get("k_var_model", []), "s--", label="modèle (xi0 intégré)")
        ax.set_xlabel("échéance (jours)")
        ax.set_ylabel(r"$K_{var}(T)$")
        ax.set_title("Courbe de variance-swap — contrôle de cohérence (écart nul attendu)")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, FIGURE_VARIANCE_SWAP)

    # (4) forward-variance curve (consistency check).
    forward = dict(report.get("forward_variance_term_structure") or {})
    if FIGURE_FORWARD_VARIANCE in wanted and forward.get("available"):
        fig, ax = plt.subplots(figsize=cfg.figure_size)
        days = forward.get("maturities_days", [])
        ax.step(days, forward.get("xi0_market", []), where="pre", label="marché")
        ax.step(days, forward.get("xi0_model", []), where="pre", linestyle="--", label="modèle")
        ax.set_xlabel("échéance (jours)")
        ax.set_ylabel(r"$\xi_0(t)$")
        ax.set_title("Variance forward — contrôle de cohérence")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, FIGURE_FORWARD_VARIANCE)

    # (5) H0 with its CI band against the calibrated H.
    params = dict(report.get("parameters") or {})
    if FIGURE_HURST in wanted and _finite(params.get("H0")):
        fig, ax = plt.subplots(figsize=cfg.figure_size)
        H0 = _f(params.get("H0"))
        ci = params.get("H0_ci95")
        if ci:
            ax.axhspan(_f(ci[0]), _f(ci[1]), alpha=0.2, label="IC95 de H0 (spec 4.5)")
        ax.axhline(H0, linestyle="--", label=f"H0 = {H0:.4f}")
        H = _f(dict(params.get("calibrated") or {}).get("H"))
        ax.plot([0], [H], "o", ms=9, label=f"H calibré = {H:.4f}")
        se = _f(params.get("H_standard_error"))
        if math.isfinite(se):
            ax.errorbar([0], [H], yerr=[se], fmt="none", capsize=6)
        ax.set_xticks([])
        ax.set_ylabel("H")
        ax.set_title("H initial (spec 4.5) contre H calibré (spec 4.10)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        _save(fig, FIGURE_HURST)

    # (6) residual heatmap over (K, T).
    if FIGURE_RESIDUAL_HEATMAP in wanted and residuals.get("available"):
        fig, ax = plt.subplots(figsize=cfg.figure_size)
        errors = [_f(x) * 100.0 for x in columns.get("iv_error", [])]
        finite = [e for e in errors if math.isfinite(e)]
        scale = max((abs(e) for e in finite), default=1.0) or 1.0
        sc = ax.scatter(
            [_f(x) for x in columns.get("K", [])],
            [_f(x) for x in columns.get("T_days", [])],
            c=errors,
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            s=42,
            edgecolors="none",
        )
        ax.set_yscale("log")
        ax.set_xlabel("strike")
        ax.set_ylabel("échéance (jours, échelle log)")
        ax.set_title("Résidus de volatilité implicite (points de vol.)")
        fig.colorbar(sc, ax=ax, label="modèle - marché (pt de vol.)")
        _save(fig, FIGURE_RESIDUAL_HEATMAP)

    # (7) spec-4.11 profile slices.
    identifiability = dict(report.get("identifiability") or {})
    profiles = list(identifiability.get("profiles") or [])
    if FIGURE_PROFILES in wanted and profiles:
        ncols = len(profiles)
        fig, axes = plt.subplots(
            1, ncols, figsize=(cfg.figure_size[0] * ncols / 2.0, cfg.figure_size[1]), squeeze=False
        )
        for idx, profile in enumerate(profiles):
            ax = axes[0][idx]
            ax.plot(profile.get("values", []), profile.get("losses", []), "o-")
            ax.axvline(_f(profile.get("optimum_value")), linestyle="--")
            ax.set_xlabel(str(profile.get("parameter", "")))
            ax.set_ylabel("coût")
            ax.set_title(f"Profil de {profile.get('parameter', '')}")
            ax.grid(alpha=0.3)
        fig.suptitle("Profils de vraisemblance (spec 4.11)")
        _save(fig, FIGURE_PROFILES)

    return written
