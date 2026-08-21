#!/usr/bin/env python
"""
End-to-end driver of the rough-Bergomi Hurst pipeline (spec 5).

Runs the WHOLE chain, in order, on one option chain::

    chain cleaning (4.1) -> forward curve (4.2) -> K_var (4.3) -> xi0 (4.4)
      -> ATM skew + H0 (4.5) -> initializer (4.9) -> joint MC calibration (4.10/4.11)
      -> validation report (9)

Two sources of quotes, mutually exclusive:

``--fixture <csv>``
    A committed CSV in the ``fetch_options_details_yahoo`` row schema. **Never
    touches the network**, whatever else is on the command line. This is the
    offline demo path and the one the end-to-end test drives.
``--ticker SPY``
    A live Yahoo fetch through ``app.model.market_data``. Needs a network; it is
    the only code path in this file that can open a socket, and it is imported
    lazily so that ``--fixture`` runs never even load the market-data module.

Layering. This is a script, not a view: it imports the model layer directly and
never goes through a controller. It writes no Streamlit, opens no UI, and the
model layer it calls stays Streamlit-free (``scripts/check_mvc_integrity.py``
only polices ``app/``, but the rule is the same one).

Rates. ``rf`` and ``div`` returned by the Yahoo fetch are hardcoded zeros and are
never read here. Discounting comes from the repo yield curve
(``app.model.yieldcurve``) unless a flat rate is pinned with ``--rate``; a
fixture that carries a machine-readable ``# rate:`` line in its provenance block
supplies its own, which is what makes an offline run reproducible on any machine
regardless of the local curve cache.

Exit status. ``0`` means *the pipeline ran end to end*; ``2`` means it could not
(unusable quotes, unreachable data, an impossible initialisation). Whether the
calibration **identified** ``H`` is a verdict, not a crash: it rides in
``calibration.success`` / ``report["success"]`` and is printed in the summary.
A deliberately cheap ``--paths`` budget legitimately returns
``success = False``; that is the calibrator being honest, not a failure of this
script.

Determinism. With ``--fixture`` and a fixed ``--seed`` the whole run is
reproducible: the market side has no RNG at all, and the Monte-Carlo side is
seeded through ``CalibratorSettings(seed=...)`` and uses common random numbers.

MEASURED REFERENCE RUN (2026-08-21, this machine, for calibration of expectations
only - none of these numbers is a claim about any market)::

    --fixture tests/fixtures/synthetic_rbergomi_chain.csv   (default budget,
    100 000 final paths, grid_n_max = 384, seed 20260821)

    market side (4.1 -> 4.9)  0.2 s     11 expiries, 162 OTM quotes, 11 K_var
    joint calibration         127.7 s   158 objective evaluations
    H0 = 0.1224 (CI95 [0.1056, 0.1392])   H = 0.1344, eta = 1.154, rho = -0.757
    IV RMSE 0.374 vol pt      price RMSE 0.0652 (5.96 %)
    verdict: success = False, FLAG_H_WEAKLY_IDENTIFIED

That last line is the point: **even at the full default budget this surface does
not pin ``H`` to the precision the calibrator demands**, and the run says so
instead of dressing 0.1344 up as a measurement. Use ``--paths`` to trade time
against Monte-Carlo noise; do not read a smaller ``--paths`` returning
``success = False`` as a bug.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model.calibration.base_calibrator import CalibratorSettings
from app.model.calibration.rough_vol.chain_cleaning import (
    CleaningConfig,
    clean_option_chains,
    cleaning_report,
)
from app.model.calibration.rough_vol.diagnostics import (
    DiagnosticsConfig,
    PipelineArtifacts,
    build_calibration_diagnostics,
    save_diagnostic_figures,
)
from app.model.calibration.rough_vol.forward_curve import (
    build_forward_curve,
    build_otm_surface,
    forward_curve_report,
)
from app.model.calibration.rough_vol.forward_variance import (
    build_forward_variance_curve,
    forward_variance_report,
)
from app.model.calibration.rough_vol.hurst_estimator import (
    estimate_hurst_from_skew,
    hurst_report,
)
from app.model.calibration.rough_vol.variance_swap import (
    build_variance_swap_curve,
    variance_swap_report,
)
from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
    JointMCConfig,
    RBergomiCalibrationError,
    calibrate_rbergomi,
    calibration_report,
)
from app.model.volatility_models.rbergomi.initializer import (
    RBergomiInitializationError,
    initial_rbergomi_params,
    initializer_report,
)

__all__ = [
    "DEFAULT_SEED",
    "FIXTURE_METADATA_KEYS",
    "ChainSource",
    "PipelineError",
    "PipelineRun",
    "build_arg_parser",
    "load_fixture",
    "load_yahoo_chain",
    "main",
    "mc_config_for_paths",
    "parse_fixture_metadata",
    "parse_short_window",
    "run_pipeline",
    "run_payload",
    "summary_fr",
]

#: Used when ``--seed`` is not given, so an unseeded run is still reproducible.
DEFAULT_SEED = 20_260_821

#: The provenance keys a fixture may declare in its ``# key: value`` header.
#: Anything else in the comment block is prose and is ignored on purpose - the
#: parser only accepts lowercase snake_case keys so that a sentence containing a
#: colon cannot be mistaken for data.
FIXTURE_METADATA_KEYS = (
    "underlying",
    "spot",
    "rate",
    "dividend_yield",
    "valuation_date",
    "H",
    "eta",
    "rho",
)


class PipelineError(RuntimeError):
    """The pipeline could not run to the end. Carries a French message."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainSource:
    """
    One option chain plus everything needed to discount it.

    Attributes
    ----------
    calls, puts:
        The two row sets, in the ``fetch_options_details_yahoo`` schema.
    spot:
        Underlying reference used as the moneyness anchor by the cleaner.
    rate:
        Flat continuously-compounded rate to pin ``D(T) = exp(-r T)``, or
        ``None`` to let the repo yield curve resolve every maturity.
    label:
        Human-readable provenance ("SPY (Yahoo)", the fixture path, ...).
    is_synthetic:
        ``True`` for a generated chain. Propagated into the report and printed
        in the summary so no number from it is ever read as market data.
    metadata:
        The fixture provenance block, verbatim, or ``{}``.
    """

    calls: Any
    puts: Any
    spot: float
    rate: float | None
    label: str
    is_synthetic: bool
    metadata: Mapping[str, str] = field(default_factory=dict)


def parse_fixture_metadata(path: Path) -> dict[str, str]:
    """
    Read the ``# key: value`` provenance block at the top of a fixture.

    Only lowercase snake_case keys (plus the single-letter model parameters
    ``H``/``eta``/``rho``) listed in :data:`FIXTURE_METADATA_KEYS` are accepted;
    everything else in the comment block is documentation. Parsing stops at the
    first non-comment line.
    """
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, sep, value = line.lstrip("#").strip().partition(": ")
            key = key.strip()
            if sep and key in FIXTURE_METADATA_KEYS and key not in out:
                out[key] = value.strip()
    return out


def load_fixture(path: Path, *, rate: float | None = None) -> ChainSource:
    """
    Load a committed CSV chain. Offline by construction - no import of the
    market-data module, no socket, no cache lookup.

    The file must carry the Yahoo row schema. ``#`` lines are the provenance
    block and are skipped. ``spot`` comes from the header when present, else from
    the ``S0`` column; the rate comes from ``rate`` (the ``--rate`` flag), else
    from the header, else stays ``None`` so the yield curve resolves it.
    """
    import pandas as pd

    if not path.is_file():
        raise PipelineError(f"Fixture introuvable : {path}")
    metadata = parse_fixture_metadata(path)
    frame = pd.read_csv(path, comment="#")
    missing = {"strike", "bid", "ask", "type", "T"} - set(frame.columns)
    if missing:
        raise PipelineError(
            "Fixture au mauvais schéma : colonnes manquantes "
            f"{sorted(missing)} (schéma attendu : celui de "
            "fetch_options_details_yahoo)."
        )
    calls = frame[frame["type"].astype(str).str.lower() == "call"]
    puts = frame[frame["type"].astype(str).str.lower() == "put"]
    if calls.empty or puts.empty:
        raise PipelineError(
            "Fixture inutilisable : il faut des calls ET des puts pour la parité "
            "put-call (spec 4.2)."
        )

    spot = _float_or_none(metadata.get("spot"))
    if spot is None and "S0" in frame.columns:
        spot = _float_or_none(frame["S0"].median())
    if spot is None or not (math.isfinite(spot) and spot > 0.0):
        raise PipelineError("Fixture sans sous-jacent exploitable (spot).")

    resolved_rate = rate if rate is not None else _float_or_none(metadata.get("rate"))
    return ChainSource(
        calls=calls,
        puts=puts,
        spot=float(spot),
        rate=resolved_rate,
        label=str(path),
        is_synthetic=True,
        metadata=metadata,
    )


def load_yahoo_chain(
    ticker: str,
    *,
    rate: float | None = None,
    max_maturity_years: float = 2.0,
    max_expiries: int = 12,
    use_cache: bool = True,
) -> ChainSource:
    """
    Fetch a live chain from Yahoo. **The only networked path in this file.**

    ``rf`` and ``div`` returned by the fetch are hardcoded zeros upstream and are
    discarded here; discounting is the yield curve's job.
    """
    from app.model.market_data.market_data import fetch_options_details_yahoo

    calls, puts, spot, _rf_unused, _div_unused = fetch_options_details_yahoo(
        ticker,
        max_maturity_years=float(max_maturity_years),
        max_expiries=int(max_expiries),
        use_cache=bool(use_cache),
    )
    if calls is None or puts is None or len(calls) == 0 or len(puts) == 0:
        raise PipelineError(
            f"Aucune chaîne d'options exploitable pour {ticker!r} "
            "(calls et puts sont tous deux nécessaires)."
        )
    spot_value = _float_or_none(spot)
    if spot_value is None or not (math.isfinite(spot_value) and spot_value > 0.0):
        raise PipelineError(f"Sous-jacent invalide pour {ticker!r} : spot={spot!r}.")
    return ChainSource(
        calls=calls,
        puts=puts,
        spot=float(spot_value),
        rate=rate,
        label=f"{ticker} (Yahoo)",
        is_synthetic=False,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass
class PipelineRun:
    """Every stage output of one end-to-end run, kept as live objects."""

    source: ChainSource
    chains: list[Any]
    forward_points: list[Any]
    surface: list[Any]
    surface_rejections: int
    variance_curve: Any
    xi0_curve: Any
    hurst: Any
    initial_params: Any
    initializer_diagnostics: dict[str, Any]
    calibration: Any
    report: dict[str, Any]
    short_window: tuple[float, float] | None
    seed: int | None
    elapsed_market_s: float
    elapsed_calibration_s: float
    figures: dict[str, str] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        return float(self.elapsed_market_s + self.elapsed_calibration_s)


def mc_config_for_paths(
    paths: int | None = None, *, grid_n_max: int | None = None
) -> JointMCConfig:
    """
    Scale the Monte-Carlo budget from one ``--paths`` number.

    ``paths`` is the FINAL evaluation's path count. Every other path count moves
    with it proportionally (floored, so a tiny budget still produces a run), and
    ``batch_paths`` never exceeds the final count. Leaving ``paths`` at ``None``
    keeps the calibrator's own defaults, which are what Phase 4 measured.
    """
    base = JointMCConfig()
    if grid_n_max is not None:
        base = replace(base, grid_n_max=int(grid_n_max))
    if paths is None:
        return base
    final = int(paths)
    if final < 1:
        raise PipelineError("--paths doit être >= 1.")
    ratio = final / float(base.final_paths)

    def _scaled(value: int, floor: int) -> int:
        return int(max(floor, round(value * ratio)))

    return replace(
        base,
        stage1_paths=_scaled(base.stage1_paths, 500),
        stage2_paths=_scaled(base.stage2_paths, 500),
        final_paths=final,
        batch_paths=int(min(base.batch_paths, final)),
    )


def parse_short_window(text: str) -> tuple[float, float]:
    """
    Parse ``--short-window``: ``"MIN,MAX"`` in years, or in days with a ``d``
    suffix (``"7d,90d"``). ``5/365`` and ``0.25`` are the spec-4.5 defaults.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise PipelineError(
            "--short-window attend deux bornes séparées par une virgule, "
            'par exemple "0.0137,0.25" (années) ou "5d,91d" (jours).'
        )
    bounds: list[float] = []
    for part in parts:
        scale = 1.0
        if part.lower().endswith("d"):
            part, scale = part[:-1], 1.0 / 365.0
        try:
            bounds.append(float(part) * scale)
        except ValueError as exc:
            raise PipelineError(f"Borne d'échéance illisible : {part!r}.") from exc
    lo, hi = bounds
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        raise PipelineError(
            f"Fenêtre courte invalide : ({lo}, {hi}). Il faut 0 < T_min < T_max."
        )
    return (lo, hi)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pair_chains_with_forwards(
    chains: Sequence[Any], points: Sequence[Any]
) -> list[tuple[Any, Any]]:
    """Match each cleaned chain to its forward point on ``T`` (exact, then close)."""
    pairs: list[tuple[Any, Any]] = []
    for chain in chains:
        T = float(chain.T)
        match = None
        for point in points:
            if abs(float(point.T) - T) <= 1e-12 * max(1.0, abs(T)):
                match = point
                break
        if match is not None:
            pairs.append((chain, match))
    return pairs


def run_pipeline(
    source: ChainSource,
    *,
    short_window: tuple[float, float] | None = None,
    seed: int | None = DEFAULT_SEED,
    mc_config: JointMCConfig | None = None,
    cleaning_config: CleaningConfig | None = None,
    diagnostics_config: DiagnosticsConfig | None = None,
    figures_dir: Path | None = None,
) -> PipelineRun:
    """
    Run spec 4.1 -> 4.11 -> 9 on ``source`` and return every stage's output.

    Raises
    ------
    PipelineError
        When a stage cannot produce an honest result: no viable expiry, no
        forward point, no variance-swap strike, an impossible ``xi0``
        reconstruction, or an initialisation with no skew to stand on. The
        message is French and names the stage.
    RBergomiCalibrationError
        Propagated untouched from spec 4.10 - it already explains itself.

    Notes
    -----
    ``xi0`` is built once, here, from the market's own ``K_var`` and handed to
    the calibrator as frozen data. Nothing downstream is allowed to move it and
    the report re-checks that it did not (spec 4.10 separation A).
    """
    started = time.perf_counter()

    # -- 4.1 cleaning ------------------------------------------------------
    chains = clean_option_chains(
        source.calls, source.puts, config=cleaning_config, spot=source.spot
    )
    if not chains:
        raise PipelineError(
            "Nettoyage (spec 4.1) : aucune échéance n'a survécu au nettoyage."
        )

    # -- 4.2 forward curve -------------------------------------------------
    rates = (
        None
        if source.rate is None
        else {float(chain.T): float(source.rate) for chain in chains}
    )
    forward_points = build_forward_curve(chains, rates=rates, S0=source.spot)
    if not forward_points:
        raise PipelineError(
            "Courbe forward (spec 4.2) : aucune échéance n'a produit de forward "
            "exploitable par parité put-call."
        )
    pairs = _pair_chains_with_forwards(chains, forward_points)

    # -- OTM surface with re-inverted implied vols -------------------------
    surface: list[Any] = []
    rejections = 0
    for chain, point in pairs:
        points_i, rejected = build_otm_surface(chain, point)
        rejections += len(rejected)
        surface.extend(points_i)
    if not surface:
        raise PipelineError(
            "Surface OTM (spec 4.2) : aucune cotation n'a pu être ré-inversée "
            "sur (F, D)."
        )

    # -- 4.3 variance swaps ------------------------------------------------
    variance_curve = build_variance_swap_curve(pairs)
    if not variance_curve.points:
        raise PipelineError(
            "Swaps de variance (spec 4.3) : aucune échéance n'a produit de K_var "
            "exploitable — "
            + " | ".join(str(f.message_fr) for f in variance_curve.failures[:3])
        )

    # -- 4.4 forward variance (xi0), the frozen data of the calibration ----
    try:
        xi0_curve = build_forward_variance_curve(variance_curve)
    except ValueError as exc:
        raise PipelineError(f"Variance forward (spec 4.4) : {exc}") from exc

    # -- 4.5 ATM skew -> H0 ------------------------------------------------
    hurst = estimate_hurst_from_skew(
        surface,
        forward_points,
        short_window,
        clean_chains=chains,
        variance_curve=variance_curve,
    )

    # -- 4.9 initializer ---------------------------------------------------
    try:
        initial_params, initializer_diagnostics = initial_rbergomi_params(
            hurst,
            surface,
            xi0_curve=xi0_curve,
            forward_curve=forward_points,
            clean_chains=chains,
            variance_curve=variance_curve,
        )
    except RBergomiInitializationError as exc:
        raise PipelineError(f"Initialisation (spec 4.9) : {exc}") from exc
    elapsed_market = time.perf_counter() - started

    # -- 4.10 / 4.11 joint calibration, xi0 frozen -------------------------
    started_calibration = time.perf_counter()
    calibration = calibrate_rbergomi(
        surface,
        xi0_curve,
        (initial_params, initializer_diagnostics),
        mc_cfg=mc_config,
        settings=CalibratorSettings(n_starts=1, seed=seed),
        clean_chains=chains,
        S0=source.spot,
    )
    elapsed_calibration = time.perf_counter() - started_calibration

    # -- 9 validation report ----------------------------------------------
    notes: list[str] = []
    if source.is_synthetic:
        notes.append(
            "SOURCE SYNTHÉTIQUE : chaîne d'options générée, aucune donnée de "
            "marché. Aucun chiffre de ce rapport ne décrit un instrument réel."
        )
    if source.rate is not None:
        notes.append(
            f"Actualisation à taux plat épinglé r = {float(source.rate):.6g} "
            "(D(T) = exp(-r T)) — la courbe de taux du dépôt n'a pas été lue."
        )
    else:
        notes.append(
            "Actualisation par la courbe de taux active du dépôt "
            "(app.model.yieldcurve) ; le résultat dépend donc du cache local."
        )
    if hurst.unstable:
        notes.append(
            "Estimation de Hurst instable (spec 4.5) : H0 est une valeur de "
            "repli, pas une mesure."
        )

    report = build_calibration_diagnostics(
        calibration,
        artifacts=PipelineArtifacts(
            market_surface=surface,
            forward_curve=forward_points,
            clean_chains=chains,
            variance_curve=variance_curve,
            xi0_curve=xi0_curve,
            hurst=hurst,
            initializer_diagnostics=initializer_diagnostics,
            source=source.label,
            is_synthetic=source.is_synthetic,
            notes_fr=tuple(notes),
        ),
        config=diagnostics_config,
    )

    run = PipelineRun(
        source=source,
        chains=list(chains),
        forward_points=list(forward_points),
        surface=list(surface),
        surface_rejections=int(rejections),
        variance_curve=variance_curve,
        xi0_curve=xi0_curve,
        hurst=hurst,
        initial_params=initial_params,
        initializer_diagnostics=dict(initializer_diagnostics),
        calibration=calibration,
        report=report,
        short_window=short_window,
        seed=seed,
        elapsed_market_s=float(elapsed_market),
        elapsed_calibration_s=float(elapsed_calibration),
    )

    if figures_dir is not None:
        run.figures = save_diagnostic_figures(
            report, figures_dir, config=diagnostics_config
        )
        report["figures"] = dict(run.figures)
    return run


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def run_payload(run: PipelineRun) -> dict[str, Any]:
    """
    The JSON payload: one stage report per pipeline stage plus the spec-9 report.

    Plain Python types only, so it survives the controller's ``_json_safe``
    untouched and ``json.dump`` needs no custom encoder. Missing numbers stay
    ``NaN`` rather than becoming ``null``: a number that could not be computed
    must remain visibly absent, and ``json.load`` reads the token back.
    """
    source = run.source
    return {
        "spec": "5",
        "source": {
            "label": source.label,
            "is_synthetic": bool(source.is_synthetic),
            "spot": float(source.spot),
            "rate_pinned": None if source.rate is None else float(source.rate),
            "n_call_rows": int(len(source.calls)),
            "n_put_rows": int(len(source.puts)),
            "metadata": {str(k): str(v) for k, v in dict(source.metadata).items()},
        },
        "stages": {
            "cleaning": cleaning_report(run.chains),
            "forward_curve": forward_curve_report(run.forward_points),
            "otm_surface": {
                "n_points": int(len(run.surface)),
                "n_rejected": int(run.surface_rejections),
                "maturities": sorted({float(p.T) for p in run.surface}),
            },
            "variance_swap": variance_swap_report(run.variance_curve),
            "forward_variance": forward_variance_report(run.xi0_curve),
            "hurst": hurst_report(run.hurst),
            "initializer": initializer_report(run.initializer_diagnostics),
            "calibration": calibration_report(run.calibration),
        },
        "report": run.report,
        "run": {
            "seed": None if run.seed is None else int(run.seed),
            "short_window": (
                None
                if run.short_window is None
                else [float(run.short_window[0]), float(run.short_window[1])]
            ),
            "elapsed_market_s": float(run.elapsed_market_s),
            "elapsed_calibration_s": float(run.elapsed_calibration_s),
            "elapsed_s": float(run.elapsed_s),
            "figures": dict(run.figures),
        },
    }


def summary_fr(run: PipelineRun) -> str:
    """A readable French summary of the whole run, stage by stage."""
    source = run.source
    report = run.report
    hurst = run.hurst
    params = run.initial_params
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PIPELINE rBergomi — estimation de H et calibration jointe (spec 5)")
    lines.append("=" * 78)
    lines.append(f"Source            : {source.label}")
    if source.is_synthetic:
        lines.append(
            "                    ⚠ DONNÉES SYNTHÉTIQUES — aucune cotation réelle."
        )
    lines.append(f"Sous-jacent       : {source.spot:.4f}")
    lines.append(
        "Actualisation     : "
        + (
            f"taux plat épinglé r = {source.rate:.6g}"
            if source.rate is not None
            else "courbe de taux active du dépôt"
        )
    )
    lines.append("")
    lines.append("-- Étapes --------------------------------------------------------------")
    n_kvar = sum(1 for c in run.chains if c.viability.usable_for_kvar)
    lines.append(
        f"4.1 Nettoyage     : {len(run.chains)} échéances, "
        f"{sum(c.n_quotes for c in run.chains)} cotations retenues, "
        f"{n_kvar} exploitables pour K_var"
    )
    lines.append(
        f"4.2 Forward       : {len(run.forward_points)} points ; surface OTM "
        f"{len(run.surface)} cotations ({run.surface_rejections} rejetées)"
    )
    k_var = [float(p.k_var) for p in run.variance_curve.points]
    lines.append(
        f"4.3 K_var         : {len(k_var)} échéances, "
        f"{len(run.variance_curve.failures)} refus ; "
        f"K_var ∈ [{min(k_var):.5f}, {max(k_var):.5f}]"
    )
    reconstruction = run.xi0_curve.reconstruction_errors()
    worst = max((abs(float(e)) for e in reconstruction), default=float("nan"))
    lines.append(
        f"4.4 ξ₀            : {len(run.xi0_curve)} paliers ({run.xi0_curve.method}), "
        f"reconstruction de K_var à {worst:.2e} près — donnée FIGÉE pour la suite"
    )
    lines.append(f"4.5 Hurst         : {hurst.message_fr}")
    lines.append(
        f"4.9 Initialisation: H0 = {params.H:.4f}, eta0 = {params.eta:.4f}, "
        f"rho0 = {params.rho:+.4f} (point de départ, pas un résultat)"
    )
    lines.append(
        f"4.10 Calibration  : {run.calibration.n_objective_evaluations} évaluations "
        f"en {run.elapsed_calibration_s:.1f} s"
    )
    lines.append("")
    lines.append("-- Rapport de validation (spec 9) --------------------------------------")
    lines.append(str(report.get("summary_fr", "")))
    open_items = list(report.get("open_items_fr") or [])
    if open_items:
        lines.append("")
        lines.append("-- Points connus, mesurés et NON corrigés -------------------------------")
        for item in open_items:
            lines.append(f"  • {item}")
    notes = list(report.get("notes_fr") or [])
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"  ! {note}")
    if run.figures:
        lines.append("")
        lines.append("-- Figures -------------------------------------------------------------")
        for name, path in sorted(run.figures.items()):
            lines.append(f"  {name:20s} {path}")
    lines.append("")
    lines.append(f"Durée totale      : {run.elapsed_s:.1f} s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_rbergomi_hurst",
        description=(
            "Chaîne complète rough Bergomi : nettoyage -> forward -> K_var -> "
            "xi0 -> skew -> H0 -> initialisation -> calibration jointe (H, eta, "
            "rho) à xi0 figé -> rapport de validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  calibrate_rbergomi_hurst.py --fixture tests/fixtures/"
            "synthetic_rbergomi_chain.csv --paths 8000 --out run.json\n"
            "  calibrate_rbergomi_hurst.py --ticker SPY --short-window 7d,91d "
            "--out spy.json\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Sous-jacent à récupérer chez Yahoo (réseau).")
    group.add_argument(
        "--fixture",
        type=Path,
        help="Chaîne d'options CSV committée (schéma Yahoo). Aucun accès réseau.",
    )
    parser.add_argument(
        "--short-window",
        default=None,
        help=(
            'Fenêtre courte de la régression de skew, "MIN,MAX". En années '
            '("0.0137,0.25") ou en jours avec le suffixe d ("5d,91d"). '
            "Défaut : la fenêtre de la spec 4.5 (5/365 à 0,25)."
        ),
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=None,
        help=(
            "Nombre de trajectoires de l'évaluation finale ; les autres budgets "
            "Monte-Carlo suivent proportionnellement. Défaut : les valeurs du "
            "calibrateur (100 000)."
        ),
    )
    parser.add_argument(
        "--grid-n-max",
        type=int,
        default=None,
        help="Pas maximal de la grille de simulation (défaut 384, spec 4.7a).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Graine Monte-Carlo (défaut {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "Taux continu plat épinglant D(T) = exp(-r T). Sans lui : la valeur "
            "du bloc de provenance de la fixture, sinon la courbe de taux active."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Fichier JSON de sortie (rapport complet). Sans lui, rien n'est écrit.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help=(
            "Répertoire où écrire les PNG de diagnostic (spec 9). Désactivé par "
            "défaut ; matplotlib n'est importé que si l'option est donnée."
        ),
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=12,
        help="--ticker seulement : nombre maximal d'échéances récupérées.",
    )
    parser.add_argument(
        "--max-maturity-years",
        type=float,
        default=2.0,
        help="--ticker seulement : échéance maximale récupérée, en années.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="N'affiche pas le résumé (le JSON est tout de même écrit).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Before parse_args: --help prints from inside it, and this summary is
    # French. A cp1252 console mangles it otherwise.
    stream = sys.stdout
    for handle in (sys.stdout, sys.stderr):
        reconfigure = getattr(handle, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = build_arg_parser().parse_args(argv)

    try:
        window = (
            None if args.short_window is None else parse_short_window(args.short_window)
        )
        mc_config = mc_config_for_paths(args.paths, grid_n_max=args.grid_n_max)
        if args.fixture is not None:
            source = load_fixture(args.fixture, rate=args.rate)
        else:
            source = load_yahoo_chain(
                args.ticker,
                rate=args.rate,
                max_maturity_years=args.max_maturity_years,
                max_expiries=args.max_expiries,
            )
        run = run_pipeline(
            source,
            short_window=window,
            seed=args.seed,
            mc_config=mc_config,
            figures_dir=args.figures_dir,
        )
    except (PipelineError, RBergomiCalibrationError) as exc:
        print(f"ÉCHEC — {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a CLI must not spit a traceback
        print(f"ÉCHEC inattendu — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    payload = run_payload(run)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    if not args.quiet:
        print(summary_fr(run), file=stream)
        if args.out is not None:
            print(f"\nRapport JSON écrit dans {args.out}", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
