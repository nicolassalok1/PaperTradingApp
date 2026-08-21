"""
End-to-end test of the rough-Bergomi Hurst pipeline on the committed fixture.

Target: scripts/calibrate_rbergomi_hurst.py (spec 5), driving
        4.1 -> 4.2 -> 4.3 -> 4.4 -> 4.5 -> 4.9 -> 4.10/4.11 -> 9.

WHAT THIS MODULE ASSERTS - AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------------
It asserts that the CHAIN HOLDS TOGETHER: every stage produces a usable object,
the invariants that link two stages hold exactly, and the final payload is plain
JSON-safe data.

It does **not** assert that the calibrated ``H`` reproduces the fixture's
generating ``H = 0.12``, and no future edit should make it:

  * the fixture's smile is the leading-order asymptotic expansion, while the
    calibrator prices with the real rough-Bergomi Monte Carlo - the two do not
    coincide away from the money or at long maturity (fixture caveat 2);
  * the spec-4.5 skew regression is measurably biased on a reference surface
    (Phase 3: ``H0 = 0.0827 +/- 0.0061`` against a truth of 0.10, i.e. the truth
    falls OUTSIDE its own 95 % interval), so agreement would be luck;
  * these tests run a deliberately cheap Monte-Carlo budget, which the
    calibrator itself reports as ``success = False`` / weakly identified. That
    verdict is the honest answer at this budget and is asserted to be *present
    and coherent*, not to be ``True``.

What IS pinned about ``H``: it is finite, it stays inside the calibrator's own
bounds, and it is reported together with ``H0``, its CI and its profile standard
error, so a reader can judge it.

Determinism: the market side has no RNG at all, so 4.1 -> 4.9 is bit-exact on
every machine; the Monte-Carlo side is seeded and is checked to reproduce itself
under the same seed.

Network: none. The suite runs socket-blocked, ``--fixture`` never imports the
market-data module, and one test asserts precisely that.

Cost: this module is slow-only (``pytestmark`` at module level). It also carries
the fixture's own integrity checks and the CLI's pure helpers, which are fast but
belong with the subject under test.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.model.calibration.rough_vol.forward_curve import black76_call_price
from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
    JointMCConfig,
    curve_fingerprint,
)

pytestmark = pytest.mark.slow

cli = importlib.import_module("scripts.calibrate_rbergomi_hurst")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_rbergomi_chain.csv"

#: Small enough to keep the module a few seconds, large enough that every stage
#: is exercised for real. NOT a production budget - see the module docstring.
_MC_CONFIG = JointMCConfig(
    grid_n_max=128,
    n_design=6,
    stage1_paths=2_000,
    top_k=1,
    stage2_paths=3_000,
    profile_paths=2_000,
    final_paths=6_000,
    batch_paths=6_000,
    profile_points=5,
    valley_points=4,
    noise_replicates=2,
    refinement_check=False,
    local_nfev_per_param=10,
)
_SEED = 20_260_821


@pytest.fixture(scope="module")
def source() -> Any:
    return cli.load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def run(source: Any) -> Any:
    """One full pipeline run, shared by the whole module."""
    return cli.run_pipeline(source, seed=_SEED, mc_config=_MC_CONFIG)


# ---------------------------------------------------------------------------
# 0. The fixture itself
# ---------------------------------------------------------------------------
def test_fixture_declares_itself_synthetic_in_its_own_header() -> None:
    """A generated chain that does not say so is a trap for the next reader."""
    header = "\n".join(
        line
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    )
    upper = header.upper()
    assert "SYNTHETIC" in upper
    assert "NOT MARKET DATA" in upper
    for key in ("underlying", "spot", "rate", "H", "eta", "rho"):
        assert f"# {key}: " in header, key


def test_fixture_has_the_yahoo_row_schema() -> None:
    frame = pd.read_csv(FIXTURE, comment="#")
    expected = [
        "underlying",
        "contractSymbol",
        "expiry",
        "expiry_ts",
        "T",
        "strike",
        "iv",
        "bid",
        "ask",
        "lastPrice",
        "openInterest",
        "volume",
        "inTheMoney",
        "type",
        "S0",
    ]
    assert list(frame.columns) == expected
    assert set(frame["type"].unique()) == {"call", "put"}
    assert (frame["ask"] >= frame["bid"]).all()
    assert (frame["bid"] >= 0.0).all()
    assert (frame["iv"] > 0.0).all(), "no vendor-IV sentinel rows in this fixture"


def test_every_expiry_has_a_distinct_expiry_ts() -> None:
    """
    The Phase-1 grouping gotcha: chains are grouped on ``expiry_ts`` first, so a
    constant timestamp would collapse the whole surface into ONE chain carrying
    the median maturity.
    """
    frame = pd.read_csv(FIXTURE, comment="#")
    pairs = frame[["expiry_ts", "T"]].drop_duplicates()
    assert len(pairs) == frame["expiry_ts"].nunique()
    assert len(pairs) == frame["T"].nunique()


def test_fixture_spans_the_spec45_window_and_beyond() -> None:
    frame = pd.read_csv(FIXTURE, comment="#")
    maturities = sorted(frame["T"].unique())
    in_window = [T for T in maturities if 5.0 / 365.0 <= T <= 0.25]
    beyond = [T for T in maturities if T > 0.25]
    assert len(in_window) >= 4, "the spec-4.5 regression needs several short expiries"
    assert len(beyond) >= 4, "xi0 must extend past the short window"
    assert max(maturities) >= 1.0


def test_fixture_prices_are_consistent_with_its_declared_vendor_iv() -> None:
    """
    Independent oracle on the DATA, not on the pipeline: re-price the OTM leg
    with Black-76 at the fixture's own ``iv``, on ``F = S0 exp(rT)``,
    ``D = exp(-rT)``, and check it lands inside the quoted bid/ask.
    """
    metadata = cli.parse_fixture_metadata(FIXTURE)
    S0, rate = float(metadata["spot"]), float(metadata["rate"])
    frame = pd.read_csv(FIXTURE, comment="#")
    checked = 0
    for row in frame.itertuples():
        T, K = float(row.T), float(row.strike)
        F = S0 * math.exp(rate * T)
        D = math.exp(-rate * T)
        if row.type == "call" and K < F:
            continue  # in-the-money leg: carried by parity, not quoted directly
        if row.type == "put" and K > F:
            continue
        call = black76_call_price(F=F, K=K, T=T, D=D, vol=float(row.iv))
        price = call if row.type == "call" else call - D * (F - K)
        # Tick rounding is +/- half a tick; the bid/ask half-spread is one tick.
        assert float(row.bid) - 0.06 <= price <= float(row.ask) + 0.06, row.contractSymbol
        checked += 1
    assert checked > 100


# ---------------------------------------------------------------------------
# 1. Stage by stage: every stage produces a usable output
# ---------------------------------------------------------------------------
def test_cleaning_keeps_every_expiry_and_marks_it_usable(run: Any) -> None:
    frame = pd.read_csv(FIXTURE, comment="#")
    assert len(run.chains) == frame["expiry_ts"].nunique()
    assert all(chain.n_quotes > 0 for chain in run.chains)
    assert all(chain.viability.usable_for_kvar for chain in run.chains)
    assert all(chain.viability.usable_for_skew for chain in run.chains)
    assert [c.T for c in run.chains] == sorted(c.T for c in run.chains)


def test_forward_curve_discounts_at_the_pinned_rate(run: Any) -> None:
    """``--rate`` (here, the fixture's own) must pin ``D = exp(-r T)`` exactly."""
    rate = float(run.source.rate)
    assert len(run.forward_points) == len(run.chains)
    for point in run.forward_points:
        assert point.D == pytest.approx(math.exp(-rate * point.T), rel=1e-15)
        assert point.r == pytest.approx(rate, rel=1e-12)
        assert point.F > 0.0
        # A positive rate and no dividend put the forward above spot.
        assert point.F > run.source.spot


def test_otm_surface_is_non_empty_and_fully_invertible(run: Any) -> None:
    assert len(run.surface) >= 100
    assert all(math.isfinite(p.iv) and p.iv > 0.0 for p in run.surface)
    assert all(math.isfinite(p.k) for p in run.surface)
    forwards = {round(float(p.T), 12): float(p.F) for p in run.forward_points}
    for p in run.surface:
        assert p.k == pytest.approx(math.log(p.K / forwards[round(float(p.T), 12)]), rel=1e-12)
        # Every point is the out-of-the-money leg.
        assert (p.option_type == "call") == (p.K >= p.F)


def test_variance_swap_curve_covers_every_expiry(run: Any) -> None:
    assert len(run.variance_curve.points) == len(run.chains)
    assert not run.variance_curve.failures
    k_var = [float(p.k_var) for p in run.variance_curve.points]
    assert all(math.isfinite(v) and v > 0.0 for v in k_var)
    assert [p.T for p in run.variance_curve.points] == sorted(
        p.T for p in run.variance_curve.points
    )


def test_xi0_reconstructs_kvar_exactly(run: Any) -> None:
    """
    The spec-4.4 invariant: ``(1/T) integral_0^T xi0 == K_var(T)`` at every
    market maturity. The piecewise-constant build is exact by construction, so
    this is a hard equality, not a tolerance to be relaxed.
    """
    curve = run.xi0_curve
    assert len(curve) == len(run.variance_curve.points)
    for point in run.variance_curve.points:
        T = float(point.T)
        assert float(curve.integrated(T)) / T == pytest.approx(
            float(point.k_var), rel=1e-12
        )
    assert max(abs(float(e)) for e in curve.reconstruction_errors()) <= 1e-12
    assert all(level > 0.0 for level in curve.levels)


def test_hurst_estimate_is_finite_and_uses_the_short_window(run: Any) -> None:
    hurst = run.hurst
    assert math.isfinite(hurst.H0) and 0.0 < hurst.H0 < 0.5
    assert math.isfinite(hurst.se) and hurst.se > 0.0
    assert hurst.ci95[0] < hurst.H0 < hurst.ci95[1]
    assert hurst.window == pytest.approx((5.0 / 365.0, 0.25))
    assert hurst.n_expiries >= 3
    assert math.isfinite(hurst.r2)


def test_initializer_returns_a_usable_starting_point(run: Any) -> None:
    params = run.initial_params
    assert math.isfinite(params.H) and 0.0 < params.H < 0.5
    assert math.isfinite(params.eta) and params.eta > 0.0
    assert math.isfinite(params.rho) and -1.0 < params.rho < 0.0
    assert params.H == pytest.approx(run.hurst.H0, rel=1e-12)
    assert run.initializer_diagnostics["H0_is_fallback"] is False


# ---------------------------------------------------------------------------
# 2. The calibration: it returns, and xi0 was frozen data throughout
# ---------------------------------------------------------------------------
def test_calibration_returns_finite_parameters_inside_its_own_bounds(run: Any) -> None:
    result = run.calibration
    for name in ("H", "eta", "rho"):
        value = float(getattr(result.params, name))
        low, high = result.bounds[name]
        assert math.isfinite(value)
        assert low <= value <= high, f"{name}={value} outside {result.bounds[name]}"
    assert result.n_objective_evaluations > 0
    assert math.isfinite(result.loss_crn) and result.loss_crn >= 0.0
    assert math.isfinite(result.loss_fresh)
    assert math.isfinite(result.loss_initial)
    assert isinstance(result.success, bool)
    assert result.message_fr


def test_xi0_was_frozen_data_and_came_back_by_identity(run: Any) -> None:
    """Spec 4.10 separation A, checked from the outside."""
    result = run.calibration
    assert result.xi0_curve is run.xi0_curve
    assert result.xi0.fingerprint == curve_fingerprint(run.xi0_curve)
    result.xi0.verify()  # raises if a single float moved
    assert "xi0" not in result.bounds
    assert result.theta.shape == (3,)


def test_the_verdict_on_H_is_reported_not_asserted(run: Any) -> None:
    """
    We pin the REPORTING, not the value. ``H`` must arrive with ``H0``, its 95 %
    interval and the profile standard error, and the success flag must agree
    with the blocking flags - so a reader can tell an identified H from this
    deliberately cheap budget's non-verdict.
    """
    result = run.calibration
    identifiability = result.identifiability
    assert identifiability is not None
    assert math.isfinite(identifiability.H0)
    assert identifiability.has_H0_ci
    assert identifiability.H_calibrated == pytest.approx(float(result.params.H))
    assert math.isfinite(identifiability.H_standard_error)
    assert identifiability.h_comparison_fr
    blocking = {
        "h_profile_flat",
        "h_weakly_identified",
        "no_improvement_over_initial",
        "optimum_not_stationary_on_profile",
    }
    assert result.success is not bool(blocking & set(result.flags))


def test_the_same_seed_reproduces_the_same_calibration(source: Any, run: Any) -> None:
    """Common random numbers + a fixed seed => the same optimum, bit for bit."""
    repeat = cli.run_pipeline(source, seed=_SEED, mc_config=_MC_CONFIG)
    assert repeat.calibration.params.H == run.calibration.params.H
    assert repeat.calibration.params.eta == run.calibration.params.eta
    assert repeat.calibration.params.rho == run.calibration.params.rho
    assert repeat.calibration.loss_crn == run.calibration.loss_crn
    # The market side has no RNG at all: it is identical whatever the seed.
    assert repeat.hurst.H0 == run.hurst.H0
    assert repeat.xi0_curve.levels == run.xi0_curve.levels


# ---------------------------------------------------------------------------
# 3. The spec-9 report and the JSON payload
# ---------------------------------------------------------------------------
def _type_offenders(node: Any, path: str = "$") -> list[str]:
    """Anything the controller's ``_json_safe`` would have to convert."""
    if node is None or isinstance(node, (bool, int, float, str)):
        return [f"{path}: numpy scalar"] if isinstance(node, np.generic) else []
    if isinstance(node, dict):
        bad: list[str] = []
        for key, value in node.items():
            if not isinstance(key, str):
                bad.append(f"{path}: non-str key {key!r}")
            bad.extend(_type_offenders(value, f"{path}.{key}"))
        return bad
    if isinstance(node, list):
        bad = []
        for i, value in enumerate(node):
            bad.extend(_type_offenders(value, f"{path}[{i}]"))
        return bad
    return [f"{path}: {type(node).__name__}"]


def test_report_is_assembled_from_the_real_result(run: Any) -> None:
    report = run.report
    assert report["spec"] == "9"
    assert report["is_synthetic"] is True
    assert report["parameters"]["calibrated"]["H"] == pytest.approx(
        float(run.calibration.params.H)
    )
    assert report["hurst"]["available"] is True
    assert report["atm_iv_term_structure"]["available"] is True
    assert report["atm_skew_term_structure"]["available"] is True
    assert len(report["per_maturity"]) == len(run.variance_curve.points)
    assert report["residuals"]["n_quotes"] == run.calibration.quotes.n_quotes
    assert math.isfinite(report["iv_error"]["rmse"])
    assert math.isfinite(report["price_error"]["rmse"])
    assert report["identifiability"] is not None
    assert report["figures"] == {}, "PNGs stay opt-in"


def test_the_consistency_check_confirms_xi0_was_not_moved(run: Any) -> None:
    section = run.report["variance_term_structure"]
    assert section["available"] is True
    assert section["role"] == "consistency_check"
    assert section["consistent"] is True
    assert section["max_abs"] <= section["tolerance"]
    assert section["xi0_fingerprint"] == curve_fingerprint(run.xi0_curve)


def test_report_and_payload_survive_json_serialisation(run: Any) -> None:
    payload = cli.run_payload(run)
    assert _type_offenders(payload) == []
    blob = json.dumps(payload, ensure_ascii=False)
    restored = json.loads(blob)
    assert set(restored) == {"spec", "source", "stages", "report", "run"}
    assert set(restored["stages"]) == {
        "cleaning",
        "forward_curve",
        "otm_surface",
        "variance_swap",
        "forward_variance",
        "hurst",
        "initializer",
        "calibration",
    }
    assert restored["source"]["is_synthetic"] is True
    assert restored["report"]["summary_fr"]


def test_the_summary_states_the_provenance_and_the_open_items(run: Any) -> None:
    text = cli.summary_fr(run)
    assert "SYNTHÉTIQUES" in text
    assert "FIGÉE" in text  # xi0 is data, and the summary says so
    assert "point de départ, pas un résultat" in text
    assert "Points connus, mesurés et NON corrigés" in text


# ---------------------------------------------------------------------------
# 4. The CLI itself
# ---------------------------------------------------------------------------
def test_main_runs_offline_and_writes_a_readable_json(tmp_path, capsys) -> None:
    out = tmp_path / "run.json"
    code = cli.main(
        [
            "--fixture",
            str(FIXTURE),
            "--paths",
            "2000",
            "--grid-n-max",
            "96",
            "--seed",
            "11",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["spec"] == "5"
    assert payload["run"]["seed"] == 11
    assert payload["source"]["is_synthetic"] is True
    printed = capsys.readouterr().out
    assert "PIPELINE rBergomi" in printed
    assert str(out) in printed


def test_the_fixture_path_never_loads_the_market_data_module(source: Any) -> None:
    """
    ``--fixture`` must be offline by construction, not merely by the suite's
    socket block: the networked module must not even be imported.
    """
    name = "app.model.market_data.market_data"
    already = name in sys.modules
    cli.run_pipeline(
        source,
        seed=3,
        mc_config=JointMCConfig(
            grid_n_max=64,
            n_design=0,
            stage1_paths=500,
            top_k=1,
            stage2_paths=500,
            profile_paths=500,
            final_paths=1_000,
            batch_paths=1_000,
            profile_points=3,
            valley_points=3,
            noise_replicates=2,
            refinement_check=False,
            local_nfev_per_param=3,
        ),
    )
    if not already:
        assert name not in sys.modules


def test_main_reports_a_bad_fixture_without_a_traceback(tmp_path, capsys) -> None:
    missing = tmp_path / "nope.csv"
    assert cli.main(["--fixture", str(missing), "--quiet"]) == 2
    assert "ÉCHEC" in capsys.readouterr().err

    truncated = tmp_path / "half.csv"
    truncated.write_text("strike,bid,ask\n100,1.0,1.1\n", encoding="utf-8")
    assert cli.main(["--fixture", str(truncated), "--quiet"]) == 2
    assert "schéma" in capsys.readouterr().err


def test_short_window_parsing_accepts_years_and_days() -> None:
    assert cli.parse_short_window("0.0137,0.25") == pytest.approx((0.0137, 0.25))
    assert cli.parse_short_window("5d,91d") == pytest.approx((5.0 / 365.0, 91.0 / 365.0))
    assert cli.parse_short_window(" 7D , 0.25 ") == pytest.approx((7.0 / 365.0, 0.25))
    for bad in ("0.25", "0.25,0.05", "0,0.25", "a,b", "1,2,3"):
        with pytest.raises(cli.PipelineError):
            cli.parse_short_window(bad)


def test_paths_scaling_moves_every_budget_together() -> None:
    default = cli.mc_config_for_paths(None)
    assert default.final_paths == JointMCConfig().final_paths
    scaled = cli.mc_config_for_paths(10_000)
    ratio = 10_000 / JointMCConfig().final_paths
    assert scaled.final_paths == 10_000
    assert scaled.stage1_paths == round(JointMCConfig().stage1_paths * ratio)
    assert scaled.stage2_paths == round(JointMCConfig().stage2_paths * ratio)
    assert scaled.batch_paths <= scaled.final_paths
    tiny = cli.mc_config_for_paths(100)
    assert tiny.stage1_paths >= 500 and tiny.stage2_paths >= 500  # floors hold
    assert cli.mc_config_for_paths(None, grid_n_max=128).grid_n_max == 128
    with pytest.raises(cli.PipelineError):
        cli.mc_config_for_paths(0)


def test_short_window_narrows_the_hurst_regression(source: Any, run: Any) -> None:
    """A user-supplied window must actually reach the spec-4.5 estimator."""
    narrow = cli.run_pipeline(
        source,
        short_window=cli.parse_short_window("7d,60d"),
        seed=_SEED,
        mc_config=JointMCConfig(
            grid_n_max=64,
            n_design=0,
            stage1_paths=500,
            top_k=1,
            stage2_paths=500,
            profile_paths=500,
            final_paths=1_000,
            batch_paths=1_000,
            profile_points=3,
            valley_points=3,
            noise_replicates=2,
            refinement_check=False,
            local_nfev_per_param=3,
        ),
    )
    assert narrow.hurst.window == pytest.approx((7.0 / 365.0, 60.0 / 365.0))
    assert narrow.hurst.n_expiries < run.hurst.n_expiries
    assert narrow.short_window == pytest.approx((7.0 / 365.0, 60.0 / 365.0))


def test_figures_are_opt_in_and_render_headless(run: Any, source: Any, tmp_path) -> None:
    """
    The seven spec-9 figures, written only when a directory is asked for. This is
    the ONLY test that touches matplotlib, and it forces the ``Agg`` backend
    through the module's own lazy import.
    """
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "figs"
    written = cli.save_diagnostic_figures(run.report, out_dir)
    assert set(written) == {
        "iv_surface",
        "atm_skew",
        "variance_swap",
        "forward_variance",
        "hurst",
        "residual_heatmap",
        "profiles",
    }
    for name, path in written.items():
        file = Path(path)
        assert file.is_file() and file.stat().st_size > 0, name
