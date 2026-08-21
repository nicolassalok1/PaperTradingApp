"""
Tests for the spec-9 validation report.

Target: app/model/calibration/rough_vol/diagnostics.py

ORACLES (all independent of the code under test):
  - The market and model implied-vol surfaces are written **forward** as exact
    local quadratics ``sigma(k) = a + b k + c k^2``. ``a`` is the ATM level and
    ``b`` the ATM skew BY CONSTRUCTION, so the ATM level error and the ATM skew
    error are known exactly before the module runs, and the report's numbers are
    compared against those constants - never read back from the module.
  - The IV / price error statistics are re-derived inline from plain numpy
    (``sqrt(mean(e^2))``, ``mean|e|``, ``max|e|``) on the very same arrays.
  - The variance-swap and forward-variance consistency checks are run against a
    REAL :class:`ForwardVarianceCurve` built by the spec-4.4 code from a
    hand-written ``K_var`` term structure whose forward variances are computed
    inline from ``(V_j - V_{j-1}) / (T_j - T_{j-1})``.
  - JSON-safety is checked by an explicit recursive type walk, plus a real
    ``json.dumps`` round trip - not by trusting the module's docstring.

Determinism: no RNG, no Monte-Carlo, no network, no matplotlib. The calibration
result is a hand-built stand-in - which is exactly the duck-typed contract the
module documents - so this module stays a fast unit test; the report is
exercised on a real ``JointCalibrationResult`` in
``tests/quant/test_rv_pipeline_e2e.py`` (slow).
"""

from __future__ import annotations

import ast
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.model.calibration.rough_vol import diagnostics as diag
from app.model.calibration.rough_vol.diagnostics import (
    CONSISTENCY_CHECK,
    FIT_METRIC,
    DiagnosticsConfig,
    PipelineArtifacts,
    atm_iv_term_structure,
    atm_skew_term_structure,
    attach_diagnostics,
    build_calibration_diagnostics,
    diagnostics_summary_fr,
    forward_variance_term_structure,
    market_surface_points,
    model_surface_points,
    per_maturity_errors,
    price_error_metrics,
    quote_table,
    residual_grid,
    variance_term_structure,
)
from app.model.calibration.rough_vol.forward_variance import (
    build_forward_variance_curve,
)
from app.model.calibration.rough_vol.variance_swap import VarianceSwapPoint

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Hand-built stand-ins for the calibration result (the module is duck typed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Quote:
    T: float
    K: float
    k: float
    F: float
    D: float
    iv: float
    price: float
    vega: float
    spread_iv: float
    option_type: str
    weight: float = 1.0


class _QuoteSet:
    def __init__(self, quotes: list[_Quote]) -> None:
        self.quotes = tuple(quotes)

    def diagnostics(self) -> dict[str, Any]:
        return {"n_quotes": int(len(self.quotes))}


@dataclass
class _Params:
    H: float
    eta: float
    rho: float


@dataclass
class _Result:
    """The subset of ``JointCalibrationResult`` the report actually reads."""

    quotes: _QuoteSet
    iv_model: np.ndarray
    price_model: np.ndarray
    weights: np.ndarray
    iv_error: np.ndarray | None = None
    params: _Params = field(default_factory=lambda: _Params(0.11, 1.3, -0.7))
    initial_params: _Params = field(default_factory=lambda: _Params(0.12, 1.2, -0.7))
    success: bool = True
    message_fr: str = "Calibration jointe rBergomi terminée."
    xi0_curve: Any = None
    flags: tuple[str, ...] = ()
    warnings_fr: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


#: Maturities of the stand-in surface, in years (calendar days / 365).
_MATURITIES = (7.0 / 365.0, 30.0 / 365.0, 91.0 / 365.0)
#: Exact local-quadratic coefficients, market side: (sigma_atm, psi, c).
_MARKET_SMILE = {
    _MATURITIES[0]: (0.2200, -1.1000, 2.5),
    _MATURITIES[1]: (0.2000, -0.7000, 1.8),
    _MATURITIES[2]: (0.1900, -0.4500, 1.2),
}
#: The model side is the SAME quadratic shifted by known offsets, so the report's
#: ATM level error and ATM skew error are known before the module runs.
_LEVEL_OFFSET = {_MATURITIES[0]: 0.0030, _MATURITIES[1]: -0.0015, _MATURITIES[2]: 0.0005}
_SKEW_OFFSET = {_MATURITIES[0]: 0.0400, _MATURITIES[1]: -0.0250, _MATURITIES[2]: 0.0100}

_F = 100.0
_D = 0.995
_K_GRID = tuple(round(-0.10 + 0.02 * i, 10) for i in range(11))  # 11 strikes, |k| <= 0.10


def _smile(T: float, k: float, *, model: bool) -> float:
    a, b, c = _MARKET_SMILE[T]
    if model:
        a = a + _LEVEL_OFFSET[T]
        b = b + _SKEW_OFFSET[T]
    return a + b * k + c * k * k


def _build_result(*, with_iv_error: bool = True) -> _Result:
    quotes: list[_Quote] = []
    iv_model: list[float] = []
    price_model: list[float] = []
    weights: list[float] = []
    for T in _MATURITIES:
        for i, k in enumerate(_K_GRID):
            K = _F * math.exp(k)
            iv_market = _smile(T, k, model=False)
            quotes.append(
                _Quote(
                    T=T,
                    K=K,
                    k=k,
                    F=_F,
                    D=_D,
                    iv=iv_market,
                    # Deterministic, strictly positive stand-in prices and vegas.
                    # Their exact level is irrelevant: every price statistic is
                    # re-derived inline from these very numbers.
                    price=1.0 + 0.1 * i + 10.0 * T,
                    vega=5.0 + 0.5 * i,
                    spread_iv=0.01,
                    option_type="call" if k >= 0.0 else "put",
                    weight=1.0 + 0.05 * i,
                )
            )
            iv_model.append(_smile(T, k, model=True))
            price_model.append(1.0 + 0.1 * i + 10.0 * T + 0.02 * (i - 5))
            weights.append(1.0 + 0.05 * i)
    market_iv = np.asarray([q.iv for q in quotes], dtype=float)
    model_iv = np.asarray(iv_model, dtype=float)
    return _Result(
        quotes=_QuoteSet(quotes),
        iv_model=model_iv,
        price_model=np.asarray(price_model, dtype=float),
        weights=np.asarray(weights, dtype=float),
        iv_error=(model_iv - market_iv) if with_iv_error else None,
    )


@pytest.fixture(scope="module")
def result() -> _Result:
    return _build_result()


@pytest.fixture(scope="module")
def xi0_curve():
    """A REAL spec-4.4 curve from a hand-written ``K_var`` term structure."""
    return build_forward_variance_curve(_variance_points())


def _variance_points() -> list[VarianceSwapPoint]:
    k_var = {_MATURITIES[0]: 0.0500, _MATURITIES[1]: 0.0440, _MATURITIES[2]: 0.0420}
    return [
        VarianceSwapPoint(
            T=float(T),
            k_var=float(value),
            k_var_trunc=float(value),
            n_puts=6,
            n_calls=6,
            F=_F,
            D=_D,
        )
        for T, value in k_var.items()
    ]


class _VarianceCurveStub:
    """Just the ``points`` attribute the report reads."""

    def __init__(self) -> None:
        self.points = tuple(_variance_points())


# ---------------------------------------------------------------------------
# 1. Quote flattening
# ---------------------------------------------------------------------------
def test_quote_table_is_rectangular_and_plain(result: _Result) -> None:
    table = quote_table(result)
    n = len(result.quotes.quotes)
    assert n == len(_MATURITIES) * len(_K_GRID)
    for name, column in table.items():
        assert len(column) == n, name
        assert isinstance(column, list), name
    assert all(isinstance(x, float) for x in table["iv_market"])
    assert table["option_type"][0] in {"call", "put"}
    # price_error is model - market, quote by quote.
    for i, quote in enumerate(result.quotes.quotes):
        assert table["price_error"][i] == pytest.approx(
            float(result.price_model[i]) - quote.price, abs=1e-15
        )


def test_iv_error_is_recomputed_when_the_result_does_not_carry_it() -> None:
    """A result without ``iv_error`` must not silently produce NaN columns."""
    carried = quote_table(_build_result(with_iv_error=True))["iv_error"]
    recomputed = quote_table(_build_result(with_iv_error=False))["iv_error"]
    assert np.allclose(np.asarray(carried), np.asarray(recomputed), atol=1e-15)
    assert all(math.isfinite(x) for x in recomputed)


def test_model_surface_drops_unusable_ivs_instead_of_inventing_them(
    result: _Result,
) -> None:
    """A NaN model IV is missing, not zero: it must lose its point, not gain one."""
    broken = _build_result()
    broken.iv_model = broken.iv_model.copy()
    broken.iv_model[0] = float("nan")
    broken.iv_model[1] = 0.0
    assert len(market_surface_points(broken)) == len(result.quotes.quotes)
    assert len(model_surface_points(broken)) == len(result.quotes.quotes) - 2


# ---------------------------------------------------------------------------
# 2. Term structures against the closed-form oracle
# ---------------------------------------------------------------------------
def test_atm_skew_term_structure_recovers_the_known_offsets(result: _Result) -> None:
    """
    Oracle: both surfaces are exact local quadratics whose ``b`` coefficient is
    written by the test. The reported error must be exactly the offset applied.
    """
    section = atm_skew_term_structure(result)
    assert section["available"] is True
    assert section["role"] == FIT_METRIC
    assert section["n_model_failed"] == 0
    assert section["maturities"] == pytest.approx(list(_MATURITIES), rel=1e-12)
    for i, T in enumerate(section["maturities"]):
        expected_market = _MARKET_SMILE[_MATURITIES[i]][1]
        assert section["psi_market"][i] == pytest.approx(expected_market, abs=1e-8)
        assert section["psi_model"][i] == pytest.approx(
            expected_market + _SKEW_OFFSET[_MATURITIES[i]], abs=1e-8
        )
        assert section["error"][i] == pytest.approx(
            _SKEW_OFFSET[_MATURITIES[i]], abs=1e-8
        )
        assert T == pytest.approx(section["maturities_days"][i] / 365.0, rel=1e-12)
    offsets = np.asarray([_SKEW_OFFSET[T] for T in _MATURITIES], dtype=float)
    assert section["rmse"] == pytest.approx(
        float(np.sqrt(np.mean(offsets**2))), abs=1e-8
    )
    assert section["mae"] == pytest.approx(float(np.mean(np.abs(offsets))), abs=1e-8)
    assert section["max_abs"] == pytest.approx(float(np.max(np.abs(offsets))), abs=1e-8)


def test_atm_iv_term_structure_recovers_the_known_level_offsets(
    result: _Result,
) -> None:
    """The spec-9 'term structure error': model ATM vol level minus market's."""
    section = atm_iv_term_structure(result)
    assert section["available"] is True
    assert section["role"] == FIT_METRIC
    for i, _T in enumerate(section["maturities"]):
        key = _MATURITIES[i]
        assert section["sigma_atm_market"][i] == pytest.approx(
            _MARKET_SMILE[key][0], abs=1e-8
        )
        assert section["error"][i] == pytest.approx(_LEVEL_OFFSET[key], abs=1e-8)
        assert section["error_vol_points"][i] == pytest.approx(
            _LEVEL_OFFSET[key] * 100.0, abs=1e-6
        )
    assert section["rmse_vol_points"] == pytest.approx(section["rmse"] * 100.0, rel=1e-12)


def test_a_maturity_the_model_cannot_fit_is_reported_not_dropped() -> None:
    """Losing a smile must show up as NaN + a counter, never as a shorter curve."""
    broken = _build_result()
    broken.iv_model = broken.iv_model.copy()
    broken.iv_model[: len(_K_GRID)] = float("nan")  # kill the first maturity entirely
    section = atm_skew_term_structure(broken)
    assert len(section["maturities"]) == len(_MATURITIES)
    assert not math.isfinite(section["psi_model"][0])
    assert math.isfinite(section["psi_market"][0])
    assert section["n_model_failed"] == 1
    assert section["n"] == len(_MATURITIES) - 1


# ---------------------------------------------------------------------------
# 3. The two consistency checks (expected ~0 BY CONSTRUCTION)
# ---------------------------------------------------------------------------
def test_variance_term_structure_is_a_consistency_check_and_is_exact(
    result: _Result, xi0_curve
) -> None:
    section = variance_term_structure(
        result, variance_curve=_VarianceCurveStub(), xi0_curve=xi0_curve
    )
    assert section["available"] is True
    assert section["role"] == CONSISTENCY_CHECK, "must not be sold as a fit metric"
    assert "cohérence" in section["definition_fr"].lower()
    # Oracle: (1/T) * integral_0^T xi0 IS K_var at every knot, by construction.
    for i, T in enumerate(section["maturities"]):
        assert section["k_var_model"][i] == pytest.approx(
            float(xi0_curve.integrated(T)) / float(T), rel=1e-15
        )
        assert section["error"][i] == pytest.approx(0.0, abs=1e-13)
    assert section["max_abs"] <= section["tolerance"]
    assert section["consistent"] is True
    assert section["market_source"] == "variance_swap_curve"


def test_forward_variance_term_structure_matches_the_inline_finite_difference(
    result: _Result, xi0_curve
) -> None:
    """Oracle: xi0 on ``(T_{j-1}, T_j]`` is ``(V_j - V_{j-1}) / (T_j - T_{j-1})``."""
    points = _variance_points()
    expected: list[float] = []
    previous_T = 0.0
    previous_V = 0.0
    for point in points:
        V = float(point.T) * float(point.k_var)
        expected.append((V - previous_V) / (float(point.T) - previous_T))
        previous_T, previous_V = float(point.T), V

    section = forward_variance_term_structure(
        result, variance_curve=_VarianceCurveStub(), xi0_curve=xi0_curve
    )
    assert section["available"] is True
    assert section["role"] == CONSISTENCY_CHECK
    assert section["xi0_market"] == pytest.approx(expected, rel=1e-12)
    assert section["xi0_model"] == pytest.approx(expected, rel=1e-12)
    assert section["max_abs"] == pytest.approx(0.0, abs=1e-13)
    assert section["levels"] == pytest.approx(expected, rel=1e-12)


def test_a_missing_curve_yields_a_reason_not_a_number(result: _Result) -> None:
    for section in (
        variance_term_structure(result, variance_curve=None, xi0_curve=None),
        forward_variance_term_structure(result, variance_curve=None, xi0_curve=None),
    ):
        assert section["available"] is False
        assert isinstance(section["reason_fr"], str) and section["reason_fr"]
        assert set(section) == {"available", "reason_fr"}, "no fabricated number"


# ---------------------------------------------------------------------------
# 4. Error statistics, re-derived inline
# ---------------------------------------------------------------------------
def test_price_error_metrics_match_plain_numpy(result: _Result) -> None:
    table = quote_table(result)
    error = np.asarray(table["price_error"], dtype=float)
    reference = np.asarray(table["price_market"], dtype=float)
    section = price_error_metrics(result)
    assert section["role"] == FIT_METRIC
    assert section["rmse"] == pytest.approx(float(np.sqrt(np.mean(error**2))), rel=1e-12)
    assert section["mae"] == pytest.approx(float(np.mean(np.abs(error))), rel=1e-12)
    assert section["max_abs"] == pytest.approx(float(np.max(np.abs(error))), rel=1e-12)
    assert section["n"] == error.size
    relative = error / reference
    assert section["rmse_relative"] == pytest.approx(
        float(np.sqrt(np.mean(relative**2))), rel=1e-12
    )


def test_per_maturity_rows_partition_the_quotes_and_match_the_global_metrics(
    result: _Result,
) -> None:
    rows = per_maturity_errors(result)
    table = quote_table(result)
    assert [row["T"] for row in rows] == pytest.approx(list(_MATURITIES), rel=1e-12)
    assert sum(row["n_quotes"] for row in rows) == len(table["T"])
    for row in rows:
        idx = [i for i, t in enumerate(table["T"]) if abs(t - row["T"]) < 1e-12]
        error = np.asarray([table["iv_error"][i] for i in idx], dtype=float)
        assert row["iv"]["rmse"] == pytest.approx(
            float(np.sqrt(np.mean(error**2))), rel=1e-12
        )
        assert row["iv"]["mae"] == pytest.approx(float(np.mean(np.abs(error))), rel=1e-12)
        assert row["iv_bias"] == pytest.approx(float(np.mean(error)), rel=1e-12)
        assert row["n_quotes"] == len(idx)
        assert row["k_min"] == pytest.approx(min(_K_GRID), abs=1e-12)
        assert row["k_max"] == pytest.approx(max(_K_GRID), abs=1e-12)
        # The per-maturity ATM numbers come from the same oracle as section 2.
        assert row["psi_error"] == pytest.approx(_SKEW_OFFSET[row["T"]], abs=1e-8)
        assert row["sigma_atm_error"] == pytest.approx(
            _LEVEL_OFFSET[row["T"]], abs=1e-8
        )


def test_residual_grid_carries_every_quote_and_honours_its_cap(result: _Result) -> None:
    full = residual_grid(result)
    n = len(result.quotes.quotes)
    assert full["available"] is True
    assert full["n_quotes"] == n and full["n_rows"] == n
    assert full["truncated"] is False
    assert len(full["columns"]["iv_error"]) == n
    assert full["columns"]["iv_error_vol_points"] == pytest.approx(
        [100.0 * x for x in full["columns"]["iv_error"]], rel=1e-12
    )
    capped = residual_grid(result, config=DiagnosticsConfig(max_residual_rows=5))
    assert capped["n_rows"] == 5 and capped["truncated"] is True
    assert capped["n_quotes"] == n, "the true count must survive the cap"


# ---------------------------------------------------------------------------
# 5. The assembled report
# ---------------------------------------------------------------------------
def _type_offenders(node: Any, path: str = "$") -> list[str]:
    """Every value whose type would NOT survive the controller's ``_json_safe``."""
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


@pytest.fixture(scope="module")
def report(result: _Result, xi0_curve) -> dict[str, Any]:
    return build_calibration_diagnostics(
        result,
        artifacts=PipelineArtifacts(
            variance_curve=_VarianceCurveStub(),
            xi0_curve=xi0_curve,
            source="stand-in",
            is_synthetic=True,
            notes_fr=("Données synthétiques.",),
        ),
    )


def test_report_covers_every_spec9_item(report: dict[str, Any]) -> None:
    for key in (
        "parameters",
        "loss",
        "iv_error",
        "iv_error_weighted",
        "price_error",
        "atm_iv_term_structure",
        "atm_skew_term_structure",
        "per_maturity",
        "variance_term_structure",
        "forward_variance_term_structure",
        "residuals",
        "hurst",
        "identifiability",
        "open_items_fr",
        "figures",
        "summary_fr",
    ):
        assert key in report, key
    assert report["parameters"]["calibrated"] == {"H": 0.11, "eta": 1.3, "rho": -0.7}
    assert report["parameters"]["xi0_frozen"] is True
    assert report["is_synthetic"] is True
    assert report["figures"] == {}, "PNG generation must be off by default"
    assert len(report["per_maturity"]) == len(_MATURITIES)
    assert len(report["open_items_fr"]) >= 3


def test_fit_metrics_and_consistency_checks_are_labelled_apart(
    report: dict[str, Any],
) -> None:
    """A reader must never mistake an arithmetic identity for a fit quality."""
    assert report["iv_error"]["role"] == FIT_METRIC
    assert report["price_error"]["role"] == FIT_METRIC
    assert report["atm_iv_term_structure"]["role"] == FIT_METRIC
    assert report["atm_skew_term_structure"]["role"] == FIT_METRIC
    assert report["variance_term_structure"]["role"] == CONSISTENCY_CHECK
    assert report["forward_variance_term_structure"]["role"] == CONSISTENCY_CHECK


def test_report_is_json_safe(report: dict[str, Any]) -> None:
    assert _type_offenders(report) == []
    blob = json.dumps(report, ensure_ascii=False)
    assert json.loads(blob).keys() == report.keys()


def test_iv_error_metrics_match_plain_numpy(report: dict[str, Any], result: _Result) -> None:
    error = np.asarray(quote_table(result)["iv_error"], dtype=float)
    assert report["iv_error"]["rmse"] == pytest.approx(
        float(np.sqrt(np.mean(error**2))), rel=1e-12
    )
    assert report["iv_error"]["rmse_vol_points"] == pytest.approx(
        report["iv_error"]["rmse"] * 100.0, rel=1e-12
    )
    weights = np.asarray(quote_table(result)["vega"], dtype=float)
    expected_vw = float(
        np.sqrt(np.sum(weights * error**2) / np.sum(weights))
    )
    assert report["iv_error_weighted"]["rmse_vw"] == pytest.approx(expected_vw, rel=1e-12)


def test_summary_names_the_parameters_and_the_consistency_verdict(
    report: dict[str, Any],
) -> None:
    text = diagnostics_summary_fr(report)
    assert text == report["summary_fr"]
    assert "H = 0.1100" in text
    assert "eta = 1.3000" in text
    assert "rho = -0.7000" in text
    assert "Contrôle de cohérence" in text


def test_report_degrades_section_by_section_without_artifacts(result: _Result) -> None:
    """No artifacts at all: sections say why, and nothing is fabricated."""
    bare = build_calibration_diagnostics(result)
    assert bare["variance_term_structure"]["available"] is False
    assert bare["forward_variance_term_structure"]["available"] is False
    assert bare["hurst"]["available"] is False
    assert bare["identifiability"] is None
    # The fit metrics that DO have data are still there.
    assert bare["atm_skew_term_structure"]["available"] is True
    assert math.isfinite(bare["price_error"]["rmse"])
    assert _type_offenders(bare) == []


def test_attach_diagnostics_writes_into_details_and_returns_the_report(
    result: _Result,
) -> None:
    target = _build_result()
    assert target.details == {}
    returned = attach_diagnostics(target, key="validation_report")
    assert target.details["validation_report"] is returned
    assert returned["spec"] == "9"


def test_hurst_section_reports_the_estimate_and_its_confidence_interval(
    result: _Result,
) -> None:
    class _Hurst:
        H0 = 0.1224
        se = 0.0086
        ci95 = (0.1056, 0.1392)
        r2 = 0.998
        n_expiries = 6
        window = (5.0 / 365.0, 0.25)
        unstable = False
        message_fr = "Estimation initiale H0 = 0.1224"

    section = build_calibration_diagnostics(
        result, artifacts=PipelineArtifacts(hurst=_Hurst())
    )["hurst"]
    assert section["available"] is True
    assert section["H0"] == pytest.approx(0.1224)
    assert section["ci95"] == pytest.approx([0.1056, 0.1392])
    assert section["window"] == pytest.approx([5.0 / 365.0, 0.25])
    assert section["unstable"] is False


# ---------------------------------------------------------------------------
# 6. Layering: matplotlib must stay lazy, the module must stay model-layer
# ---------------------------------------------------------------------------
def test_matplotlib_is_never_imported_at_module_level() -> None:
    """
    Structural check: no module-level ``import matplotlib`` anywhere in the file,
    and the ``Agg`` backend is forced before ``pyplot`` is touched.
    """
    source = Path(diag.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "matplotlib" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "matplotlib"
    use_index = source.index('matplotlib.use("Agg")')
    assert use_index < source.index("import matplotlib.pyplot")


def test_building_the_report_does_not_pull_matplotlib_in(
    result: _Result, xi0_curve
) -> None:
    already_loaded = "matplotlib" in sys.modules
    build_calibration_diagnostics(
        result, artifacts=PipelineArtifacts(xi0_curve=xi0_curve)
    )
    if not already_loaded:
        assert "matplotlib" not in sys.modules


def test_the_model_layer_imports_no_view_and_no_controller() -> None:
    """
    The MVC rule, checked on the imports themselves (the prose may well *name*
    streamlit to say it is absent - ``scripts/check_mvc_integrity.py`` reads the
    AST for the same reason).
    """
    forbidden = ("streamlit", "app.vue", "app.controller")
    tree = ast.parse(Path(diag.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    for name in imported:
        assert not any(
            name == bad or name.startswith(bad + ".") for bad in forbidden
        ), name
