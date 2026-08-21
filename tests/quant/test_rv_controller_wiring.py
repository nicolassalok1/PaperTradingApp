"""Phase 5B — controller wiring of the rough-volatility pipeline.

Scope: the *plumbing*, not the numerics. Every heavy stage is either stubbed or
proven not to run:

* the spec-4.10 joint calibrator is registered under its own dispatch key and
  reachable through ``run_advanced_surface_calibration``;
* ``run_rbergomi_hurst_pipeline`` chains spec 4.1 -> 4.9 off a synthetic chain
  built here (closed-form Black-76, no Monte-Carlo anywhere) and **refuses to
  start the expensive fit unless asked**;
* ``success=False`` is carried to the caller with its French reason and its flag
  labels instead of being swallowed, and the parameter triple that comes with it
  is explicitly marked unusable — the repo guardrail "never present a
  meaningless H as a result" enforced at the controller boundary;
* everything returned survives ``_json_safe`` and ``json.dumps``.

The synthetic chain is generated from Black-76 with an analytic smile, so it is
an INDEPENDENT oracle for the pipeline's own inversions: the implied vols the
pipeline recovers must be the ones this module priced, and the parity forward
must be ``S0 * exp(r * T)`` because the puts are priced from the calls by exact
parity.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from app.controller import calibration_controller as cc
from app.controller.calibration_controller import (
    ROUGH_VOL_MODEL_KEY,
    ROUGH_VOL_STAGE_FULL,
    ROUGH_VOL_STAGE_PREPARE,
    CalibrationController,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic option chain — the independent oracle
# ---------------------------------------------------------------------------
S0 = 100.0
RATE = 0.02
EXPIRY_DAYS = (7, 14, 30, 60, 90, 180, 365)
SPREAD_REL = 0.01


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _black76_call(*, F: float, K: float, T: float, D: float, vol: float) -> float:
    if not (T > 0.0 and vol > 0.0):
        return D * max(F - K, 0.0)
    s = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    return D * (F * _norm_cdf(d1) - K * _norm_cdf(d1 - s))


def _smile_iv(k: float, T: float) -> float:
    """A rough-flavoured smile: ATM level flat, skew decaying like ``T**(H-1/2)``."""
    psi = -0.30 * (T ** (0.10 - 0.5)) / (0.25 ** (0.10 - 0.5))
    return float(0.22 + psi * 0.05 * k + 0.60 * k * k)


def synthetic_chain() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls_df, puts_df) in the ``fetch_options_details_yahoo`` row schema.

    Puts are priced from the calls by EXACT put-call parity, so the spec-4.2
    slope-pinned regression must recover ``F = S0 exp(r T)`` and ``D = exp(-r T)``.
    """
    call_rows: list[dict] = []
    put_rows: list[dict] = []
    for days in EXPIRY_DAYS:
        T = days / 365.0
        D = math.exp(-RATE * T)
        F = S0 / D
        # log-moneyness ladder, dense enough for the ATM skew fit and the K_var
        # quadrature on both wings.
        for k in np.linspace(-0.30, 0.30, 25):
            K = float(F * math.exp(float(k)))
            iv = _smile_iv(float(k), T)
            call = _black76_call(F=F, K=K, T=T, D=D, vol=iv)
            put = call - D * (F - K)  # exact parity
            for price, opt_type, sink in ((call, "call", call_rows), (put, "put", put_rows)):
                if price <= 1e-6:
                    continue
                half = 0.5 * SPREAD_REL * price
                sink.append(
                    {
                        "underlying": "TEST",
                        "contractSymbol": f"TEST{days}{opt_type[0].upper()}{K:.2f}",
                        "expiry": f"day+{days}",
                        # Distinct per expiry: `clean_option_chains` groups on this
                        # first, and a constant would collapse every expiry into one
                        # chain at the MEDIAN maturity.
                        "expiry_ts": int(days),
                        "T": T,
                        "strike": K,
                        "iv": iv,
                        "bid": price - half,
                        "ask": price + half,
                        "lastPrice": price,
                        "openInterest": 250,
                        "volume": 120,
                        "inTheMoney": bool(K < F) if opt_type == "call" else bool(K > F),
                        "type": opt_type,
                        "S0": S0,
                    }
                )
    return pd.DataFrame(call_rows), pd.DataFrame(put_rows)


def _pipeline_payload(**overrides):
    calls, puts = synthetic_chain()
    payload = {
        "ticker": "TEST",
        "calls": calls,
        "puts": puts,
        "S0": S0,
        "r": RATE,
        "q": 0.0,
        "seed": 0,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Stub calibrator — stands in for the expensive spec-4.10 fit
# ---------------------------------------------------------------------------
class _StubJointCalibrator:
    """Records its call and returns a canned ``SurfaceCalibrationResult``."""

    model = "rbergomi"
    method = "joint_h_mc"
    PARAM_ORDER = ("H", "eta", "rho")
    DEFAULT_BOUNDS = {"H": (0.01, 0.49), "eta": (0.05, 5.0), "rho": (-0.999, 0.999)}

    calls: list[dict] = []
    success = True
    message = "Calibration jointe rBergomi réussie."
    flags: tuple[str, ...] = ()
    warnings_fr: tuple[str, ...] = ()

    def calibrate(self, surface, *, constraints=None, settings=None):
        from app.model.calibration.base_calibrator import SurfaceCalibrationResult

        type(self).calls.append(
            {"surface": surface, "constraints": constraints, "settings": settings}
        )
        n_t = len(np.asarray(surface.t_grid, dtype=float))
        n_m = len(np.asarray(surface.m_grid, dtype=float))
        iv_model = np.full((n_t, n_m), 0.22, dtype=float)
        return SurfaceCalibrationResult(
            success=bool(type(self).success),
            message=str(type(self).message),
            model=self.model,
            method=self.method,
            params={"H": 0.11, "eta": 1.42, "rho": -0.68},
            metrics={"mae": 0.004, "rmse": 0.005, "max_abs": 0.01, "n": float(n_t * n_m)},
            metrics_vw={"mae_vw": 0.004, "rmse_vw": 0.005, "max_abs_vw": 0.01},
            iv_model=iv_model,
            iv_error=np.zeros((n_t, n_m), dtype=float),
            vega_weights=np.ones((n_t, n_m), dtype=float),
            details={
                "flags": [str(f) for f in type(self).flags],
                "warnings_fr": [str(w) for w in type(self).warnings_fr],
                "report": {
                    "success": bool(type(self).success),
                    "flags": [str(f) for f in type(self).flags],
                    "warnings_fr": [str(w) for w in type(self).warnings_fr],
                    "message_fr": str(type(self).message),
                    "H": 0.11,
                    "eta": 1.42,
                    "rho": -0.68,
                },
            },
        )


@pytest.fixture
def stub_joint(monkeypatch):
    """Swap the real joint calibrator for the stub, at its definition site."""
    import app.model.volatility_models.rbergomi.calibrator_joint_mc as joint

    _StubJointCalibrator.calls = []
    _StubJointCalibrator.success = True
    _StubJointCalibrator.message = "Calibration jointe rBergomi réussie."
    _StubJointCalibrator.flags = ()
    _StubJointCalibrator.warnings_fr = ()
    monkeypatch.setattr(joint, "RBergomiJointHCalibrator", _StubJointCalibrator)
    return _StubJointCalibrator


@pytest.fixture
def no_monte_carlo(monkeypatch):
    """Make ANY construction of the real joint calibrator an immediate failure."""
    import app.model.volatility_models.rbergomi.calibrator_joint_mc as joint

    def _boom(*args, **kwargs):
        raise AssertionError("the expensive joint calibrator must not be constructed here")

    monkeypatch.setattr(joint, "RBergomiJointHCalibrator", _boom)
    monkeypatch.setattr(joint, "calibrate_rbergomi", _boom)


# ---------------------------------------------------------------------------
# 1. registration
# ---------------------------------------------------------------------------
def test_the_joint_calibrator_is_registered_as_an_expensive_advanced_model():
    specs = CalibrationController().get_advanced_models()
    by_key = {s["key"]: s for s in specs}

    assert ROUGH_VOL_MODEL_KEY in by_key, sorted(by_key)
    spec = by_key[ROUGH_VOL_MODEL_KEY]
    assert spec["expensive"] is True
    assert spec["calibration"] == "joint_h_mc"
    assert spec["pricing"] == "mc"
    assert "xi0_curve" in spec["requires_constraints"]

    # The pre-existing surrogate entry is pinned by
    # tests/quant/test_advanced_calibration_roundtrip.py: it must be untouched.
    assert by_key["rbergomi"]["calibration"] == "mc_surrogate"
    assert by_key["rbergomi"]["label"] == "rBergomi (MC + surrogate)"
    assert ROUGH_VOL_MODEL_KEY != "rbergomi"

    # Every spec keeps the five keys the advanced tab reads.
    for spec in specs:
        assert {"key", "label", "pricing", "calibration", "expensive"} <= set(spec)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_every_advanced_model_key_resolves_in_the_calibrator_map():
    """The dispatcher must know every key `get_advanced_models` advertises.

    An unknown key returns "Modèle inconnu", which is exactly what a missing
    `calibrator_map` entry would produce — so this catches the half-wiring where
    the model is advertised but not dispatchable.
    """
    ctrl = CalibrationController()
    df = pd.DataFrame(
        [{"K": 100.0, "T": 0.5, "S0": 100.0, "iv": 0.2, "type": "call"}]
    )
    for spec in ctrl.get_advanced_models():
        res = ctrl.run_advanced_surface_calibration(
            {"model": spec["key"], "df": df, "S0": 100.0, "constraints": {}}
        )
        assert isinstance(res, dict)
        assert "Modèle inconnu" not in str(res.get("message")), spec["key"]


def test_an_unknown_model_key_is_still_refused_by_name():
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": "rbergomi_joint_h_typo",
            "df": pd.DataFrame([{"K": 100.0, "T": 0.5, "S0": 100.0, "iv": 0.2}]),
        }
    )
    assert res["success"] is False
    assert "Modèle inconnu" in res["message"]


def test_the_dispatcher_reaches_the_joint_calibrator(stub_joint):
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": ROUGH_VOL_MODEL_KEY,
            "df": pd.DataFrame(
                [
                    {"K": 90.0 + 5 * i, "T": T, "S0": 100.0, "iv": 0.2}
                    for i in range(5)
                    for T in (0.25, 0.5, 1.0)
                ]
            ),
            "S0": 100.0,
            "constraints": {"xi0_curve": object()},
        }
    )
    assert len(stub_joint.calls) == 1
    assert res["success"] is True
    assert res["model"] == "rbergomi"
    assert res["method"] == "joint_h_mc"
    # The constraints reached the calibrator untouched.
    assert "xi0_curve" in stub_joint.calls[0]["constraints"]


def test_the_real_calibrator_refuses_without_an_xi0_curve_in_french():
    """No Monte-Carlo runs: the calibrator returns before any simulation.

    An IV grid does not determine a variance-swap curve, so the absence of
    `constraints["xi0_curve"]` must be a named refusal, never a flat-xi0 fallback.
    """
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": ROUGH_VOL_MODEL_KEY,
            "df": pd.DataFrame(
                [{"K": 100.0, "T": T, "S0": 100.0, "iv": 0.2} for T in (0.25, 0.5)]
            ),
            "S0": 100.0,
            "constraints": {},
        }
    )
    assert res["success"] is False
    assert "xi0_curve" in res["message"]
    assert "variance forward" in res["message"]
    assert res["params"] == {}


# ---------------------------------------------------------------------------
# 2. the pipeline, prepare stage (no Monte-Carlo)
# ---------------------------------------------------------------------------
def test_prepare_is_the_default_stage_and_runs_no_monte_carlo(no_monte_carlo):
    res = CalibrationController().run_rbergomi_hurst_pipeline(_pipeline_payload())

    assert res["success"] is True, res.get("message")
    assert res["stage"] == ROUGH_VOL_STAGE_PREPARE  # the DEFAULT: never a casual fit
    assert res["failed_step"] is None
    for key in ("cleaning", "forward_curve", "variance_swap", "forward_variance",
                "hurst", "initializer", "cost"):
        assert key in res, key
    # Nothing is calibrated at this stage.
    assert res["params"] == {}
    assert res["params_usable"] is False
    assert "point" in res["message"].lower() or "départ" in res["message"]


def test_prepare_recovers_the_forward_and_the_smile_it_was_priced_from(no_monte_carlo):
    """Independent-oracle check that the plumbing feeds the right numbers through."""
    res = CalibrationController().run_rbergomi_hurst_pipeline(_pipeline_payload())

    assert res["n_maturities"] == len(EXPIRY_DAYS)
    assert res["S0"] == pytest.approx(S0)
    assert sorted(res["t_grid"]) == pytest.approx(sorted(d / 365.0 for d in EXPIRY_DAYS))

    # Puts were priced from the calls by exact parity => F = S0 exp(rT), D = exp(-rT).
    for point in res["forward_curve"]["points"]:
        T = float(point["T"])
        assert float(point["F"]) == pytest.approx(S0 * math.exp(RATE * T), rel=1e-9)
        assert float(point["D"]) == pytest.approx(math.exp(-RATE * T), rel=1e-12)

    # The variance-swap strikes sit in the neighbourhood of the ATM variance the
    # smile was built with (0.22**2), never negative.
    kvars = [float(v) for v in res["variance_swap"]["k_var"]]
    assert kvars
    assert all(kv > 0.0 for kv in kvars)
    assert min(kvars) > 0.5 * 0.22**2 and max(kvars) < 2.0 * 0.22**2

    # xi0 is DATA: it exists before any (H, eta, rho) does.
    assert res["forward_variance"]["n_maturities"] >= 2
    assert len(res["forward_variance"]["levels"]) == res["forward_variance"]["n_maturities"]


def test_prepare_never_hands_back_a_calibrated_h(no_monte_carlo):
    """H0 is an initialiser and says so; it is not reported as a result."""
    res = CalibrationController().run_rbergomi_hurst_pipeline(_pipeline_payload())

    assert res["hurst"]["H0_is_fallback"] in (True, False)
    assert "calibration" in res["hurst"]["message_fr"].lower() or res["hurst"]["H0_is_fallback"]
    assert res["initializer"]["warning_fr"]
    assert "jamais un résultat" in res["initializer"]["warning_fr"]
    assert res["params_usable"] is False


def test_prepare_costs_the_fit_it_did_not_run(no_monte_carlo):
    res = CalibrationController().run_rbergomi_hurst_pipeline(_pipeline_payload())
    cost = res["cost"]
    assert cost["success"] is True
    assert cost["expensive"] is True
    assert cost["n_evaluations"] > cost["local_stage_only_evaluations"]
    assert cost["n_paths_total"] > 0


# ---------------------------------------------------------------------------
# 3. refusals, named by the step that produced them
# ---------------------------------------------------------------------------
def test_the_pipeline_refuses_without_data_and_names_the_step():
    res = CalibrationController().run_rbergomi_hurst_pipeline({})
    assert res["success"] is False
    assert res["failed_step"] == "chain"
    assert "Ticker manquant" in res["message"]
    assert res["params"] == {}
    assert res["params_usable"] is False
    assert res["steps"][-1]["ok"] is False


def test_an_empty_chain_stops_at_the_cleaning_step():
    empty = pd.DataFrame(columns=["type", "strike", "T", "bid", "ask", "expiry_ts", "S0"])
    res = CalibrationController().run_rbergomi_hurst_pipeline(
        {"ticker": "TEST", "calls": empty, "puts": empty, "S0": S0, "r": RATE}
    )
    assert res["success"] is False
    assert res["failed_step"] == "chain"
    assert res["params_usable"] is False


def test_every_reported_step_carries_a_french_label():
    ctrl = CalibrationController()
    steps = ctrl.get_rough_vol_steps()
    assert [s["step"] for s in steps][0] == "chain"
    assert [s["step"] for s in steps][-1] == "calibration"
    assert all(s["label_fr"] and s["label_fr"] != s["step"] for s in steps)


# ---------------------------------------------------------------------------
# 4. success=False is a verdict, and it reaches the caller
# ---------------------------------------------------------------------------
def test_a_false_verdict_is_surfaced_with_its_french_reason(stub_joint):
    from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
        FLAG_H_PROFILE_FLAT,
        FLAG_H_WEAKLY_IDENTIFIED,
        JOINT_CALIBRATION_LABELS_FR,
    )

    stub_joint.success = False
    stub_joint.message = (
        "Calibration en échec : cette surface n'identifie pas H au-delà du bruit "
        "Monte-Carlo mesuré."
    )
    stub_joint.flags = (FLAG_H_PROFILE_FLAT, FLAG_H_WEAKLY_IDENTIFIED)
    stub_joint.warnings_fr = tuple(
        JOINT_CALIBRATION_LABELS_FR[f] for f in stub_joint.flags
    )

    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=ROUGH_VOL_STAGE_FULL)
    )

    assert len(stub_joint.calls) == 1
    assert res["success"] is False
    assert res["failed_step"] == "calibration"
    # NOT swallowed: the model's own French verdict is the message.
    assert res["message"] == stub_joint.message
    assert set(res["flags"]) == {FLAG_H_PROFILE_FLAT, FLAG_H_WEAKLY_IDENTIFIED}
    assert res["warnings_fr"] and all("identifi" in w or "PLAT" in w for w in res["warnings_fr"])
    # The triple is still reported for diagnosis, but is explicitly NOT a result.
    assert res["params_usable"] is False
    assert res["params"]["H"] == pytest.approx(0.11)
    assert res["steps"][-1] == {
        "step": "calibration",
        "label_fr": dict(cc.ROUGH_VOL_STEPS)["calibration"],
        "ok": False,
        "message_fr": stub_joint.message,
    }


def test_a_true_verdict_marks_the_parameters_usable(stub_joint):
    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=ROUGH_VOL_STAGE_FULL)
    )
    assert res["success"] is True, res.get("message")
    assert res["params_usable"] is True
    assert res["model"] == "rbergomi"
    assert res["method"] == "joint_h_mc"
    assert set(res["params"]) == {"H", "eta", "rho"}
    assert np.asarray(res["iv_model"], dtype=float).shape == (
        len(res["t_grid"]),
        len(res["m_grid"]),
    )


def test_the_full_stage_freezes_xi0_and_refuses_to_constrain_it(stub_joint):
    """xi0 rides in `constraints["xi0_curve"]` as DATA and can never be a parameter."""
    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(
            stage=ROUGH_VOL_STAGE_FULL,
            constraints={"xi0": 0.04, "H": [0.05, 0.30]},
        )
    )
    assert res["success"] is True, res.get("message")
    constraints = stub_joint.calls[0]["constraints"]
    assert "xi0" not in constraints
    assert constraints["H"] == [0.05, 0.30]
    assert constraints["xi0_curve"] is not None
    # The curve handed to the calibrator is the very object the pipeline built.
    assert hasattr(constraints["xi0_curve"], "xi0")
    assert constraints["option_surface"] and constraints["clean_chains"]
    assert constraints["initial_params"] is not None


def test_the_full_stage_forwards_the_settings(stub_joint):
    CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=ROUGH_VOL_STAGE_FULL, max_nfev=200, n_starts=3, seed=7)
    )
    settings = stub_joint.calls[0]["settings"]
    assert settings.max_nfev == 200
    assert settings.n_starts == 3
    assert settings.seed == 7


# ---------------------------------------------------------------------------
# 5. cost estimate — the two Phase-4 notes
# ---------------------------------------------------------------------------
def test_max_nfev_80_is_reported_as_ambiguous():
    """`settings.max_nfev == 80` cannot be told apart from "not passed"."""
    from app.model.calibration.base_calibrator import CalibratorSettings

    ctrl = CalibrationController()
    default = int(CalibratorSettings().max_nfev)
    assert default == 80

    ambiguous = ctrl.rbergomi_joint_cost_estimate({"max_nfev": default})
    assert ambiguous["max_nfev_is_ambiguous"] is True
    assert ambiguous["max_nfev_requested"] == default
    assert ambiguous["max_nfev_effective"] == 165  # local_nfev_per_param(55) * 3 free
    assert ambiguous["max_nfev_source"] == "config"
    assert "indistinguable" in ambiguous["max_nfev_ambiguity_fr"]

    explicit = ctrl.rbergomi_joint_cost_estimate({"max_nfev": 120})
    assert explicit["max_nfev_is_ambiguous"] is False
    assert explicit["max_nfev_effective"] == 120
    assert explicit["max_nfev_source"] == "settings"

    omitted = ctrl.rbergomi_joint_cost_estimate({})
    assert omitted["max_nfev_effective"] == ambiguous["max_nfev_effective"]


def test_the_cost_estimate_exceeds_the_advanced_tab_heuristic():
    """`per_eval * max_nfev * n_starts` under-states this calibrator by ~2x.

    Oracle: the Stage-2 term is `n_starts_eff * max_nfev`; everything the
    calibrator additionally runs (Stage-1 design, profiles, valley, noise floor,
    grid-bias refinement, final repricing) is missing from it.
    """
    cost = CalibrationController().rbergomi_joint_cost_estimate({"n_starts": 1})
    assert cost["local_stage_only_evaluations"] == 165
    assert cost["ratio_vs_local_stage_only"] > 1.5
    assert cost["n_evaluations"] == sum(int(s["n_evaluations"]) for s in cost["stages"])
    assert cost["n_paths_total"] == sum(int(s["n_paths_total"]) for s in cost["stages"])
    assert str(cost["n_evaluations"]) in cost["message_fr"]
    # No wall time is invented.
    assert "dépend de la machine" in cost["wall_time_fr"]
    # ...and the FFT-calibrated `per_eval` constant is named as a second,
    # independent error on top of the counting error.
    assert "FFT" in cost["wall_time_fr"]


def test_the_heuristic_ratio_uses_the_numbers_the_tab_would_multiply():
    """`tab_advanced_calibration.py:499` multiplies the REQUESTED nfev/starts.

    Oracle: with the tab's own "Normal" profile the heuristic counts 80 * 2 = 160
    evaluations, while the calibrator ignores the ambiguous 80 and runs its own
    165-per-start budget plus every spec-4.11 diagnostic.
    """
    ctrl = CalibrationController()

    normal = ctrl.rbergomi_joint_cost_estimate({"max_nfev": 80, "n_starts": 2})
    assert normal["heuristic_evaluations"] == 160
    assert normal["max_nfev_effective"] == 165  # the 80 was ignored as ambiguous
    assert normal["ratio_vs_heuristic"] == pytest.approx(
        normal["n_evaluations"] / 160.0
    )
    assert normal["ratio_vs_heuristic"] > 1.5
    assert "160" in normal["message_fr"]

    # When the caller imposes a budget, requested == effective and the two
    # ratios coincide.
    imposed = ctrl.rbergomi_joint_cost_estimate({"max_nfev": 120, "n_starts": 1})
    assert imposed["heuristic_evaluations"] == 120
    assert imposed["local_stage_only_evaluations"] == 120
    assert imposed["ratio_vs_heuristic"] == pytest.approx(
        imposed["ratio_vs_local_stage_only"]
    )


def test_a_cheap_mc_config_shrinks_the_estimate():
    ctrl = CalibrationController()
    full = ctrl.rbergomi_joint_cost_estimate({})
    cheap = ctrl.rbergomi_joint_cost_estimate(
        {
            "mc_cfg": {
                "n_design": 0,
                "stage2_paths": 2_000,
                "profile_paths": 2_000,
                "final_paths": 4_000,
                "batch_paths": 4_000,
                "profile_points": 4,
                "valley_points": 4,
                "noise_replicates": 2,
                "refinement_check": False,
                "local_nfev_per_param": 20,
            }
        }
    )
    assert cheap["n_paths_total"] < full["n_paths_total"]
    assert cheap["max_nfev_effective"] == 60
    assert cheap["n_starts_effective"] == 1  # no Stage 1 => a single start


def test_pinning_a_parameter_lowers_the_free_count():
    cost = CalibrationController().rbergomi_joint_cost_estimate(
        {"constraints": {"rho": -0.7}}
    )
    assert cost["pinned_parameters"] == ["rho"]
    assert cost["n_free_parameters"] == 2
    assert cost["max_nfev_effective"] == 110  # 55 * 2


# ---------------------------------------------------------------------------
# 6. JSON safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", [ROUGH_VOL_STAGE_PREPARE, ROUGH_VOL_STAGE_FULL])
def test_every_pipeline_result_survives_json_dumps(stub_joint, stage):
    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=stage)
    )
    dumped = json.dumps(res, allow_nan=True)
    assert json.loads(dumped)["stage"] == stage
    assert _no_numpy(res)


def test_a_refusal_also_survives_json_dumps():
    res = CalibrationController().run_rbergomi_hurst_pipeline({})
    json.dumps(res, allow_nan=True)
    assert _no_numpy(res)


def test_the_cost_estimate_survives_json_dumps():
    cost = CalibrationController().rbergomi_joint_cost_estimate({})
    json.dumps(cost, allow_nan=True)
    assert _no_numpy(cost)


# ---------------------------------------------------------------------------
# 7. flag labels — the view must classify a verdict without importing the model
# ---------------------------------------------------------------------------
def test_the_controller_exposes_every_flag_label_and_marks_the_blocking_ones():
    from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
        BLOCKING_FLAGS,
        JOINT_CALIBRATION_LABELS_FR,
    )

    labels = CalibrationController().get_rough_vol_flag_labels()

    # Nothing the model can raise may reach the screen as a bare identifier.
    assert set(labels) == set(JOINT_CALIBRATION_LABELS_FR)
    for flag, entry in labels.items():
        assert entry["label_fr"] == JOINT_CALIBRATION_LABELS_FR[flag]
        assert entry["blocking"] is (flag in BLOCKING_FLAGS)
    assert {f for f, e in labels.items() if e["blocking"]} == set(BLOCKING_FLAGS)
    json.dumps(labels, allow_nan=False)


def test_a_false_verdict_pairs_each_flag_with_its_blocking_status(stub_joint):
    from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
        FLAG_H_WEAKLY_IDENTIFIED,
        FLAG_PARAMETER_AT_BOUND,
        JOINT_CALIBRATION_LABELS_FR,
    )

    stub_joint.success = False
    stub_joint.message = "Calibration en échec : H n'est pas identifié."
    # One blocking, one advisory -> the view must be able to tell them apart.
    stub_joint.flags = (FLAG_H_WEAKLY_IDENTIFIED, FLAG_PARAMETER_AT_BOUND)
    stub_joint.warnings_fr = tuple(JOINT_CALIBRATION_LABELS_FR[f] for f in stub_joint.flags)

    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=ROUGH_VOL_STAGE_FULL)
    )

    assert res["success"] is False
    assert res["params_usable"] is False
    assert res["blocking_flags"] == [FLAG_H_WEAKLY_IDENTIFIED]
    by_flag = {d["flag"]: d for d in res["flag_details"]}
    assert set(by_flag) == set(stub_joint.flags)
    assert by_flag[FLAG_H_WEAKLY_IDENTIFIED]["blocking"] is True
    assert by_flag[FLAG_PARAMETER_AT_BOUND]["blocking"] is False
    for flag, entry in by_flag.items():
        assert entry["label_fr"] == JOINT_CALIBRATION_LABELS_FR[flag]
    json.dumps(res, allow_nan=True)


@pytest.mark.parametrize("stage", [ROUGH_VOL_STAGE_PREPARE, ROUGH_VOL_STAGE_FULL])
def test_flag_details_are_always_present_even_when_empty(stub_joint, stage):
    """Uniform shape: the view never has to guess whether the keys exist."""
    res = CalibrationController().run_rbergomi_hurst_pipeline(
        _pipeline_payload(stage=stage)
    )
    assert res["flag_details"] == []
    assert res["blocking_flags"] == []

    refusal = CalibrationController().run_rbergomi_hurst_pipeline({})
    assert refusal["flag_details"] == []
    assert refusal["blocking_flags"] == []


# ---------------------------------------------------------------------------
# 8. the tab — registered, MVC-clean, and gated on the cost
# ---------------------------------------------------------------------------
def test_the_rough_vol_tab_is_registered_and_visible():
    """A `tab_*.py` is auto-discovered but INVISIBLE until listed in TAB_GROUPS."""
    from app.vue import main_app
    from app.vue.tabs import tab_rough_vol

    label = tab_rough_vol.TAB_LABEL
    assert main_app.DEFAULT_LABEL_OVERRIDES["tab_rough_vol"] == label
    grouped = [lbl for labels in main_app.TAB_GROUPS.values() for lbl in labels]
    assert label in grouped, grouped
    assert label not in main_app.EXCLUDED_LABELS
    assert callable(tab_rough_vol.render_tab)


def test_every_mc_profile_key_is_a_real_config_field():
    """`_config_from_mapping` SILENTLY DROPS unknown keys.

    A typo in the tab's cheap profile would therefore be invisible: the screen
    would quote a small budget and the calibrator would run the expensive
    defaults. Pin every key against the dataclass, and check the cheap profile
    really is cheaper end to end.
    """
    from app.model.volatility_models.rbergomi.calibrator_joint_mc import JointMCConfig
    from app.vue.tabs.tab_rough_vol import _MC_PROFILES

    fields = set(JointMCConfig().__dataclass_fields__)
    ctrl = CalibrationController()
    costs = {}
    for label, cfg in _MC_PROFILES.items():
        unknown = set(cfg) - fields
        assert not unknown, f"{label}: {sorted(unknown)} are not JointMCConfig fields"
        estimate = ctrl.rbergomi_joint_cost_estimate({"mc_cfg": cfg} if cfg else {})
        assert estimate["success"] is True, label
        costs[label] = estimate

    cheap = min(costs.values(), key=lambda c: c["n_paths_total"])
    dear = max(costs.values(), key=lambda c: c["n_paths_total"])
    assert cheap["n_paths_total"] < dear["n_paths_total"]
    assert cheap["grid_n_max"] < dear["grid_n_max"]


def test_the_rough_vol_tab_talks_only_to_controllers():
    """MVC gate, asserted at the import level too: no `app.model` / `app.utils`."""
    import ast
    import pathlib

    # cc.__file__ is app/controller/calibration_controller.py -> parents[1] is app/
    source = (
        pathlib.Path(cc.__file__).resolve().parents[1] / "vue" / "tabs" / "tab_rough_vol.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(m.startswith(("app.model", "app.utils")) for m in imported), imported
    assert "app.controller.calibration_controller" in imported


def test_the_expensive_fit_is_gated_behind_an_explicit_acknowledgement():
    """No casual click: the "full" stage is only reachable from a ticked checkbox.

    Read off the AST rather than the text, so a rename cannot quietly pass: the
    call that carries ``stage="full"`` must sit inside the ``if st.button(...)``
    whose ``disabled=`` argument reads the acknowledgement flag.
    """
    import ast
    import inspect
    import textwrap

    from app.vue.tabs import tab_rough_vol

    tree = ast.parse(textwrap.dedent(inspect.getsource(tab_rough_vol.render_tab)))

    # 1. every literal stage used by the tab is one the controller knows.
    literals = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value in (ROUGH_VOL_STAGE_PREPARE, ROUGH_VOL_STAGE_FULL)
    }
    assert literals == {ROUGH_VOL_STAGE_PREPARE, ROUGH_VOL_STAGE_FULL}

    # 2. the "full" assignment lives under an `if st.button(..., disabled=...)`.
    def _assigns_full(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Constant) and n.value == ROUGH_VOL_STAGE_FULL
            for n in ast.walk(node)
        )

    gated = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and any(kw.arg == "disabled" for kw in n.test.keywords)
        and any(_assigns_full(child) for child in n.body)
    ]
    assert gated, "the stage='full' run is not gated behind a disabled-able button"

    # 3. and the cheap stage is the one that runs outside any such gate.
    cheap = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value == ROUGH_VOL_STAGE_PREPARE
    ]
    assert cheap


def _no_numpy(obj) -> bool:
    """No numpy scalar or array survives anywhere in the payload."""
    if isinstance(obj, (np.generic, np.ndarray)):
        return False
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _no_numpy(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_no_numpy(v) for v in obj)
    return isinstance(obj, (str, int, float, bool, type(None)))
