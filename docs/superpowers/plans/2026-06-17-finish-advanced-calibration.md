# Finish Advanced Calibration Tabs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 5 gated Advanced Calibration sub-tabs (`sabr`, `merton_jump_diffusion`, `rheston`, `rbergomi`, `volterra`) actually usable and honest, and lock the behaviour with offline quant tests.

**Architecture:** The model calibrators and controller dispatch already exist and work for 4 of 5 models. The work is: (1) a generic anti-false-success guard in the controller so a degenerate (all-NaN) surface can never report `success=True`; (2) a targeted numerical fix to the rHeston Markovian-CF Riccati integration that currently overflows to an all-NaN surface; (3) an offline quant test suite (round-trip per model + guard unit tests + rHeston regression + one controller end-to-end); (4) removing the UI gate that hides the 5 tabs.

**Tech Stack:** Python 3.10 (conda env `papertrading`), numpy, scipy, pandas, Streamlit (view only), pytest 8 (markers enforced, `--disable-socket`).

## Global Constraints

- **Python/env:** run everything via `conda run -n papertrading python ...`. scipy/numpy/pandas are present.
- **Tests offline:** `--disable-socket` is active via `pyproject.toml addopts`. New tests must NOT touch the network (all inputs are synthetic numpy/pandas).
- **Every test MUST carry a registered marker** or collection hard-fails (`conftest.pytest_collection_modifyitems`). Registered markers: `unit`, `smoke`, `characterization`, `integration`, `slow`. Convention for this plan: pure finite-check / logic tests → `@pytest.mark.unit`; tests that run a real calibration → `@pytest.mark.slow` (the `slow` marker doc explicitly covers "calibration"); module-import guards → `@pytest.mark.smoke`.
- **No `quant` marker exists** — quant tests live in `tests/quant/` but are marked `unit`/`slow`. Mirror `tests/quant/test_heston_pricing.py`.
- **MVC:** the view (`app/vue/...`) talks to the controller only; never import `app.model.*` into the view. Tests may import `app.model.*` directly.
- **Surgical changes (Karpathy):** touch only what each task needs; match existing style; don't refactor unrelated code.
- **Commits:** conventional (`feat:`/`fix:`/`test:`/`docs:`). End each commit message body with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Branch:** work on `feat/finish-advanced-calibration` (already created).

### Key verbatim API facts (used across tasks)

- `SurfaceGrid(S0, r, q, m_grid, t_grid, iv_market, mask=None)` and `CalibratorSettings(fit_to_observed_only=True, max_nfev=80, n_starts=1, seed=None)` — both **frozen** dataclasses in `app/model/calibration/base_calibrator.py`. `SurfaceCalibrationResult(success, message, model, method, params, metrics=None, metrics_vw=None, iv_model=None, iv_error=None, vega_weights=None, details=None)` — **mutable** dataclass.
- `BaseSurfaceCalibrator.calibrate(self, surface, *, constraints=None, settings=None) -> SurfaceCalibrationResult`.
- `iv_error_metrics(...)` returns keys `{"mae","rmse","max_abs","n"}`.
- All 3 analytic/CF models expose `implied_vol_surface(self, surface: SurfaceGrid, params: Dict, **kwargs) -> np.ndarray` of shape `(len(t_grid), len(m_grid))`; placeholder `iv_market` in the input `SurfaceGrid` is ignored:
  - SABR `app/model/volatility_models/sabr/model.py` `SABRAnalyticModel(beta=0.5)` — params `{"alpha","rho","nu"}` (beta via constructor, NOT params).
  - Merton `app/model/volatility_models/jump_diffusion/model.py` `MertonJumpDiffusionModel()` — params `{"sigma","lam","muj","sigj"}`.
  - rHeston `app/model/volatility_models/rheston/model_fft.py` `RHestonFFTMarkovianModel()` — params `{"H","kappa","theta","xi","rho","v0"}` (vol-of-vol is **`xi`**).
- MC calibrators read a tiny config override only from `constraints["mc_cfg"]` (a dict). RBergomi keys: `n_design,n_surrogate_candidates,n_paths,n_steps`; bounds `{"H":(0.02,0.49),"eta":(0.05,5.0),"rho":(-0.999,0.999),"xi0":(1e-4,1.5)}`; param keys `H,eta,rho,xi0`. Volterra keys: `n_design,n_paths,n_steps,kernel_type,H,lam`; bounds `{"kappa":(0.0,10.0),"theta":(1e-4,1.5),"xi":(1e-3,5.0),"rho":(-0.999,0.999),"v0":(1e-4,1.5)}`; param keys `kappa,theta,xi,rho,v0`. Seed reproducibility via `CalibratorSettings(seed=...)`.
- Controller `run_advanced_surface_calibration` (`app/controller/calibration_controller.py`): `heston_v1` is special-cased and returns early (lines 916-938); the `calibrator_map` path produces `result = calibrator.calibrate(...)` (line 953) then `return self._json_safe({...})` (lines 957-978). `result` is the mutable `SurfaceCalibrationResult`.

---

## File Structure

- **Modify** `app/model/calibration/base_calibrator.py` — add pure helper `apply_degeneracy_guard(result)` (+ ensure `import numpy as np`). [Task 1]
- **Modify** `app/controller/calibration_controller.py` — import + call `apply_degeneracy_guard` in the `calibrator_map` path of `run_advanced_surface_calibration`. [Task 1]
- **Modify** `app/model/volatility_models/rheston/cf_markovian.py` — freeze diverging `u`-columns + cap `Re(A)` so the CF stays finite. [Task 2]
- **Modify** `app/vue/tabs/tab_advanced_calibration.py` — lift the gate to module-level `_IN_PROGRESS_MODELS = set()` (empty) + honest ETA caption for MC models. [Task 6]
- **Create** `tests/quant/test_calibration_guard.py` — unit tests for `apply_degeneracy_guard`. [Task 1]
- **Create** `tests/quant/test_rheston_calibration.py` — rHeston repro + round-trip. [Task 2]
- **Create** `tests/quant/test_advanced_calibration_roundtrip.py` — SABR/Merton round-trip [Task 3] + rBergomi/Volterra robustness [Task 4].
- **Create** `tests/quant/test_advanced_calibration_controller.py` — controller end-to-end + guard wiring. [Task 5]
- **Create** `tests/smoke/test_advanced_calibration_unlocked.py` — gate-empty import guard. [Task 6]

---

## Task 1: Anti-false-success guard (generic)

**Files:**
- Modify: `app/model/calibration/base_calibrator.py`
- Modify: `app/controller/calibration_controller.py:850-955`
- Test: `tests/quant/test_calibration_guard.py`

**Interfaces:**
- Produces: `apply_degeneracy_guard(result: SurfaceCalibrationResult) -> SurfaceCalibrationResult` (mutates `result.success`/`result.message` in place when the model surface is entirely non-finite or its metrics are non-finite; returns the same object).
- Consumes: `SurfaceCalibrationResult` (existing dataclass).

- [ ] **Step 1: Write the failing tests**

Create `tests/quant/test_calibration_guard.py`:

```python
"""Unit tests for the generic anti-false-success calibration guard.

A calibrator must never report success=True while returning a degenerate
(all-NaN) model surface or non-finite metrics. See the rHeston overflow bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import (
    SurfaceCalibrationResult,
    apply_degeneracy_guard,
)

pytestmark = pytest.mark.unit


def _result(*, success, iv_model, metrics):
    return SurfaceCalibrationResult(
        success=success,
        message="OK",
        model="test",
        method="test",
        params={},
        metrics=metrics,
        iv_model=iv_model,
    )


def test_guard_flips_all_nan_surface_to_failure():
    res = _result(
        success=True,
        iv_model=np.full((3, 5), np.nan),
        metrics={"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "n": 0.0},
    )
    out = apply_degeneracy_guard(res)
    assert out is res
    assert res.success is False
    assert "dégénér" in res.message.lower()


def test_guard_flips_non_finite_metrics_to_failure():
    iv = np.full((3, 5), np.nan)
    iv[0, 0] = 0.2  # one finite cell, but metrics are non-finite
    res = _result(
        success=True,
        iv_model=iv,
        metrics={"mae": float("inf"), "rmse": 0.1, "max_abs": 0.2, "n": 1.0},
    )
    apply_degeneracy_guard(res)
    assert res.success is False


def test_guard_keeps_partial_nan_success():
    # Legitimate SABR-style result: some maturities NaN, but finite metrics.
    iv = np.full((3, 5), np.nan)
    iv[1:, :] = 0.2
    res = _result(
        success=True,
        iv_model=iv,
        metrics={"mae": 0.001, "rmse": 0.002, "max_abs": 0.004, "n": 10.0},
    )
    apply_degeneracy_guard(res)
    assert res.success is True
    assert res.message == "OK"


def test_guard_is_noop_on_already_failed_result():
    res = _result(success=False, iv_model=None, metrics=None)
    apply_degeneracy_guard(res)
    assert res.success is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n papertrading python -m pytest tests/quant/test_calibration_guard.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_degeneracy_guard'`.

- [ ] **Step 3: Implement the guard**

In `app/model/calibration/base_calibrator.py`: ensure `import numpy as np` is present at the top (add it with the other imports if missing). Then append this function after the `SurfaceCalibrationResult` dataclass definition:

```python
def apply_degeneracy_guard(result: "SurfaceCalibrationResult") -> "SurfaceCalibrationResult":
    """Flip a falsely-successful calibration to an explicit failure.

    A model that returns success=True while producing an entirely non-finite
    (all-NaN/inf) IV surface, or non-finite error metrics, is degenerate: the UI
    would show blank heatmaps while claiming success. Detect that and set
    success=False with a clear message. Mutates and returns `result`.

    Does NOT flag a partially-NaN surface (e.g. SABR with unobserved maturities)
    as long as at least one model cell is finite and the metrics are finite.
    """
    if not result.success:
        return result

    iv = result.iv_model
    iv_arr = np.asarray(iv, dtype=float) if iv is not None else None
    all_nan = iv_arr is None or iv_arr.size == 0 or not np.isfinite(iv_arr).any()

    metrics = result.metrics or {}
    metric_vals = [
        float(v)
        for k, v in metrics.items()
        if k != "n" and isinstance(v, (int, float))
    ]
    metrics_bad = bool(metric_vals) and any(not np.isfinite(v) for v in metric_vals)

    if all_nan or metrics_bad:
        result.success = False
        result.message = f"Calibration dégénérée: surface modèle non finie (NaN). [{result.message}]"
    return result
```

Add `apply_degeneracy_guard` to the module's `__all__` if one exists.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n papertrading python -m pytest tests/quant/test_calibration_guard.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Wire the guard into the controller**

In `app/controller/calibration_controller.py`, in `run_advanced_surface_calibration`, extend the local import block (currently `from app.model.calibration.base_calibrator import CalibratorSettings, SurfaceGrid` near line 850) to also import the guard:

```python
        from app.model.calibration.base_calibrator import CalibratorSettings, SurfaceGrid, apply_degeneracy_guard
```

Then, between the `try/except` that produces `result` (ends ~line 955) and the `return self._json_safe(` (line 957), insert:

```python
        apply_degeneracy_guard(result)

```

- [ ] **Step 6: Verify controller still imports and guard tests pass**

Run: `conda run -n papertrading python -c "from app.controller.calibration_controller import CalibrationController; print('ok')"`
Expected: prints `ok`.
Run: `conda run -n papertrading python -m pytest tests/quant/test_calibration_guard.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/model/calibration/base_calibrator.py app/controller/calibration_controller.py tests/quant/test_calibration_guard.py
git commit -m "feat(calibration): generic anti-false-success guard

A calibration returning an all-NaN model surface or non-finite metrics can no
longer report success=True. Pure helper apply_degeneracy_guard in
base_calibrator, called in the controller's advanced-surface dispatch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: rHeston numerical fix (Riccati overflow)

**Files:**
- Modify: `app/model/volatility_models/rheston/cf_markovian.py:73-92`
- Test: `tests/quant/test_rheston_calibration.py`

**Interfaces:**
- Consumes: `rheston_log_return_cf_markovian(...)` (unchanged signature), `RHestonFFTMarkovianCalibrator`, `RHestonFFTMarkovianModel`, `SurfaceGrid`, `CalibratorSettings`.
- Produces: a finite (non-all-NaN) rHeston CF / IV surface on normal inputs.

**Root cause:** explicit-Euler integration of the Riccati ODE in `cf_markovian.py` (the `+ 0.5*xi*xi*(S*S)` term, line 84) diverges for large `|u|` (the FFT evaluates the CF over a wide `u`-grid). `B` → inf → `A` → inf → `np.exp(A)` (line 89) = inf/NaN, and the NaN poisons the whole FFT price row. Fix: freeze any `u`-column whose state goes non-finite (its true large-`u` CF tail → 0) and cap `Re(A)` before exponentiating.

- [ ] **Step 1: Write the failing tests**

Create `tests/quant/test_rheston_calibration.py`:

```python
"""rHeston FFT calibration: regression for the Riccati-overflow all-NaN bug,
plus a round-trip on a model-generated surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import SurfaceGrid, CalibratorSettings
from app.model.volatility_models.rheston.calibrator_fft import RHestonFFTMarkovianCalibrator
from app.model.volatility_models.rheston.model_fft import RHestonFFTMarkovianModel

pytestmark = pytest.mark.slow

_M = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
_T = np.array([0.25, 0.5, 1.0])


def _smooth_smile_surface():
    M, T = np.meshgrid(_M, _T)
    iv = 0.2 + 0.1 * (M - 1.0) ** 2 + 0.03 * (T - 0.25)
    return SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=iv, mask=np.isfinite(iv),
    )


def test_rheston_calibrator_does_not_return_all_nan_surface():
    # Mirrors the verified bug: a smooth smile produced an all-NaN model surface.
    surface = _smooth_smile_surface()
    res = RHestonFFTMarkovianCalibrator().calibrate(
        surface, settings=CalibratorSettings(max_nfev=40, n_starts=1, seed=0)
    )
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model).any(), "rHeston returned an all-NaN surface (overflow bug)"


def test_rheston_model_surface_is_finite():
    # The model's own IV surface must be finite for normal params.
    model = RHestonFFTMarkovianModel()
    src = SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )
    iv = np.asarray(
        model.implied_vol_surface(
            src, {"H": 0.1, "kappa": 2.0, "theta": 0.04, "xi": 0.6, "rho": -0.5, "v0": 0.04}
        ),
        dtype=float,
    )
    assert iv.shape == (len(_T), len(_M))
    assert np.isfinite(iv).sum() > 0, "rHeston model surface is entirely NaN (overflow bug)"


def test_rheston_roundtrip_finite_and_reasonable():
    model = RHestonFFTMarkovianModel()
    src = SurfaceGrid(
        S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )
    iv = np.asarray(
        model.implied_vol_surface(
            src, {"H": 0.1, "kappa": 2.0, "theta": 0.04, "xi": 0.6, "rho": -0.5, "v0": 0.04}
        ),
        dtype=float,
    )
    mask = np.isfinite(iv) & (iv > 0)
    assert mask.sum() > 0  # generation must not collapse to all-NaN
    market = SurfaceGrid(S0=100.0, r=0.02, q=0.0, m_grid=_M, t_grid=_T, iv_market=iv, mask=mask)
    res = RHestonFFTMarkovianCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=60, n_starts=1, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 5e-2  # loose: Markovian approx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n papertrading python -m pytest tests/quant/test_rheston_calibration.py -v`
Expected: FAIL — `test_rheston_calibrator_does_not_return_all_nan_surface` and `test_rheston_model_surface_is_finite` fail because the surface is entirely NaN (overflow). (`test_rheston_roundtrip_...` likely fails at `mask.sum() > 0`.)

- [ ] **Step 3: Implement the fix**

In `app/model/volatility_models/rheston/cf_markovian.py`, replace the maturity loop body (current lines 73-92) with the guarded version. Add a module-level constant near the top (after the imports):

```python
_RE_CAP = 700.0  # exp(700) is near float64 max; beyond this np.exp overflows
```

Then the loop becomes:

```python
    out: dict[float, np.ndarray] = {}
    t_prev = 0.0

    for T in maturities_sorted:
        dt_total = float(T) - float(t_prev)
        if dt_total <= 0:
            out[float(T)] = np.exp(A).astype(complex)
            continue

        n_steps = max(1, int(round(float(cfg.steps_per_year) * dt_total)))
        dt = dt_total / n_steps

        for _ in range(n_steps):
            S = np.sum(B, axis=0)  # shape (n_u,)
            G = (-half_u2_iu) + (-kappa * S) + (iu * rho * xi * S) + (0.5 * xi * xi) * (S * S)

            B = B + dt * ((-rates[:, None]) * B + (weights[:, None]) * G)
            A = A + dt * (iu * (r - q) + kappa * theta * S + v0 * G)

            # Freeze any u-column whose Riccati state diverged. The true CF tail
            # for large |u| decays to 0, so a blown-up column is set to exp(A)->0
            # instead of letting inf/NaN poison the whole FFT price row.
            bad = ~(np.isfinite(B).all(axis=0) & np.isfinite(A))
            if bad.any():
                B[:, bad] = 0.0
                A[bad] = complex(-_RE_CAP, 0.0)

        # Cap Re(A) to avoid overflow on borderline-huge finite values, then
        # zero out any residual non-finite entry.
        a_re = np.clip(A.real, -np.inf, _RE_CAP)
        phi = np.exp(a_re + 1j * A.imag)
        phi = np.where(np.isfinite(phi), phi, 0.0 + 0.0j)
        out[float(T)] = phi.reshape(u.shape).astype(complex)
        t_prev = float(T)

    return out
```

(The `A`/`B` initialisation above the loop, lines 67-71, is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n papertrading python -m pytest tests/quant/test_rheston_calibration.py -v`
Expected: PASS (3 passed).

> If `test_rheston_roundtrip_finite_and_reasonable` passes the finiteness asserts but fails `mae < 5e-2`, raise `RHestonMarkovianConfig.steps_per_year` (e.g. 120 → 240) for finer, more stable integration and re-run. If it still cannot meet a loose tol, escalate to a semi-implicit/RK4 Riccati step per the spec's escalation clause. The finiteness asserts (the actual bug) must pass regardless.

- [ ] **Step 5: Commit**

```bash
git add app/model/volatility_models/rheston/cf_markovian.py tests/quant/test_rheston_calibration.py
git commit -m "fix(rheston): guard Riccati integration against overflow

Explicit-Euler integration of the Markovian-approx CF diverged for large |u|,
producing an all-NaN surface (silently reported as success). Freeze diverging
u-columns to the correct vanishing tail and cap Re(A) before exp, so the CF
stays finite. Regression + round-trip tests added.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: SABR + Merton round-trip tests

**Files:**
- Test: `tests/quant/test_advanced_calibration_roundtrip.py`

**Interfaces:**
- Consumes: `SABRAnalyticModel`, `SABRAnalyticCalibrator`, `MertonJumpDiffusionModel`, `MertonJumpDiffusionCalibrator`, `SurfaceGrid`, `CalibratorSettings`.

- [ ] **Step 1: Write the tests**

Create `tests/quant/test_advanced_calibration_roundtrip.py`:

```python
"""Round-trip / robustness tests for the advanced surface calibrators.

For analytic models (SABR, Merton): generate an IV surface from the model with
known params, calibrate, and assert the fit recovers it within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.model.calibration.base_calibrator import SurfaceGrid, CalibratorSettings

pytestmark = pytest.mark.slow

_M = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
_T = np.array([0.25, 0.5, 1.0])


def _placeholder_surface(S0=100.0, r=0.02, q=0.0):
    return SurfaceGrid(
        S0=S0, r=r, q=q, m_grid=_M, t_grid=_T,
        iv_market=np.full((len(_T), len(_M)), np.nan), mask=None,
    )


def _market_from_model(model, params, S0=100.0, r=0.02, q=0.0):
    iv = np.asarray(model.implied_vol_surface(_placeholder_surface(S0, r, q), params), dtype=float)
    mask = np.isfinite(iv) & (iv > 0)
    market = SurfaceGrid(S0=S0, r=r, q=q, m_grid=_M, t_grid=_T, iv_market=iv, mask=mask)
    return market, mask


def test_sabr_roundtrip_recovers_surface():
    from app.model.volatility_models.sabr.model import SABRAnalyticModel
    from app.model.volatility_models.sabr.calibrator import SABRAnalyticCalibrator

    model = SABRAnalyticModel(beta=0.5)
    market, mask = _market_from_model(model, {"alpha": 0.3, "rho": -0.4, "nu": 0.6})
    assert mask.sum() > 0
    res = SABRAnalyticCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=200, n_starts=2, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 1e-2  # analytic per-maturity fit -> near-exact


def test_merton_roundtrip_recovers_surface():
    from app.model.volatility_models.jump_diffusion.model import MertonJumpDiffusionModel
    from app.model.volatility_models.jump_diffusion.calibrator import MertonJumpDiffusionCalibrator

    model = MertonJumpDiffusionModel()
    market, mask = _market_from_model(model, {"sigma": 0.2, "lam": 0.5, "muj": -0.1, "sigj": 0.3})
    assert mask.sum() > 0
    res = MertonJumpDiffusionCalibrator().calibrate(
        market, settings=CalibratorSettings(max_nfev=200, n_starts=3, seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert np.isfinite(iv_model[mask]).all()
    assert res.metrics["mae"] < 3e-2  # global 4-param fit + FFT/IV discretisation
```

- [ ] **Step 2: Run the tests**

Run: `conda run -n papertrading python -m pytest tests/quant/test_advanced_calibration_roundtrip.py -v`
Expected: PASS (2 passed). If a `mae` tol is marginally exceeded due to FFT discretisation, loosen the Merton tol to `5e-2` (analytic SABR must stay tight).

- [ ] **Step 3: Commit**

```bash
git add tests/quant/test_advanced_calibration_roundtrip.py
git commit -m "test(calibration): SABR + Merton round-trip recovery

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: rBergomi + Volterra robustness tests

**Files:**
- Modify: `tests/quant/test_advanced_calibration_roundtrip.py` (append)

**Interfaces:**
- Consumes: `RBergomiMCSurrogateCalibrator`, `VolterraSDECalibrator`, `SurfaceGrid`, `CalibratorSettings`. MC config via `constraints["mc_cfg"]`.

For MC models, exact parameter recovery is noisy with a tiny config; assert robustness instead: success, finite metrics, correct 2D shape, params within bounds.

- [ ] **Step 1: Append the tests**

Append to `tests/quant/test_advanced_calibration_roundtrip.py`:

```python
def _smooth_smile_market(S0=100.0, r=0.02, q=0.0):
    M, T = np.meshgrid(_M, _T)
    iv = 0.2 + 0.1 * (M - 1.0) ** 2 + 0.03 * (T - 0.25)
    return SurfaceGrid(S0=S0, r=r, q=q, m_grid=_M, t_grid=_T, iv_market=iv, mask=np.isfinite(iv))


def test_rbergomi_calibration_is_robust():
    from app.model.volatility_models.rbergomi.calibrator_mc_surrogate import (
        RBergomiMCSurrogateCalibrator,
    )

    market = _smooth_smile_market()
    cons = {"mc_cfg": {"n_paths": 300, "n_steps": 12, "n_design": 4, "n_surrogate_candidates": 16}}
    res = RBergomiMCSurrogateCalibrator().calibrate(
        market, constraints=cons, settings=CalibratorSettings(seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert iv_model.shape == (len(_T), len(_M))
    assert np.isfinite(iv_model).any()
    assert all(np.isfinite(float(v)) for v in res.metrics.values())
    bounds = {"H": (0.02, 0.49), "eta": (0.05, 5.0), "rho": (-0.999, 0.999), "xi0": (1e-4, 1.5)}
    for k, (lo, hi) in bounds.items():
        assert lo <= float(res.params[k]) <= hi


def test_volterra_calibration_is_robust():
    from app.model.volatility_models.volterra.calibrator_mc import VolterraSDECalibrator

    market = _smooth_smile_market()
    cons = {"mc_cfg": {"n_paths": 300, "n_steps": 12, "n_design": 4}}
    res = VolterraSDECalibrator().calibrate(
        market, constraints=cons, settings=CalibratorSettings(seed=0)
    )
    assert res.success is True
    iv_model = np.asarray(res.iv_model, dtype=float)
    assert iv_model.shape == (len(_T), len(_M))
    assert np.isfinite(iv_model).any()
    assert all(np.isfinite(float(v)) for v in res.metrics.values())
    bounds = {"kappa": (0.0, 10.0), "theta": (1e-4, 1.5), "xi": (1e-3, 5.0),
              "rho": (-0.999, 0.999), "v0": (1e-4, 1.5)}
    for k, (lo, hi) in bounds.items():
        assert lo <= float(res.params[k]) <= hi
```

- [ ] **Step 2: Run the tests**

Run: `conda run -n papertrading python -m pytest tests/quant/test_advanced_calibration_roundtrip.py -v`
Expected: PASS (4 passed total). If RBFInterpolator (scipy) is unavailable the rBergomi calibrator returns `success=False` — scipy is present in `papertrading`, so this must pass.

- [ ] **Step 3: Commit**

```bash
git add tests/quant/test_advanced_calibration_roundtrip.py
git commit -m "test(calibration): rBergomi + Volterra MC robustness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Controller end-to-end + guard wiring test

**Files:**
- Test: `tests/quant/test_advanced_calibration_controller.py`

**Interfaces:**
- Consumes: `CalibrationController.run_advanced_surface_calibration(payload)` — payload `{"model","df","r","q","S0","fit_to_observed_only","max_nfev","n_starts","seed","constraints"}`; returns a JSON-safe dict with keys `success, message, model, method, params, metrics, metrics_vw, S0, r, q, m_grid, t_grid, iv_market, iv_model, iv_error, vega_weights, mask, details`.

- [ ] **Step 1: Write the test**

Create `tests/quant/test_advanced_calibration_controller.py`:

```python
"""End-to-end test of the advanced-calibration controller path, including the
anti-false-success guard wiring. Uses SABR (fast, analytic) through the full
DataFrame -> dispatch -> result-dict pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.controller.calibration_controller import CalibrationController

pytestmark = pytest.mark.slow


def _call_surface_df():
    rows = []
    for T in (0.25, 0.5, 1.0):
        for mny in (0.9, 0.95, 1.0, 1.05, 1.1):
            iv = 0.2 + 0.1 * (mny - 1.0) ** 2 + 0.03 * (T - 0.25)
            rows.append({"K": 100.0 * mny, "T": T, "S0": 100.0, "iv": iv, "type": "call"})
    return pd.DataFrame(rows)


def test_controller_sabr_end_to_end():
    res = CalibrationController().run_advanced_surface_calibration(
        {
            "model": "sabr",
            "df": _call_surface_df(),
            "r": 0.02,
            "q": 0.0,
            "S0": 100.0,
            "fit_to_observed_only": True,
            "max_nfev": 80,
            "n_starts": 1,
            "seed": 0,
            "constraints": {},
        }
    )
    assert res["success"] is True
    assert res["model"] == "sabr"
    iv_model = np.asarray(res["iv_model"], dtype=float)
    assert iv_model.ndim == 2
    assert np.isfinite(iv_model).any()
    assert all(np.isfinite(float(v)) for v in res["metrics"].values())
```

- [ ] **Step 2: Run the test**

Run: `conda run -n papertrading python -m pytest tests/quant/test_advanced_calibration_controller.py -v`
Expected: PASS (1 passed). The controller re-grids onto the default 6×9 grid; the shortest maturities may be NaN (no data) but the surface is not all-NaN, so the guard does not flip success.

- [ ] **Step 3: Commit**

```bash
git add tests/quant/test_advanced_calibration_controller.py
git commit -m "test(calibration): controller advanced-surface end-to-end (SABR)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Unlock the UI tabs

**Files:**
- Modify: `app/vue/tabs/tab_advanced_calibration.py` (gate + ETA caption)
- Test: `tests/smoke/test_advanced_calibration_unlocked.py`

**Interfaces:**
- Produces: module-level `_IN_PROGRESS_MODELS: set[str]` (empty) in `tab_advanced_calibration.py`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/smoke/test_advanced_calibration_unlocked.py`:

```python
"""Smoke guard: no advanced-calibration model is gated behind the
'en cours d'implémentation' placeholder.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_no_advanced_calibration_models_are_gated():
    from app.vue.tabs import tab_advanced_calibration as t

    assert hasattr(t, "_IN_PROGRESS_MODELS")
    assert t._IN_PROGRESS_MODELS == set()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n papertrading python -m pytest tests/smoke/test_advanced_calibration_unlocked.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_IN_PROGRESS_MODELS'`.

- [ ] **Step 3: Lift the gate to a module constant (empty)**

In `app/vue/tabs/tab_advanced_calibration.py`, add a module-level constant after `TAB_LABEL = "🧪 Calibration avancée"` (line 29):

```python
# Models still hidden behind the "en cours d'implémentation" placeholder.
# All advanced models are functional + tested, so this is now empty.
_IN_PROGRESS_MODELS: set[str] = set()
```

Then change the in-function definition (line 401) from:

```python
    in_progress_models = {"rheston", "rbergomi", "volterra", "merton_jump_diffusion", "sabr"}
```

to:

```python
    in_progress_models = _IN_PROGRESS_MODELS
```

(Leave the `if model_key in in_progress_models:` block at lines 408-410 as-is — harmless and future-proof now that the set is empty.)

- [ ] **Step 4: Make the ETA caption honest for MC models**

In `app/vue/tabs/tab_advanced_calibration.py`, replace the ETA caption block (current lines 466-471):

```python
            per_eval = 0.05
            if model_key in {"rheston", "rbergomi", "volterra"}:
                per_eval *= 5.0
            eta_seconds = per_eval * float(max_nfev) * float(max(1, n_starts))
            eta_label = _eta_human(eta_seconds)
            st.caption(f"ETA estimée: ~{eta_label} (heuristique; dépend du modèle et des données).")
```

with:

```python
            if model_key in {"rbergomi", "volterra"}:
                st.caption(
                    "ETA: pilotée par la config Monte-Carlo interne (n_paths/n_design), "
                    "indépendante des réglages nfev/starts. Compter quelques secondes."
                )
            else:
                per_eval = 0.05
                if model_key == "rheston":
                    per_eval *= 5.0
                eta_seconds = per_eval * float(max_nfev) * float(max(1, n_starts))
                eta_label = _eta_human(eta_seconds)
                st.caption(f"ETA estimée: ~{eta_label} (heuristique; dépend du modèle et des données).")
```

- [ ] **Step 5: Run the smoke test + the existing offline import smoke test**

Run: `conda run -n papertrading python -m pytest tests/smoke/test_advanced_calibration_unlocked.py tests/smoke/test_offline_imports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/vue/tabs/tab_advanced_calibration.py tests/smoke/test_advanced_calibration_unlocked.py
git commit -m "feat(calibration): unlock the 5 advanced calibration tabs

All advanced models (SABR, Merton, rHeston, rBergomi, Volterra) are functional
and tested; remove the in-progress UI gate and make the ETA caption honest for
MC-driven models.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] **Run the full new quant + smoke suite (incl. slow):**

Run: `conda run -n papertrading python -m pytest tests/quant/ tests/smoke/ -v -m "unit or smoke or slow"`
Expected: all green (existing + new).

- [ ] **Run the CI-default selection to confirm nothing unmarked / no regression:**

Run: `conda run -n papertrading python -m pytest -m "unit or smoke" -q`
Expected: green, no `UsageError` about unmarked tests.

- [ ] **Manual sanity (optional):** launch the app and confirm each of the 5 tabs renders a calibration form, calibrates, and shows finite metrics + non-empty heatmaps. rHeston must NOT show a success with blank heatmaps.

---

## Self-Review

**Spec coverage:**
- Spec §1 Déverrouillage UI → Task 6 (gate empty + honest ETA). ✓
- Spec §2 Fix rHeston (reproduce → targeted guard → finer-steps escalation → finite + non-NaN) → Task 2. ✓
- Spec §3 Anti-false-success guard (centralised, "entirely NaN OR non-finite metrics", must not flag partial-NaN SABR) → Task 1. ✓
- Spec §4 Quant tests (per-model round-trip, anti-false-success unit, rHeston regression, offline, mirror test_heston_pricing) → Tasks 1–5. ✓
- Spec "Out of scope" (heston_v1 LS bug, perf, variance reduction) → not touched. ✓
- Spec success criteria (5 tabs calibrate + finite metrics + non-empty heatmaps; pytest green; rHeston never success-with-NaN) → Final verification + Tasks 1/2/6. ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases". The two conditional notes (rHeston steps_per_year escalation; Merton tol loosening) are concrete fallbacks with exact values and a verifiable gate, not placeholders.

**Type consistency:** `apply_degeneracy_guard(result) -> result` used identically in Task 1 def, controller call, and tests. `SurfaceGrid`/`CalibratorSettings`/`SurfaceCalibrationResult` field names match the verbatim dataclass defs. `implied_vol_surface(surface, params)` param keys match per model (SABR alpha/rho/nu; Merton sigma/lam/muj/sigj; rHeston H/kappa/theta/xi/rho/v0). MC `constraints["mc_cfg"]` keys and bounds match the verbatim configs. `metrics` keys (mae/rmse/max_abs/n) consistent across guard + tests.
