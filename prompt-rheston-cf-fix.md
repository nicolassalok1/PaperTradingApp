# ultracode — Remise en service du rough Heston (CF markovienne + FFT) dans Calibration avancée (PaperTradingApp)

> **How to use this prompt:** paste it as a single message into Claude Code (CLI) with Opus 5 selected, from the root of the PaperTradingApp working copy (e.g. `claude --model opus` then paste, or `claude --model opus -p "$(cat prompt-rheston-cf-fix.md)"`). The literal keyword **ultracode** opts the session into multi-agent orchestration — the contract is §3. This is a SURGICAL repair task with a validated reference implementation already in the repo: the job is to port, wire, harden and test it — not to re-derive it, and not to redesign the model.

**ultracode** — orchestrate per §3.

---

## 0. Role and mission

You are a senior quantitative developer (stochastic volatility, rough volatility, Fourier pricing) working inside the existing **PaperTradingApp** repository (Streamlit paper-trading app, strict MVC, Python 3.11).

The app's rough Heston brick in the *Calibration avancée* tab — `RHestonFFTMarkovianModel` / `RHestonFFTMarkovianCalibrator`, registry key `"rheston"` — is currently **out of service**: the multi-factor Riccati ODE for the characteristic function is integrated with explicit Euler on a stiff system (factor rates up to x·dt ≈ 558 against a stability limit of 2), so the CF collapses to ≈0 (or blows up to ~1e304 at short maturities), the rendered IV surface is NaN + spurious 1–2% wing vols, the Hurst parameter has **zero** effect on the output, and the calibrator returns its own starting point after 1 evaluation with `success=True` and ~15 vol points of MAE. All of this is demonstrated, measured and root-caused in the in-repo review; a numerically validated replacement scheme ships with it.

Your mission, in order:

1. **Port the validated CF scheme** (fully implicit Riccati step + kernel constant factor + per-maturity step floor + health guard) from `scripts/rheston_cf_reference.py` into `app/model/volatility_models/rheston/cf_markovian.py`, preserving the public API.
2. **Make the calibrator honest**: vol-space vega-weighted objective, explicit penalties for missing model values (today a NaN residual counts as a *perfect fit*), evaluation-level rejection on degenerate CFs, meaningful `success` criteria, tightened ξ bounds, LHS multi-start, optional warm start, params-at-bounds reporting.
3. **Replace the dead tests** with oracle tests that would have caught the regression (closed-form Heston at H→0.5; fractional Adams at H≈0.07 — both oracles are given verbatim in §6).
4. **Light UI/controller wiring**: "Fixer H" (default ON, 0.0725), recalibrated ETA, config plumbing.

## 1. Source-of-truth artifacts (in the repo — read them FIRST)

- **`docs/review-2026-08-rough-heston-qrhplus.md`** — the full review: findings C1–C3 (blocking), M1–M6, m1–m6, each with measured numbers, file/line anchors, and the validation table of the fix. The acceptance thresholds in §5 below come from it.
- **`scripts/rheston_cf_reference.py`** — the VALIDATED reference implementation (kernel with constant factor `kernel_geom_with_const`, fully implicit CF `rheston_cf_implicit`, closed-form Heston oracle `heston_cf_closed_form`, self-validation `__main__`). Run `python scripts/rheston_cf_reference.py` in Phase 0: expected output ≈ `max|phi|=1.000000` on real u for all T; `max|phi_rh − phi_heston|` ∈ [3e-4, 1.3e-3] at H=0.499 for T ∈ {0.1, 0.25, 1.0}; H-sensitivity ≈ 0.087. If you get materially different numbers under the repo's pinned numpy/scipy, STOP and report before proceeding.
- Methodological background (why these choices): Bourgey–Noble–Petursson–Rosenbaum–Szymanski, *The Quadratic Rough Heston+ Model for Short-Dated Options* (SSRN 6072928, 2026) — the implicit-in-the-linear-part OU-factor update `U^j ← (U^j + √V δW)/(1+γ_j Δ)`, N=200 steps per maturity including ultra-short ones, vega-floored vol-space calibration objective. You do NOT need the paper to execute this task; the review encodes what matters.

## 2. Repository facts (verified 2026-08-21 on the working copy — re-verify in Phase 0 before coding)

**The brick.** `app/model/volatility_models/rheston/`: `markovian_kernel.py` (geometric quadrature of the Laplace representation — currently MISSING the [0, x_min) mass), `cf_markovian.py` (the broken integrator: explicit Euler line ~98; `x_max = x_max_mult·steps_per_year` line ~52 ⇒ 120 000 by default; `n_steps = max(1, round(steps_per_year·dt_total))` line ~84 ⇒ 2 steps for T=1/52; freeze-to-zero guard lines ~104–109; `Re(A)` clip at **+700** line ~113 which can emit φ≈e⁷⁰⁰), `model_fft.py` (surface builder, per-maturity CF calls), `calibrator_fft.py` (relative-price residuals with `scale=max(1e-4, px)` line ~139; **`res = np.where(np.isfinite(res), res, 0.0)` line ~249 — a missing model price contributes ZERO cost**; unconditional `success=True, "OK (rHeston approx, expensive)"` line ~332; `DEFAULT_BOUNDS` with `xi: (1e-3, 5.0)` lines ~58–65; hand-rolled multi-start loop lines ~252–278; `FFTConfig` re-parse with inconsistent default `n=1024` line ~157 vs 2048 in `common/fft.py`). `common/fft.py` = Carr-Madan (α=1.5, n=2048, η=0.25) — correct, DO NOT MODIFY.

**Wiring.** Registry key `"rheston"` already exists (no new key, no `desired_order` edit): spec entry in `CalibrationController.get_advanced_models()` (~line 837), instance in `calibrator_map` inside `run_advanced_surface_calibration()` (~line 972). The tab (`app/vue/tabs/tab_advanced_calibration.py`) fixes the profile for non-Heston models to Normal = `max_nfev=80, n_starts=2` (~lines 455–478); ETA heuristic `per_eval=0.05·5` for rheston (~lines 496–501); `constraints: Dict[str, Any] = {}` is created ONCE at ~line 132 and shared by every model tab in the loop — a per-model copy is required before adding model-specific keys (payload at ~line 521). Views import ONLY controllers (MVC gate).

**Shared calibration contract.** `app/model/calibration/base_calibrator.py`: frozen `SurfaceGrid`, `CalibratorSettings(fit_to_observed_only, max_nfev=80, n_starts=1, seed)`, `SurfaceCalibrationResult`, `apply_degeneracy_guard` (flips success only on ALL-non-finite surfaces — **do not change its shared semantics**; the stricter checks go inside the rheston calibrator). Constraint convention: scalar or `{"value": v}` pins a param, `[min,max]`/`{"min","max"}` tightens bounds, nested dicts (`fft_cfg`, `markovian_cfg`) override numeric knobs. Helpers that already exist and must be reused: `app/model/calibration/optimizers.py` → `latin_hypercube_samples(*, n, bounds: list[tuple[float,float]], rng) -> (n,d)` and `multi_start_least_squares(residuals, *, x0_candidates, bounds, max_nfev, method="trf") -> (best_run | None, summary_dict)` (summary carries `best_run`, `runs` — the tab already displays these keys); `loss_surface.py` → `compute_bs_vega_grid`, `effective_mask`, `iv_error_metrics`, `iv_error_metrics_weighted`; `implied_vol.py` → `bs_call_price`, `implied_vol_call` (Brent on [1e-4, 5], NaN outside no-arb bounds).

**Gates & conventions.** CI: `python scripts/check_mvc_integrity.py && python scripts/scan_secrets.py && python -m pytest -m "unit or smoke" --cov=app` (Python 3.11, 20-min budget, coverage floor `fail_under=12`). Pytest 8, `--strict-markers --disable-socket`; every test module carries one registered marker; heavy tests go in **slow-ONLY modules** (`pytestmark = pytest.mark.slow`). Pinned deps: numpy==1.26.4, scipy==1.11.4, pandas==2.1.4, numba==0.59.1; **no new dependencies, no torch on CI paths**. The reference script was validated under numpy 2.4.4/scipy 1.17.1 — it uses nothing version-sensitive, but Phase 1's tests must pass under the repo pins. French user-facing messages, English identifiers/docstrings, `np.random.default_rng(seed)`, explicit `__all__`. `tests/quant/test_rheston_calibration.py` currently passes in 0.75 s against the fully broken code (assertions `isfinite(...).any()`, tautological round-trip) — you are explicitly authorized to REWRITE that file.

**Do-not-touch list.** `app/model/volatility_models/{rbergomi,volterra,sabr,heston,jump_diffusion,kalman}/`, `app/model/calibration/rough_vol/`, `common/fft.py`, `base_calibrator.apply_degeneracy_guard`, all `tests/quant/test_rv_*.py` (active rBergomi-Hurst pipeline — parallel workstream), and every unrelated defect listed in the 360° audit (LSMC discounting, put delta, exotic pricers…): out of scope, do not "fix in passing". Do not rename `rheston_log_return_cf_markovian`, `RHestonMarkovianConfig`, `RHestonFFTMarkovianCalibrator`, `RHestonFFTMarkovianModel`, or the `"rheston"` registry key.

## 3. Orchestration contract (ultracode)

Run §7's phases as a **sequence, one workflow per phase**; you stay in the loop between phases. Inside a phase, fan out subagents only for genuinely independent files. Every phase ends with an adversarial verification fan-out before its gate:

1. **Math auditor** — prompted to REFUTE the implementation against the exact equations of §4 (hand it the formulas verbatim, not a summary).
2. **Test runner** — runs the phase's new tests plus the CI-equivalent fast gate, reports raw output.
3. **Numerical skeptic** — attacks with pathological inputs: H=0.02 and H=0.49; ξ at both bounds; ρ=±0.999; T=0.02 and T=2.0; u real, u with the Carr-Madan shift −i(α+1), u=0 (φ must equal exactly 1); a surface with 2 observed points; an all-NaN market row; κ pinned to its lower bound.

For the implicit-scheme port (the highest-risk component), use a **3-lens panel** (independent agents, majority must confirm): (a) formula-vs-spec re-derivation of the quadratic step and the Citardauq branch choice; (b) reproduction of the §5 oracle numbers from a clean run; (c) degenerate-input behavior (ξ→0 linear limit, dt→0 ⇒ S→P, u=0 ⇒ φ=1). Confirmed findings are fixed before the next phase. No silent scope cuts — anything skipped or capped goes in the final report.

## 4. Normative specification

### 4.1 Kernel (fixes M1)

`K(t)=t^{H−1/2}/Γ(H+½) ≈ w_∞ + Σᵢ wᵢ e^{−xᵢt}` — geometric quadrature of the Laplace representation on [x_min, x_max] **plus the constant factor** for the [0, x_min) mass:

```
power = 1/2 − H ;  c = 1/(Γ(H+1/2)·Γ(1/2−H))
edges = geomspace(x_min, x_max, n+1) ;  x_i = √(edge_i·edge_{i+1})
w_i = c·(edge_{i+1}^power − edge_i^power)/power ;   w_∞ = c·x_min^power/power  (rate 0)
```

Implement in `markovian_kernel.py` behind a new keyword `add_const: bool = True` on `fractional_kernel_markovian_approx` (default True; the constant factor is prepended as rate 0). Justification of w_∞ (document): e^{−xt} ≈ 1 on [0, x_min] since x_min·T ≤ x_min_mult = 0.1. Measured impact of omitting it: at H=0.499 the smile is flat at 20.0% vs a true 16.8–24.4% (439 bp).

### 4.2 CF integrator (fixes C1, M2 — the core change)

Replace the body of the maturity integration in `cf_markovian.py` by the **fully implicit scheme** of the reference. Per maturity T, with `rates`/`weights` from §4.1 (`x_min = x_min_mult/max(T,1e-6)`, `x_max = x_max_mult·steps_per_year` — unchanged semantics, = 120 000 by default), `n_steps = max(n_steps_min, round(steps_per_year·T))` with **`n_steps_min: int = 200` added to `RHestonMarkovianConfig`** (frozen dataclass, new field with default; also parsed from the `markovian_cfg` constraints dict in the calibrator), `dt = T/n_steps`:

```
iu = i·u ;  half = ½(u² + iu) ;  β = iuρξ − κ ;  γ = ξ²/2
denom_i = 1 + x_i·dt ;  W = Σ_i w_i/denom_i          (scalar)
per step:
  P  = Σ_i B_i/denom_i
  solve  dtWγ·S² + (dtWβ−1)·S + (P − dtW·half) = 0   for S = S_{n+1}:
     disc = sqrt(b1² − 4·a2·c0)  with a2=dtWγ, b1=dtWβ−1, c0=P−dtW·half
     qq   = −½·(b1 + disc·sign)  where sign = +1 if Re(conj(b1)·disc) ≥ 0 else −1
     S    = c0/qq                                     (Citardauq: stable root, S→P as dt→0)
  G  = −half + β·S + γ·S²
  A += dt·( iu(r−q) + κθ·½(S_prev+S) + v0·½(G_prev+G) )     (trapezoid)
  B_i ← (B_i + dt·w_i·G)/denom_i
  S_prev, G_prev ← S, G
```

Sanity anchors the math auditor must confirm: G is F(u,x) = −½(u²+iu) + (iuρξ−κ)x + ½ξ²x² evaluated at the multifactor h=ΣBᵢ; the A-terms implement κθ·I¹h + V₀·I¹F (with V₀I^{1−α}h = V₀I¹F, α=H+½ the fractional order); φ(0)=1 exactly; the ξ→0 degenerate quadratic reduces to the correct linear solve through the same Citardauq expression; unconditional stability (no x·dt restriction).

**Multi-maturity contract**: keep the public signature `rheston_log_return_cf_markovian(u, maturities, *, r, q, kappa, theta, xi, rho, v0, H, cfg) -> dict[float, ndarray]`, but integrate **each maturity independently from 0** with its own step count (the current chained integration is incompatible with the per-maturity step floor; no in-repo caller passes more than one maturity — verify in Phase 0 and document the change in the docstring).

### 4.3 Health guard (fixes M3, replaces the freeze/clip machinery)

Delete the in-loop freeze logic and the `Re(A) ≤ +700` clip. At maturity exit only: `bad = ~isfinite(A) | (A.real ≥ 50.0)` → φ[bad] = 0, and **count them**. Rationale (document): a legitimate damped-CF value has Re(A)=O(10); anything at e⁵⁰⁺ is a moment-explosion artefact, and zeroing must stay an *exceptional-tail* patch, never the main path. Internal core returns `(phi_map, health_map)` with `health_map[T] = n_zeroed`; the legacy public name stays a thin wrapper returning `phi_map` alone; `model_fft.py` and `calibrator_fft.py` call the core and surface the health numbers (`details["cf_health"]`, per-maturity, plus the max fraction).

### 4.4 Calibrator (fixes C2, C3, M4, M5, m2, m5)

- **Objective** (default, `constraints["objective"] = "iv_vega"`): residuals in **vol space**. Per observed point: `res_k = √w_k · (iv_model_k − iv_mkt_k)`, with iv_model inverted from the FFT price via `implied_vol_call`. Weights: `w_raw = clip(vega, f_floor·max_vega_of_expiry, ∞)` with `f_floor = 0.05` (vega floor so wings still contribute — reuse `compute_bs_vega_grid`), normalized to sum 1 **within each expiry**, expiries equally weighted, then globally normalized. Legacy behavior available as `"price_rel"` (same residuals as today EXCEPT the NaN rule below).
- **Penalties, not rewards** (C3): any point whose model price/IV is non-finite gets `res_k = PEN = 1.0` (one full vol point of penalty, ≫ any real residual). Evaluation-level rejection: if the zeroed-CF fraction (§4.3) exceeds 10% on any maturity, or fewer than 50% of masked points produced a finite model value, ALL residuals = 2·PEN for that evaluation. Never `where(isfinite, res, 0)` anywhere.
- **Multi-start** (M5): keep the improved heuristic x0 (fix m5: `iv_atm` read from the first row of the mask with an observed ATM value, not blindly row 0), accept `constraints["x0"]` (partial dict of params, clipped to bounds) as a warm start, and fill remaining starts with `latin_hypercube_samples` over the (possibly constraint-tightened) box. Replace the hand-rolled loop with `multi_start_least_squares` so `details` keeps the `runs`/`best_run` shape the tab already renders.
- **Success criteria** (C2): `success = (best run exists ∧ finite cost) ∧ (finite-model fraction on mask ≥ 0.8) ∧ (max zeroed-CF fraction ≤ 0.1) ∧ (metrics["rmse"] ≤ seuil)` with `seuil = constraints["success_rmse_max"]` (default 0.05). On failure, a French message stating WHICH criterion failed and its value (e.g. `"Calibration rHeston dégénérée: fraction de surface modèle finie 43% < 80%."`). Report `details["at_bounds"]` = list of params within 1e-6·(ub−lb) of a bound. Do NOT touch the shared `apply_degeneracy_guard`.
- **Bounds** (M3): `DEFAULT_BOUNDS["xi"] = (0.01, 1.5)` — measured basis: with Carr-Madan damping α=1.5 and H=0.07, the required 2.5-moment explodes for ξ ≥ 1.5 (17/121 non-finite CF columns at ξ=1.5, 89/121 at ξ=5). Other bounds unchanged. H stays free at the calibrator level; pinning is the UI's default (§4.5).
- **m2**: the `fft_cfg` constraints re-parse uses default `n=2048` (consistent with `FFTConfig`).

### 4.5 UI + controller (thin)

In the rheston tab branch only, working on a per-model copy `constraints_model = dict(constraints)`: a checkbox **"Fixer H (recommandé)"**, default True, with `st.number_input` default **0.0725** (= α = −0.4275 in the QRH+ paper's parametrization) → `constraints_model["H"] = {"value": h}`; pass `constraints_model` in the payload. Recalibrate the ETA heuristic for rheston: `per_eval ≈ 0.6 s` and multiply by `(1 + n_free_params)` for the finite-difference Jacobian, labelled "ordre de grandeur" (measured reality: ≈ 305 s for 50 nfev × 1 start × 4 maturities on the fixed scheme). Controller: no dispatch changes; verify `markovian_cfg["n_steps_min"]` flows through. Views import only controllers; French labels.

## 5. Acceptance thresholds (from the validated runs — these become test assertions)

| Check | Threshold |
|---|---|
| \|φ(u)\| on real u, all T ∈ {1/52, 0.05, 0.25, 1}, H ∈ {0.07, 0.3}, ξ=0.6 | ≤ 1 + 1e-9, zero zeroed columns |
| φ(0) | = 1 to 1e-12 |
| vs closed-form Heston (Albrecher, §6a), H=0.499, u ∈ [0,60], T ∈ {0.1, 0.25, 1} | max\|Δφ\| ≤ 2e-3 |
| same, smiles through the app's own FFT+implied-vol path, m ∈ [0.85, 1.15] | ≤ 10 bp |
| H-sensitivity: max\|φ(H=0.07) − φ(H=0.40)\|, T=0.25, u ∈ [0,30] | ≥ 0.01 |
| vs fractional Adams (§6b), H=0.07, ξ=0.3, T=1/52, smiles (damping 0.75 in the test) | ≤ 15 bp |
| same, CF at T=0.05 on the damped grid | max\|Δφ\| ≤ 2e-3 |
| Adams oracle self-check vs closed-form Heston at H=0.499, real u, T=0.25 | ≈ 3.3e-4 (assert ≤ 1e-3) |
| round-trip (slow): truth (H=0.10 pinned, κ=1.5, θ=0.05, ξ=0.35, ρ=−0.65, v0=0.045), offset start via `constraints["x0"]` | success=True, masked MAE ≤ 1.5e-2, cost strictly improved vs start, `at_bounds` empty |
| honesty (fast): CF monkeypatched to all-zeros / all-NaN | success=False with the French criterion message |
| timing (informative, report only) | ≈ 0.12 s/maturity CF eval (2048 u, 21 factors, 200 steps) |

## 6. Oracles (embed VERBATIM as inline test helpers — independent of the code under test)

**(a) Closed-form Heston (Albrecher et al., no little-trap)** — the rough Heston must converge onto it as H→½ (kernel → 1):

```python
def heston_cf_oracle(u, T, *, kappa, theta, xi, rho, v0, r, q):
    u = np.asarray(u, dtype=complex); iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d); e = np.exp(-d * T)
    C = (kappa * theta / (xi * xi)) * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((b - d) / (xi * xi)) * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * (r - q) * T + C + D * v0)
```

**(b) Fractional Adams predictor-corrector (El Euch–Rosenbaum) for the rough Riccati** — true rough-regime reference, itself validated against (a) at H=0.499 (3.3e-4):

```python
def rheston_cf_adams_oracle(u, T, *, r, q, kappa, theta, xi, rho, v0, H, n_steps=400):
    from scipy.special import gamma as G
    u = np.asarray(u, dtype=complex); alpha = H + 0.5; dt = T / n_steps
    iu = 1j * u; half = 0.5 * (u * u + iu)
    F = lambda h: -half + (iu * rho * xi - kappa) * h + 0.5 * xi * xi * h * h
    h = np.zeros((n_steps + 1, u.size), dtype=complex); Fv = np.zeros_like(h); Fv[0] = F(h[0])
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n_steps):
            j = np.arange(k + 1)
            b_w = (dt ** alpha / alpha) * ((k + 1 - j) ** alpha - (k - j) ** alpha)
            hP = (b_w[:, None] * Fv[:k + 1]).sum(axis=0) / G(alpha)
            c = np.empty(k + 1)
            if k > 0:
                jj = np.arange(1, k + 1)
                c[1:] = (k - jj + 2) ** (alpha + 1) + (k - jj) ** (alpha + 1) - 2.0 * (k - jj + 1) ** (alpha + 1)
            c[0] = k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha
            h[k + 1] = (dt ** alpha / (alpha * (alpha + 1.0)) * (c[:, None] * Fv[:k + 1]).sum(axis=0)
                        + (dt ** alpha / (alpha * (alpha + 1.0))) * F(hP)) / G(alpha)
            Fv[k + 1] = F(h[k + 1])
        I1h = dt * (h[0] * 0.5 + h[1:-1].sum(axis=0) + h[-1] * 0.5)
        I1F = dt * (Fv[0] * 0.5 + Fv[1:-1].sum(axis=0) + Fv[-1] * 0.5)
        A = iu * (r - q) * T + kappa * theta * I1h + v0 * I1F
        return np.where(np.isfinite(A) & (A.real < 700.0),
                        np.exp(np.where(np.isfinite(A) & (A.real < 700.0), A, 0.0)), np.nan)
```

O(N²·n_u): keep it in **slow** modules with modest grids (n_steps ≤ 500, FFT n=2048). Test files: `tests/quant/test_rheston_cf.py` (**unit** — everything in §5 involving oracle (a), \|φ\|≤1, φ(0)=1, H-sensitivity, determinism, plus the fast honesty tests with a small config `markovian_cfg={"n_steps_min": 50}`, `fft_cfg={"n": 512, "eta": 0.5}` to stay in CI budget) and a REWRITTEN `tests/quant/test_rheston_calibration.py` (**slow-only** — oracle (b) comparisons, the anti-tautological round-trip, objective-switch and warm-start behavior). Follow the house exemplar `tests/quant/test_mc_pricing.py`: inline oracles, docstrings explaining why each oracle is valid, fixed seeds.

## 7. Execution plan

**Phase 0 — Scout (inline, before any code).** Read the two §1 artifacts. Re-verify every §2 fact and line anchor against the current working copy (the repo may have moved — in particular check whether anything now calls `rheston_log_return_cf_markovian` with multiple maturities, and how `constraints` is built in the tab). Run `python scripts/rheston_cf_reference.py` and the three baseline gates; record the green baseline (`pytest -m "unit or smoke" -q` count included). Present a SHORT delta plan if anything conflicts with this spec — resolve conflicts per the STOP rule below, do not pick silently.

**Phase 1 — CF core.** §4.1 + §4.2 + §4.3 in `markovian_kernel.py` / `cf_markovian.py`; `model_fft.py` switched to the health-aware core; new `tests/quant/test_rheston_cf.py`. 3-lens panel on the scheme, then gate (MVC + secrets + fast tests, all green under the repo's pinned numpy/scipy).

**Phase 2 — Calibrator.** §4.4 in `calibrator_fft.py`; rewrite `tests/quant/test_rheston_calibration.py`; run the slow module locally and report raw output. Verification fan-out, then gate.

**Phase 3 — UI/controller.** §4.5; extend `tests/integration/test_advanced_calibration_tab.py` or `tests/quant/test_advanced_calibration_controller.py` (whichever the repo pattern favors — Phase 0 tells you) with a dispatch test asserting the pinned-H constraint reaches the calibrator. Gate.

**Phase 4 — Final.** Full run: fast gate + `pytest tests/quant/test_rheston_cf.py tests/quant/test_rheston_calibration.py -m "unit or smoke or slow"` + the untouched neighbors' tests (`test_heston_ls_calibrator.py`, `test_advanced_calibration_roundtrip.py`, `test_market_surface_grid.py`) to prove no collateral damage. Adversarial review panel over the whole diff. Final report: files created/modified; §5 table with MEASURED values side by side; raw test counts before/after; deviations from this spec (each with rationale); known limitations (e.g. per-maturity integration cost, moment-explosion behavior near ξ=1.5, damping α=1.5 kept); exact commands to reproduce.

**Guardrails.** If any §4 requirement conflicts with the repo's current state, STOP, state the conflict and your proposed resolution, get it on record in the final report — never pick silently. Fail closed, never invent numbers, never weaken a threshold to make a test pass (if a §5 threshold genuinely cannot be met, that is a finding to report, not to paper over). No behavior change for any other model key. Keep every gate green at every phase; never proceed on red.

---

*Context for the reviewer-agents: the §5 thresholds are not aspirational — they were all achieved by `scripts/rheston_cf_reference.py` during the review session of 2026-08-21 (see the review's annexe for the measurement scripts). The two most subtle spots, flagged for extra scrutiny: the Citardauq sign choice (`Re(conj(b1)·disc) ≥ 0` — picking the wrong root makes S→c0·qq-garbage instead of S→P as dt→0, which a dt-refinement test catches), and the per-expiry weight normalization in §4.4 (skipping it silently reintroduces the wing-domination defect M4 the objective change exists to remove).*
