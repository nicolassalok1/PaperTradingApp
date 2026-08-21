# The rough-volatility Hurst pipeline

**What this document is.** A complete, auditable description of the pipeline that turns a raw
listed option chain into a calibrated rough Bergomi triple `(H, eta, rho)` on a **frozen**
forward-variance curve `xi0`. It is written for someone who has to *trust or challenge* the number
`H` that comes out the other end: every stage is stated as mathematics first, then as the
implementation decision that was actually taken, then as the measurement that justifies that
decision.

**What this document is not.** It is not a tutorial on rough volatility, and it is not marketing.
Wherever a number appears it is either derived in closed form here, or it was **measured** in this
repository and the measurement is quoted with its conditions. Where something is *not* measured,
this document says so in those words. There are no market numbers in this file: every figure below
is either analytic or comes from a synthetic reference surface or from a simulator experiment.

**Day count.** Calendar days divided by 365, everywhere, with no exception. `T = 7/365` is
`0.0191781…` years. See [§14.6](#146-day-count).

---

## Table of contents

- [§0 Notation and conventions](#0-notation-and-conventions)
- [§1 The pipeline end to end](#1-the-pipeline-end-to-end)
- [§2 Stage 0 — quote cleaning](#2-stage-0--quote-cleaning-spec-41)
- [§3 Stage 1 — forward, discount and the OTM surface](#3-stage-1--forward-discount-and-the-otm-surface-spec-42)
- [§4 Stage 2 — the log contract and `K_var`](#4-stage-2--the-log-contract-and-k_var-spec-43)
- [§5 Stage 3 — forward variance and the isotonic repair](#5-stage-3--forward-variance-and-the-isotonic-repair-spec-44)
- [§6 Stage 4 — short-maturity skew and `H`](#6-stage-4--short-maturity-skew-and-h-spec-45)
- [§7 The fractional machinery — Davies–Harte fBm vs the RL Volterra driver](#7-the-fractional-machinery--daviesharte-fbm-vs-the-rl-volterra-driver-spec-47)
- [§8 rBergomi dynamics and the correlation through the shared driver](#8-rbergomi-dynamics-and-the-correlation-through-the-shared-driver-spec-4648)
- [§9 The pricing layer](#9-the-pricing-layer-spec-48)
- [§10 Initialisation and `c(H)`](#10-initialisation-and-ch-spec-49)
- [§11 The joint calibration — objective and weights](#11-the-joint-calibration--objective-and-weights-spec-410)
- [§12 Parameter identifiability](#12-parameter-identifiability-spec-411)
- [§13 Numerical issues](#13-numerical-issues)
- [§14 The deliberate deviations](#14-the-deliberate-deviations)
- [§15 Limitations — the honest list](#15-limitations--the-honest-list)
- [§16 How to challenge this number](#16-how-to-challenge-this-number)
- [§17 References](#17-references)

---

## §0 Notation and conventions

| Symbol | Meaning |
|---|---|
| $S_0$ | spot of the underlying at the observation time |
| $T$ | maturity in years, **calendar days / 365** |
| $D(T)$ | discount factor from the repo yield curve; $r(T) = -\ln D(T)/T$ **always**, so $e^{-rT}=D$ to machine precision |
| $F(T)$ | forward, from put–call parity on cleaned mids (never $S_0e^{rT}$) |
| $k$ | log-forward-moneyness $\ln(K/F(T))$ — the *only* moneyness the pipeline regresses in |
| $\sigma(k,T)$ | Black-76 implied volatility, **recomputed** from cleaned mids; vendor IV is never an input |
| $\psi(T)$ | ATM skew $\partial_k\sigma|_{k=0}$ |
| $K_{\mathrm{var}}(T)$ | variance-swap strike (fair variance rate) at $T$ |
| $V(0,T)$ | total variance $T\,K_{\mathrm{var}}(T)$ |
| $\xi_0(t)$ | forward variance curve, $\mathbb E[V_t]$ under the model |
| $\widetilde W$ | Riemann–Liouville Volterra driver of rBergomi |
| $B, B^\perp$ | the driving Brownian motion and its independent complement |
| $H,\eta,\rho$ | Hurst index, vol-of-vol, spot/vol correlation |

Parameter box (hard, spec 4.6/4.10): $0.01 \le H \le 0.49$, $0.05 \le \eta \le 5.0$,
$|\rho| \le 0.999$.

Recovery tolerances the slow suite judges a calibration against, and against which every
"is this material?" verdict in the pipeline is expressed:
$\mathrm{TOL}_H = 0.05$, $\mathrm{TOL}_\eta = 0.35$, $\mathrm{TOL}_\rho = 0.12$
(`PARAM_SCALE`). Deliberately **not** the bound-to-bound width: the box is $0.48$ wide in $H$ and
$4.95$ wide in $\eta$, so a threshold defined against the width would only ever complain about
errors already larger than the tolerance the result is accepted on.

---

## §1 The pipeline end to end

The pipeline is a chain of five measurements, each of which is a *different kind* of statement
about the market. They are separated on purpose: the separation is what makes the last one
identifiable.

| Stage | Question it answers | Nature of the answer | Module |
|---|---|---|---|
| 0 | which quotes are information? | filter + audit log | `rough_vol/chain_cleaning.py` |
| 1 | what is the forward and the discount? | model-free, parity | `rough_vol/forward_curve.py` |
| 2 | what is the market's total variance to $T$? | model-free, static replication | `rough_vol/variance_swap.py` |
| 3 | what is $\xi_0(t)$? | bootstrap + isotonic repair | `rough_vol/forward_variance.py` |
| 4 | how rough is it? (initial) | short-maturity **asymptotic** | `rough_vol/hurst_estimator.py` |
| 5 | $(H,\eta,\rho)$ jointly, $\xi_0$ frozen | Monte-Carlo fit + identifiability verdict | `rbergomi/calibrator_joint_mc.py` |

Supporting numerical layers: `rbergomi/volterra_gaussian.py` (exact joint Gaussian construction and
grid policy), `rbergomi/simulator_xi_curve.py` (log-Euler spot on the $\xi_0$ curve),
`rbergomi/pricing.py` (estimators, IVs), `rbergomi/initializer.py` (the seed),
`rbergomi/fbm.py` (Davies–Harte standard fBm, **diagnostics only**).

The whole chain, in one line:

$$
\text{quotes} \;\longrightarrow\; \{F(T),D(T),\sigma(k,T)\}
\;\longrightarrow\; K_{\mathrm{var}}(T) \;\longrightarrow\; \xi_0(t)
\;\longrightarrow\; H_0 \text{ (asymptotic)} \;\longrightarrow\; (H,\eta,\rho)\ \text{with } \xi_0 \text{ frozen.}
$$

Two structural rules govern the whole chain and are non-negotiable:

1. **$\xi_0$ is data.** It is measured in Stage 2/3 by a model-free replication, and the Stage-5
   optimiser is *structurally* unable to modify it — it is not a component of $\theta$, not a key
   of the bounds dictionary, and not reachable from the objective's argument. It rides in a
   `FrozenXi0` holder that records a SHA-256 content fingerprint at construction and re-verifies it
   after the optimisation; the *same object* is handed back so a caller can assert identity, not
   merely equality. Passing `"xi0"` through the constraints protocol is rejected with an explicit
   French error, never silently ignored.
2. **A refusal is never a number.** Every stage that cannot honestly produce a value returns a
   typed failure with a French message instead of a plausible-looking float.

---

## §2 Stage 0 — quote cleaning (spec 4.1)

Input: the raw row schema of `market_data.fetch_options_details_yahoo`. Output: one `CleanChain`
per expiry, carrying the surviving quotes, an **exhaustive removal log** (one machine-readable
reason code per dropped contract, with a French label), and a `ViabilityReport` saying whether the
expiry can support a $K_{\mathrm{var}}$ and/or a skew.

Filters applied, in order:

1. **Structural** — bad type, non-positive strike, non-positive maturity.
2. **Quote validity** — missing bid/ask, negative bid, non-positive ask, crossed quote
   (`ask < bid`), non-positive mid.
3. **Staleness** — `volume == 0` and `openInterest == 0`
   ("*Contrat dormant (volume et open interest nuls)*").
4. **Duplicate strikes** — Yahoo occasionally returns two contracts on the same `(type, strike)`;
   duplicates break the strike-ordered arbitrage scans (zero-width butterflies), so the
   tighter-spread quote is kept and the other is logged.
5. **Relative spread ceiling** — $(\text{ask}-\text{bid})/\text{mid}$ above `s_max_otm = 0.5`
   (`s_max_atm = 0.25` inside the ATM band $|k| \le 0.05$).
6. **The CBOE zero-bid wall** — see [deviation 1](#deviation-1--the-cboe-zero-bid-wall).
7. **Arbitrage repair** — vertical monotonicity and butterfly convexity on the strike-ordered
   ladder.

The vendor `iv` column is **never** an input to any decision. It is carried through untouched on
`CleanQuote.vendor_iv` purely as a cross-check against the IVs the pipeline recomputes itself, and
flagged when it looks like a placeholder. (A live Yahoo fetch cannot even produce a zero-IV
sentinel: `market_data._norm_contract` already drops any contract with vendor `iv <= 0`. Such rows
are reachable only from committed fixtures. They are handled anyway, deliberately.)

**Exercise style is stated, not hidden.** Yahoo single-name and ETF chains are American; the
log-contract replication and the Black-76 inversion both assume European exercise. The pipeline
does not de-Americanise. It stamps `FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN` on every chain so the
assumption propagates into the report. See [§15.1](#151-american-exercise).

---

## §3 Stage 1 — forward, discount and the OTM surface (spec 4.2)

### 3.1 Discount

$D(T)$ comes from the repo yield curve (`yieldcurve.service.get_active_curve`), never from the
chain fetch, whose `rf` and `div` columns are hardcoded `0.0`. The rate is *defined* from the
discount, $r(T) = -\ln D(T)/T$, so `exp(-r*T) == D` holds to machine precision by construction and
the two can never drift apart. Precedence: explicit `D` → explicit `r` → injected curve → lazy
`get_active_curve()`.

### 3.2 Forward from put–call parity

Parity on cleaned mids at a common strike is exact for European options:

$$
C(K,T) - P(K,T) = D(T)\,\bigl(F(T)-K\bigr).
$$

The slope is *known* — it is $-D(T)$, and $D(T)$ has already been measured. So $F$ is estimated by
**weighted least squares with the slope pinned**, over the `max_pairs` strike pairs nearest the
money, with weights $w_i = 1/\sigma_i^2$ where $\sigma_i$ is the average half-spread of the two
legs (floored, so a suspiciously tight quote cannot dominate).

The free two-parameter regression is run **alongside** as a data-quality diagnostic and reported in
`ParitySlopeDiag` — the fitted slope must come out near $-D$; the gap is reported, never swallowed
— but it never feeds $F$. Verified in the suite: $F$, $D$ and $q$ recovery to $\le 1.4\times10^{-16}$,
fitted slope against $-D$ to $3.7\times10^{-15}$.

A diagnostic implied dividend yield $q_{\text{impl}}(T) = r(T) - \ln(F/S_0)/T$ is reported and used
nowhere.

### 3.3 Black-76 inversion — the exact substitution

The repo's inverter `implied_vol_call` is parametrised on spot/carry. A price quoted on forward $F$
with discount $D$ is inverted by calling it with $S_0 \to F D$, $r \to -\ln(D)/T$, $q \to 0$:

$$
\mathrm{bs\_call}(FD,K,T,-\tfrac{\ln D}{T},0,v)
= FD\,N(d_1) - K e^{\ln D} N(d_2) = D\bigl(F N(d_1) - K N(d_2)\bigr),
\qquad d_1=\frac{\ln(F/K)+\tfrac12 v^2T}{v\sqrt T}.
$$

That is Black-76 **exactly**, not an approximation, and it is pinned by a round-trip test. No second
inverter exists anywhere in the pipeline.

### 3.4 The OTM surface

Convention: $k<0$ (i.e. $K<F$) → puts, $k \ge 0$ → calls. Put quotes are converted to their
parity-equivalent call price $C = P + D(F-K)$ **before** inversion, so both wings are produced by
exactly the same inverter with exactly the same numerical properties. Output: `SurfacePoint(T, K,
k, F, D, iv, …)`, plus a typed rejection list.

---

## §4 Stage 2 — the log contract and `K_var` (spec 4.3)

This is the model-free heart of the pipeline. It is worth reading slowly, because *this* is the
stage whose systematic errors land directly on $\xi_0$ and therefore on everything downstream.

### 4.1 The continuous replication, derived

Assume $S$ is a positive continuous semimartingale under the pricing measure with
$dS_t/S_t = \mu_t\,dt + \sigma_t\,dW_t$. Then

$$
d\ln S_t = \frac{dS_t}{S_t} - \tfrac12\sigma_t^2\,dt
\qquad\Longrightarrow\qquad
\int_0^T \sigma_t^2\,dt \;=\; 2\int_0^T \frac{dS_t}{S_t} \;-\; 2\ln\frac{S_T}{S_0}.
$$

The first term is the payoff of a **continuously rebalanced** position holding $2/S_t$ shares — a
self-financing dynamic strategy requiring no option. The second is a static European payoff, the
**log contract**. Taking expectations under the risk-neutral measure, the dynamic term contributes
the drift and the fair variance rate is the price of $-2\ln(S_T/S_\star)$ plus a deterministic
correction.

The log payoff is replicated statically by the Carr–Madan / Breeden–Litzenberger decomposition
around any reference strike $S_\star$:

$$
-\ln\frac{S_T}{S_\star}
= -\frac{S_T-S_\star}{S_\star}
+ \int_0^{S_\star}\frac{(K-S_T)^+}{K^2}\,dK
+ \int_{S_\star}^{\infty}\frac{(S_T-K)^+}{K^2}\,dK ,
$$

which is the second-order Taylor formula with remainder applied to $f(S)=-\ln(S/S_\star)$, whose
second derivative is $f''(K) = 1/K^2$. Pricing each leg gives the Demeterfi–Derman–Kamal–Zou (1999)
formula, with **undiscounted** option mids:

$$
K_{\mathrm{var}}(T) = \frac{2}{T}\Bigl[rT - \Bigl(\frac{S_0e^{rT}}{S_\star}-1\Bigr) - \ln\frac{S_\star}{S_0}\Bigr]
+ \frac{2e^{rT}}{T}\Bigl[\int_0^{S_\star}\frac{P(K,T)}{K^2}dK + \int_{S_\star}^{\infty}\frac{C(K,T)}{K^2}dK\Bigr].
$$

Choosing $S_\star = F(T)$ — the parity forward of Stage 1 — makes the first bracket vanish
identically and collapses this to the clean form

$$
K_{\mathrm{var}}(T) = \frac{2}{T\,D(T)}\Bigl[\int_0^{F}\frac{P(K,T)}{K^2}dK + \int_{F}^{\infty}\frac{C(K,T)}{K^2}dK\Bigr].
$$

**Nothing about rBergomi has been used.** This is a static no-arbitrage identity plus the
continuity of $S$. The continuity assumption is the one that matters and is treated honestly in
[§15.2](#152-jumps-the-log-contract-is-not-realised-variance).

### 4.2 The discrete strike integration

Quotes live on a finite, irregular ladder, so the integral is replaced by the CBOE VIX
white-paper sum on the **cleaned** strike set:

$$
K_{\mathrm{var}}(T) = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}\,\frac{Q(K_i,T)}{D(T)}
\;+\; \underbrace{\frac{2}{T}\Bigl[\ln\frac{F}{K_0} - \Bigl(\frac{F}{K_0}-1\Bigr)\Bigr]}_{\text{correction term}} .
$$

with

- $K_0$ = the **largest strike $\le F$** — not the nearest strike;
- $Q(K_i)$ = the put mid for $K_i<K_0$, the call mid for $K_i>K_0$, and the **average of the call
  and put mids at $K_0$**;
- $\Delta K_i = (K_{i+1}-K_{i-1})/2$, one-sided at both edges
  ($\Delta K_0 = K_1-K_0$, $\Delta K_n = K_n - K_{n-1}$);
- $1/D(T)$ from the Stage-1 curve, **never** $e^{rT}$ with a guessed flat $r$;
- implied volatilities are **never averaged**: the sum runs on prices.

Missing strikes and irregular spacing need no special code — they are absorbed by $\Delta K_i$ by
construction, which is precisely why the CBOE weights are used rather than a trapezoid on a
resampled grid.

### 4.3 The correction term: exact, not truncated

Setting $S_\star = K_0$ in the continuous formula and substituting $F = S_0e^{rT}$ turns the first
bracket into $(2/T)[1 - F/K_0 + \ln(F/K_0)]$. With $x = F/K_0 - 1 \ge 0$ (non-negative because
$K_0$ is the largest strike *at or below* $F$) this is

$$
\frac{2}{T}\bigl[\ln(1+x) - x\bigr] \;=\; -\frac{1}{T}x^2 + \frac{2}{3T}x^3 - O(x^4),
$$

a negative quantity that vanishes when $K_0 = F$. CBOE truncates it at second order as $-x^2/T$.
This pipeline uses the **exact closed form** at identical cost. Since the subtracted quantity
$x - \ln(1+x) = x^2/2 - x^3/3 + \dots$ is *larger* under the CBOE truncation, CBOE biases
$K_{\mathrm{var}}$ one-sidedly **low** by $\tfrac{2}{3T}x^3 + O(x^4)$ — a term that grows without
bound as $T\to0$, i.e. precisely at the short end that Stage 4 regresses to obtain $H$. See
[deviation 2](#deviation-2--the-exact-ddkz-correction-term). The CBOE value is still computed and
reported as `diagnostics.correction_term_cboe`.

### 4.4 The strike-grid discretisation bias — derived and quantified

The discrete portfolio interpolates the log payoff **piecewise linearly** between listed strikes.
For the convex payoff $-2\ln(S/F)$ the interpolant lies *above* the function, so the discrete
replication systematically **over-replicates**. Integrating the interpolation error
$f''(K)(K-K_i)(K_{i+1}-K)/2$ against the risk-neutral density with $f'' = 2/K^2$ gives $f''h^2/12$
per cell, i.e. a total-variance excess

$$
V_{\text{bias}} = \frac{h^2}{6F^2}
\qquad (h = \text{strike spacing at the money}),
$$

**independent of $T$ and of the volatility level**. Measured against this law, the module
reproduces it to a ratio of `1.0000` across $h \in [0.5,4]$, $F \in [10,5000]$ and
$\sigma \in [0.10,0.80]$.

In $K_{\mathrm{var}}$ terms the bias is $h^2/(6F^2T)$: it **decays like $1/T$**. On a \$5 ladder
over a \$100 underlying that is **+54 % at 7 days and +0.5 % at 2 years**. Because it is constant in
total variance $V$, it cancels in every forward-variance increment **except the first**, so it lands
**entirely on $\xi_0$ over $[0,T_1]$**.

The returned `k_var` is the faithful CBOE-style sum. The bias is quantified in
`diagnostics.discretisation_bias`, expressed relative to `k_var` in `discretisation_bias_rel`, and
raises `FLAG_COARSE_STRIKE_LADDER` past 2 %. See
[deviation 3](#deviation-3--the-strike-grid-bias-is-reported-not-corrected).

### 4.5 Tail truncation

Two numbers are produced per expiry:

- `k_var_trunc` — the sum on the **quoted** strikes only;
- `k_var` — the same sum plus the two missing tails, completed by **flat-IV extrapolation** of the
  last reliable two-sided OTM quote on each side. The anchor IV is inverted with Stage 1's
  `implied_vol_from_forward`; Black-76 prices at that flat IV are then integrated in $K$ outwards
  until $Q/K^2 < \varepsilon_{\text{tail}} = 10^{-10}$, or the hard cap
  `tail_n_std = 12` standard deviations binds, whichever comes first.

`k_var - k_var_trunc >= 0` is the **truncation-error estimate**, attached to every point, and
`FLAG_LARGE_TRUNCATION` fires past 5 %. Quadrature is **composite Simpson on a uniform grid in $K$**
— the tail integrand is $C^\infty$ on a bounded interval, has no kink and no singularity, so
$O(h^4)$ sits many orders of magnitude below the tail contribution, needs no adaptive branching (so
the result is bit-reproducible), and costs one vectorised Black-76 evaluation. The tail interval
starts at the *cell edge* of the discrete sum ($K_{\min}-\Delta K_{\text{first}}/2$ below,
$K_{\max}+\Delta K_{\text{last}}/2$ above) because the CBOE weights already cover those half-cells;
starting at the quoted strikes themselves would double-count them.

Deep put tails are priced by exact parity off the call pricer ($P = C - D(F-K)$) rather than by a
second pricer. That subtraction cancels catastrophically once $P$ is tiny, with an absolute floor
around $\varepsilon_{\text{mach}} D F$; the $\varepsilon_{\text{tail}}$ stop is reached long before
that floor matters (relative error $\approx \varepsilon_{\text{mach}}F/(\varepsilon_{\text{tail}}K^2)
\approx 4\times10^{-8}$ on a 100-priced underlying), and residual negative noise is clipped at zero
and **counted** in the diagnostics.

The known weakness of flat-IV completion is in [§15.3](#153-flat-iv-wing-completion).

### 4.6 Refusals

An expiry returns a `VarianceSwapFailure` — French message, **no number** — when: viability fails;
fewer than 3 usable strikes survive; no strike sits at or below $F$; $T$, $F$ or $D$ is not finite
and strictly positive; no forward point matches the expiry; the forward point carries a different
maturity than the chain; or $K_{\mathrm{var}}$ comes out negative or non-finite. That last one is
not pedantry — see [§5.2](#52-why-a-negative-k_var-is-a-hard-error).

---

## §5 Stage 3 — forward variance and the isotonic repair (spec 4.4)

### 5.1 Construction

Stage 2 gives, at every market maturity $T_j$, the **total** variance

$$
V(0,T_j) = T_j\,K_{\mathrm{var}}(T_j) = \int_0^{T_j}\xi_0(u)\,du .
$$

$\xi_0$ is therefore a *derivative* of market data, and market data is noisy. Differentiating
$(T_j,V_j)$ with raw finite differences amplifies quote noise into a saw-toothed forward-variance
curve, which the simulator would then faithfully reproduce as a saw-toothed term structure. The
pipeline never does that. Two steps:

**Step 1 — calendar-arbitrage repair.** $V(0,T)$ must be non-decreasing in $T$; a decreasing total
variance is a calendar-spread arbitrage, not information. It is repaired by **isotonic regression**
via the pool-adjacent-violators algorithm (PAVA, Ayer et al. 1955), which returns the exact,
unique least-squares non-decreasing fit
$\arg\min\sum_j (V_j - \hat V_j)^2$ s.t. $\hat V_1 \le \dots \le \hat V_n$. It is implemented
in-module with **uniform weights** — `scipy.optimize.isotonic_regression` does not exist in the
pinned scipy 1.11.4 and scikit-learn is not a dependency. Every adjusted point and the size of its
adjustment is logged in `ForwardVarianceMetadata.isotonic_adjustments`
("*Réparation isotone appliquée (arbitrage calendaire)*"); the repair is never silent. A
scale-relative floor of a few ulps prevents PAVA's rewriting of a genuinely flat segment to its own
mean from being reported as a repaired arbitrage.

**Step 2 — piecewise-constant levels.**

$$
\xi_0(t) = \begin{cases}
V_1/T_1 & t \in [0,T_1]\\[2pt]
\dfrac{V_{j+1}-V_j}{T_{j+1}-T_j} & t \in (T_j,T_{j+1}]\\[6pt]
\xi_0(T_{\text{last}}) & t > T_{\text{last}}\quad\text{(flat)}
\end{cases}
$$

Positivity is enforced by a floor `eps_xi = 1e-6`. Flooring a level **breaks** the exact
reconstruction on that interval by construction, so it raises `FLAG_XI0_FLOORED`
("*Variance forward plancherée à eps_xi*"), records the affected indices and reports the resulting
reconstruction gap. After the isotonic repair the increments are non-negative, so the floor only
bites on a genuinely flat or negative first increment.

Why piecewise-constant is the default rather than a smooth curve: [deviation 4](#deviation-4--piecewise-constant-xi_0-by-default).

The resulting `ForwardVarianceCurve` is a **frozen** dataclass of tuples plus non-writeable numpy
views — nothing a consumer can reach lets it mutate the curve, so it is safe to share across
threads and across calibration restarts. It exposes a vectorised `xi0(t)` (scalar in → scalar out,
array in → same-shape array out; `NaN` for `t<0` and for `NaN` input), `integrated(T)` (exact at
every market maturity to $10^{-12}$), and `reconstruction_errors()`.

### 5.2 Why a negative `K_var` is a hard error

PAVA **cannot** catch a negative $V$: a negative-then-rising sequence is already non-decreasing, so
the repair is a no-op. The piecewise-constant level on the following interval,
$(V_1-V_0)/(T_1-T_0)$, then absorbs the whole negative offset. A measured case produced
$\xi_0 = 12.58$ — **355 % instantaneous volatility**, with no flag at all. Both `variance_swap`
(refusal) and `forward_variance` (`ValueError` with a French message) refuse rather than propagate
it.

### 5.3 Where the Stage-2 biases land

This is the load-bearing consequence of §4.4. Because the discretisation bias is a *constant* in
total variance, it telescopes out of every increment $V_{j+1}-V_j$ and survives only in the very
first level $V_1/T_1$. So a coarse strike ladder does **not** smear across the curve; it inflates
$\xi_0$ on $[0,T_1]$ and nowhere else. Since $\xi_0$ is then *frozen* as data, that inflation is
not absorbed by $(H,\eta,\rho)$ — it shows up as a residual at the shortest maturity. That is
better than the alternative (an error that the optimiser would silently trade against $H$), but it
is still an error, and `FLAG_COARSE_STRIKE_LADDER` is how a reviewer sees it.

---

## §6 Stage 4 — short-maturity skew and `H` (spec 4.5)

> **This stage produces an asymptotic *initial* estimate, not a calibration result.** It must never
> be reported as "the" Hurst exponent of the surface, and it must never be pinned as a calibrated
> parameter.

### 6.1 The relationship and its asymptotic status

For a rough volatility model with Hurst index $H$, the short-maturity ATM skew behaves as

$$
\psi(T) \;=\; \frac{\partial\sigma_{BS}}{\partial k}\Big|_{k=0} \;\sim\; A\,T^{H-1/2},
\qquad T \to 0,
$$

so that $\log|\psi(T)| = \log|A| + (H-\tfrac12)\log T$ and $H = \text{slope} + \tfrac12$. This is
the Alòs–León–Vives (2007) / Fukasawa (2011) short-time expansion, popularised in the rough
setting by Bayer–Friz–Gatheral (2016) and used as the empirical signature of roughness by
Gatheral–Jaisson–Rosenbaum (2014).

**The status of that "$\sim$" is the whole point.** It is a $T\to0$ statement. Every listed
maturity is finite; the default window here ends at a quarter of a year. The observed exponent is
therefore contaminated by

- the sub-leading terms of the expansion (an $O(T^{H+1/2})$ correction, relatively $O(T)$),
- the term structure of $\xi_0$ itself (a non-flat forward variance tilts $\psi$ maturity by
  maturity),
- quote noise, which is *worst* exactly where the asymptotics are *best* (the shortest expiries
  carry the widest spreads and the coarsest ladders),
- and, from §4.4, a $1/T$ strike-grid bias that is largest inside this very window.

The last three all push in maturity-dependent ways, which is why this estimate is a **start** and a
**diagnostic**, and why the joint fit of Stage 5 exists at all. On the synthetic reference surface
the bias is measurable and material: see [§15.5](#155-the-spec-45-regression-is-biased-low-on-the-reference-surface).

### 6.2 Per-expiry skew

A **weighted local quadratic** regression of the recomputed market IV against $k$:

$$
\sigma(k) \approx a + b\,k + c\,k^2,
\qquad \psi(T) = b,
\qquad \partial^2_k\sigma|_{k=0} = 2c .
$$

Window: $|k| \le c_w\,\sigma_{\mathrm{ATM}}(T)\sqrt T$ with $c_w = 1.5$. An expiry is **skipped**
unless the window holds $\ge 5$ strikes with $\ge 2$ strictly on each side of the money.
$\mathrm{SE}(b)$ comes from the weighted-regression covariance
$\hat\sigma^2_{\text{resid}}(X^\top W X)^{-1}$ — computed, never assumed.

The window is self-referential ($\sigma_{\mathrm{ATM}}$ is what the fit produces). It is resolved by
a **robust seed plus one fixed-point pass**: seed on the IV of the quote nearest $k=0$; fit; take
$\sigma_{\mathrm{ATM}} = a$, the fitted intercept (*not* the nearest quote, which is biased by the
skew whenever the nearest strike is not exactly at the money); rebuild the window and refit. The
reported `(fit, window)` pair is always *consistent* — the reported coefficients are the ones
produced by the reported window. Non-convergence is flagged (`FLAG_WINDOW_NOT_CONVERGED`), and a
refined window that no longer satisfies the strike counts reverts to the previous valid fit
(`FLAG_WINDOW_REVERTED`) rather than being silently widened.

Both strike-convention derivatives are exposed so a consumer never has to redo the chain rule:

$$
\psi = F\,\frac{\partial\sigma}{\partial K}\Big|_{K=F},
\qquad
\frac{\partial^2\sigma}{\partial K^2}\Big|_{K=F} = \frac{2c-b}{F^2}.
$$

The regression itself is *always* run in $k$: dimensionless, centred at 0 by construction, far
better conditioned than raw strikes.

### 6.3 Weights

Spec 4.5 asks for vega weights or $1/\text{spread}^2$. The regression is on **implied volatility**,
so the only dimensionally consistent reading of $1/\text{spread}^2$ is *in vol units*: a price
half-spread $s$ maps to an IV uncertainty $s/\text{vega}$ at first order, hence

$$
w_i = \frac{1}{\text{spread}_{\text{iv},i}^2} = \Bigl(\frac{\text{vega}_i}{s_i}\Bigr)^{2}
\qquad(\texttt{"iv\_spread"}, \text{ the default when spreads are available}).
$$

`SurfacePoint` carries only `mid`, so spreads must be supplied by handing the cleaned chains in via
`clean_chains=`; the lookup is keyed on contract symbol first, then `(option_type, strike, T)`.
Without them the documented fallback is $w_i = \text{vega}_i^2$, which is the same weighting under
homoskedastic price noise. `"price_spread"` (the literal, dimensionally inconsistent form) and
`"uniform"` remain available for sensitivity work. Weights are normalised to mean 1 before the
solve; the covariance formula is exactly invariant to that rescaling, so the reported standard
errors do not depend on it.

### 6.4 The cross-maturity regression

Weighted least squares of $\log|\psi(T)|$ on $\log T$ over the window $[5/365,\ 0.25]$, with
$H_0 = \text{slope} + 1/2$. Weights use the **delta-method** transform into the space the
regression actually lives in: $\mathrm{SE}(\log|\psi|) = \mathrm{SE}(\psi)/|\psi|$, hence
$w = (|\psi|/\mathrm{SE}(\psi))^2$. (The literal $1/\mathrm{SE}(\psi)^2$ is available as
`"raw_psi"` for sensitivity work.)

Guard rails, all of them reported:

- $|\psi| < 10^{-6}$ dropped before any log is taken;
- a sign flip across the window is flagged (`FLAG_SIGN_FLIP`) and the fit continues on $|\psi|$
  with the sign census in the diagnostics;
- at least 3 usable expiries;
- $\mathrm{SE}(H_0)$, a 95 % CI and $R^2$ are always reported;
- the WLS slope is cross-checked against a **Theil–Sen** slope (median of pairwise slopes, exact
  for the 3–8 points this regression sees); a gap larger than one SE raises
  `FLAG_ROBUST_DISAGREEMENT` **before** the rejection thresholds run;
- rejection as UNSTABLE when $R^2 < 0.6$, $\mathrm{SE}(H_0) > 0.15$, or $H_0 \notin (0.01,0.49)$.
  An UNSTABLE estimate returns $H_0 = 0.1$ with `unstable=True` — a fallback **for the optimiser
  start**, never a result. The measured value stays visible in `diagnostics["H0_estimated"]`.

Carry-over from Stage 2: an expiry whose variance-swap point carries `FLAG_COARSE_STRIKE_LADDER`
gets `FLAG_COARSE_LADDER_IN_WINDOW` on its `SkewPoint` and is listed in
`diagnostics["coarse_strike_ladder"]`, so a reviewer can see when the $H$ regression leaned on a
coarse ladder.

---

## §7 The fractional machinery — Davies–Harte fBm vs the RL Volterra driver (spec 4.7)

### 7.1 The convention decision, stated once and for all

Two *different* Gaussian processes appear in this codebase. They share only their marginal
variance. Confusing them is the classic silent error of this model family, so the distinction is a
recorded convention decision:

$$
\textbf{standard fBm (Mandelbrot–Van Ness):}\quad
\mathrm{Cov}(B^H_u,B^H_v) = \tfrac12\bigl(u^{2H}+v^{2H}-|u-v|^{2H}\bigr)
$$

$$
\textbf{RL Volterra driver (rBergomi):}\quad
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dB_s,
\qquad
\mathrm{Cov}(\widetilde W_u,\widetilde W_v) = 2H\!\!\int_0^{u\wedge v}\!\!(u-s)^{H-1/2}(v-s)^{H-1/2}ds
$$

Both give $\mathrm{Var} = t^{2H}$. They agree **nowhere else**: $B^H$ has stationary increments,
$\widetilde W$ does not (its kernel is truncated at $0$, not at $-\infty$).

- **`rbergomi/fbm.py` (Davies–Harte)** produces *standard* fBm. It powers the fBm utility, the
  statistical validation of the fractional machinery, and the **roughness diagnostics** (e.g.
  estimating $H$ from a realised-variance series). It is exact, cheap and stationary — exactly what
  such an estimator needs.
- **`rbergomi/volterra_gaussian.py`** produces $\widetilde W$ jointly with the increments of the
  Brownian motion that drives it. **The pricing path uses this and only this.**

Never feed Davies–Harte output to an rBergomi pricer as if it were $\widetilde W$.

### 7.2 Davies–Harte, derived

Target: $n$ samples of standard fGn $X_0,\dots,X_{n-1}$ (unit step), stationary centred Gaussian
with

$$
\gamma(j) = \tfrac12\bigl(|j+1|^{2H} + |j-1|^{2H} - 2|j|^{2H}\bigr), \qquad \gamma(0)=1 .
$$

1. **Circulant embedding.** A circulant $C$ of size $m$ with first row $c$ satisfies
   $C[i,j] = c[(j-i)\bmod m]$. Its leading $n\times n$ block equals the fGn Toeplitz covariance iff
   $c[k]=\gamma(k)$ for $k \le n-1$ *and* $c[m-k]=\gamma(k)$ for $1 \le k \le n-1$. Both hold for
   the **reflected extension** $c[k] = \gamma(\min(k, m-k))$, which needs $m \ge 2(n-1)$. The module
   takes $m = 2(n-1)$ (`padding="minimal"`) or the next power of two (`padding="power_of_two"`, the
   default — the radix-2 FFT is markedly faster than the mixed-radix/Bluestein path numpy must take
   when $2(n-1)$ carries a large prime factor). Padding extends the autocovariance to lags up to
   $m/2$; it never truncates nor approximates the lags actually needed.
2. **Eigenvalues.** $\lambda = \mathrm{FFT}(c)$; $c$ is symmetric so $\lambda$ is real up to
   round-off with $\lambda_k = \lambda_{m-k}$. $C$ is a valid covariance iff $\lambda \ge 0$ —
   validated, with a tolerance, and a typed `CirculantEmbeddingError` on failure.
3. **Spectral synthesis.** With $w = e^{2\pi i/m}$ and a Hermitian complex sequence $Z$
   ($Z_{m-k} = \overline{Z_k}$, $\mathbb E|Z_k|^2 = 1$, $\mathbb E[Z_k\overline{Z_l}]=\delta_{kl}$),

$$
Y_j = \frac{1}{\sqrt m}\sum_k \sqrt{\lambda_k}\,Z_k\,w^{jk}
\qquad\Longrightarrow\qquad
\mathbb E[Y_jY_{j'}] = \frac1m\sum_k \lambda_k w^{(j-j')k} = c[(j-j')\bmod m],
$$

which is exactly $\gamma(|j-j'|)$ whenever $|j-j'|\le n-1$. So $X = Y[0{:}n]$ is **exact** fGn, not
an approximation. Building $Z$ from i.i.d. normals: $k=0$ and $k=m/2$ are self-conjugate so $Z_k$
must be real; for $0<k<m/2$, $Z_k = (V'+iV'')/\sqrt2$. Folding in $1/\sqrt m$ gives the coefficients
$\sqrt{\lambda_k/m}\,V$ and $\sqrt{\lambda_k/(2m)}(V'+iV'')$ — the $2$ is the variance split between
real and imaginary parts.

**Scaling to $[0,T]$:** multiply the unit-step increments by $dt^{H}$ with $dt = T/n$ (not
$dt^{2H}$, not $\sqrt{dt}$), so that $\mathrm{Var}(B^H_T) = T^{2H}$. Verified in the suite:
$\gamma(j)$ matches for $H \in \{0.15, 0.5, 0.75\}$ within $5\cdot$stderr, and
$\mathrm{Var}(B^H_T) = T^{2H}$ across $T$.

A Cholesky fallback exists for the cases circulant embedding rejects, behind a typed
`CholeskyFallbackError`; nothing degrades silently.

### 7.3 The exact joint construction for pricing

On the grid $0 = t_0 < t_1 < \dots < t_n$, the vector
$(\widetilde W_{t_1},\dots,\widetilde W_{t_n},\ \Delta B_1,\dots,\Delta B_n)$ is centred Gaussian
with a $2n\times2n$ covariance $\Sigma = \begin{pmatrix}A & C\\ C^\top & D\end{pmatrix}$:

**Block $A$.** With $u = \min(t_i,t_j)$, $v = \max(t_i,t_j)$, substituting $s = u\sigma$ in the
defining integral and applying Euler's integral representation of the Gauss hypergeometric function
gives the exact closed form

$$
\boxed{\;\mathrm{Cov}(\widetilde W_u,\widetilde W_v)
= \frac{2H}{H+\tfrac12}\;u^{H+1/2}\,v^{H-1/2}\;
{}_2F_1\!\Bigl(\tfrac12-H,\;1;\;H+\tfrac32;\;\frac uv\Bigr)\;}
$$

(with $b=1$, $c=H+\tfrac32$, $a=\tfrac12-H$, $z=u/v$ and $B(1,H+\tfrac12)=1/(H+\tfrac12)$).
This arrangement is numerically **stable** as $u\to v$ — the argument tends to $1^-$, where
${}_2F_1$ converges because $c-a-b = 2H > 0$ — unlike the naive $(v-u)$ arrangement whose
$(v-u)^{H-1/2}$ factor blows up. At $u=v=t$, Gauss's theorem gives
${}_2F_1(\tfrac12-H,1;H+\tfrac32;1) = (H+\tfrac12)/(2H)$, so the diagonal collapses to $t^{2H}$
**exactly**. Verified: diagonal to $\le 8.9\times10^{-16}$ absolute; the closed form against
brute-force quadrature (endpoint singularity declared) to a worst relative $1.5\times10^{-11}$ over
$H\in\{0.05,\dots,0.49\}$; the implementation against the independent oracle to $1.6\times10^{-11}$.
$A$ is symmetric *by construction*, bit for bit, because $\min$ and $\max$ are symmetric functions
of the index pair.

**Block $C$.** Itô isometry against the deterministic integrand gives, for
$a = t_{j-1} < \min(b,t)$ with $b = t_j$, $t = t_i$:

$$
\mathrm{Cov}(\widetilde W_t, B_b-B_a) = \frac{\sqrt{2H}}{H+\tfrac12}
\Bigl[(t-a)^{H+1/2} - \bigl(t-\min(b,t)\bigr)^{H+1/2}\Bigr],
\qquad = 0 \text{ when } a \ge t .
$$

On an increasing grid this makes $C$ **lower triangular** ($C_{ij}=0$ for $j>i$): the increment
$\Delta B_j$ lives in the future of $t_i$ and the Volterra kernel is causal. The difference of two
nearly equal powers is evaluated cancellation-free as

$$
(t-a)^{H+1/2}\Bigl(-\mathrm{expm1}\bigl[(H+\tfrac12)\,\mathrm{log1p}\bigl(-\tfrac{b-a}{t-a}\bigr)\bigr]\Bigr),
$$

exact to full precision even when $(b-a) \lll (t-a)$ — fine grids at long lags, where the naive
subtraction loses digits.

**Block $D$.** $\mathrm{Cov}(\Delta B_i,\Delta B_j) = \delta_{ij}\,\Delta t_i$.

$\Sigma$ is Cholesky-factored **once** per $(H,\text{grid})$ and every path is $L Z$ with
$Z \sim N(0,I_{2n})$: $O(n^3)$ once, $O(n^2\cdot\text{paths})$ per draw. Conditioning and jitter:
[§13.1](#131-cholesky-conditioning-and-jitter).

---

## §8 rBergomi dynamics and the correlation through the shared driver (spec 4.6/4.8)

### 8.1 The model

$$
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}dB_s
\qquad (\mathrm{Var}\,\widetilde W_t = t^{2H})
$$
$$
V_t = \xi_0(t)\exp\Bigl(\eta\,\widetilde W_t - \tfrac12\eta^2 t^{2H}\Bigr)
\qquad\Longrightarrow\qquad \mathbb E[V_t] = \xi_0(t)
$$
$$
dS_t = (r(t)-q)S_t\,dt + S_t\sqrt{V_t}\,dW^S_t,
\qquad W^S = \rho B + \sqrt{1-\rho^2}\,B^\perp .
$$

The exponential is normalised so that $\mathbb E[V_t] = \xi_0(t)$ *identically*. This is the single
most important structural fact in the whole document: **the level of the variance term structure is
$\xi_0$ and nothing but $\xi_0$**; $(H,\eta,\rho)$ do not move $\mathbb E[V_t]$ at all. Everything
they do is dispersion *around* that level. [§12](#12-parameter-identifiability) is built on this.

### 8.2 Why the correlation must go through the shared driver $B$

$\widetilde W$ is **not a semimartingale** — its quadratic variation is infinite for $H<1/2$ — so
the usual shorthand $d\langle W^S,\widetilde W\rangle_t = \rho\,dt$ has no literal meaning. What
*is* well defined, and what the implementation does, is that the spot noise and the Volterra driver
are built from **the same Brownian motion**:

$$
\widetilde W_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,\underbrace{dB_s}_{\text{same }B},
\qquad
\Delta W^S_i = \rho\,\underbrace{\Delta B_i}_{\text{the very same increment}} + \sqrt{1-\rho^2}\,\Delta B^\perp_i .
$$

`volterra_gaussian` returns $(\widetilde W_{t_1..t_n}, \Delta B_{1..n})$ as **one exactly-correlated
Gaussian vector** from a single Cholesky factor. The consequence is the exact cross moment

$$
\mathbb E\bigl[W^S_t\,\widetilde W_t\bigr] = \rho\,\frac{\sqrt{2H}}{H+\tfrac12}\,t^{H+1/2},
$$

which holds **by construction, with no discretisation error whatsoever** — it is §7.3's block-$C$
formula with $a=0$, $b=t$, multiplied by $\rho$. The suite checks it statistically for several
$\rho$ including a negative one.

> **Correlating fBm *values* with the spot noise is wrong.** Drawing a standard fBm from
> Davies–Harte and correlating its marginals against a Brownian motion imposes a correlation on the
> **marginal laws** instead of on the **driving noise**. Three things break: the joint law is not
> the rough Bergomi law; the cross moment above is violated; and the skew the construction produces
> is wrong — which means it lands directly on $\rho$, on $\eta$ and, through the maturity scaling of
> $\psi$, on $H$. There is no tolerance for this: it is a different model, not an approximation.

### 8.3 The log-Euler scheme and what it does *not* approximate

*Exact* — no discretisation error at all: the joint law of
$(\widetilde W_{t_0..t_n}, \Delta B, \Delta B^\perp)$ on the grid (it is sampled from its exact
covariance), hence $V_{t_i}$ at every node, hence the spot/vol correlation.

*Approximated* — the spot, by log-Euler with **left-point (adapted) variance**:

$$
\ln S_{i+1} = \ln S_i + (r_i-q)\Delta t_i - \tfrac12 V_{t_i}\Delta t_i
+ \sqrt{V_{t_i}}\bigl(\rho\,\Delta B_i + \sqrt{1-\rho^2}\,\Delta B^\perp_i\bigr).
$$

Freezing the variance at the **left** endpoint is not cosmetic. $V_{t_i}$ is
$\mathcal F_{t_i}$-measurable (the kernel is causal, so $\mathrm{Cov}(\widetilde W_{t_i},\Delta B_j)=0$
for every increment strictly after $t_i$), while $\Delta B_i$ and $\Delta B^\perp_i$ span
$[t_i,t_{i+1}]$ and are independent of $\mathcal F_{t_i}$. Hence

$$
\mathbb E\Bigl[\exp\bigl(\sqrt{V_{t_i}}\,\Delta W^S_i - \tfrac12 V_{t_i}\Delta t_i\bigr)\Big|\mathcal F_{t_i}\Bigr] = 1
\quad\text{exactly},
$$

so the **discounted, dividend-adjusted spot is an exact martingale of the discrete scheme**:
$\mathbb E[D(T)S_Te^{qT}] = S_0$ with *no* discretisation bias, only Monte-Carlo error. A midpoint
or right-point variance would destroy that. What the scheme *does* approximate is
$\int_{t_i}^{t_{i+1}}V_u\,du \approx V_{t_i}\Delta t_i$; the error is $O(\Delta t)$ in the usual weak
sense and shrinks with `n_max`. Its one materially biased consequence is the short-end ATM skew:
[§15.4](#154-the-log-euler-short-end-skew-bias).

**Every occurrence of a step length uses the real $\Delta t_i$.** The grid is deliberately
log-spaced, so $\Delta t_i$ varies by orders of magnitude; the drift, the Itô compensator, the
integrated variance, the discount factors and the variance of the Brownian increments all use
`grid.dt[i]`. Using a constant $T/n$ anywhere is the classic silent bug of this family of
simulators, and the suite pins it with a strongly non-uniform grid, a per-step rate array and an
$\eta\to0$ degeneracy that would move by percent if a constant $\Delta t$ crept in.

### 8.4 $\xi_0$ on the grid

Node evaluation vs cell average is a real decision with a measured consequence — see
[deviation 5](#deviation-5--xi0_cell_average-by-default). The gap between the left-point sum and the
true integral is deterministic and computable in closed form,

$$
\sum_i \xi_0(t_i)\Delta t_i - \int_0^T\xi_0
= \sum_{\text{knots } T_j \in (0,T)} \bigl(\xi_0(T_j) - \xi_0(T_j^+)\bigr)\,\Delta t_{k_j},
$$

and it is exposed as `RBergomiPathSet.xi0_quadrature_error` at every grid node. It is never silent
under either mode.

---

## §9 The pricing layer (spec 4.8)

One path set prices the **whole surface**: the grid carries every quoted maturity with exact float
equality, so all strikes at all maturities come from a single Cholesky factorisation and a single
set of random numbers. That is what makes common-random-number calibration affordable, and it
removes the maturity-snapping bias of the legacy `pricing_mc.py`.

**Two estimators.**

- `ESTIMATOR_PLAIN` — the textbook $D(T)\,\overline{(S_T-K)^+}$.
- `ESTIMATOR_CONDITIONAL` — the **mixed/conditional** estimator (Romano–Touzi 1997, in the rough
  Bergomi form of McCrickerd–Pakkanen 2018). Conditionally on the whole variance path *and* on $B$,
  the only remaining randomness in $\ln S_T$ is the single Gaussian
  $\sqrt{1-\rho^2}\int_0^T\sqrt V\,dB^\perp$, so $S_T$ is conditionally lognormal and the inner
  expectation is Black in closed form:

$$
\mathbb E\bigl[(S_T-K)^+\,\big|\,V,B\bigr] = \mathrm{Black}(F_{\text{cond}},K,w_{\text{cond}}),
$$
$$
F_{\text{cond}} = S_0e^{\int_0^T(r-q)}\exp\Bigl(\rho\!\int_0^T\!\!\sqrt V\,dB - \tfrac12\rho^2\!\int_0^T\!\!V\,du\Bigr),
\qquad
w_{\text{cond}} = (1-\rho^2)\!\int_0^T\!\!V\,du .
$$

The $\int\sqrt V\,dB$ uses **the same $dB$** that drives $\widetilde W$ — available *exactly* in
the Cholesky scheme, which is precisely why this estimator is exact here rather than approximate.
Integrating out the perpendicular noise analytically removes the dominant variance source; this is
the recommended estimator for calibration and the default of the joint calibrator. It is validated
against the plain estimator to within the standard error.

**Standard errors are honest.** Antithetic sampling makes rows $k$ and $k + n/2$ perfectly
dependent, so every standard error in the package is computed from the $n_{\text{paths}}/2$
independent **pair means**. Reporting $\mathrm{std}/\sqrt{n_{\text{paths}}}$ instead would be an
outright lie about the accuracy. Under scrambled-Sobol QMC the sample standard deviation is not an
estimate of the randomised-QMC error at all; the result is flagged `stderr_is_conservative` and
must be read as a pseudo-Monte-Carlo upper bound.

**Exact sample-wise invariants.** Because calls and puts come from the same sample, put–call parity
is an *algebraic identity*, not a statistical statement:
$C(K,T)-P(K,T) = D(T)(\text{sample\_forward}(T)-K)$ to round-off, strike by strike, because
$(x)^+-(-x)^+=x$ pathwise and $\mathrm{Black}_{\text{call}}-\mathrm{Black}_{\text{put}} = F-K$
identically.

**IVs.** `implied_vol_surface` inverts against the **sample** forward (the MC price is consistent
with it, not with the theoretical forward) through the same Black-76 substitution, and returns
**`NaN`** — never a fabricated `1e-4` — wherever the sample resolved no time value or the price
sits outside the no-arbitrage band.

---

## §10 Initialisation and `c(H)` (spec 4.9)

> **Everything this stage returns is a starting point, never a result.**

$H_0$ comes straight from Stage 4, `unstable` flag and all. $(\eta_0,\rho_0)$ come from the skew
*level*: at leading order

$$
\psi(T) \simeq c(H)\,\rho\,\eta\,T^{H-1/2},
$$

which is a function of the **product** $\rho\eta$ only. The degeneracy is broken here **by a prior,
not by data**: $\rho_0 = \mathrm{sign}(\psi)\cdot0.7$ (the equity-index prior, configurable), then

$$
\eta_0 = \mathrm{clip}\!\left(\frac{|\psi(T_{\text{ref}})|}{c(H_0)\,|\rho_0|\,T_{\text{ref}}^{H_0-1/2}},\;0.5,\;3.5\right).
$$

The degeneracy is stated explicitly in `diagnostics["rho_eta_degeneracy"]` together with its
invariant: changing $|\rho_0|$ rescales $\eta_0$ by exactly the inverse factor and leaves
$\rho_0\eta_0$ unchanged. **Stage 1 of the joint calibration is what actually breaks it.**

`c(H)` was **measured**, not assumed — see
[deviation 6](#deviation-6--ch-with-the-measured-factor-12).

**An independent $\eta$ from the ATM curvature**, reported alongside and never used as $\eta_0$.
The leading-order rBergomi curvature is

$$
\kappa(T) = \frac{\eta^2T^{2H-1}}{\sigma_{\mathrm{ATM}}}\bigl[a(H) - 2c(H)^2\rho^2\bigr],
\qquad a(H) = \frac{H}{4(H+\tfrac12)^2(H+1)} .
$$

$a(H)$ is **derived, not borrowed**: at $\rho=0$ the mixing (Romano–Touzi) representation gives
$w(k) = w + \tfrac12\mathrm{Var}(I)[k^2/(2w^2) - 1/(2w) - 1/8] + O(\eta^3)$ with
$I=\int_0^T V_t\,dt$ and $w=\xi_0T$; since $\partial^2_kw|_0 = \mathrm{Var}(I)/(2w^2)$ and
$\sigma=\sqrt{w/T}$ with a vanishing ATM slope at $\rho=0$, one gets
$\kappa = \mathrm{Var}(I)/(4w^{5/2}\sqrt T)$, and the exact RL covariance gives
$\mathrm{Var}(I) = \xi_0^2\eta^2HT^{2H+2}/((H+\tfrac12)^2(H+1))$, which collapses to $a(H)$.
Two independent confirmations: at $H=1/2$ rBergomi *is* lognormal SABR with $\nu = \eta/2$, and
Hagan's $\beta=1$ expansion gives $\partial^2_k\sigma = (2-3\rho^2)\nu^2/(6\alpha)$, hence
$a(1/2) = 1/12$ — **exact match** — and the $\rho$ law $(2-3\rho^2)/24 = a - 2c(1/2)^2\rho^2$ fixes
the $\rho^2$ coefficient at $2c(H)^2$ there. Measured against this repo's own simulator at $\rho=0$
(150 000 antithetic paths, `n_max=256`, flat $\xi_0=0.04$, $T\in\{20,40,80\}/365$),
$\hat g/a(H) = 1.034 / 1.006 / 0.993 / 0.986 / 0.978 / 0.972$ at
$H = 0.05/0.10/0.20/0.30/0.40/0.45$: the level is good to ~3 %. The $\rho^2$ law, measured as the
*ratio* $g(\rho)/g(0)$ at matched $(H,T,\eta,\text{grid})$ so grid and higher-order biases cancel,
gives $b(H)/a(H) = 1.63/1.68/1.67/1.65/1.63/1.63$ on the best-resolved cells against
$2c(H)^2/a(H) = 1.75/1.72/1.66/1.61/1.55/1.53$ (residual RMS 0.073), versus 0.150 and a systematic
low bias for the rival flat-$1.5$ law.

**The implemented form avoids the $\rho$ prior entirely.** Since
$\psi^2 = c(H)^2\rho^2\eta^2T^{2H-1}$ exactly, the identity rearranges to

$$
\kappa\,\sigma_{\mathrm{ATM}} + 2\psi^2 = a(H)\,\eta^2T^{2H-1}
\qquad\Longrightarrow\qquad
\eta_{\text{curv}} = \sqrt{\frac{(\kappa\,\sigma_{\mathrm{ATM}} + 2\psi^2)\,T^{1-2H}}{a(H)}} .
$$

That matters twice: (i) an estimate meant to *test* the $\rho_0$ prior must not consume it, and
(ii) the bracket $a-2c^2\rho^2$ **changes sign** near $|\rho| = \sqrt{a/(2c^2)}$ (about 0.76 at
$H=0.1$, 0.81 at $H=0.45$), so at the equity prior $|\rho_0|=0.7$ inverting it directly is a badly
conditioned small difference of comparable numbers; the $\rho$-free form is their *sum* and is well
conditioned.

End-to-end verification (150 000 paths, `n_max=256`, $T\in\{10,20,40,80\}/365$, $\xi_0=0.04$,
$H\in\{0.05,\dots,0.45\}$, $\eta\in\{0.3,0.5,1,2\}$, $\rho\in\{0,\pm0.4,\pm0.7\}$):
$\eta_{\text{curv}}/\eta_{\text{true}} \in [0.95,1.09]$ in every cell with $\eta\ge1$ or
$|\rho|\le0.4$. In the near-cancellation corner ($|\rho|\approx0.7$ **and** $\eta\le0.5$) single
cells ran 0.48 to 1.45. That corner is not hidden: the diagnostics report
`curvature_information_share` $= |\kappa\sigma|/(|\kappa\sigma|+2\psi^2)$ and raise
`FLAG_CURVATURE_ILL_CONDITIONED` below the configured minimum. For a typical equity index
($|\rho|\approx0.7$) that share is naturally only ~0.16 at $H=0.1$ and ~0.25 at $H=0.45$ — which is
exactly why this is a *cross-check with a factor-2 tolerance* and not a second calibration. A
disagreement beyond a factor 2 raises `FLAG_ETA_DISAGREEMENT`.

$T_{\text{ref}}$ is chosen among the skew points the Hurst regression actually used, as the one with
the **smallest relative standard error** $\mathrm{SE}(\psi)/|\psi|$, ties broken toward the shorter
maturity. Rationale: asymptotic validity favours the shortest maturity, precision favours the
longest — and the $c(H)$ measurement settles the arbitration, since the closed form is accurate to
~2 % roughly *uniformly* over 5–80 days, so precision is the binding constraint inside the window.

---

## §11 The joint calibration — objective and weights (spec 4.10)

### 11.1 Matching the model forward to the market forward

Market IVs are inverted on the parity forward $F(T)$ and the curve discount $D(T)$. Comparing them
to a model whose forward is $S_0e^{\int(r-q)}$ for some unrelated $(r,q)$ would inject a pure
forward mismatch straight into $(H,\eta,\rho)$. So the drift is built **from the market forwards
themselves**: with $q=0$,

$$
r_i = \frac{\ln F(t_{i+1}) - \ln F(t_i)}{\Delta t_i},
\qquad \ln F \text{ interpolated linearly in } T \text{ through } (0,\ln S_0),
$$

which makes `paths.model_forward(T) == F_market(T)` **exactly** at every quoted maturity. The path
set's own discount is then $D_{\text{paths}}(T) = S_0/F(T)$, which is *not* the market discount and
does not have to be: an implied volatility is invariant under a common rescaling of the price and
of $FD$, so the IV objective is unaffected. The price-based objective is not, so there every model
price is rescaled by $D_{\text{market}}/D_{\text{paths}}$ — an exact algebraic rescale, not an
approximation.

### 11.2 The objective

Default (`OBJECTIVE_IV`), on the **cleaned quote set**, in volatility units:

$$
\mathcal L(\theta) = \sum_i w_i\Bigl(\mathrm{IV}^{\text{model}}(K_i,T_i;\theta) - \mathrm{IV}^{\text{market}}(K_i,T_i)\Bigr)^2,
\qquad \sum_i w_i = 1 .
$$

`OBJECTIVE_PRICE_RELATIVE` replaces the residual by
$(\text{px}^{\text{model}} - \text{px}^{\text{market}})/\max(\text{price\_floor},\text{px}^{\text{market}})$.

Each quote is priced **out of the money** (put below the forward, call above). Put–call parity is
exact on a shared sample, so the two legs carry the same information — but the OTM leg carries far
less Monte-Carlo variance, and `implied_vol_surface` converts the put back to a call with that exact
sample parity before inverting.

A quote whose model price cannot be inverted charges a fixed vol-unit penalty
(`UNINVERTIBLE_PENALTY_VOL = 0.5`) and raises `FLAG_UNINVERTIBLE_MODEL_PRICE`, rather than
disappearing from the sum (which would let the optimiser *earn* loss by making prices
uninvertible).

### 11.3 Weights

$$
s_{\mathrm{iv},i} = \mathrm{clip}\Bigl(\frac{0.5\,(\text{ask}-\text{bid})_i}{\text{vega}_i},\;
s_{\text{floor}} = 2\times10^{-3},\; s_{\text{cap}} = 0.5\Bigr)
\quad\text{(vol units)},
$$
$$
w_i = \Bigl(\frac{\mathrm{median}_j\, s_{\mathrm{iv},j}}{s_{\mathrm{iv},i}}\Bigr)^{2},
\qquad\text{then normalised so every maturity contributes exactly } 1/n_{\text{maturities}} .
$$

The spread is converted into **volatility** units through the vega *before* being inverted and
squared, so $w_i$ is a plain inverse variance in the units the residual lives in. The literal
$1/\text{spread}^2$ of the spec is dimensionally inconsistent with a vol-space objective and,
worse, it *rewards* the cheap deep wings whose absolute spread is a half tick — precisely the "do
not overweight illiquid deep-OTM options" failure the spec warns about. The per-maturity
normalisation stops a densely quoted expiry from drowning the others.

**There is deliberately no separate vega factor.** Substituting the definition gives
$w \sim \text{vega}^2/s^2$: the vega already enters twice, which is the entire content of "in vol
units". An earlier version multiplied by a further $\text{vega}_i/\max_T\text{vega}$; on a
constant-absolute-spread surface the measured exponent was then exactly $p=3.00$ wherever no clip
binds, and the two largest quotes of each maturity took **43–67 %** of that maturity's whole weight
(11 strikes, so a uniform split is 18 %). That strips the wings, which are what carry the
$\rho$/skew information. At $p=2$ the same measurement gives a top-two share of ~50 % and a
single-quote maximum of ~28 %. (`WeightConfig.vega_floor_rel` survives as a dead field so existing
configs keep constructing; the role it played is now played by `spread_iv_cap`.)

**Mixing known and unknown spreads.** A quote whose own spread is unknown, while others are known,
is assigned the **median of the known ones after the clip** — not the constant `default_spread_iv`.
Measured on a set where half the quotes carry a 0.02 absolute spread and the median $s_{\mathrm{iv}}$
equals `default_spread_iv` exactly, the constant assignment gave the known-spread quotes **98.89 %**
of the total weight (mean weight ratio 117×, overall max/min $1.25\times10^7$): an ATM known-spread
quote has $0.5\times0.02/\text{vega}$ far below the floor, so it was clipped up and received
$(0.02/0.002)^2 = 100$, while every unknown-spread quote received exactly 1.

**Hard exclusions**, applied before any weighting: $|k| > k_{\max} = 1.0$; relative spread above
0.25; non-finite or non-positive market IV; zero vega; a maturity left with fewer than
`min_quotes_per_maturity = 2` quotes (dropped whole — a single quote would otherwise receive the
entire $1/n_{\text{maturities}}$ share). A **standardised** cut
$|k| > 3\,\sigma_{\mathrm{ATM}}(T)\sqrt T$ is applied too; it is an addition to the letter of the
spec, documented and disableable with `None`, because an absolute $k$ threshold is ten standard
deviations at one week and two at two years.

### 11.4 Two stages, common random numbers, budget

*Stage 1 (coarse)* — a Latin-hypercube design over the box (`n_design = 24`) with the Stage-4/4.9
initial point always inserted as an anchor, evaluated at reduced paths (`stage1_paths = 8 000`) on
**one** common draw. The best `top_k` survive.

*Stage 2 (local)* — a derivative-free local search (Nelder–Mead by default; Powell and the repo's
`multi_start_least_squares` on $\sqrt{w_i}e_i$ are available) from each of the `n_starts` best
Stage-1 points, at `stage2_paths = 12 000`. Each restart gets its own CRN draw (spec-literal: one
$Z$ per restart), and every restart optimum is then **re-evaluated on one common selection draw**,
so the winner is picked from comparable numbers rather than from three different noise
realisations.

*Final repricing* — one high-accuracy evaluation at the optimum with a **fresh** seed and
`final_paths = 100 000`, run in batches of `batch_paths = 20 000` and pooled exactly (equal batches
⇒ pooled mean is the mean of batch means, pooled variance is $\sum \mathrm{SE}_b^2/B^2$).

**CRN is the load-bearing choice** — see [deviation 7](#deviation-7--common-random-numbers).

**The evaluation budget is the calibrator's own.** `CalibratorSettings.max_nfev` defaults to 80
across the repo, which is right for the analytic calibrators that dataclass was written for and
**truncating** here. Measured on the 8-maturity reference surface (7 d … 2 y, 88 quotes, truth
$H=0.10$, $\eta=1.5$, $\rho=-0.70$), Nelder–Mead over three free parameters needs **115–162**
evaluations to stop on its own convergence test:

| `max_nfev` | $H$ (3 seeds) | `loss_crn` | converged |
|---|---|---|---|
| 80 | 0.0757 / 0.0899 / 0.1013 | 6.3e-6 … 3.5e-6 | **False** |
| 400 | 0.1015 / 0.1039 / 0.0994 | 7.4e-7 … 3.4e-6 | True |

Truncation cost up to $\Delta H = -0.026$ and $\Delta\rho = -0.09$ — about ten times the grid
systematic measured elsewhere — and it broke the module's own stationarity invariant. So the
default budget is `local_nfev_per_param * n_free` $= 55\times3 = 165$, and `settings.max_nfev` is
honoured only when the caller changed it from the shared class default. The consequence is
recorded in [§15.7](#157-the-max_nfev--80-ambiguity).

---

## §12 Parameter identifiability (spec 4.11)

This is the section a challenger should read first. Everything above is measurement; this is where
the pipeline states what its number can and cannot mean.

### 12.1 Why $\xi_0$ comes from the variance term structure

From §8.1, $\mathbb E[V_t] = \xi_0(t)$ **identically**, for every $(H,\eta,\rho)$. Therefore

$$
\int_0^T \xi_0(u)\,du = \mathbb E\Bigl[\int_0^T V_u\,du\Bigr] = T\,K_{\mathrm{var}}(T)
$$

is a statement about $\xi_0$ **alone**. Nothing about roughness, vol-of-vol or leverage enters it.
And $K_{\mathrm{var}}$ is measured by a *static replication* (§4) that never mentions rBergomi: it
is an integral against the risk-neutral density, valid for any continuous price process. So the
separation is not a numerical convenience — the two pieces of information have genuinely different
provenance. $\xi_0$ is a **price**; $(H,\eta,\rho)$ are a **shape**.

### 12.2 Why $H$ comes from the roughness / short-skew structure

$H$ enters through the kernel exponent. Two independent signatures:

- **The skew power law.** $\psi(T)\sim c(H)\rho\eta\,T^{H-1/2}$: the *maturity scaling* of the ATM
  skew is a direct read of $H$, and it is a scaling, so it is insensitive to the overall level
  (which is where $\rho\eta$ lives). This is what Stage 4 regresses, and it is why $H$ is
  fundamentally a **term-structure** parameter.
- **The roughness of the variance path itself**, $\mathrm{Var}(\widetilde W_t) = t^{2H}$ with
  non-stationary, strongly negatively correlated increments for small $H$ — visible in the whole
  smile shape across maturities, not just at the money.

The direct consequence: **with a single maturity, $H$ is not identifiable by construction.** The
calibrator says so in as many words —
`FLAG_SINGLE_MATURITY`: "*Une seule échéance dans le jeu de calibration : H n'est pas identifiable
par construction (il vit dans la structure par terme).*"

### 12.3 The $\rho$–$\eta$ degeneracy

At leading order the short-dated ATM skew depends on $\rho$ and $\eta$ **only through their
product**. This is not a theoretical worry: it was *measured* on this repo's own simulator. The
fitted constant $\hat c$ is independent of $\rho$ to three decimals —
$H=0.05$, $\eta=0.5$: $\hat c = 0.18070 / 0.18079 / 0.18097$ at $\rho = -0.9/-0.7/-0.4$
(MATH_ORACLE §8). Any procedure whose only information is the short-dated ATM slope therefore
determines $\rho\eta$ and nothing more; that is exactly why the initializer breaks the tie with a
**prior** ($|\rho_0| = 0.7$) and says so.

What breaks it in the joint fit:

- the **ATM curvature**, whose leading order $\propto \eta^2[a(H) - 2c(H)^2\rho^2]$ separates
  $\eta^2$ from $\rho^2$ (§10) — though with a small information share at the equity prior;
- the **wings**, whose asymmetry is a function of $\rho$ at fixed $\rho\eta$;
- the **longer maturities**, where the leading-order short-time expansion is no longer the whole
  story;
- **Stage 1's global design**, which searches the whole box on the whole surface rather than
  following the short-dated ATM slope.

And the pipeline *checks* whether it actually got broken, rather than assuming: spec 4.11 computes
the **$(\eta,\rho)$ valley at fixed $\rho\eta$** and raises `FLAG_ETA_RHO_VALLEY_FLAT` —
"*Vallée (eta, rho) à produit rho\*eta constant PLATE : la dégénérescence du skew court terme n'est
pas levée par la surface complète.*"

### 12.4 Why freezing $\xi_0$ is what makes $(H,\eta,\rho)$ identifiable at all

Suppose $\xi_0$ were free — say an $m$-node curve fitted jointly with $(H,\eta,\rho)$. The ATM
level at each quoted maturity is then reproducible for **any** $(H,\eta,\rho)$: whatever the triple
does to the ATM total variance, the corresponding $\xi_0$ node absorbs it exactly, because
$\mathbb E[V_t] = \xi_0(t)$ is an identity and the ATM row of the surface is essentially $m$
numbers. The loss surface acquires an $m$-dimensional valley along which $\xi_0$ compensates the
triple, and the only information left about $(H,\eta,\rho)$ is second-order smile *shape* — with
$m$ extra free parameters absorbing the first-order signal. That is the standard way a rough-vol
calibration produces a confident-looking $H$ that means nothing.

Freezing $\xi_0$ removes that compensating direction. The ATM level stops being a free fit and
becomes a **prediction**: a wrong $(H,\eta,\rho)$ now costs loss at the money as well as in the
wings, at every maturity, and the maturity scaling of that cost is what pins $H$.

This is why the repo guardrail is structural rather than procedural. $\xi_0$ is not "held fixed by
convention"; it is not in $\theta$, not in the bounds, not reachable from the objective's argument,
content-fingerprinted with SHA-256 at construction, re-verified after the optimisation, and handed
back by identity. `constraints["xi0_curve"]` is **required** — there is no honest way to
manufacture a forward-variance curve out of an IV grid, so its absence is an explicit failure, not
a silent fallback.

### 12.5 The diagnostics that are actually computed

- **1-D profile slices** of the loss along $H$, $\eta$ and $\rho$, each spanning the
  **bound-to-bound** interval with the optimum and its two finite-difference neighbours inserted, so
  gradient and curvature come from the same evaluations.
- **The $(\eta,\rho)$ valley** at fixed $\rho\eta$ (§12.3).
- **$H_0$ versus $H_{\text{calibrated}}$**, with the Stage-4 95 % CI; disagreement raises
  `FLAG_H_OUTSIDE_H0_CI`. When $H_0$ is the fallback, `FLAG_H0_FALLBACK` says the comparison has no
  probative value.
- **A measured Monte-Carlo noise floor.** Every quantity tested here is a *difference* of two losses
  inside ONE shared draw, and CRN cancels the sampling noise in exactly those differences — so the
  floor is measured on the differences themselves: the whole CRN set is re-drawn `noise_replicates`
  times and the run-to-run standard deviation of the *same* one-step difference is taken. Measured
  up to **25× smaller** than the scatter of the loss *level* across seeds, which is what a naive
  version used. The level-based quantity is still reported, for the one comparison genuinely made
  across draws.
- **Stationarity**, checked rather than assumed: $\theta^\*$ must be the cheapest point of its own
  profile, up to that floor. Otherwise `FLAG_PROFILE_NOT_STATIONARY`.
- **Standard errors from the profile curvature** — the number that answers the question.

### 12.6 `SE(H)` and why the span flag is not enough

$$
\boxed{\;\mathrm{SE}(p) = \sqrt{\frac{2\,\sigma_{\mathcal L}}{\partial^2\mathcal L/\partial p^2}}\;}
$$

the half-width of the interval over which the loss rises by less than one Monte-Carlo sigma above
its minimum — i.e. **the set of parameter values this surface cannot tell apart from the optimum**.
Both ingredients are already measured, so it costs nothing extra. On the reference surface,
$\partial^2\mathcal L/\partial H^2 = 1.226\times10^{-2}$ (stable across steps 0.005–0.02) and
$\sigma_{\mathcal L} = 1.375\times10^{-6}$ give $\mathrm{SE}(H) = 0.015$, directly comparable to the
Stage-4 $\mathrm{SE}(H_0) = 0.0061$ — a ratio of 2.5, well inside the `se_vs_h0_factor = 10`
default. A non-positive curvature means the optimum is not a minimum along that axis at all and
yields $\mathrm{SE} = \infty$.

`FLAG_H_WEAKLY_IDENTIFIED` fires when $\mathrm{SE}(H) > 0.4\times\mathrm{TOL}_H = 0.02$
(`se_material_ratio = 0.4`). Why 0.4 and not 1.0: at 1.0 a parameter is only flagged once its
one-sigma uncertainty *alone* consumes the entire recovery tolerance, which is far too late.
Measured on a deliberately thin surface (quotes at 1 y and 2 y only, truth $H=0.10$), three seeds
returned $H = 0.1136 / 0.1064 / 0.1621$ with $\mathrm{SE}(H) = 0.0118 / 0.0186 / 0.0303$. The third
is outside $\mathrm{TOL}_H = 0.05$, and at ratio 1.0 it was reported `success=True` **with no
warning at all**, even though a $\pm1\sigma$ band 0.061 wide provably cannot resolve $H$ to the
tolerance the result is accepted on. At 0.4 that run is correctly flagged.

**The bound-to-bound span flag is secondary and known to be blind.** `span` is dominated by the far
arms of the slice and is very nearly independent of how sharply the optimum is resolved. On the same
thin surface the span sat **264–1115×** above its floor in all three seeds and
`FLAG_H_PROFILE_FLAT` never fired; on another thin set (1 y and 2 y, truth $H=0.10$) three seeds
returned $H = 0.1260 / 0.0447 / 0.1780$ — a spread of 0.133, i.e. $2.7\times\mathrm{TOL}_H$ — while
the span sat 250–320× above the floor throughout. A span-based flatness test could never fire on
either. It is kept as a secondary diagnostic; the standard error decides.

### 12.7 `success` is a verdict

`success` is **computed from the diagnostics, never a constant**. It is `False` whenever the result
carries no usable information about $H$:

```
BLOCKING_FLAGS = (FLAG_H_PROFILE_FLAT,          # bound-to-bound span under the measured noise floor
                  FLAG_H_WEAKLY_IDENTIFIED,     # SE(H) = sqrt(2 sigma_L / d2L/dH2) over threshold
                  FLAG_NO_IMPROVEMENT,          # optimum no better than the start, up to the floor
                  FLAG_PROFILE_NOT_STATIONARY)  # theta* is not the cheapest point of its own profile
```

`FLAG_H_WEAKLY_IDENTIFIED` is the load-bearing one. Everything else is a warning carried in `flags`.

This is not cosmetic. `apply_degeneracy_guard` only trips on an all-NaN surface, so a hard-coded
`success = True` shipped a meaningless $H$ to the controller **as a calibrated result** — against
the repo guardrail *never hard-code $H$ (or any calibrated parameter) as an output*. See
[deviation 8](#deviation-8--success-is-a-verdict).

The French label is explicit about what the returned number is *not*:
"*Profil de H PLAT : … cette surface n'identifie pas H. La valeur rendue est celle où l'optimiseur
s'est arrêté sur une surface plate ; elle ne mesure rien et n'est PAS l'initialisation (le simplexe
se déplace même sans signal). Calibration en échec : ne pas consommer ce H.*"

**Corollary for the reader: `success=False` is not a bug — and it is not confined to cheap
configurations.** The measured reference run on the committed fixture, at the FULL default budget
(100 000 final paths, `grid_n_max = 384`), returns `success = False` with
`FLAG_H_WEAKLY_IDENTIFIED`: `H0 = 0.1224` (CI95 `[0.1056, 0.1392]`), `H = 0.1344`, `eta = 1.154`,
`rho = -0.757`, IV RMSE 0.374 vol pt. Four seeds agree (`H` in `[0.1338, 0.1366]`, all
`NON CONCLUANTE`). That surface genuinely does not pin `H` to the precision the calibrator
demands, and the pipeline says so rather than dressing `0.1344` up as a measurement.
It is the pipeline saying the run does not identify $H$.

---

## §13 Numerical issues

### 13.1 Cholesky conditioning and jitter

$\Sigma$ is positive semi-definite but can be **singular**: at $H = 1/2$ the kernel degenerates to
1, $\widetilde W_t = B_t$ exactly, and $\widetilde W$ is a deterministic function of the increments
— the joint law is genuinely rank $n$ and `numpy.linalg.cholesky` rejects it. Round-off can also
push a legitimately positive-definite but badly conditioned $\Sigma$ ($H$ very close to 1/2, or a
grid with a very small first step) marginally negative.

Policy: a **documented and logged** diagonal jitter
$\texttt{jitter\_rel}\times\max(\mathrm{diag}\,\Sigma)$ with `jitter_rel = 1e-12`, retried with a
factor-10 escalation per attempt, up to 6 attempts. Nothing is silently regularised: the amount
actually applied is reported in `VolterraFactor.jitter_applied` / `.diagnostics()` and emitted
through `logging.warning`. Exhausting the attempts raises `CovarianceFactorizationError` rather
than returning a wrong factor. The jitter policy is part of the LRU cache key, because it changes
$L$ (the spec's $(H,\text{grid-hash})$ key is a strict prefix of it).

Two conditioning choices upstream of the factorisation matter as much as the jitter:

- the **stable ${}_2F_1$ arrangement** of the $A$ block (§7.3), which is well behaved as $u\to v$
  where the naive $(v-u)^{H-1/2}$ arrangement blows up;
- the **cancellation-free `expm1`/`log1p` form** of the $C$ block, exact to full precision when the
  step is tiny relative to the lag.

### 13.2 Grid policy

The grid is the sorted union of:

- **every quoted maturity, verbatim** — no snapping, no rounding: the float handed in is the float
  stored, and `grid.index_of(T)` recovers its exact position. (The legacy `simulator.py` snaps with
  `round(T/dt)`; that maturity-dependent bias is exactly what this policy removes.)
- a **log-spaced short-end block** of `min_steps - 1` points geometrically spaced on
  $[\texttt{short\_end\_ratio}\cdot T_{\min},\,T_{\min})$, so a 7-day expiry is not resolved by two
  points;
- a **log-spaced bulk fill** on $[T_{\min},T_{\max}]$, dense near the short end where the kernel is
  most singular and the variance curve moves fastest.

The short-end block scales with the budget: measured **29 / 61 / 124 / 252** steps to $T_{\min}$ at
`n_max` = 64 / 128 / 256 / 512. A geometric block cannot reach 0, so the first cell
$[0,\texttt{short\_end\_ratio}\cdot T_{\min}]$ is the grading floor and is the one cell not part of
the progression. That is inherent to any finite grid on a rough process — $\mathrm{Var}(\widetilde
W_t) = t^{2H}$ already accumulates most of its short-end mass inside the first cell whatever the
floor. Lower `short_end_ratio` or raise `min_steps` to push it down.

Fill points within `dedup_rtol = 1e-9` of a quoted maturity (or of each other) are dropped — the
quoted maturity always wins — because two nodes a round-off apart make $\Sigma$ numerically
singular for no modelling benefit. The budget $n \le n_{\max}$ is spent on quoted maturities first,
then the short-end block, then the fill; when `n_max` cannot hold even the first two, the builder
raises `GridConfigurationError` instead of silently violating either requirement.

**Calibration uses `grid_n_max = 384`, not the pipeline default 256**, for the reason in
[§15.4](#154-the-log-euler-short-end-skew-bias). Note that `grid_min_steps` is *not* the lever: the
short-end block already receives about half the node budget, so the shortest maturity sits at index
~188 out of 384 whatever `min_steps` says.

### 13.3 Monte-Carlo noise versus the optimiser

Three separate things are done so a derivative-free optimiser can work on a Monte-Carlo objective:

1. **CRN makes the objective bit-deterministic and smooth in $\theta$.** The grid is built once, so
   the draw's shape never changes; the seed is fixed for the whole stage, so $Z$ is bit-identical
   whatever $(H,\eta,\rho)$; only $L(H)$ changes, and $\widetilde W = ZL(H)^\top$ is a *continuous*
   function of $H$ because the Cholesky map is continuous on the positive-definite cone. So the loss
   is a smooth function of $\theta$ on one draw — which is exactly what lets a derivative-free
   method (and finite differences) work at all.
2. **The noise floor is measured on differences, not levels** (§12.5), because differences are what
   CRN cancels and what every test actually looks at.
3. **Variance reduction is real and its accounting is honest.** Antithetic pairs are exact
   negations, so standard errors come from pair means (§9). The conditional estimator removes the
   dominant variance source, which is what makes a 12 000-path objective usable.

**Over-fitting to the draw is tested at matched path counts.** Comparing `loss_fresh` at
`final_paths = 100 000` against `loss_crn` at `stage2_paths = 12 000` compares two different
estimators: measured $\mathbb E[\mathcal L(12\text{k})] = 6.647\times10^{-6}$ against
$\mathbb E[\mathcal L(100\text{k})] = 5.163\times10^{-6}$, a ratio of 0.777, so the fresh loss is
structurally *smaller* and a naive flag at ratio 3 needed a genuine 3.9× over-fit before it could
fire. Instead $\theta^\*$ is repriced on a fresh seed at the **matched** `stage2_paths` and compared
to the loss on the draw the local stage actually fitted (`loss_crn_matched` / `loss_fresh_matched`),
with `fresh_seed_gap_ratio = 12.0` sized from that measurement.

**Reproducibility.** `np.random.default_rng(seed)` throughout, sub-seeds via
`rng.integers(0, 2**31-1)`; the joint block draws $2n$ normals then $n$ for $B^\perp$ per base path,
so a fixed seed gives a bit-identical path set; antithetic draws consume normals for
`n_paths // 2` base paths only and mirror them by exact negation. `settings.seed` makes the whole
calibration run bit-identical.

---

## §14 The deliberate deviations

Eight decisions in this pipeline depart from the letter of the specification or from the obvious
default. Every one of them is a **recorded decision with a measurement attached**, not an accident.
They are listed here so a reviewer can attack them individually.

### Deviation 1 — the CBOE zero-bid wall

**Spec conflict.** Spec 4.1 bullet 1 protects one-sided quotes ("*a zero bid means the contract is
usable for the tails*"); bullet 3 imposes a relative-spread ceiling. These contradict: a zero-bid
quote has $(\text{ask}-\text{bid})/\text{mid} = 2$ **identically**, so the ratio filter annihilates
exactly the quotes the one-sided rule was written to protect.

**Arbitration.** One-sided quotes are **exempt** from the ratio test
(`apply_spread_filter_to_one_sided = False`) and governed instead by a second gate: the CBOE VIX
white-paper rule. Walking **outward from the money on the OTM side only**, once
`zero_bid_stop_count = 2` consecutive zero-bid strikes appear, that run and everything beyond it is
dropped (`REASON_ZERO_BID_TAIL` — "*Queue tronquée (deux bids nuls consécutifs, règle CBOE)*").
Isolated zero bids inside the wall survive, flagged `FLAG_ONE_SIDED`, so bullet 1 stays alive.

**Measured justification.** Without the wall, worthless contracts survive at their half-tick ask and
their $Q(K)\Delta K/K^2$ terms inflate $K_{\mathrm{var}}$ by **+6.9 % at 7 days versus +0.0 % at
6 months**. Because those contracts cluster at short maturities, the distortion does **not** cancel:
it tilts the $\log|\psi(T)|$ regression of Stage 4 and biases **$H$ itself**. The CBOE rule is
adopted rather than an invented price floor because the CBOE white paper is the reference the spec
itself names for this stage. Do not weaken or bypass it.

### Deviation 2 — the exact DDKZ correction term

**Deviation.** The correction term is evaluated in its **exact** closed form
$(2/T)[\ln(F/K_0) - (F/K_0 - 1)]$ instead of CBOE's second-order truncation $-(1/T)x^2$.

**Measured/derived justification.** The truncation error is one-sided and unbounded at the short
end: $x - \ln(1+x) = x^2/2 - x^3/3 + \dots$, so the CBOE subtrahend is *larger* than the exact one
and CBOE biases $K_{\mathrm{var}}$ **low** by $\tfrac{2}{3T}x^3 + O(x^4)$, a term that grows without
bound as $T\to0$ — precisely at the maturities Stage 4 regresses for $H$. The exact form costs
nothing extra. The CBOE value is still computed and reported as
`diagnostics.correction_term_cboe` so a user can reconcile against a VIX-style number.

### Deviation 3 — the strike-grid bias is *reported*, not corrected

**Deviation.** The returned `k_var` is the faithful CBOE-style sum. The known discretisation bias is
quantified and flagged but **not subtracted**.

**Derived and measured justification.** The discrete replication interpolates the log payoff
piecewise-linearly between listed strikes, over-replicating by

$$
V_{\text{bias}} = \frac{h^2}{6F^2}\quad\text{in TOTAL variance}
$$

(verified to a ratio of **1.0000** across $h\in[0.5,4]$, $F\in[10,5000]$, $\sigma\in[0.10,0.80]$).
In $K_{\mathrm{var}}$ it decays like $1/T$: **+54 % at 7 days versus +0.5 % at 2 years** on a \$5
ladder over a \$100 underlying. Being constant in $V$, it lands **entirely on $\xi_0$ over
$[0,T_1]$**.

**Why flagged rather than subtracted.**
(i) Spec 4.3 mandates the CBOE form, and the returned number must remain the number the spec names —
a silently debiased `k_var` would no longer be comparable with any external variance-swap or VIX-style
quote.
(ii) $h$ is the **at-the-money spacing**, a single number standing in for a genuinely irregular
ladder; the closed form is exact for a uniform ladder against a smooth density, so subtracting it
would replace a *known, bounded, reported* bias with a correction whose own error is unquantified —
and there is no measurement in this repo of that correction's error on an irregular ladder.
(iii) The bias lands on $\xi_0$, which the pipeline treats as **frozen data**. Silently editing a
data input is worse than reporting a flagged one.

What downstream may do: subtract `diagnostics.discretisation_bias`, or drop the flagged maturity.
What it must not do is consume the number unaware — hence `FLAG_COARSE_STRIKE_LADDER` past 2 % and
`discretisation_bias_rel` on every point.

### Deviation 4 — piecewise-constant $\xi_0$ by default

**Deviation.** The source spec's headline asks for a "smooth" $\xi_0$; the default here is
piecewise-constant, with the smooth monotone-PCHIP variant behind
`ForwardVarianceConfig.method = "pchip_monotone"`.

**Justification.** Piecewise-constant is **exact by construction**
($\int_0^{T_j}\xi_0 = V_j$ at every market maturity, to machine precision), **oscillation-free** (no
interpolant can invent a wiggle between two maturities), and it is what the market itself does when
bootstrapping forward variance from variance-swap quotes. Given that the whole downstream
calibration then *freezes* $\xi_0$ as data, an interpolant's invented structure would be frozen too.

The smooth variant is not merely offered — it is **validated or rejected**. PCHIP is accepted only
if it passes three tests, otherwise the builder falls back to piecewise-constant and records the
reason: **positivity** ($\xi_0 > \varepsilon_\xi$ on a fine grid); **no oscillation** (material
turning points of $\xi_0$ bounded by the material turning points of the piecewise-constant level
sequence, "material" meaning a swing above `oscillation_rel_tol` of the level range — a smaller
wiggle is Hermite end-condition noise, far below the accuracy of the `K_var` inputs themselves);
and **quadrature reconstruction** ($|\int_0^{T_j}\xi_0 - V_j| < 10^{-8}$ at every market maturity,
by composite Simpson interval by interval — $\xi_0$ is piecewise quadratic so Simpson is exact, and
the check really tests that the differentiated interpolant still integrates back to the data).

### Deviation 5 — `XI0_CELL_AVERAGE` by default

**Deviation.** Spec 4.8 writes $V_{t_i} = \xi_0(t_i)\exp(\cdot)$ — the **node** value.
`SimulationConfig.xi0_evaluation` defaults to `XI0_CELL_AVERAGE`, the exact cell average
$\frac{1}{\Delta t_i}\int_{t_i}^{t_{i+1}}\xi_0(u)\,du$. `XI0_NODE` remains available and fully
tested for anyone who wants spec 4.8 verbatim; it is a single documented flag.

**Justification.** The cell average is a strictly better quadrature of the **same continuous
model** (whose variance is $\xi_0(t)\exp(\cdot)$), not a different model. The node value interacts
badly with the default piecewise-constant curve: $\xi_0$ is left-continuous with jumps sitting
exactly on the quoted maturities, which are exactly grid nodes, so $\xi_0(t_k)$ returns the level of
the cell that *ends* at $t_k$ while the Euler cell $[t_k,t_{k+1}]$ is governed by the *next* level.
Three measured consequences:

- spec 4.4 requires $\int_0^{T_j}\xi_0 = T_jK_{\mathrm{var}}(T_j)$ **exactly** at every quoted
  maturity and spec 3 requires $\xi_0$ to stay frozen data; under node evaluation the *simulated*
  variance-swap strike does not reproduce the $K_{\mathrm{var}}$ that $\xi_0$ was bootstrapped from,
  so the guarantee is broken at the source;
- the resulting error is maturity-dependent and **changes sign** across the term structure (measured
  positive at 1–4 months, negative past 6 months) — the worst possible shape, because it **tilts**
  $\log|\psi(T)|$ versus $\log T$ rather than shifting it, and therefore lands directly on $H$;
- it fails the $\eta\to0$ Black–Scholes degeneracy by a few hundredths of a vol point, maturity by
  maturity.

Under `XI0_CELL_AVERAGE` the quadrature gap is identically zero. Under either mode it is computed
exactly and exposed as `xi0_quadrature_error`. **Do not flip this back for calibration.**

### Deviation 6 — `c(H)` with the measured factor 1/2

**Deviation.** Spec 4.9 gives the literature constant $\sqrt{2H}/((H+\tfrac12)(H+\tfrac32))$ "up to
convention" and explicitly instructs *not* to trust it blindly. The measurement was run, and the
implemented constant is

$$
\boxed{\;c(H) = \tfrac12\,\frac{\sqrt{2H}}{(H+\tfrac12)(H+\tfrac32)}\;}
$$

**Measurement conditions.** This repo's own simulator; flat $\xi_0 = 0.04$; conditional estimator;
60 000 antithetic paths; central difference $dk = 0.01$; $T \in \{5,10,20,40,80\}/365$; over
$H\in\{0.05,0.10,0.20,0.35,0.45\}\times\eta\in\{0.5,1,2\}\times\rho\in\{-0.9,-0.7,-0.4\}$.

**Evidence.**

- $\hat c\,/\,[\sqrt{2H}/((H+\tfrac12)(H+\tfrac32))] = $ **0.487–0.493 at $\eta=0.5$ for every $H$
  tested**, i.e. it converges on 1/2 as $\eta\to0$, which is the leading-order limit the formula
  describes. **The factor is 1/2.**
- $\hat c$ is **independent of $\rho$ to three decimals** (0.18070 / 0.18079 / 0.18097 at
  $\rho=-0.9/-0.7/-0.4$, $H=0.05$, $\eta=0.5$) — a direct confirmation of the leading-order
  $\rho\eta$ degeneracy (§12.3).
- $\hat c$ *does* drift with $\eta$ (a higher-order correction), worst at small $H$: 0.1807 / 0.1723
  / 0.1416 at $\eta = 0.5/1/2$ for $H=0.05$ (spread over $\eta$ of 0.0399 at $H=0.05$ versus 0.0058
  at $H=0.45$).

| $H$ | $\hat c$ ($\eta=0.5$) | $0.5\times$lit | ratio |
|---|---|---|---|
| 0.05 | 0.18070 | 0.18547 | 0.974 |
| 0.10 | 0.22867 | 0.23292 | 0.982 |
| 0.20 | 0.26188 | 0.26574 | 0.985 |
| 0.35 | 0.26201 | 0.26603 | 0.985 |
| 0.45 | 0.25188 | 0.25605 | 0.984 |

So the closed form is good to ~2 % for $\eta\le1$, degrading to ~10–15 % at $\eta=2$, worst at small
$H$. **Ample for a seed, useless as a result.** `C_OF_H_PROVENANCE` repeats this in every
diagnostics payload so the number can never be mistaken for a fitted quantity. Reference:
MATH_ORACLE §8.

### Deviation 7 — common random numbers

**Deviation.** The joint calibrator uses **common random numbers**: one fixed seed per stage, one
grid built once, so the objective is bit-deterministic and smooth in $\theta$. The pre-existing
surrogate calibrator (`rbergomi/calibrator_mc_surrogate.py`) draws an **independent seed per design
point**.

**Why the surrogate's choice is a defect.** With independent seeds the objective is a *noisy*
function whose **differences** are dominated by sampling noise. Every quantity an optimiser, a
profile slice, a flatness test or a finite-difference gradient ever looks at is a difference of two
losses — so under independent seeds the signal a three-parameter Nelder–Mead is chasing can be
smaller than the noise separating any two evaluations, and the "optimum" is then a draw of the noise.
Under CRN the noise cancels to first order in exactly those differences: the loss becomes a smooth
function of $\theta$ on one draw ($\widetilde W = ZL(H)^\top$ with $Z$ fixed and $L$ continuous in
$H$ on the positive-definite cone), which is what makes a derivative-free local method work at all.

The corollary is that the *residual* noise must then be measured on differences too, not on levels —
and it is, at a floor measured up to **25× smaller** than the level-based one (§12.5). CRN is also
what makes the whole approach affordable: one Cholesky per $H$, one path set for the whole surface,
memoised across strikes, maturities and restarts.

Note the honesty cost that comes with CRN: an optimiser *can* fit its own draw. That is why
$\theta^\*$ is re-priced on a fresh seed at **matched** path count (§13.3), not compared against a
different estimator.

### Deviation 8 — `success` is a verdict

**Deviation.** `SurfaceCalibrationResult.success` is computed from the identifiability diagnostics,
not set to `True` on completion.

**Criterion.** `FLAG_H_WEAKLY_IDENTIFIED`, from
$\mathrm{SE}(H) = \sqrt{2\sigma_{\mathcal L}\,/\,(\partial^2\mathcal L/\partial H^2)}$ against
$0.4\times\mathrm{TOL}_H = 0.02$, is the load-bearing test, alongside `FLAG_NO_IMPROVEMENT` and the
module's own stationarity invariant.

**Why the obvious alternative fails.** The bound-to-bound span flag (`FLAG_H_PROFILE_FLAT`) is
**secondary and known to be blind**: on a thin surface (quotes at 1 y and 2 y only, truth
$H = 0.10$) three seeds returned $H = 0.1136 / 0.1064 / 0.1621$ — the third outside
$\mathrm{TOL}_H$ — while the span verdict said "informative" every time, at 264–1115× its floor. The
span is dominated by the far arms of the slice and is nearly independent of how sharply the optimum
is resolved.

**Why this matters at all.** `apply_degeneracy_guard` only trips on an all-NaN surface. A hard-coded
`success = True` therefore shipped a meaningless $H$ to the Phase-5 controller **as a calibrated
result**, in direct violation of the repo guardrail *never hard-code $H$ (or any calibrated
parameter) as an output*. A configuration reporting `success = False` — cheap **or** at the full
default budget, as the measured reference run does — is the pipeline working
correctly, not failing.

---

## §15 Limitations — the honest list

Every item here is either measured (with the number) or explicitly labelled **unmeasured in this
repository**. None of them is corrected silently.

### 15.1 American exercise

Yahoo single-name and ETF chains are **American**. The log-contract replication (§4), the Black-76
inversion (§3.3) and the whole IV objective (§11) assume **European** exercise. The pipeline does
not de-Americanise; it stamps `FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN` on every chain so the
assumption reaches the report.

The stated assumption is that the early-exercise premium is negligible for **short-dated OTM**
options on **low-dividend** underlyings. It is **not** negligible for deep ITM puts, for
high-dividend names, or for long maturities — which is one more reason the pipeline consumes OTM
quotes only. Prefer cash-settled European index proxies (SPX-like) for demonstrations; ETF proxies
such as SPY are American and only approximately compliant.

**The size of the residual bias on $H$ is unmeasured in this repository.** Direction of the
first-order effect: an American premium inflates the option price, hence inflates
$K_{\mathrm{var}}$, and it does so more at long maturities and on the put wing — a maturity-dependent
distortion of exactly the kind that tilts a term-structure regression.

### 15.2 Jumps: the log contract is *not* realised variance

§4.1 derived the replication under the assumption that $S$ is **continuous**. The replication prices
the **log contract** exactly, in every case. What is only true under continuity is the identification
of that price with the fair strike of realised variance.

Take $d\ln S_t = (\mu_t - \tfrac12\sigma_t^2)dt + \sigma_t dW_t + J\,dN_t$ with jump intensity
$\lambda$ and log jump size $J$. Risk-neutral drift compensation gives
$\mu = r - \lambda\,\mathbb E[e^J-1]$, so

$$
-2\,\mathbb E\Bigl[\ln\frac{S_T}{F}\Bigr]
= \mathbb E\Bigl[\int_0^T\sigma_t^2dt\Bigr] + 2\lambda T\,\mathbb E\bigl[e^J - 1 - J\bigr],
$$

whereas the quadratic variation of the log price is $\int_0^T\sigma_t^2dt + \sum J^2$. The gap is

$$
\underbrace{2\lambda\,\mathbb E[e^J-1-J]}_{\text{log contract}} - \underbrace{\lambda\,\mathbb E[J^2]}_{\text{realised variance}}
= \lambda\,\mathbb E\Bigl[\tfrac13 J^3 + \tfrac1{12}J^4 + \dots\Bigr] .
$$

For **negatively skewed** jumps ($\mathbb E[J^3] < 0$) — the empirically relevant case for equity
indices — the log contract therefore sits **below** the fair variance strike at third order. This is
the classical result (Carr–Wu 2009; Broadie–Jain 2008).

**Consequence in this pipeline.** rBergomi is a continuous model, so any jump content in the market
is mapped into $\xi_0$ *as if it were diffusive variance*. Since $\xi_0$ is subsequently **frozen**,
that misspecification does not get traded against $(H,\eta,\rho)$ by the optimiser — it surfaces as
a residual instead. But it is not harmless: the jump contribution is not proportional to $T$, so it
distorts the *shape* of the term structure that $\xi_0$ encodes, and the short end is where jump
risk is proportionally largest. **The magnitude of this effect is unmeasured in this repository**;
this pipeline neither estimates a jump component nor corrects for one.

### 15.3 Flat-IV wing completion

The missing tails are completed by extrapolating the **last reliable two-sided OTM quote's implied
volatility flat** outward (§4.5). A real equity index tail is not flat: the put wing typically keeps
steepening. A flat extrapolation therefore **under-prices the deep put tail** and biases
$K_{\mathrm{var}}$ **low**, by an amount that grows with how early the quoted ladder stops — i.e.
worst on thin chains and at short maturities where fewer strikes are listed.

Mitigations in place: the truncation error `k_var - k_var_trunc` is reported on every point;
`FLAG_LARGE_TRUNCATION` fires past 5 %; the anchor must be a genuinely two-way quote
(`tail_anchor_two_sided_only`), because a one-sided quote has no reliable mid and would extrapolate
the whole tail off a half-tick ask; the anchor IV must fall inside an acceptance window
$[10^{-4}, 5.0]$ or the next quote inward is tried.

Residual weaknesses that remain: `eps_tail = 1e-10` is the spec's literal criterion and is therefore
an **absolute** threshold, hence scale-dependent — on a very high-priced underlying the ATM
integrand is itself small and the threshold binds sooner (the stop reason actually used is always
reported, so the interplay with the scale-free `tail_n_std = 12` cap is never hidden). **The size of
the flat-wing bias against a realistically skewed tail is unmeasured in this repository.**

### 15.4 The log-Euler short-end skew bias

The log-Euler scheme freezes the variance at the left endpoint of each cell (§8.3), so it
**under-states the model's own ATM skew, and does so more at the short end than at the long end**.
Measured at the pipeline default `GridConfig(n_max = 256)`: roughly **9.7 % at 7 days against 3.0 %
at 2 years**, which displaces the model's own $\log|\psi|$ versus $\log T$ exponent by about
**0.02** — a systematic error on the very parameter being calibrated.

Two things are done, neither of them a silent correction:

1. **A finer grid is the calibration default**, `grid_n_max = 384`. Measured on the 8-maturity
   reference term structure (7 d … 2 y, 60 000 antithetic paths, conditional estimator, $H=0.10$,
   $\eta=1.5$, $\rho=-0.70$, $dk = 0.01$), $\psi(T)$ at `n_max = 384` agrees with its
   `n_max = 1024` value to **0.3–1.0 % out to six months and 2.8 % at one and two years** (the
   latter of the order of the Monte-Carlo noise of that very measurement), whereas `n_max = 256` is
   still about 2 % low at the short end ($\psi(7\text{d}) = -1.0914$ against $-1.1111$) — which is
   the maturity-dependent tilt that lands on $H$. Cost: at 12 000 paths and 88 quotes over 8
   maturities, one full objective evaluation measured **0.32 s at `n_max=256` versus 0.43 s at
   `n_max=384`** on the reference machine, a ~33 % surcharge per evaluation.
2. **The residual bias is measured and reported in parameter units.** With
   `refinement_check` on, the loss gradient is recomputed at the optimum on a grid
   `refinement_factor = 2` times finer. At the optimum the gradient on the *calibration* grid is
   zero by construction, so the gradient on the refined grid **is** the grid-induced tilt; combined
   with the Hessian diagonal read off the profile slices it gives a first-order Newton estimate of
   how far $\theta$ would move on the finer grid. That triple is reported as `GridBiasReport` and,
   past `grid_bias_material = 0.25` (a quarter of the recovery tolerance, i.e. $|\Delta H| > 0.0125$),
   raises `FLAG_GRID_BIAS_MATERIAL`. The precise estimator is

   $$
   \text{shift}_p = -\,\frac{\partial_p\mathcal L|_{\text{refined}} - \partial_p\mathcal L|_{\text{calibration}}}{\partial^2_p\mathcal L|_{\text{calibration}}},
   $$

   both gradients measured **on one fresh draw**, so the residual tilt that has nothing to do with
   discretisation (θ\* minimises the loss on the local stage's draw, not on this one) cancels in
   the difference. On the reference synthetic surface the estimate came out at
   $(\Delta H,\Delta\eta,\Delta\rho) = (+0.0019, -0.0002, +0.0068)$ between $n = 384$ and
   $n = 768$ — an order of magnitude below the recovery tolerances, which is the evidence that 384
   is enough for this term structure. A second reference run, recorded in the Phase-4 handoff,
   reports a residual shift of $(-0.0049, +0.0100, +0.0105)$ against tolerances
   $(0.098, 0.029, 0.088)$ and is likewise not material. *Consistency note for a reviewer:* that
   second tolerance triple is quoted verbatim from the phase record and does **not** equal
   `grid_bias_material * PARAM_SCALE` $= (0.0125, 0.0875, 0.03)$, which is the threshold the
   shipped `material` verdict uses; the two triples come from different acceptance criteria and the
   discrepancy is reported here rather than reconciled.

**It is an estimate of the residual bias, not a correction: nothing is subtracted from the
calibrated parameters.** At the calibration default `n_max = 384` the underlying scheme bias is
measured and reported, not corrected.

**"Not measured" is `NaN`, never `0`.** Two situations make a parameter's shift unmeasurable: the
finite-difference step does not fit inside its bounds, or the profile curvature is non-positive.
The first is not hypothetical for this pipeline — $\text{step}_H = 0.02\times(0.49-0.01) = 0.0096$,
so **any calibrated $H < 0.0196$** is unmeasurable, and that is precisely the very-rough regime the
whole exercise is about. An earlier version reported `theta_shift = 0.0` there, which the
`material` verdict then read as "no bias": a live run landing at $H = 0.01832$ reported
`theta_shift = {'H': 0.0, …}, material = False`, i.e. it **fabricated** the one number the caller
asked for. Both paths now report `NaN`, `GridBiasReport.unmeasured` names the parameters concerned,
`material` is computed over the measured ones only, and `FLAG_GRID_BIAS_NOT_MEASURED` is raised.
The practical limitation stands: **at very small $H$ the grid bias is not quantified at all.**

### 15.5 The spec-4.5 regression is biased low on the reference surface

On the synthetic reference surface, whose truth is $H = 0.10$, the Stage-4 asymptotic regression
returns

$$
H_0 = 0.0827 \pm 0.0061 ,
$$

so **the truth sits outside the estimate's own 95 % confidence interval.** This is not a bug in the
regression; it is the asymptotic status of §6.1 made quantitative — the finite-maturity window, the
sub-leading terms, the $\xi_0$ term structure and the discretisation biases all bite, and the
reported CI measures only the *statistical* scatter of the fit, not that systematic. It is the
sharpest single argument for why $H_0$ is a seed and the joint fit of Stage 5 is the result, and it
is why `FLAG_H_OUTSIDE_H0_CI` is a *warning* rather than a blocking flag.

### 15.6 Day count

Calendar days / 365, everywhere, with no business-day or trading-time weighting. A 7-day expiry is
$T = 0.0191781$ years regardless of holidays or weekends.

A *uniform* rescale $T \to cT$ would shift the intercept of the $\log|\psi|$ versus $\log T$
regression and leave the slope — hence $H_0$ — untouched. A **non-uniform** one would not: if
non-trading days cluster unevenly across the short-maturity window (a long weekend inside a 7-day
expiry but not inside a 30-day one), a trading-day count would compress some maturities more than
others and move the slope itself. **This effect is unmeasured in this repository**, and no
alternative day count is implemented. The convention is stated so that anyone comparing this
pipeline's $H$ against a published figure computed on a business-day convention knows the two are
not directly comparable.

### 15.7 The `max_nfev == 80` ambiguity

`settings.max_nfev` is honoured only when it differs from the shared `CalibratorSettings` class
default of 80 (§11.4). A caller who *deliberately* passes `max_nfev = 80` is therefore
indistinguishable from one who passed nothing, and gets the calibrator's own budget of 165 instead.
That ambiguity is the price of not mutating a dataclass shared by every calibrator in the repo. It
is documented and latent; it becomes visible the day a UI exposes the budget.

### 15.8 Vendor data quality

- The `rf` and `div` columns of the fetch are **hardcoded 0.0** and are never used; rates come from
  the yield curve. Anyone wiring a new data source must preserve that.
- The vendor `iv` column is never an input to any decision — carried as a cross-check only.
  `market_data._norm_contract` already drops any contract with vendor `iv <= 0`, so a live fetch
  cannot produce a zero-IV sentinel; those rows are reachable only from committed fixtures, and are
  handled anyway.
- `fetch_iv_surface` **drops bid/ask** and is therefore the wrong entry point for this pipeline: the
  spreads are what carry the weights of §6.3 and §11.3.
- Mids are the midpoint of bid and ask, with no microstructure model behind them. Stale quotes are
  dropped on `volume == 0 and openInterest == 0`, which is a crude proxy for staleness and does not
  detect a quote that is merely old within an active contract.
- **No measurement of vendor data quality is made in this repository**; the pipeline's defence is
  the exhaustive removal log, not a statistical filter.

### 15.9 Reporting beyond the quoted maturities

$\xi_0$ extrapolates **flat** past the last quoted maturity, and if a reporting grid extends beyond
the last quote the forward is extrapolated at the last known rate. `FLAG_REPORT_BEYOND_QUOTES` says
so: "*…le forward y est EXTRAPOLÉ au dernier taux connu et aucune cotation ne contraint le modèle
au-delà — surface indicative.*" Nothing beyond the last quoted maturity is calibrated.

### 15.10 Dead but retained

`WeightConfig.vega_floor_rel` no longer participates in the weight formula (deviation 7's sibling
measurement, §11.3) and is retained only so existing configurations keep constructing. It is
mentioned here rather than removed, per the repo's rule on unrelated changes.

---

## §16 How to challenge this number

A concrete audit path, in the order that finds problems fastest.

1. **Is the quote set real?** Read `cleaning_report(chains)` and the per-chain
   `removals_by_reason()`. A maturity that lost most of its ladder to `wide_spread` or
   `zero_bid_tail` is not a maturity you should let into the $H$ regression.
2. **Does the forward hold up?** `ParitySlopeDiag.slope_abs_error` against $-D$, and
   `forward_minus_free`. A parity slope that is off says the mids are not a coherent European chain.
3. **Is $K_{\mathrm{var}}$ trustworthy at the short end?** `discretisation_bias_rel` and
   `FLAG_COARSE_STRIKE_LADDER` (deviation 3); `truncation_error` and `FLAG_LARGE_TRUNCATION`
   (§15.3). If the shortest maturity is flagged on either, $\xi_0$ on $[0,T_1]$ is suspect and so is
   every short-dated residual.
4. **Did $\xi_0$ need repairing?** `isotonic_adjustments` and `FLAG_XI0_FLOORED`. A large isotonic
   adjustment means the raw total variance was decreasing — a data problem, not a model problem.
5. **Is $H_0$ a measurement or a fallback?** `HurstEstimate.unstable`, `r2`, `se`,
   `FLAG_ROBUST_DISAGREEMENT`, and `diagnostics["coarse_strike_ladder"]`. And remember §15.5: even a
   healthy $H_0$ can be systematically low.
6. **Is $H$ identified?** This is the decisive one. `SE(H)` from the profile curvature,
   `FLAG_H_WEAKLY_IDENTIFIED`, `FLAG_PROFILE_NOT_STATIONARY`, and `success` itself (§12.6–12.7). The
   bound-to-bound span is *not* the test.
7. **Was the $\rho\eta$ degeneracy broken?** `FLAG_ETA_RHO_VALLEY_FLAT` and the reported valley.
8. **Is the answer a property of the grid?** `GridBiasReport` and `FLAG_GRID_BIAS_MATERIAL`
   (§15.4). Then re-run at a higher `grid_n_max` and check the shift against
   $\mathrm{TOL}_H = 0.05$.
9. **Is the answer a property of the draw?** `loss_crn_matched` versus `loss_fresh_matched` at
   matched path counts, and `FLAG_FRESH_SEED_LOSS_GAP` (§13.3). Then re-run with a different
   `settings.seed`: three seeds disagreeing by more than $\mathrm{TOL}_H$ is the same verdict as a
   large `SE(H)`, arrived at the expensive way.
10. **Is $\xi_0$ still the curve you handed in?** `FrozenXi0.verify()` and object identity on
    `JointCalibrationResult.xi0_curve`.

If steps 6, 7 and 9 all pass and steps 3 and 8 are unflagged, the number is as good as this
pipeline can make it — subject, always, to §15.1 and §15.2, which no amount of numerical care can
remove.

---

## §17 References

**Required by the specification**

1. Gatheral, J., Jaisson, T. & Rosenbaum, M. (2014/2018). *Volatility is rough*. arXiv:1410.3394;
   **Quantitative Finance** 18(6), 933–949. — The empirical case for $H \ll 1/2$ and the
   term-structure signature this pipeline regresses.
2. Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility*. **Quantitative
   Finance** 16(6), 887–904. — The rBergomi model as implemented here: $\widetilde W$, $\xi_0$,
   $\eta$, $\rho$, and the short-maturity skew.
3. Alòs, E., León, J. A. & Vives, J. (2007). *On the short-time behavior of the implied volatility
   for jump-diffusion models with stochastic volatility*. **Finance and Stochastics** 11(4),
   571–589. — The ATM skew short-time expansion underlying $\psi(T)\sim A T^{H-1/2}$.
4. Fukasawa, M. (2011). *Asymptotic analysis for stochastic volatility: martingale expansion*.
   **Finance and Stochastics** 15(4), 635–654. — The martingale-expansion route to the same
   asymptotics, and the source of the $T^{H-1/2}$ exponent's status as a limit statement.
5. Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary
   processes*. **Finance and Stochastics** 21(4), 931–965. — The reference discretisation of the
   Volterra kernel; this pipeline instead uses the *exact* joint Gaussian construction of §7.3, at
   $O(n^3)$ once per $(H,\text{grid})$.
6. McCrickerd, R. & Pakkanen, M. S. (2018). *Turbocharging Monte Carlo pricing for the rough Bergomi
   model*. **Quantitative Finance** 18(11), 1877–1886. — The conditional/mixed estimator of §9.
7. Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). *More Than You Ever Wanted To Know About
   Volatility Swaps*. Goldman Sachs Quantitative Strategies Research Notes. — The continuous
   log-contract replication of §4.1 and the exact correction term of §4.3.
8. CBOE. *The CBOE Volatility Index — VIX* (white paper). — The discrete strike sum, the $\Delta K_i$
   weights, the $K_0$ rule and the two-consecutive-zero-bid wall of deviation 1.
9. Davies, R. B. & Harte, D. S. (1987). *Tests for Hurst effect*. **Biometrika** 74(1), 95–101. —
   The circulant-embedding synthesis of §7.2.
10. Dietrich, C. R. & Newsam, G. N. (1997). *Fast and exact simulation of stationary Gaussian
    processes through circulant embedding of the covariance matrix*. **SIAM Journal on Scientific
    Computing** 18(4), 1088–1107. — The general circulant-embedding framework and its validity
    conditions.

**Supporting, used in specific derivations above**

11. Romano, M. & Touzi, N. (1997). *Contingent claims and market completeness in a stochastic
    volatility model*. **Mathematical Finance** 7(4), 399–412. — The mixing/conditioning
    representation behind §9 and the $a(H)$ derivation of §10.
12. Carr, P. & Madan, D. (2001). *Optimal positioning in derivative securities*. **Quantitative
    Finance** 1(1), 19–37. — The static payoff replication of §4.1 (with Breeden, D. T. &
    Litzenberger, R. H. (1978), **Journal of Business** 51(4), 621–651).
13. Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. & Silverman, E. (1955). *An empirical
    distribution function for sampling with incomplete information*. **Annals of Mathematical
    Statistics** 26(4), 641–647. — PAVA, the isotonic repair of §5.1.
14. Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). *Managing smile risk*.
    **Wilmott Magazine**, 84–108. — The $\beta = 1$ SABR expansion used as the exact cross-check of
    $a(1/2) = 1/12$ in §10.
15. Carr, P. & Wu, L. (2009). *Variance risk premiums*. **Review of Financial Studies** 22(3),
    1311–1341; and Broadie, M. & Jain, A. (2008). *The effect of jumps and discrete sampling on
    volatility and variance swaps*. **International Journal of Theoretical and Applied Finance**
    11(8), 761–797. — The log-contract-versus-realised-variance gap under jumps, §15.2.

---

## Appendix — where each claim can be verified in the source

| Claim | Source of truth |
|---|---|
| Cleaning rules, reason codes, zero-bid wall, +6.9 % measurement | `app/model/calibration/rough_vol/chain_cleaning.py` (module docstring, `CleaningConfig`, `_truncate_zero_bid_tail`) |
| Parity WLS forward, Black-76 substitution, OTM surface | `app/model/calibration/rough_vol/forward_curve.py` |
| Log-contract derivation, exact vs CBOE correction, $h^2/(6F^2)$ bias, tails | `app/model/calibration/rough_vol/variance_swap.py` (module docstring, `VarianceSwapConfig`) |
| Isotonic repair, piecewise-constant levels, PCHIP validations, negative-$K_{\mathrm{var}}$ refusal | `app/model/calibration/rough_vol/forward_variance.py` |
| Skew fit, window fixed point, weights, $H_0$ regression and its guard rails | `app/model/calibration/rough_vol/hurst_estimator.py` |
| Davies–Harte derivation and the fBm-vs-$\widetilde W$ convention | `app/model/volatility_models/rbergomi/fbm.py` |
| Exact covariances, ${}_2F_1$ arrangement, jitter policy, grid policy | `app/model/volatility_models/rbergomi/volterra_gaussian.py` |
| rBergomi dynamics, shared-driver correlation, log-Euler martingale property, `XI0_CELL_AVERAGE` | `app/model/volatility_models/rbergomi/simulator_xi_curve.py` |
| Estimators, honest stderr, sample parity, IV `NaN` policy | `app/model/volatility_models/rbergomi/pricing.py` |
| $c(H)$ measurement, $a(H)$ derivation, $\eta_{\text{curv}}$, $T_{\text{ref}}$ | `app/model/volatility_models/rbergomi/initializer.py`; `MATH_ORACLE.md` §8 |
| Objective, weights, CRN, budget, profiles, `SE(H)`, `success` | `app/model/volatility_models/rbergomi/calibrator_joint_mc.py` (module docstring, `WeightConfig`, `JointMCConfig`, `ProfileSlice`, `BLOCKING_FLAGS`) |
