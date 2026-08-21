"""
OU mean-level estimation + scalar Kalman filter + mean-reversion band trading.

Adapted from the Quant Guild "Kalman Trading System" (kts.py) and the
"What Mean Reversion Should Look Like" notebook (ktsmr.ipynb):

- AR(1) fit on closes  ->  (phi, mu, sigma_eps), the discrete OU parameters.
- Scalar Kalman filter whose STATE is the estimated mean level (fair value):
    predict:  x = phi * x + (1 - phi) * mu ,  P = phi^2 P + Q
    update :  K = P / (P + R) ,  x += K (z - x) ,  P = (1 - K) P
  with Q = sigma_eps^2 * (1 - phi^2) and R = sigma_eps^2 * obs_scale.
  The observation-noise lever maps [0, 100] to obs_scale (0 = trust prices,
  100 = trust the OU model: once P << R the gain is ~0 and the level follows
  pure OU). The prior is P0 = R (kts.py parity), so the first gain is
  phi^2 / (1 + phi^2) ~ 0.5 whatever the lever and decays like 1/n: at lever
  100 the level starts as a running mean of the prices and only then freezes.
- Trading bands around a center (frozen mu-hat, or the adaptive Kalman level):
    upper/lower = center +/- k * sigma_stat ,  sigma_stat = sigma_eps / sqrt(1 - phi^2)
  (discrete analog of sigma / sqrt(2 theta)). The Kalman center used for the
  decision at t is the one-step PRIOR x_{t|t-1} (the level expected before
  z_t), so both variants compare z_t with information up to t-1.
- Band strategy (notebook rules): flat -> short above upper, long below lower;
  close when price crosses back through the center. Mark-to-market PnL.

The point of running BOTH centers (frozen vs Kalman) is the notebook's
"mean reversion trap": a mean estimated on a short window is structurally
biased and bleeds PnL; the Kalman level adapts and is the remedy.

Pure numpy — no streamlit, no controller imports (MVC model layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_PHI_MIN = 1e-4
_PHI_MAX = 0.9999
_MIN_BARS = 5


# ---------------------------------------------------------------------------
# OU / AR(1) estimation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OUParams:
    """Discrete OU (= AR(1)) parameters estimated from closes (dt = 1 bar)."""

    phi: float            # AR(1) slope in (0, 1)
    mu: float             # mean level used by the filter (see mu_mode)
    sigma_eps: float      # residual (innovation) std per bar
    mu_window: float      # sample mean of the calibration window (kts.py choice)
    mu_ar1: float         # regression-implied mean c / (1 - phi)
    n_bars: int           # number of bars used for the fit

    @property
    def theta(self) -> float:
        """Mean-reversion speed per bar: phi = exp(-theta)."""
        return float(-np.log(self.phi))

    @property
    def half_life_bars(self) -> float:
        """Bars for a deviation to halve: ln(2) / theta."""
        return float(np.log(2.0) / self.theta)

    @property
    def sigma_stat(self) -> float:
        """Stationary std of the AR(1): sigma_eps / sqrt(1 - phi^2)."""
        return float(self.sigma_eps / np.sqrt(max(1.0 - self.phi**2, 1e-12)))


def estimate_ou_ar1(closes, *, mu_mode: str = "window") -> OUParams | None:
    """
    Fit X_{t+1} = c + phi X_t + eps on the given closes (OLS).

    mu_mode:
      - "window": mu = sample mean of the window (kts.py behaviour, robust)
      - "ar1"   : mu = c / (1 - phi) (regression-implied, noisy when phi ~ 1)

    Returns None when there are fewer than 5 finite closes or the fit fails.
    """
    if closes is None:
        return None
    y = np.asarray(closes, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < _MIN_BARS:
        return None
    try:
        x_lag, x_curr = y[:-1], y[1:]
        X = np.column_stack([np.ones_like(x_lag), x_lag])
        beta, *_ = np.linalg.lstsq(X, x_curr, rcond=None)
        c, phi = float(beta[0]), float(beta[1])
    except Exception:
        return None

    phi_clipped = float(np.clip(phi, _PHI_MIN, _PHI_MAX))
    if phi_clipped != phi:
        # The OLS intercept belongs to the unclipped slope; keep the residuals
        # centred by refitting it for the slope actually used.
        phi = phi_clipped
        c = float(np.mean(x_curr) - phi * np.mean(x_lag))
    resid = x_curr - (c + phi * x_lag)
    dof = max(resid.size - 2, 1)  # two fitted parameters (c, phi)
    sigma_eps = float(np.sqrt(np.sum(resid**2) / dof))
    if not np.isfinite(sigma_eps) or sigma_eps <= 0.0:
        sigma_eps = float(max(np.std(y) * 0.01, 1e-9))

    mu_window = float(np.mean(y))
    denom = 1.0 - phi
    mu_ar1 = float(c / denom) if abs(denom) > 1e-12 else mu_window
    if not np.isfinite(mu_ar1):
        mu_ar1 = mu_window

    mu = mu_ar1 if str(mu_mode).lower() == "ar1" else mu_window
    return OUParams(
        phi=phi,
        mu=mu,
        sigma_eps=sigma_eps,
        mu_window=mu_window,
        mu_ar1=mu_ar1,
        n_bars=int(y.size),
    )


# ---------------------------------------------------------------------------
# Noise lever (kts.py): [0, 100] -> observation-noise scale
# ---------------------------------------------------------------------------
def noise_lever_to_scale(lever_percent: float) -> float:
    """
    0   -> 0.1  (low R: trust prices, the level hugs the price)
    100 -> 1e8  (trust the OU model: once the P0 = R prior has been absorbed the
                 gain is ~1e-8 and the level is frozen at its running mean)
    Linear in between on [0.1, 10].
    """
    p = max(0.0, min(100.0, float(lever_percent))) / 100.0
    if p >= 1.0:
        return 1e8
    return 0.1 + (10.0 - 0.1) * p


# ---------------------------------------------------------------------------
# Scalar Kalman filter: state = OU mean level (fair value)
# ---------------------------------------------------------------------------
@dataclass
class KalmanOUState:
    """Mutable filter state; params (phi, mu, Q, R) are fixed at construction."""

    phi: float
    mu: float
    Q: float
    R: float
    x: float
    P: float

    @classmethod
    def from_params(cls, params: OUParams, *, obs_scale: float = 1.0) -> "KalmanOUState":
        Q = params.sigma_eps**2 * max(1.0 - params.phi**2, 1e-6)
        R = params.sigma_eps**2 * max(float(obs_scale), 0.01)
        return cls(phi=params.phi, mu=params.mu, Q=Q, R=R, x=params.mu, P=R)

    def predict(self) -> None:
        self.x = self.phi * self.x + (1.0 - self.phi) * self.mu
        self.P = self.phi**2 * self.P + self.Q

    def update(self, z: float) -> float:
        """Predict then measurement-update with observed price z; returns the gain."""
        self.predict()
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return float(K)

    def forecast(self, steps: int) -> np.ndarray:
        """Pure OU mean forecast 1..steps ahead (no new observations)."""
        k = np.arange(1, int(steps) + 1, dtype=float)
        return self.mu + (self.phi**k) * (self.x - self.mu)


@dataclass(frozen=True)
class KalmanRun:
    """Filter output over a close series."""

    x_filt: np.ndarray    # estimated mean level after each observation (posterior)
    x_pred: np.ndarray    # level expected BEFORE each observation (one-step prior)
    P_filt: np.ndarray    # posterior variance after each observation
    gain: np.ndarray      # Kalman gain used at each observation
    state: KalmanOUState  # final state (for live continuation / forecasting)


def run_kalman_ou(closes, params: OUParams, *, obs_scale: float = 1.0) -> KalmanRun:
    """Run the scalar OU Kalman filter over a close series."""
    y = np.asarray(closes, dtype=float).ravel()
    st = KalmanOUState.from_params(params, obs_scale=obs_scale)
    n = y.size
    x_filt = np.empty(n, dtype=float)
    x_pred = np.empty(n, dtype=float)
    P_filt = np.empty(n, dtype=float)
    gain = np.empty(n, dtype=float)
    for i in range(n):
        z = y[i]
        x_pred[i] = st.phi * st.x + (1.0 - st.phi) * st.mu
        if not np.isfinite(z):
            st.predict()  # missing observation: time update only
            gain[i] = 0.0
        else:
            gain[i] = st.update(z)
        x_filt[i] = st.x
        P_filt[i] = st.P
    return KalmanRun(x_filt=x_filt, x_pred=x_pred, P_filt=P_filt, gain=gain, state=st)


def forecast_with_bands(
    state: KalmanOUState,
    params: OUParams,
    *,
    steps: int,
    n_sd: float = 1.0,
) -> dict:
    """
    OU forecast of the PRICE started from the filtered level:
      z_T ~ N(x, P_eff) ,  z_{T+k} = mu + phi (z_{T+k-1} - mu) + sigma_eps eps
      mean_k = mu + phi^k (x - mu)
      var_k  = phi^{2k} P_eff  +  sigma_eps^2 (1 - phi^{2k}) / (1 - phi^2)
    with P_eff = min(P, sigma_stat^2): the start point cannot be more uncertain
    than the stationary law of the price (the P0 = R prior at lever 100 is never
    fully absorbed when phi ~ 1 and would otherwise swamp the band).
    """
    steps = int(steps)
    k = np.arange(1, steps + 1, dtype=float)
    phi2k = params.phi ** (2.0 * k)
    mean = state.mu + (params.phi**k) * (state.x - state.mu)
    one_minus_phi2 = max(1.0 - params.phi**2, 1e-12)
    innov_var = params.sigma_eps**2 * (1.0 - phi2k) / one_minus_phi2
    p_eff = min(max(state.P, 0.0), params.sigma_eps**2 / one_minus_phi2)
    var = phi2k * p_eff + innov_var
    sd = np.sqrt(np.maximum(var, 0.0))
    return {"mean": mean, "sd": sd, "upper": mean + n_sd * sd, "lower": mean - n_sd * sd}


# ---------------------------------------------------------------------------
# Mean-reversion band strategy (notebook rules) + backtest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandBacktest:
    positions: np.ndarray          # position held AFTER the decision at each bar
    pnl: np.ndarray                # mark-to-market PnL per bar (per unit)
    equity: np.ndarray             # cumulative PnL
    trades: list = field(default_factory=list)  # {index, price, action, side}
    stats: dict = field(default_factory=dict)


def backtest_bands(closes, center, upper, lower) -> BandBacktest:
    """
    Notebook trading rules, mark-to-market:
      pnl_t = pos_{t-1} * (P_t - P_{t-1})   (marked BEFORE the decision at t)
      flat : P_t > upper_t -> enter short ; P_t < lower_t -> enter long
      long : P_t >= center_t -> close ; short: P_t <= center_t -> close
    center/upper/lower may be scalars or arrays aligned with closes.
    """
    y = np.asarray(closes, dtype=float).ravel()
    n = y.size
    c = np.broadcast_to(np.asarray(center, dtype=float), (n,))
    u = np.broadcast_to(np.asarray(upper, dtype=float), (n,))
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,))

    positions = np.zeros(n, dtype=float)
    pnl = np.zeros(n, dtype=float)
    trades: list[dict] = []
    pos = 0.0
    entry_price = np.nan
    last_mark = np.nan  # last finite close: a missing bar marks nothing, the next one marks the gap

    for t in range(n):
        price = y[t]
        if not np.isfinite(price):
            positions[t] = pos
            continue
        if np.isfinite(last_mark):
            pnl[t] = pos * (price - last_mark)
        last_mark = price

        if pos == 0.0:
            if price > u[t]:
                pos = -1.0
                entry_price = price
                trades.append({"index": t, "price": float(price), "action": "entry", "side": "short"})
            elif price < lo[t]:
                pos = 1.0
                entry_price = price
                trades.append({"index": t, "price": float(price), "action": "entry", "side": "long"})
        elif pos > 0.0:
            if price >= c[t]:
                trades.append(
                    {
                        "index": t,
                        "price": float(price),
                        "action": "exit",
                        "side": "long",
                        "trade_pnl": float(price - entry_price),
                    }
                )
                pos = 0.0
                entry_price = np.nan
        else:  # pos < 0
            if price <= c[t]:
                trades.append(
                    {
                        "index": t,
                        "price": float(price),
                        "action": "exit",
                        "side": "short",
                        "trade_pnl": float(entry_price - price),
                    }
                )
                pos = 0.0
                entry_price = np.nan
        positions[t] = pos

    equity = np.cumsum(pnl)
    stats = _backtest_stats(equity, trades, positions)
    return BandBacktest(positions=positions, pnl=pnl, equity=equity, trades=trades, stats=stats)


def _backtest_stats(equity: np.ndarray, trades: list, positions: np.ndarray) -> dict:
    exits = [t for t in trades if t.get("action") == "exit"]
    entries = [t for t in trades if t.get("action") == "entry"]
    wins = sum(1 for t in exits if float(t.get("trade_pnl", 0.0)) > 0.0)
    total = float(equity[-1]) if equity.size else 0.0
    if equity.size:
        running_max = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity - running_max))
    else:
        max_dd = 0.0
    open_pos = float(positions[-1]) if positions.size else 0.0
    return {
        "total_pnl": total,
        "n_entries": int(len(entries)),
        "n_round_trips": int(len(exits)),
        "hit_rate": float(wins / len(exits)) if exits else None,
        "max_drawdown": max_dd,
        "open_position": open_pos,
    }


# ---------------------------------------------------------------------------
# Full pipeline: calibrate on the head window, filter everything, backtest
# out-of-sample with BOTH centers (frozen mu-hat vs adaptive Kalman level).
# ---------------------------------------------------------------------------
def run_ou_kalman_pipeline(
    closes,
    *,
    calib_window: int,
    noise_lever: float = 50.0,
    band_k: float = 1.0,
    forecast_steps: int = 5,
    mu_mode: str = "window",
    calib_anchor: str = "head",
) -> dict | None:
    """
    calib_anchor="head" (notebook): estimate (phi, mu, sigma) on the FIRST
    `calib_window` bars, then trade the remaining bars out-of-sample.
    calib_anchor="tail" (kts.py live behaviour): estimate on the LAST
    `calib_window` bars — parameters track the latest regime, but every bar
    is then in-sample, so no backtest is produced.

    The Kalman filter runs over the whole series with those parameters;
    both band variants share sigma_stat and k, only the CENTER differs:
      - "static":  frozen mu-hat from the calibration window (the trap)
      - "kalman":  the level expected before z_t, x_{t|t-1} (the remedy); the
        posterior x_{t|t} already contains K (z_t - x_{t|t-1}) and would widen
        the effective band by 1/(1-K).

    Returns None when the series is too short to calibrate.
    """
    y = np.asarray(closes, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = y.size
    w = int(max(_MIN_BARS, min(int(calib_window), n)))
    anchor = "tail" if str(calib_anchor).lower() == "tail" else "head"
    calib_slice = y[-w:] if anchor == "tail" else y[:w]
    params = estimate_ou_ar1(calib_slice, mu_mode=mu_mode)
    if params is None:
        return None

    obs_scale = noise_lever_to_scale(noise_lever)
    run = run_kalman_ou(y, params, obs_scale=obs_scale)
    sigma_stat = params.sigma_stat
    band = float(band_k) * sigma_stat

    fc = forecast_with_bands(run.state, params, steps=int(forecast_steps), n_sd=1.0)

    out: dict = {
        "params": params,
        "obs_scale": obs_scale,
        "kalman": run,
        "sigma_stat": sigma_stat,
        "band_halfwidth": band,
        "forecast": fc,
        "calib_window": w,
        "calib_anchor": anchor,
        "n_bars": n,
        "backtests": {},
    }

    # Out-of-sample slice (notebook phase 2). Needs at least 2 bars to mark
    # PnL, and only exists when the calibration is anchored at the head.
    if anchor == "head" and n - w >= 2:
        oos = y[w:]
        static_bt = backtest_bands(oos, params.mu, params.mu + band, params.mu - band)
        center_kf = run.x_pred[w:]
        kalman_bt = backtest_bands(oos, center_kf, center_kf + band, center_kf - band)
        out["backtests"] = {"static": static_bt, "kalman": kalman_bt}
        out["oos_start"] = w
    return out


__all__ = [
    "OUParams",
    "KalmanOUState",
    "KalmanRun",
    "BandBacktest",
    "estimate_ou_ar1",
    "noise_lever_to_scale",
    "run_kalman_ou",
    "forecast_with_bands",
    "backtest_bands",
    "run_ou_kalman_pipeline",
]
