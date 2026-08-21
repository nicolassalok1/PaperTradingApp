"""
OU mean-level Kalman filter + band backtest — unit tests.

Oracle policy (repo red line: never a code-produced value as expected):
  - AR(1) recovery      -> the TRUE parameters of a hand-rolled synthetic path;
  - filter variance/gain -> the closed-form fixed point of the scalar Riccati
                            equation, re-derived in the test;
  - filter level        -> hand-computed exact fractions for a 3-step recursion,
                            and the closed-form fixed point under constant input;
  - forecast            -> the OU forecast formula evaluated with literals;
  - backtest            -> a tiny path whose trades and PnL are computed by hand.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from app.model.volatility_models.kalman.ou_kalman import (
    KalmanOUState,
    backtest_bands,
    estimate_ou_ar1,
    forecast_with_bands,
    noise_lever_to_scale,
    run_kalman_ou,
    run_ou_kalman_pipeline,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# helpers (test-local, independent of the model implementation)
# ---------------------------------------------------------------------------
def _hand_rolled_ar1(n: int, phi: float, mu: float, sigma_eps: float, x0: float, seed: int) -> np.ndarray:
    """X_{t+1} = mu + phi (X_t - mu) + sigma_eps * eps  (test-local generator)."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma_eps * rng.standard_normal()
    return x


def _riccati_fixed_point(phi: float, Q: float, R: float) -> tuple[float, float, float]:
    """
    Steady state of the scalar Kalman recursion (re-derived independently):
      prior     M = phi^2 * (M R / (M + R)) + Q
      =>        M^2 + M (R (1 - phi^2) - Q) - Q R = 0   (positive root)
      posterior P = M R / (M + R),  gain K = M / (M + R)
    """
    b = R * (1.0 - phi**2) - Q
    M = (-b + math.sqrt(b * b + 4.0 * Q * R)) / 2.0
    K = M / (M + R)
    return M, M * R / (M + R), K


# ---------------------------------------------------------------------------
# AR(1) / OU estimation
# ---------------------------------------------------------------------------
def test_ar1_recovers_true_parameters_of_synthetic_path():
    phi_true, mu_true, sig_true = 0.9, 100.0, 0.5
    y = _hand_rolled_ar1(4000, phi_true, mu_true, sig_true, x0=mu_true, seed=7)
    p = estimate_ou_ar1(y)
    assert p is not None
    assert abs(p.phi - phi_true) < 0.03
    assert abs(p.mu_window - mu_true) < 0.5
    assert abs(p.sigma_eps - sig_true) / sig_true < 0.10
    # derived quantities against their definitions (literal formulas)
    assert p.theta == pytest.approx(-math.log(p.phi))
    assert p.half_life_bars == pytest.approx(math.log(2.0) / -math.log(p.phi))
    assert p.sigma_stat == pytest.approx(p.sigma_eps / math.sqrt(1.0 - p.phi**2))


def test_ar1_regression_implied_mean_exact_on_noiseless_recursion():
    # deterministic x_{t+1} = 50 + 0.5 x_t has fixed point 100: OLS is exact.
    x = [200.0]
    for _ in range(11):
        x.append(50.0 + 0.5 * x[-1])
    p = estimate_ou_ar1(np.array(x), mu_mode="ar1")
    assert p is not None
    assert p.phi == pytest.approx(0.5, abs=1e-8)
    assert p.mu_ar1 == pytest.approx(100.0, abs=1e-6)
    assert p.mu == pytest.approx(p.mu_ar1)  # mu_mode="ar1" is honoured


def test_ar1_sigma_eps_is_ols_residual_std_with_two_parameter_ddof():
    # OLS by hand on y = [0, 1, 3, 2, 4, 3]:
    #   x_lag = [0,1,3,2,4] (mean 2), x_curr = [1,3,2,4,3] (mean 2.6)
    #   Sxx = 4+1+1+0+4 = 10 ; Sxy = 3.2-0.4-0.6+0+0.8 = 3.0  ->  phi = 0.3, c = 2.6-0.3*2 = 2.0
    #   resid = x_curr - 2 - 0.3 x_lag = [-1, 0.7, -0.9, 1.4, -0.2] ; SS = 4.30
    #   two fitted parameters on 5 pairs -> sigma^2 = 4.30 / (5-2)
    p = estimate_ou_ar1([0.0, 1.0, 3.0, 2.0, 4.0, 3.0], mu_mode="ar1")
    assert p is not None
    assert p.phi == pytest.approx(0.3, abs=1e-12)
    assert p.sigma_eps == pytest.approx(math.sqrt(4.30 / 3.0), abs=1e-12)
    assert p.mu_ar1 == pytest.approx(2.0 / 0.7, abs=1e-12)
    assert p.mu_window == pytest.approx(13.0 / 6.0, abs=1e-12)


def test_ar1_residuals_are_recentred_after_phi_clip():
    # Alternating y = [100,110,100,110,100,110]: OLS gives phi = -1, c = 210.
    # phi is clipped to 1e-4; the intercept must be refitted for the clipped slope
    # (c = mean(x_curr) - phi mean(x_lag) = 106 - 1e-4*104 = 105.9896), otherwise the
    # residuals carry the bias (phi_ols - phi_clip) * price ~ 105 -> sigma_eps ~ 105.
    # By hand with the refitted c:
    #   resid(lag 100 -> 110) = 110 - 105.9896 - 0.0100 =  4.0004   (x3)
    #   resid(lag 110 -> 100) = 100 - 105.9896 - 0.0110 = -6.0006   (x2)
    #   sigma^2 = (3*4.0004^2 + 2*6.0006^2) / (5-2)
    p = estimate_ou_ar1([100.0, 110.0, 100.0, 110.0, 100.0, 110.0], mu_mode="ar1")
    assert p is not None
    assert p.phi == pytest.approx(1e-4)
    assert p.sigma_eps == pytest.approx(math.sqrt((3 * 4.0004**2 + 2 * 6.0006**2) / 3.0), rel=1e-9)
    assert p.mu_ar1 == pytest.approx(105.9896 / (1.0 - 1e-4), rel=1e-9)


def test_ar1_needs_five_finite_bars():
    assert estimate_ou_ar1([1.0, 2.0, 3.0, 4.0]) is None
    assert estimate_ou_ar1(np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0])) is None
    assert estimate_ou_ar1(None) is None


# ---------------------------------------------------------------------------
# Noise lever (spec from kts.py: 0 -> 0.1, 100 -> 1e8, linear on [0.1, 10])
# ---------------------------------------------------------------------------
def test_noise_lever_endpoints_and_midpoint():
    assert noise_lever_to_scale(0) == pytest.approx(0.1)
    assert noise_lever_to_scale(50) == pytest.approx(0.1 + (10.0 - 0.1) * 0.5)  # = 5.05
    assert noise_lever_to_scale(100) == pytest.approx(1e8)
    assert noise_lever_to_scale(-5) == pytest.approx(0.1)  # clamped
    assert noise_lever_to_scale(400) == pytest.approx(1e8)


# ---------------------------------------------------------------------------
# Kalman recursion
# ---------------------------------------------------------------------------
def test_kalman_three_steps_match_hand_computed_fractions():
    # phi=0.5, mu=100, sigma_eps=1, scale=1  =>  Q=0.75, R=1, x0=100, P0=R=1
    # z = [100, 101, 99]; recursion done by hand with exact fractions:
    #  t1: x=100,               P=1/2
    #  t2: x=100 + 7/15,        P=7/15
    #  t3: x=100 + 7/30 - 481/840,  P=13/28
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    params = OUParams(phi=0.5, mu=100.0, sigma_eps=1.0, mu_window=100.0, mu_ar1=100.0, n_bars=60)
    run = run_kalman_ou(np.array([100.0, 101.0, 99.0]), params, obs_scale=1.0)
    assert run.x_filt[0] == pytest.approx(100.0, abs=1e-12)
    assert run.P_filt[0] == pytest.approx(0.5, abs=1e-12)
    assert run.x_filt[1] == pytest.approx(100.0 + 7.0 / 15.0, abs=1e-12)
    assert run.P_filt[1] == pytest.approx(7.0 / 15.0, abs=1e-12)
    assert run.x_filt[2] == pytest.approx(100.0 + 7.0 / 30.0 - 481.0 / 840.0, abs=1e-12)
    assert run.P_filt[2] == pytest.approx(13.0 / 28.0, abs=1e-12)


def test_kalman_exposes_one_step_prior_level():
    # Same recursion as above; the prior at t is phi x_{t-1|t-1} + (1-phi) mu, i.e. the
    # level expected BEFORE seeing z_t:  t1: 100 ; t2: 100 ; t3: 0.5 (100 + 7/15) + 50.
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    params = OUParams(phi=0.5, mu=100.0, sigma_eps=1.0, mu_window=100.0, mu_ar1=100.0, n_bars=60)
    run = run_kalman_ou(np.array([100.0, 101.0, 99.0]), params, obs_scale=1.0)
    assert list(run.x_pred) == pytest.approx([100.0, 100.0, 100.0 + 7.0 / 30.0], abs=1e-12)


def test_kalman_variance_and_gain_converge_to_riccati_fixed_point():
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    phi, sigma_eps, scale = 0.8, 2.0, 2.0
    params = OUParams(phi=phi, mu=100.0, sigma_eps=sigma_eps, mu_window=100.0, mu_ar1=100.0, n_bars=60)
    Q = sigma_eps**2 * (1.0 - phi**2)
    R = sigma_eps**2 * scale
    _, P_star, K_star = _riccati_fixed_point(phi, Q, R)

    z = 100.0 + 5.0 * np.cos(np.arange(3000))  # any bounded input
    run = run_kalman_ou(z, params, obs_scale=scale)
    assert run.P_filt[-1] == pytest.approx(P_star, rel=1e-9)
    assert run.gain[-1] == pytest.approx(K_star, rel=1e-9)


def test_kalman_trust_prices_fixed_point_under_constant_input():
    # scale=0.01 (the R floor): the level must converge to the closed-form
    # fixed point  x* = [K z0 + (1-K)(1-phi) mu] / (1 - (1-K) phi).
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    phi, sigma_eps, scale, mu, z0 = 0.6, 1.0, 0.01, 100.0, 110.0
    params = OUParams(phi=phi, mu=mu, sigma_eps=sigma_eps, mu_window=mu, mu_ar1=mu, n_bars=60)
    Q = sigma_eps**2 * (1.0 - phi**2)
    R = sigma_eps**2 * scale
    _, _, K = _riccati_fixed_point(phi, Q, R)
    x_star = (K * z0 + (1.0 - K) * (1.0 - phi) * mu) / (1.0 - (1.0 - K) * phi)

    run = run_kalman_ou(np.full(500, z0), params, obs_scale=scale)
    assert run.x_filt[-1] == pytest.approx(x_star, abs=1e-8)
    # trust-prices means the level hugs the price: within 1% of the gap to mu
    assert abs(run.x_filt[-1] - z0) < 0.01 * abs(z0 - mu)


def test_kalman_missing_observation_is_time_update_only():
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    params = OUParams(phi=0.5, mu=100.0, sigma_eps=1.0, mu_window=100.0, mu_ar1=100.0, n_bars=60)
    run = run_kalman_ou(np.array([100.0, np.nan]), params, obs_scale=1.0)
    # after t1 (above): x=100, P=1/2 ; NaN at t2 -> predict only:
    #   x = 0.5*100 + 0.5*100 = 100 ; P = 0.25*0.5 + 0.75 = 0.875 ; gain 0
    assert run.x_filt[1] == pytest.approx(100.0, abs=1e-12)
    assert run.P_filt[1] == pytest.approx(0.875, abs=1e-12)
    assert run.gain[1] == 0.0


# ---------------------------------------------------------------------------
# OU forecast
# ---------------------------------------------------------------------------
def test_forecast_band_matches_monte_carlo_of_ou_price_from_uncertain_start():
    # The band is the OU price forecast started from the filtered level, whose
    # uncertainty is P: z_T ~ N(x, P), z_{T+k} = mu + phi (z_{T+k-1} - mu) + sigma eps.
    # Oracle: a seeded Monte-Carlo of exactly that process (no formula from the code).
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    phi, mu, sigma, x0, P0 = 0.5, 100.0, 1.0, 104.0, 0.4
    params = OUParams(phi=phi, mu=mu, sigma_eps=sigma, mu_window=mu, mu_ar1=mu, n_bars=60)
    st = KalmanOUState(phi=phi, mu=mu, Q=sigma**2 * (1 - phi**2), R=1.0, x=x0, P=P0)
    fc = forecast_with_bands(st, params, steps=3, n_sd=1.0)

    rng = np.random.default_rng(2026)
    z = x0 + math.sqrt(P0) * rng.standard_normal(400_000)
    for k in range(3):
        z = mu + phi * (z - mu) + sigma * rng.standard_normal(z.size)
        assert fc["mean"][k] == pytest.approx(z.mean(), abs=0.01)
        assert fc["sd"][k] == pytest.approx(z.std(), rel=0.01)
        assert fc["upper"][k] == pytest.approx(fc["mean"][k] + fc["sd"][k], abs=1e-12)
        assert fc["lower"][k] == pytest.approx(fc["mean"][k] - fc["sd"][k], abs=1e-12)


def test_forecast_band_caps_start_uncertainty_at_stationary_variance():
    # A start point cannot be more uncertain than the stationary law of the price:
    # with P >> sigma_stat^2 (the P0 = R prior at lever 100, never fully absorbed when
    # phi ~ 1) the band must collapse to exactly sigma_stat at every horizon:
    #   var_k = phi^{2k} sigma_stat^2 + sigma^2 (1 - phi^{2k}) / (1 - phi^2) = sigma_stat^2.
    from app.model.volatility_models.kalman.ou_kalman import OUParams

    phi, sigma = 0.5, 1.0
    sigma_stat = sigma / math.sqrt(1.0 - phi**2)
    params = OUParams(phi=phi, mu=100.0, sigma_eps=sigma, mu_window=100.0, mu_ar1=100.0, n_bars=60)
    st = KalmanOUState(phi=phi, mu=100.0, Q=0.75, R=1e8, x=104.0, P=1e6)
    fc = forecast_with_bands(st, params, steps=4, n_sd=1.0)
    assert list(fc["sd"]) == pytest.approx([sigma_stat] * 4, rel=1e-12)


def test_state_forecast_matches_ou_decay_formula():
    st = KalmanOUState(phi=0.9, mu=50.0, Q=1.0, R=1.0, x=60.0, P=1.0)
    fc = st.forecast(3)
    assert fc[0] == pytest.approx(50.0 + 0.9 * 10.0)
    assert fc[1] == pytest.approx(50.0 + 0.81 * 10.0)
    assert fc[2] == pytest.approx(50.0 + 0.729 * 10.0)


# ---------------------------------------------------------------------------
# Band strategy backtest (notebook rules), hand-computed paths
# ---------------------------------------------------------------------------
def test_backtest_short_then_long_round_trips_hand_computed():
    closes = [100.0, 103.0, 101.0, 100.0, 97.0, 99.0, 100.0, 100.0]
    bt = backtest_bands(closes, center=100.0, upper=102.0, lower=98.0)
    # hand-computed: short@103 exits @100 (+3), long@97 exits @100 (+3)
    assert list(bt.positions) == [0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert list(bt.pnl) == [0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0]
    assert bt.equity[-1] == pytest.approx(6.0)
    assert bt.stats["n_entries"] == 2
    assert bt.stats["n_round_trips"] == 2
    assert bt.stats["hit_rate"] == pytest.approx(1.0)
    assert bt.stats["max_drawdown"] == pytest.approx(0.0)
    assert bt.stats["open_position"] == 0.0
    sides = [(t["action"], t["side"], t["index"]) for t in bt.trades]
    assert sides == [("entry", "short", 1), ("exit", "short", 3), ("entry", "long", 4), ("exit", "long", 6)]


def test_backtest_adverse_move_marks_drawdown():
    closes = [100.0, 103.0, 104.0, 102.5, 100.0]
    bt = backtest_bands(closes, center=100.0, upper=102.0, lower=98.0)
    # short@103; equity path 0,0,-1,0.5,3 -> max drawdown = -1
    assert list(bt.pnl) == [0.0, 0.0, -1.0, 1.5, 2.5]
    assert bt.equity[-1] == pytest.approx(3.0)
    assert bt.stats["max_drawdown"] == pytest.approx(-1.0)
    assert bt.stats["hit_rate"] == pytest.approx(1.0)


def test_backtest_position_left_open_has_no_exit():
    closes = [100.0, 103.0, 104.0]
    bt = backtest_bands(closes, center=100.0, upper=102.0, lower=98.0)
    assert bt.stats["open_position"] == -1.0
    assert bt.stats["n_entries"] == 1
    assert bt.stats["n_round_trips"] == 0
    assert bt.stats["hit_rate"] is None  # no round trip: undefined, and strict-JSON safe


def test_backtest_marks_pnl_across_a_missing_close():
    # short@103 ; NaN bar (no decision, nothing marked) ; 110 marks the whole 103->110
    # move against the short (-7) ; 100 closes it (+10). Hand: equity -7 then +3.
    closes = [100.0, 103.0, np.nan, 110.0, 100.0]
    bt = backtest_bands(closes, center=100.0, upper=102.0, lower=98.0)
    assert list(bt.positions) == [0.0, -1.0, -1.0, -1.0, 0.0]
    assert list(bt.pnl) == pytest.approx([0.0, 0.0, 0.0, -7.0, 10.0])
    assert bt.equity[-1] == pytest.approx(3.0)
    assert bt.stats["max_drawdown"] == pytest.approx(-7.0)
    assert bt.trades[-1]["trade_pnl"] == pytest.approx(3.0)


def test_backtest_moving_center_uses_per_bar_bands():
    # Same closes, two centers. Fixed center 100 (bands 99/101): 103 breaks the
    # upper band at t1 -> one short entry. Moving center [100, 104, 104] with
    # bands +/-1: at t1 price 103 is inside [103, 105] (lower bound requires
    # STRICTLY below) -> no trade at all. Proves the arrays are read per bar.
    closes = [100.0, 103.0, 103.0]
    fixed = backtest_bands(closes, 100.0, 101.0, 99.0)
    assert fixed.stats["n_entries"] == 1
    center = np.array([100.0, 104.0, 104.0])
    moving = backtest_bands(closes, center, center + 1.0, center - 1.0)
    assert moving.stats["n_entries"] == 0
    assert moving.equity[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full pipeline (structure + determinism, no strategy-performance claims)
# ---------------------------------------------------------------------------
def test_pipeline_head_anchor_produces_out_of_sample_backtests():
    y = _hand_rolled_ar1(300, 0.9, 100.0, 0.5, x0=103.0, seed=11)
    res = run_ou_kalman_pipeline(y, calib_window=60, noise_lever=50.0, band_k=1.0, forecast_steps=5)
    assert res is not None
    assert res["calib_anchor"] == "head"
    assert res["oos_start"] == 60
    assert res["params"].n_bars == 60
    assert res["kalman"].x_filt.shape == (300,)
    assert set(res["backtests"].keys()) == {"static", "kalman"}
    for bt in res["backtests"].values():
        assert bt.equity.shape == (240,)
    assert len(res["forecast"]["mean"]) == 5
    # deterministic: same inputs, same outputs
    res2 = run_ou_kalman_pipeline(y, calib_window=60, noise_lever=50.0, band_k=1.0, forecast_steps=5)
    assert np.array_equal(res["kalman"].x_filt, res2["kalman"].x_filt)
    assert res["backtests"]["kalman"].stats == res2["backtests"]["kalman"].stats


def test_pipeline_kalman_center_is_the_prior_level_and_still_trades_at_lever_zero():
    # The decision at t compares z_t with the level expected BEFORE z_t (the prior):
    # centring on the posterior x_{t|t} would subtract K (z_t - prior) from the
    # deviation first and widen the band by 1/(1-K) -- at lever 0 (K ~ 0.7) the
    # "remedy" variant would stop trading altogether on a textbook OU path.
    y = _hand_rolled_ar1(400, 0.9, 100.0, 1.0, x0=100.0, seed=21)
    res = run_ou_kalman_pipeline(y, calib_window=60, noise_lever=0.0, band_k=1.0)
    assert res is not None
    w = res["oos_start"]
    kalman_bt = res["backtests"]["kalman"]
    assert kalman_bt.stats["n_entries"] > 0
    # wiring: the Kalman bands are centred on x_pred over the out-of-sample slice
    band = res["band_halfwidth"]
    center = res["kalman"].x_pred[w:]
    ref = backtest_bands(y[w:], center, center + band, center - band)
    assert list(kalman_bt.positions) == list(ref.positions)


def test_pipeline_tail_anchor_disables_backtests():
    y = _hand_rolled_ar1(300, 0.9, 100.0, 0.5, x0=100.0, seed=3)
    res = run_ou_kalman_pipeline(y, calib_window=60, calib_anchor="tail")
    assert res is not None
    assert res["calib_anchor"] == "tail"
    assert res["backtests"] == {}
    assert "oos_start" not in res


def test_pipeline_too_short_returns_none():
    assert run_ou_kalman_pipeline(np.array([1.0, 2.0, 3.0]), calib_window=60) is None


# ---------------------------------------------------------------------------
# Controller bridge (JSON-safety + tick path), no network involved
# ---------------------------------------------------------------------------
def test_controller_run_pipeline_is_json_safe_and_complete():
    from app.controller import kalman_controller as kc

    y = _hand_rolled_ar1(200, 0.85, 50.0, 0.3, x0=51.0, seed=5)
    out = kc.run_pipeline({"closes": y.tolist(), "calib_window": 50, "noise_lever": 30.0, "band_k": 1.5})
    assert out["success"] is True
    json.dumps(out, allow_nan=False)  # strict JSON: plain types, no NaN/inf anywhere
    assert len(out["x_filt"]) == 200
    assert len(out["x_pred"]) == 200
    assert out["oos_start"] == 50
    assert {"phi", "mu", "sigma_eps", "half_life_bars", "sigma_stat"} <= set(out["params"].keys())
    assert {"static", "kalman"} == set(out["backtests"].keys())
    assert out["band_halfwidth"] == pytest.approx(1.5 * out["params"]["sigma_stat"])


def test_controller_run_pipeline_without_round_trip_is_strict_json():
    from app.controller import kalman_controller as kc

    # 3 flat out-of-sample bars: no trade can close -> hit_rate undefined, must not be NaN
    y = np.concatenate([_hand_rolled_ar1(60, 0.8, 100.0, 0.5, x0=100.0, seed=8), np.full(3, 100.0)])
    out = kc.run_pipeline({"closes": y.tolist(), "calib_window": 60})
    assert out["success"] is True
    json.dumps(out, allow_nan=False)
    assert out["backtests"]["static"]["stats"]["hit_rate"] is None


def test_controller_run_pipeline_honours_explicit_zero_arguments():
    from app.controller import kalman_controller as kc

    y = _hand_rolled_ar1(120, 0.8, 100.0, 0.5, x0=100.0, seed=12)
    out = kc.run_pipeline({"closes": y.tolist(), "calib_window": 60, "band_k": 0.0, "forecast_steps": 0})
    assert out["success"] is True
    assert out["band_halfwidth"] == 0.0
    assert out["forecast"]["mean"] == []


def test_controller_run_pipeline_rejects_missing_or_bad_closes():
    from app.controller import kalman_controller as kc

    assert kc.run_pipeline(None)["success"] is False
    assert kc.run_pipeline({})["success"] is False
    assert kc.run_pipeline({"closes": ["a", "b"]})["success"] is False


def test_controller_kalman_step_moves_between_prediction_and_price():
    from app.controller import kalman_controller as kc

    y = _hand_rolled_ar1(100, 0.9, 100.0, 0.5, x0=100.0, seed=9)
    out = kc.run_pipeline({"closes": y.tolist(), "calib_window": 100, "noise_lever": 50.0})
    state = out["state"]
    x_old, phi, mu = float(state["x"]), float(state["phi"]), float(state["mu"])
    z = x_old + 5.0
    step = kc.kalman_step(state, z, forecast_steps=3)
    assert step["success"] is True
    x_pred = phi * x_old + (1.0 - phi) * mu  # model-free bracket: pred < new x < z
    assert x_pred < step["x"] < z
    assert len(step["forecast_mean"]) == 3
    json.dumps(step)


def test_controller_kalman_step_returns_the_prior_level_it_decided_against():
    from app.controller import kalman_controller as kc

    # phi=0.5, mu=100, x=104: the level expected before the tick is 0.5*104 + 50 = 102
    state = {"phi": 0.5, "mu": 100.0, "Q": 0.75, "R": 1.0, "x": 104.0, "P": 0.4}
    step = kc.kalman_step(state, 103.0)
    assert step["success"] is True
    assert step["x_pred"] == pytest.approx(102.0, abs=1e-12)
    assert 102.0 < step["x"] < 103.0


@pytest.mark.parametrize(
    "bad",
    [
        {"x": float("nan")},
        {"P": -1.0},
        {"R": 0.0},
        {"phi": 1.5},
    ],
)
def test_controller_kalman_step_rejects_invalid_state(bad):
    from app.controller import kalman_controller as kc

    state = {"phi": 0.9, "mu": 100.0, "Q": 1.0, "R": 1.0, "x": 100.0, "P": 1.0, **bad}
    out = kc.kalman_step(state, 101.0)
    assert out["success"] is False


def test_controller_load_history_rejects_empty_ticker_without_network():
    from app.controller import kalman_controller as kc

    out = kc.load_history("", "5 min")
    assert out["success"] is False
    assert kc.bar_choice_labels()  # non-empty catalogue
    assert "1 jour" in kc.BAR_CHOICES


def test_controller_load_history_drops_non_finite_closes_and_names_the_last_bar(monkeypatch):
    import pandas as pd

    from app.controller import kalman_controller as kc
    from app.model.market_data import market_data as md

    dates = pd.date_range("2026-08-21 13:30", periods=8, freq="5min", tz="UTC")
    closes = [100.0, 101.0, np.inf, 102.0, np.nan, 103.0, 104.0, 105.0]
    df = pd.DataFrame({"Date": dates, "Open": closes, "High": closes, "Low": closes, "Close": closes})
    monkeypatch.setattr(md, "fetch_intraday_ohlc", lambda *a, **k: df)

    out = kc.load_history("AAPL", "5 min")
    assert out["success"] is True
    assert [b["c"] for b in out["bars"]] == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert all(np.isfinite([b[k] for b in out["bars"] for k in ("o", "h", "l", "c")]))
    assert out["bars"][-1]["t"].endswith("+00:00")  # timezone is explicit, not naive
    assert "2026-08-21T14:05:00+00:00" in out["message"]  # the user sees how fresh the data is


def test_controller_latest_price_is_none_without_alpaca_keys(monkeypatch):
    # No keys: the Stooq path would serve a disk-cached close of unknown age as a
    # "spot". That is not a live price; the controller must say so (None) instead.
    from app.controller import kalman_controller as kc
    from app.model.market_data import market_data as md

    monkeypatch.setattr(md, "_alpaca_credentials", lambda: (None, None, "https://paper-api.alpaca.markets"))
    monkeypatch.setattr(md, "_fetch_stooq_spot", lambda sym: 123.45)
    assert kc.latest_price("AAPL") is None


def test_model_fetch_intraday_guard_empty_symbol_no_network():
    from app.model.market_data.market_data import fetch_intraday_ohlc

    df = fetch_intraday_ohlc("")
    assert df is not None and df.empty


def test_model_fetch_intraday_daily_path_refreshes_a_stale_cache(monkeypatch):
    import pandas as pd

    from app.model.market_data import market_data as md

    def _frame(last_day: str) -> pd.DataFrame:
        d = pd.date_range(end=last_day, periods=30, freq="B")
        return pd.DataFrame({"Date": d, "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 0})

    today = pd.Timestamp.today().normalize()
    stale = _frame((today - pd.Timedelta(days=200)).strftime("%Y-%m-%d"))
    fresh = _frame(today.strftime("%Y-%m-%d"))
    monkeypatch.setattr(md, "fetch_ohlc_history", lambda *a, **k: stale)
    monkeypatch.setattr(md, "_fetch_yahoo_ohlc", lambda *a, **k: fresh)
    assert md.fetch_intraday_ohlc("AAPL", interval="1d")["Date"].iloc[-1] == fresh["Date"].iloc[-1]

    # Yahoo unavailable: the stale series is still better than nothing
    monkeypatch.setattr(md, "_fetch_yahoo_ohlc", lambda *a, **k: pd.DataFrame())
    assert md.fetch_intraday_ohlc("AAPL", interval="1d")["Date"].iloc[-1] == stale["Date"].iloc[-1]

    # a fresh cache is not refetched
    calls = []
    monkeypatch.setattr(md, "fetch_ohlc_history", lambda *a, **k: fresh)
    monkeypatch.setattr(md, "_fetch_yahoo_ohlc", lambda *a, **k: calls.append(1) or pd.DataFrame())
    md.fetch_intraday_ohlc("AAPL", interval="1d")
    assert calls == []
