import math

import numpy as np
import pandas as pd

from app.model.options.core import pricing_lib
from app.model.options.core.trees import (
    price_barrier_vanilla,
    price_lookback_fixed_mc,
    price_lookback_mc,
    simulate_gbm_paths,
)
from app.model.options.engines.black_scholes import price_call_spread, price_put_spread
from app.model.options.core.shared import get_cached_iv_for
from app.model.options.engines.crr import price_american_crr


def test_american_crr_discount_uses_r_not_r_minus_q():
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.02
    T = 1.0
    sigma = 0.2
    steps = 1

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    payoff_up = max(S0 * u - K, 0.0)
    payoff_dn = max(S0 * d - K, 0.0)

    expected = math.exp(-r * dt) * (p * payoff_up + (1.0 - p) * payoff_dn)
    got = price_american_crr(S0=S0, K=K, r=r, q=q, T=T, sigma=sigma, steps=steps, option_type="call")

    assert abs(got - expected) < 1e-12


def test_view_european_payoff_matches_intrinsic_grid():
    s0 = 100.0
    K = 110.0
    T = 0.5
    r = 0.03
    q = 0.01
    sigma = 0.2

    view = pricing_lib.view_european(s0, K, option_type="call", r=r, q=q, sigma=sigma, T=T)
    s_grid = np.asarray(view["s_grid"], dtype=float)
    payoff = np.asarray(view["payoff"], dtype=float)

    expected_payoff = np.maximum(s_grid - K, 0.0)
    assert np.allclose(payoff, expected_payoff, atol=1e-12, rtol=0.0)


def test_get_cached_iv_for_accepts_iv_column_and_type_filter():
    df = pd.DataFrame(
        [
            {"K": 100.0, "T": 1.0, "iv": 0.25, "type": "call"},
            {"K": 100.0, "T": 1.0, "iv": 0.30, "type": "put"},
        ]
    )

    iv_call = get_cached_iv_for(df, 100.0, 1.0, "call", k_tol=0.01, t_tol=0.01, ticker="AAPL")
    iv_put = get_cached_iv_for(df, 100.0, 1.0, "put", k_tol=0.01, t_tol=0.01, ticker="AAPL")

    assert abs(float(iv_call) - 0.25) < 1e-12
    assert abs(float(iv_put) - 0.30) < 1e-12


def test_price_lookback_mc_matches_path_extremes_definition():
    params = dict(S0=100.0, r=0.02, q=0.0, T=0.5, sigma=0.25, steps=8, n_paths=64, seed=123)
    paths = simulate_gbm_paths(**params)
    disc = math.exp(-params["r"] * params["T"])

    min_path = paths.min(axis=1)
    max_path = paths.max(axis=1)
    terminal = paths[:, -1]

    expected_call = float(disc * np.maximum(terminal - min_path, 0.0).mean())
    expected_put = float(disc * np.maximum(max_path - terminal, 0.0).mean())

    price_call = price_lookback_mc(option_type="call", **params)
    price_put = price_lookback_mc(option_type="put", **params)

    assert math.isclose(price_call, expected_call, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_put, expected_put, rel_tol=0.0, abs_tol=1e-12)


def test_price_lookback_fixed_mc_uses_max_for_call_and_min_for_put():
    params = dict(
        S0=95.0, K=90.0, r=0.015, q=0.0, T=1.2, sigma=0.3, steps=10, n_paths=128, seed=321
    )
    paths = simulate_gbm_paths(
        S0=params["S0"],
        r=params["r"],
        q=params["q"],
        sigma=params["sigma"],
        T=params["T"],
        steps=params["steps"],
        n_paths=params["n_paths"],
        seed=params["seed"],
    )
    disc = math.exp(-params["r"] * params["T"])

    maxima = paths.max(axis=1)
    minima = paths.min(axis=1)

    expected_call = float(disc * np.maximum(maxima - params["K"], 0.0).mean())
    expected_put = float(disc * np.maximum(params["K"] - minima, 0.0).mean())

    price_call = price_lookback_fixed_mc(option_type="call", **params)
    price_put = price_lookback_fixed_mc(option_type="put", **params)

    assert math.isclose(price_call, expected_call, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_put, expected_put, rel_tol=0.0, abs_tol=1e-12)


def test_price_call_and_put_spreads_match_vanilla_differences():
    params = dict(S0=100.0, r=0.03, q=0.01, T=0.75, sigma=0.22)
    k_long_call, k_short_call = 95.0, 110.0
    k_long_put, k_short_put = 110.0, 90.0

    call_long = pricing_lib.bs_price_call(params["S0"], k_long_call, r=params["r"], q=params["q"], sigma=params["sigma"], T=params["T"])
    call_short = pricing_lib.bs_price_call(params["S0"], k_short_call, r=params["r"], q=params["q"], sigma=params["sigma"], T=params["T"])
    put_long = pricing_lib.bs_price_put(params["S0"], k_long_put, r=params["r"], q=params["q"], sigma=params["sigma"], T=params["T"])
    put_short = pricing_lib.bs_price_put(params["S0"], k_short_put, r=params["r"], q=params["q"], sigma=params["sigma"], T=params["T"])

    expected_call_spread = call_long - call_short
    expected_put_spread = put_long - put_short

    price_cs = price_call_spread(
        params["S0"], k_long_call, k_short_call, params["r"], params["q"], params["T"], params["sigma"]
    )
    price_ps = price_put_spread(
        params["S0"], k_long_put, k_short_put, params["r"], params["q"], params["T"], params["sigma"]
    )

    assert math.isclose(price_cs, expected_call_spread, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_ps, expected_put_spread, rel_tol=0.0, abs_tol=1e-12)


def test_price_barrier_vanilla_respects_barrier_hit_logic_and_casing():
    params = dict(
        S0=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        T=1.0,
        sigma=0.25,
        barrier=110.0,
        barrier_type="UP",  # intentional uppercase to exercise normalization
        knock="Out",
        option_type="CALL",
        steps=12,
        n_paths=256,
        seed=77,
    )
    # Recompute payoff directly on generated paths to ensure hit logic is consistent
    paths = simulate_gbm_paths(
        params["S0"],
        params["r"],
        params["q"],
        params["sigma"],
        params["T"],
        params["steps"],
        params["n_paths"],
        seed=params["seed"],
    )
    hit = paths.max(axis=1) >= params["barrier"]
    payoff_terminal = np.maximum(paths[:, -1] - params["K"], 0.0)
    expected = float(math.exp(-params["r"] * params["T"]) * np.mean(payoff_terminal * (~hit)))

    price = price_barrier_vanilla(**params)
    assert math.isclose(price, expected, rel_tol=0.0, abs_tol=1e-12)
