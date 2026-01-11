import math

import numpy as np


def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
):
    dt = float(T) / max(int(steps), 1)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    vol = float(sigma) * math.sqrt(dt)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((int(n_paths), max(int(steps), 1)))
    increments = drift + vol * Z
    log_paths = np.cumsum(increments, axis=1)
    S = float(S0) * np.exp(log_paths)
    S = np.concatenate([np.full((int(n_paths), 1), float(S0)), S], axis=1)
    return S


def price_asian_mc(S0, K, r, q, T, sigma, steps=50, n_paths=2000, option_type="call", seed=None):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
    avg = paths.mean(axis=1)
    if str(option_type).lower().startswith("c"):
        payoff = np.maximum(avg - float(K), 0.0)
    else:
        payoff = np.maximum(float(K) - avg, 0.0)
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def price_asian_geo_mc(
    S0, K, r, q, T, sigma, steps=50, n_paths=2000, option_type="call", seed=None
):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
    log_avg = np.log(paths + 1e-12).mean(axis=1)
    geo_avg = np.exp(log_avg)
    if str(option_type).lower().startswith("c"):
        payoff = np.maximum(geo_avg - float(K), 0.0)
    else:
        payoff = np.maximum(float(K) - geo_avg, 0.0)
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def price_lookback_mc(S0, r, q, T, sigma, steps=50, n_paths=2000, option_type="call", seed=None):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
    if str(option_type).lower().startswith("c"):
        extrema = paths.min(axis=1)
        payoff = np.maximum(paths[:, -1] - extrema, 0.0)
    else:
        extrema = paths.max(axis=1)
        payoff = np.maximum(extrema - paths[:, -1], 0.0)
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def price_lookback_fixed_mc(
    S0, K, r, q, T, sigma, steps=50, n_paths=2000, option_type="call", seed=None
):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
    maxima = paths.max(axis=1)
    minima = paths.min(axis=1)
    if str(option_type).lower().startswith("c"):
        payoff = np.maximum(maxima - float(K), 0.0)
    else:
        payoff = np.maximum(float(K) - minima, 0.0)
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def price_cliquet(S0, r, q, T, sigma, n_periods=4, cap=0.1, floor=0.0, seed=None):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_periods, 2000, seed=seed)
    returns = paths[:, 1:] / paths[:, :-1] - 1.0
    cliquet_payoff = np.clip(returns, floor, cap).sum(axis=1)
    disc = math.exp(-float(r) * float(T))
    return float(disc * cliquet_payoff.mean())


def price_barrier_vanilla(
    S0,
    K,
    r,
    q,
    T,
    sigma,
    barrier,
    barrier_type="up",
    knock="out",
    option_type="call",
    steps=100,
    n_paths=5000,
    seed=None,
):
    # Normalize textual parameters to make the function tolerant to different casings
    dir_norm = str(barrier_type).lower()
    knock_norm = str(knock).lower()
    opt_norm = str(option_type).lower()

    if dir_norm.startswith("up"):
        paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
        hit = paths.max(axis=1) >= barrier
    elif dir_norm.startswith("down"):
        paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
        hit = paths.min(axis=1) <= barrier
    else:
        raise ValueError(f"Unknown barrier_type='{barrier_type}', expected 'up' or 'down'.")

    active = ~hit if knock_norm.startswith("out") else hit
    payoff_terminal = (
        np.maximum(paths[:, -1] - K, 0.0)
        if opt_norm.startswith("c")
        else np.maximum(K - paths[:, -1], 0.0)
    )
    payoff = payoff_terminal * active
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def price_barrier_digital(
    S0,
    K,
    r,
    q,
    T,
    sigma,
    barrier,
    barrier_type="up",
    knock="out",
    payout=1.0,
    steps=100,
    n_paths=5000,
    seed=None,
):
    paths = simulate_gbm_paths(S0, r, q, sigma, T, steps, n_paths, seed=seed)
    if barrier_type == "up":
        hit = paths.max(axis=1) >= barrier
    else:
        hit = paths.min(axis=1) <= barrier
    active = ~hit if knock == "out" else hit
    intrinsic = paths[:, -1] > K
    payoff = payout * intrinsic * active
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())
