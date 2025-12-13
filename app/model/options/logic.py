from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from pathlib import Path
import os
import re
import datetime
import logging
import json
import time
from urllib.parse import quote_plus
import requests
from scipy.stats import norm
from app.model.market_data.market_data import fetch_closing_prices as md_fetch_closing_prices

from app.model.market_data.market_data import fetch_spot_price

try:
    from alpaca_trade_api import REST as AlpacaREST
    from alpaca_trade_api.rest import TimeFrame
except Exception:  # pragma: no cover - optional dependency
    AlpacaREST = None  # type: ignore
    TimeFrame = None  # type: ignore

from app.model.options.exotic.american.pricing import (
    GeometricBrownianMotion,
    Option,
    black_scholes_merton,
    crr_pricing,
)
from app.model.options.exotic.lookback.lookback_call import lookback_call_option
from app.utils.paths import CACHE_CSV_DIR
from app.utils.secrets import get_secret

CLOSING_CACHE_FILE = CACHE_CSV_DIR / "closing_cache.csv"

# Reuse global minimum maturity for IV-related grids
MIN_IV_MATURITY = 0.1
CACHE_DIR = CACHE_CSV_DIR

def _load_alpaca_credentials():
    """Load Alpaca creds from env or app secrets."""
    key = get_secret("APCA_API_KEY_ID")
    secret = get_secret("APCA_API_SECRET_KEY")
    base = get_secret("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    return key, secret, base


def fetch_alpaca_option_tickers(limit: int | None = None) -> list[str]:
    """
    Fetch a universe of underlying tickers from Alpaca that are active/tradable.
    Used to populate UI selectors for options trading.
    If limit is None or <= 0, the full set is returned.
    """
    key, secret, base = _load_alpaca_credentials()
    if not key or not secret or not base:
        return []

    headers = {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
    }
    url = f"{(base or '').rstrip('/')}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        assets = resp.json() or []
    except Exception as exc:
        logging.warning(f"[alpaca-options] failed to fetch assets universe: {exc}")
        return []

    symbols: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if not asset.get("tradable", False):
            continue
        sym = (asset.get("symbol") or "").strip().upper()
        if sym:
            symbols.append(sym)

    symbols = sorted(set(symbols))
    if limit and limit > 0 and len(symbols) > limit:
        symbols = symbols[:limit]
    return symbols


def _period_to_days(period: str) -> int:
    """Approx convert period string to day count (1y->365, 6mo->180, etc.)."""
    if not period:
        return 365
    p = period.lower().strip()
    try:
        if p.endswith("y"):
            return int(float(p[:-1]) * 365)
        if p.endswith("mo"):
            return int(float(p[:-2]) * 30)
        if p.endswith("w"):
            return int(float(p[:-1]) * 7)
        if p.endswith("d"):
            return int(float(p[:-1]))
        return int(float(p))
    except Exception:
        return 365


def save_csv(df: pd.DataFrame, filename: str):
    """Save CSV to cache folder if not empty."""
    try:
        if df is None or df.empty:
            return
        path = CACHE_DIR / filename
        df.to_csv(path, index=False)
    except Exception as exc:
        logging.warning(f"[cache] Failed to save {filename}: {exc}")


def _normalize_symbol(sym: str) -> str:
    """Uppercase & trim."""
    return (sym or "").strip().upper()


def simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=42):
    """
    Simulate GBM paths under the risk-neutral measure:
        dS_t = (r - q) S_t dt + sigma S_t dW_t
    Returns:
        S : array of shape (M+1, N_paths)
        dt: time step
    """
    dt = T / M
    rng = np.random.default_rng(seed)
    S = np.empty((M + 1, N_paths))
    S[0, :] = S0
    Z = rng.normal(size=(M, N_paths))
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)
    for t in range(1, M + 1):
        S[t, :] = S[t - 1, :] * np.exp(drift + vol_term * Z[t - 1, :])
    return S, dt


def price_bermudan_lsmc(
    S0,
    K,
    T,
    r,
    q,
    sigma,
    cpflag="p",
    M=50,
    N_paths=100_000,
    degree=3,
    n_ex_dates=6,
    seed: int = 42,
):
    """
    LongstaffÃ¢â‚¬â€œSchwartz Monte Carlo pricing for a Bermudan option
    under risk-neutral GBM (BlackÃ¢â‚¬â€œScholes dynamics).
    """
    S, dt = simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=seed)
    disc = np.exp(-r * dt)

    if cpflag == "c":
        Y = np.maximum(S - K, 0.0)
    elif cpflag == "p":
        Y = np.maximum(K - S, 0.0)
    else:
        raise ValueError("cpflag must be 'c' or 'p'")

    C = Y[-1, :].copy()
    ex_indices = np.linspace(1, M - 1, max(1, n_ex_dates - 1), dtype=int)
    ex_set = set(ex_indices.tolist())

    for j in range(M - 1, 0, -1):
        C *= disc
        if j in ex_set:
            S_j = S[j, :]
            Y_j = Y[j, :]
            itm = Y_j > 0.0
            if np.any(itm):
                X = np.vstack([S_j[itm] ** k for k in range(degree + 1)]).T
                y_reg = C[itm]
                beta, *_ = np.linalg.lstsq(X, y_reg, rcond=None)
                X_all = np.vstack([S_j**k for k in range(degree + 1)]).T
                C_hat = X_all @ beta
                exercise = (Y_j > C_hat) & itm
                C[exercise] = Y_j[exercise]

    C *= disc
    price = np.mean(C)
    return float(price)


def longstaff_schwartz_price(option: Option, process, n_paths: int, n_steps: int) -> float:
    """
    ImplÃƒÂ©mentation locale de l'algorithme LS, basÃƒÂ©e sur Longstaff/pricing.py,
    mais qui renvoie le prix comme float.
    """
    simulated_paths = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n_paths, m=n_steps)
    payoffs = option.payoff(s=simulated_paths)

    continuation_values = np.zeros_like(payoffs)
    continuation_values[-1] = payoffs[-1]

    dt = option.T / n_steps
    discount = np.exp(-process.mu * dt)

    for time_index in range(n_steps - 1, 0, -1):
        polynomial = Polynomial.fit(
            simulated_paths[time_index], discount * continuation_values[time_index + 1], 5
        )
        continuation = polynomial(simulated_paths[time_index])
        continuation_values[time_index] = np.where(
            payoffs[time_index] > continuation,
            payoffs[time_index],
            discount * continuation_values[time_index + 1],
        )

    price = discount * np.mean(continuation_values[1])
    return float(price)


def _compute_bsm_heatmaps(
    s_values: np.ndarray, k_values: np.ndarray, maturity: float, rate: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, call=False)
            call_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_call)
            put_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_put)
    return call_matrix, put_matrix


def _compute_mc_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    mu: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    process = GeometricBrownianMotion(mu=mu, sigma=sigma)
    discount = np.exp(-mu * maturity)
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)

    for j, spot in enumerate(s_values):
        simulated_paths = process.simulate(s0=spot, T=maturity, n=n_paths, m=n_steps, v0=None)
        terminal_prices = simulated_paths[-1]
        for i, strike in enumerate(k_values):
            call_matrix[i, j] = np.mean(np.maximum(terminal_prices - strike, 0)) * discount
            put_matrix[i, j] = np.mean(np.maximum(strike - terminal_prices, 0)) * discount

    return call_matrix, put_matrix


def _compute_american_ls_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    process,
    n_paths: int,
    n_steps: int,
    v0=None,
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, v0=v0, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, v0=v0, call=False)
            call_matrix[i, j] = longstaff_schwartz_price(option_call, process, n_paths, n_steps)
            put_matrix[i, j] = longstaff_schwartz_price(option_put, process, n_paths, n_steps)
    return call_matrix, put_matrix


def _compute_american_crr_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    rate: float,
    sigma: float,
    n_tree: int,
) -> tuple[np.ndarray, np.ndarray]:
    call_matrix = np.zeros((len(k_values), len(s_values)))
    put_matrix = np.zeros_like(call_matrix)
    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            option_call = Option(s0=spot, T=maturity, K=strike, call=True)
            option_put = Option(s0=spot, T=maturity, K=strike, call=False)
            call_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_call, n=n_tree)
            put_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_put, n=n_tree)
    return call_matrix, put_matrix


# ---------------------------------------------------------------------------
# Medium-computation helpers (data/heatmap/pricing utilities)
# ---------------------------------------------------------------------------


def _vanilla_price_with_dividend(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type.lower() in {"call", "c"} else max(K - S0, 0.0)
        return float(intrinsic)
    sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
    d2 = d1 - sqrt_T
    if option_type.lower() in {"call", "c"}:
        price = S0 * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend * T) * norm.cdf(-d1)
    return float(max(price, 0.0))


def _digital_cash_or_nothing_price(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
    payout: float = 1.0,
) -> float:
    if T <= 0 or sigma <= 0 or payout <= 0:
        return 0.0
    sqrt_T = sigma * np.sqrt(T)
    d2 = (np.log(S0 / K) + (r - dividend - 0.5 * sigma * sigma) * T) / sqrt_T
    disc = payout * np.exp(-r * T)
    if option_type.lower() in {"call", "c"}:
        return float(disc * norm.cdf(d2))
    return float(disc * norm.cdf(-d2))


def _asset_or_nothing_price(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    if T <= 0 or sigma <= 0 or S0 <= 0:
        return 0.0
    sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
    disc = S0 * np.exp(-dividend * T)
    if option_type.lower() in {"call", "c"}:
        return float(disc * norm.cdf(d1))
    return float(disc * norm.cdf(-d1))


def _chooser_option_price(
    S0: float,
    K: float,
    T: float,
    t_choice: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    call_price = _vanilla_price_with_dividend("call", S0, K, T, r, dividend, sigma)
    tau = max(0.0, T - t_choice)
    if tau <= 0:
        return call_price
    K_adj = K * np.exp(-r * (T - t_choice))
    sqrt_tau = sigma * np.sqrt(tau)
    d1 = (np.log(S0 / K_adj) + (r - dividend + 0.5 * sigma * sigma) * tau) / sqrt_tau
    d2 = d1 - sqrt_tau
    put_piece = K_adj * np.exp(-r * t_choice) * norm.cdf(-d2) - S0 * np.exp(
        -dividend * t_choice
    ) * norm.cdf(-d1)
    return float(max(call_price + put_piece, 0.0))


def _forward_start_price_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T_start: float,
    T_end: float,
    k: float,
    n_paths: int = 5000,
    n_steps: int = 200,
    option_type: str = "call",
) -> float:
    if T_end <= T_start or sigma <= 0 or n_paths <= 0 or n_steps <= 0:
        return 0.0
    dt = T_end / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diff = sigma * np.sqrt(dt)
    disc = np.exp(-r * T_end)
    payoffs = []
    step_choice = int(T_start / dt)
    for _ in range(n_paths):
        s = S0
        s_start = None
        for step in range(n_steps):
            z = np.random.normal()
            s *= np.exp(drift + diff * z)
            if s_start is None and step >= step_choice:
                s_start = s
        s_start = s if s_start is None else s_start
        strike_dyn = k * s_start
        if option_type.lower() in {"call", "c"}:
            payoff = max(s - strike_dyn, 0.0)
        else:
            payoff = max(strike_dyn - s, 0.0)
        payoffs.append(payoff)
    return float(disc * np.mean(payoffs))


def _binary_barrier_mc(
    option_type: str,
    barrier_type: str,
    direction: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
    payout: float,
    n_paths: int,
    n_steps: int,
) -> float:
    if payout <= 0 or barrier <= 0 or n_paths <= 0 or n_steps <= 0:
        return 0.0
    dt = T / n_steps
    drift = (r - dividend - 0.5 * sigma * sigma) * dt
    diff = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)
    hits = []
    for _ in range(n_paths):
        s = S0
        touched = False
        for _ in range(n_steps):
            z = np.random.normal()
            s *= np.exp(drift + diff * z)
            if (barrier_type == "up" and s >= barrier) or (barrier_type == "down" and s <= barrier):
                touched = True
                break
        if direction == "out":
            pay = 0.0 if touched else payout
        else:
            pay = payout if touched else 0.0
        if pay == 0.0:
            if option_type.lower() in {"call", "c"}:
                pay = payout if s >= K else 0.0
            else:
                pay = payout if s <= K else 0.0
        hits.append(pay)
    return float(disc * np.mean(hits))


def _cliquet_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_periods: int,
    cap: float,
    floor: float,
    n_paths: int = 2000,
    seed: int | None = None,
) -> float:
    if n_periods <= 0 or n_paths <= 0 or T <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    dt = T / n_periods
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diff = sigma * np.sqrt(dt)
    disc = np.exp(-r * T)
    payoffs = []
    for _ in range(n_paths):
        s = S0
        coupons = []
        for _ in range(n_periods):
            z = rng.normal()
            s_next = s * np.exp(drift + diff * z)
            ret = (s_next / s) - 1.0
            coupons.append(np.clip(ret, floor, cap))
            s = s_next
        payoffs.append(sum(coupons))
    return float(disc * np.mean(payoffs))


def _quanto_vanilla_price(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r_dom: float,
    q_for: float,
    sigma_asset: float,
    sigma_fx: float,
    rho: float,
) -> float:
    q_adj = q_for + rho * sigma_asset * sigma_fx
    return _vanilla_price_with_dividend(option_type, S0, K, T, r_dom, q_adj, sigma_asset)


def _rainbow_two_asset_mc(
    payoff_on: str,
    S0_a: float,
    S0_b: float,
    sigma_a: float,
    sigma_b: float,
    rho: float,
    K: float,
    T: float,
    r: float,
    q_a: float,
    q_b: float,
    n_paths: int = 5000,
    n_steps: int = 200,
    option_type: str = "call",
) -> float:
    if n_paths <= 0 or n_steps <= 0 or T <= 0:
        return 0.0
    dt = T / n_steps
    disc = np.exp(-r * T)
    payoff_list = []
    for _ in range(n_paths):
        s_a, s_b = S0_a, S0_b
        for _ in range(n_steps):
            z1 = np.random.normal()
            z2 = np.random.normal()
            z_b = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2
            s_a *= np.exp((r - q_a - 0.5 * sigma_a * sigma_a) * dt + sigma_a * np.sqrt(dt) * z1)
            s_b *= np.exp((r - q_b - 0.5 * sigma_b * sigma_b) * dt + sigma_b * np.sqrt(dt) * z_b)
        s_star = max(s_a, s_b) if payoff_on == "max" else min(s_a, s_b)
        if option_type.lower() in {"call", "c"}:
            payoff = max(s_star - K, 0.0)
        else:
            payoff = max(K - s_star, 0.0)
        payoff_list.append(payoff)
    return float(disc * np.mean(payoff_list))


def _barrier_closed_form_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    if barrier <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("ParamÃƒÂ¨tres invalides pour la formule fermÃƒÂ©e barriÃƒÂ¨re.")
    if barrier_type == "up" and S0 >= barrier:
        return 0.0
    if barrier_type == "down" and S0 <= barrier:
        return 0.0

    option_flag = option_type.lower()
    phi = 1.0 if option_flag in {"call", "c"} else -1.0
    eta = 1.0 if barrier_type == "down" else -1.0
    mu = (r - dividend - 0.5 * sigma * sigma) / (sigma * sigma)
    sigma_sqrt_T = sigma * np.sqrt(T)
    if sigma_sqrt_T == 0:
        return 0.0
    x1 = (np.log(S0 / K) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
    y1 = (np.log((barrier * barrier) / (S0 * K)) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
    power1 = (barrier / S0) ** (2.0 * (mu + eta))
    power2 = (barrier / S0) ** (2.0 * mu)
    term1 = phi * S0 * np.exp(-dividend * T) * (norm.cdf(phi * x1) - power1 * norm.cdf(eta * y1))
    term2 = (
        phi
        * K
        * np.exp(-r * T)
        * (
            norm.cdf(phi * x1 - phi * sigma_sqrt_T)
            - power2 * norm.cdf(eta * y1 - eta * sigma_sqrt_T)
        )
    )
    price = term1 - term2
    return max(float(price), 0.0)


def _knock_in_closed_form_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> float:
    vanilla = _vanilla_price_with_dividend(
        option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
    )
    barrier_out_price = _barrier_closed_form_price(
        option_type=option_type,
        barrier_type=barrier_type,
        S0=S0,
        K=K,
        barrier=barrier,
        T=T,
        r=r,
        dividend=dividend,
        sigma=sigma,
    )
    return max(vanilla - barrier_out_price, 0.0)


def _barrier_monte_carlo_price(
    option_type: str,
    barrier_type: str,
    S0: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    knock_in: bool = False,
) -> float:
    if barrier <= 0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("ParamÃƒÂ¨tres invalides pour le Monte Carlo barriÃƒÂ¨re.")
    option_type_lower = option_type.lower()
    if barrier_type == "up" and S0 >= barrier:
        if knock_in:
            return _vanilla_price_with_dividend(
                option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
            )
        return 0.0
    if barrier_type == "down" and S0 <= barrier:
        if knock_in:
            return _vanilla_price_with_dividend(
                option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
            )
        return 0.0
    dt = T / n_steps
    drift = (r - dividend - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)
    payoffs = []
    for _ in range(n_paths):
        s = S0
        barrier_hit = False
        for _ in range(n_steps):
            z = np.random.normal()
            s *= np.exp(drift + diffusion * z)
            if barrier_type == "up" and s >= barrier:
                barrier_hit = True
                if not knock_in:
                    break
            elif barrier_type == "down" and s <= barrier:
                barrier_hit = True
                if not knock_in:
                    break
        if knock_in and not barrier_hit:
            payoffs.append(0.0)
            continue
        if not knock_in and barrier_hit:
            payoffs.append(0.0)
            continue
        if option_type_lower in {"call", "c"}:
            payoff = max(s - K, 0.0)
        else:
            payoff = max(K - s, 0.0)
        payoffs.append(payoff)
    return discount * (float(np.mean(payoffs)) if payoffs else 0.0)


def _compute_barrier_heatmap_matrix(
    option_type: str,
    barrier_type: str,
    strike_values: np.ndarray,
    offset_values: np.ndarray,
    S0: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(strike_values), len(offset_values)))
    ratio_axis = np.zeros(len(offset_values))

    for j, offset in enumerate(offset_values):
        ratio = 1.1 + offset if barrier_type == "up" else max(0.01, 0.9 - offset)
        ratio_axis[j] = ratio

        for i, strike in enumerate(strike_values):
            barrier = max(strike * ratio, 0.01)
            try:
                price = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type=barrier_type,
                    S0=S0,
                    K=float(strike),
                    barrier=float(barrier),
                    T=T,
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                price = 0.0
            matrix[i, j] = price

    if np.any(np.diff(ratio_axis) < 0):
        order = np.argsort(ratio_axis)
        ratio_axis = ratio_axis[order]
        matrix = matrix[:, order]

    return matrix, ratio_axis


def _compute_up_and_out_strike_heatmap(
    option_type: str,
    barrier: float,
    strike_values: np.ndarray,
    maturity_values: np.ndarray,
    spot: float,
    r: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    matrix = np.zeros((len(maturity_values), len(strike_values)))
    for i, maturity in enumerate(maturity_values):
        for j, strike in enumerate(strike_values):
            if strike <= 0.0:
                matrix[i, j] = 0.0
                continue
            try:
                price = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type="up",
                    S0=float(spot),
                    K=float(strike),
                    barrier=float(barrier),
                    T=float(maturity),
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                price = 0.0
            matrix[i, j] = price
    return matrix


def _compute_up_and_in_strike_heatmap(
    option_type: str,
    barrier: float,
    strike_values: np.ndarray,
    maturity_values: np.ndarray,
    spot: float,
    r: float,
    dividend: float,
    sigma: float,
) -> np.ndarray:
    matrix = np.zeros((len(maturity_values), len(strike_values)))
    for i, maturity in enumerate(maturity_values):
        for j, strike in enumerate(strike_values):
            if strike <= 0.0:
                matrix[i, j] = 0.0
                continue
            vanilla = _vanilla_price_with_dividend(
                option_type, spot, float(strike), float(maturity), r, dividend, sigma
            )
            try:
                barrier_out = _barrier_closed_form_price(
                    option_type=option_type,
                    barrier_type="up",
                    S0=float(spot),
                    K=float(strike),
                    barrier=float(barrier),
                    T=float(maturity),
                    r=r,
                    dividend=dividend,
                    sigma=sigma,
                )
            except ValueError:
                matrix[i, j] = 0.0
                continue
            matrix[i, j] = max(vanilla - barrier_out, 0.0)
    return matrix


def _compute_lookback_exact_heatmap(
    s_values: np.ndarray,
    t_values: np.ndarray,
    t_current: float,
    rate: float,
    sigma: float,
) -> np.ndarray:
    matrix = np.zeros((len(t_values), len(s_values)))
    for i, maturity in enumerate(t_values):
        for j, spot in enumerate(s_values):
            lookback_opt = lookback_call_option(
                T=float(maturity),
                t=float(t_current),
                S0=float(spot),
                r=float(rate),
                sigma=float(sigma),
            )
            matrix[i, j] = lookback_opt.price_exact()
    return matrix


def _compute_lookback_mc_heatmap(
    s_values: np.ndarray,
    t_values: np.ndarray,
    t_current: float,
    rate: float,
    sigma: float,
    n_iters: int,
) -> np.ndarray:
    matrix = np.zeros((len(t_values), len(s_values)))
    for i, maturity in enumerate(t_values):
        for j, spot in enumerate(s_values):
            lookback_opt = lookback_call_option(
                T=float(maturity),
                t=float(t_current),
                S0=float(spot),
                r=float(rate),
                sigma=float(sigma),
            )
            matrix[i, j] = lookback_opt.price_monte_carlo(n_iters)
    return matrix


def _compute_down_in_heatmap(
    option_type: str,
    strike_values: np.ndarray,
    offset_values: np.ndarray,
    S0: float,
    T: float,
    r: float,
    dividend: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_out, ratio_axis = _compute_barrier_heatmap_matrix(
        option_type=option_type,
        barrier_type="down",
        strike_values=strike_values,
        offset_values=offset_values,
        S0=S0,
        T=T,
        r=r,
        dividend=dividend,
        sigma=sigma,
    )
    matrix_in = np.zeros_like(matrix_out)
    for i, strike in enumerate(strike_values):
        vanilla = _vanilla_price_with_dividend(
            option_type=option_type,
            S0=S0,
            K=float(strike),
            T=T,
            r=r,
            dividend=dividend,
            sigma=sigma,
        )
        matrix_in[i, :] = np.maximum(vanilla - matrix_out[i, :], 0.0)
    return matrix_in, ratio_axis


# ---------------------------------------------------------------------------
# Closing prices cache helpers and numeric utilities
# ---------------------------------------------------------------------------


def _fetch_ohlc_alpaca(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Attempt to pull OHLC from Alpaca; returns empty DataFrame on failure."""
    if AlpacaREST is None or TimeFrame is None:
        return pd.DataFrame()

    key, secret, base = _load_alpaca_credentials()
    if not key or not secret:
        return pd.DataFrame()

    days = _period_to_days(period)
    try:
        client = AlpacaREST(key, secret, base, api_version="v2")
        bars = client.get_bars(symbol, TimeFrame.Day, limit=days).df
        if bars is None or bars.empty:
            logging.info(f"[closing] Alpaca returned no bars for {symbol} (days={days})")
            return pd.DataFrame()
        bars = bars.reset_index()
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = [c[-1] if isinstance(c, tuple) else c for c in bars.columns]
        if "timestamp" in bars.columns:
            bars["Date"] = pd.to_datetime(bars["timestamp"], errors="coerce")
        elif "time" in bars.columns:
            bars["Date"] = pd.to_datetime(bars["time"], errors="coerce")
        else:
            bars["Date"] = pd.to_datetime(bars.index, errors="coerce")
        if "close" in bars.columns:
            bars["Close"] = pd.to_numeric(bars["close"], errors="coerce")
        return bars[["Date", "Close"]].dropna()
    except Exception as exc:
        logging.warning(f"[closing] Alpaca OHLC failed for {symbol}: {exc}")
        return pd.DataFrame()


def fetch_closing_prices(
    tickers: str | Iterable[str], period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Unified closing price fetcher used by market_data.fetch_closing_prices.
    Accepts single ticker or list of tickers. Returns Date + ticker columns.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    frames: list[pd.DataFrame] = []
    for sym in tickers:
        sym_norm = _normalize_symbol(sym)
        df = md_fetch_closing_prices(sym_norm, period=period, interval=interval)
        if df is None or df.empty:
            continue
        frames.append(df.rename(columns={sym_norm: sym_norm}))

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for df in frames[1:]:
        try:
            out = out.merge(df, on="Date", how="outer")
        except Exception:
            pass

    try:
        CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(CLOSING_CACHE_FILE, index=False)
    except Exception:
        pass
    return out


def compute_corr_from_prices(prices_df: pd.DataFrame):
    price_cols = [c for c in prices_df.columns if c.lower() != "date"]
    returns = np.log(prices_df[price_cols] / prices_df[price_cols].shift(1)).dropna(how="any")
    if returns.empty:
        raise RuntimeError("Pas assez de donnÃƒÂ©es pour calculer la corrÃƒÂ©lation.")
    return returns.corr()


def load_closing_prices_with_tickers(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.exists():
        return None, []
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, []
    ticker_cols: list[str] = []
    for col in df.columns:
        col_str = str(col).strip()
        if not col_str or col_str.lower() == "date":
            continue
        ticker_cols.append(col_str)
    return df, ticker_cols


def load_close_series_for_ticker(ticker: str, fallback_value: float | None = None) -> pd.Series:
    """
    Charge une série de clôtures pour un ticker.
    - Lit cache_csv/closing_cache.csv si la colonne du ticker est présente.
    - Sinon télécharge via fetch_closing_prices (1y, 1d), normalise Date/Close, met à jour le cache.
    - Sinon retourne une série fallback si fournie.
    """
    ticker_norm = (ticker or "").strip().upper()
    try:
        from app.model.market_data.market_data import fetch_closing_prices as _fetch_md_prices
    except Exception:
        _fetch_md_prices = None

    def _to_series(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        date_col = next(
            (c for c in df.columns if str(c).lower() in {"date", "datetime", "index"}),
            df.columns[0],
        )
        close_col = next(
            (
                c
                for c in df.columns
                if str(c).lower() in {"close", "adjclose", "adj_close", "adj close"}
            ),
            None,
        )
        if close_col is None and len(df.columns) > 1:
            close_col = df.columns[-1]
        dates = pd.to_datetime(df[date_col], errors="coerce")
        series = pd.Series(df[close_col].values, index=dates, name="Close")
        return series.dropna()

    def _load_from_cache() -> pd.Series:
        try:
            if not CLOSING_CACHE_FILE.exists():
                return pd.Series(dtype=float)
            df_cache = pd.read_csv(CLOSING_CACHE_FILE, parse_dates=["Date"])
            if ticker_norm not in df_cache.columns:
                return pd.Series(dtype=float)
            dates = pd.to_datetime(df_cache["Date"], errors="coerce")
            series = pd.Series(df_cache[ticker_norm].values, index=dates, name="Close")
            return series.dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _write_cache(series: pd.Series) -> None:
        if series is None or series.empty:
            return
        try:
            CLOSING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            if CLOSING_CACHE_FILE.exists():
                df_cache = pd.read_csv(CLOSING_CACHE_FILE, parse_dates=["Date"])
            else:
                df_cache = pd.DataFrame({"Date": series.index})
            df_series = pd.DataFrame({"Date": series.index, ticker_norm: series.values})
            df_cache["Date"] = pd.to_datetime(df_cache["Date"], errors="coerce")
            merged = df_cache.merge(df_series, on="Date", how="outer")
            merged = merged.sort_values("Date")
            merged.to_csv(CLOSING_CACHE_FILE, index=False)
        except Exception:
            pass

    # 1) Cache
    series = _load_from_cache()
    if series is not None and not series.empty:
        return series

    # 2) Download (already cached to cache/ohlc_*)
    try:
        if _fetch_md_prices is not None:
            df_dl = _fetch_md_prices(ticker_norm, period="2y", interval="1d")
            series = _to_series(df_dl)
            if series is not None and not series.empty:
                cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
                series_1y = series[series.index >= cutoff]
                if series_1y.empty:
                    series_1y = series.tail(252)
                if series_1y is None or series_1y.empty:
                    series_1y = series
                _write_cache(series_1y)
                return series_1y
    except Exception:
        pass

    # 3) Fallback
    if fallback_value is not None:
        return pd.Series([fallback_value], index=pd.Index([datetime.date.today()]), name="Close")
    return pd.Series(dtype=float)


def _update_closing_cache_series(ticker: str, series: pd.Series) -> None:
    """Ãƒâ€°crit/merge une sÃƒÂ©rie de clÃƒÂ´tures dans closing_cache.csv (Date + colonnes ticker)."""
    if series is None or series.empty:
        return
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return
    df_new = series.reset_index()
    df_new.columns = ["Date", tkr]
    try:
        if CLOSING_CACHE_FILE.exists():
            df_existing = pd.read_csv(CLOSING_CACHE_FILE, parse_dates=["Date"])
        else:
            df_existing = pd.DataFrame(columns=["Date"])
        if "Date" not in df_existing.columns:
            df_existing["Date"] = pd.to_datetime(df_existing.get("Date", pd.NaT))
        df_merged = pd.merge(df_existing, df_new, on="Date", how="outer")
        df_merged = df_merged.sort_values("Date")
        CLOSING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(CLOSING_CACHE_FILE, index=False)
    except Exception:
        pass


def _closing_cache_path_for_ticker(ticker: str, period: str = "1y", interval: str = "1d") -> Path:
    """Build a per-ticker CSV path for cached closing prices."""
    ticker_norm = (ticker or "").strip().upper()
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker_norm) or "TICKER"
    suffix = f"{period}_{interval}".replace("/", "-")
    return CACHE_CSV_DIR / f"ohlc_{safe}_{suffix}.csv"


def load_or_fetch_closing_history(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
) -> tuple[pd.DataFrame | None, Path | None, bool]:
    """
    Retourne les clÃ´tures 1 an du ticker en lisant le cache CSV ou via fetch_closing_prices.
    Loggue chaque Ã©tape pour visibilitÃ© console/UI.
    """

    def _log(msg: str) -> None:
        logging.info(msg)
        try:
            print(msg)
        except Exception:
            pass

    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return None, None, False
    cache_path = _closing_cache_path_for_ticker(ticker_norm, period=period, interval=interval)

    df_hist = None
    from_cache = False
    if cache_path.exists():
        try:
            df_hist = pd.read_csv(cache_path)
            from_cache = True
            _log(
                f"[closing] cache hit for {ticker_norm} ({period}/{interval}) rows={df_hist.shape[0]}"
            )
        except Exception:
            df_hist = None
            from_cache = False

    if df_hist is None or df_hist.empty:
        _log(f"[closing] cache miss -> download {ticker_norm} ({period}/{interval})")
        try:
            df_hist = fetch_closing_prices(ticker_norm, period=period, interval=interval)
            if df_hist is not None and not df_hist.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df_hist.to_csv(cache_path, index=False)
                _log(f"[closing] wrote cache {cache_path} rows={df_hist.shape[0]}")
                from_cache = False
        except Exception as exc:
            _log(f"[closing] error while fetching {ticker_norm}: {exc}")
            df_hist = None
            from_cache = False

    if df_hist is not None and not df_hist.empty:
        _log(
            f"[closing] return history for {ticker_norm} rows={df_hist.shape[0]} cols={df_hist.shape[1]} from_cache={from_cache}"
        )
    else:
        _log(f"[closing] no history for {ticker_norm}")

    return df_hist, cache_path, from_cache


def clear_closing_history_cache(ticker: str, *, period: str = "1y", interval: str = "1d") -> None:
    """Supprime le cache CSV des clÃƒÂ´tures pour un ticker (best-effort)."""
    try:
        path = _closing_cache_path_for_ticker(ticker, period=period, interval=interval)
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _safe_float(val, default):
    try:
        return float(default if val is None else val)
    except Exception:
        return float(default)


def build_grid(
    df: pd.DataFrame,
    spot: float,
    n_k: int = 200,
    n_t: int = 200,
    k_min: float | None = None,
    k_max: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    k_span: float | None = None,
    margin_frac: float = 0.02,
):
    if k_min is None or k_max is None:
        if k_span is not None:
            k_min = spot - k_span
            k_max = spot + k_span
        else:
            data_k_min = float(df["K"].min())
            data_k_max = float(df["K"].max())
            delta_k = data_k_max - data_k_min
            pad = delta_k * margin_frac
            k_min = data_k_min - pad
            k_max = data_k_max + pad

    if t_min is None:
        t_min = float(df["T"].min())
    if t_max is None:
        t_max = float(df["T"].max())

    if k_min >= k_max:
        raise ValueError("k_min doit ÃƒÂªtre infÃƒÂ©rieur ÃƒÂ  k_max.")
    if t_min >= t_max:
        raise ValueError("t_min doit ÃƒÂªtre infÃƒÂ©rieur ÃƒÂ  t_max.")

    k_vals = np.linspace(k_min, k_max, n_k)
    t_vals = np.linspace(t_min, t_max, n_t)

    df = df[(df["K"] >= k_min) & (df["K"] <= k_max)].copy()
    df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

    if df.empty:
        raise ValueError("Aucun point n'appartient au domaine dÃƒÂ©fini par la grille.")

    df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left").clip(0, n_k - 1)
    df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left").clip(0, n_t - 1)

    grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

    iv_grid = np.full((n_t, n_k), np.nan, dtype=float)
    for _, row in grouped.iterrows():
        iv_grid[int(row["T_idx"]), int(row["K_idx"])] = row["iv"]

    k_grid, t_grid = np.meshgrid(k_vals, t_vals)
    return k_grid, t_grid, iv_grid


def download_options_alpaca(
    symbol: str,
    *,
    feed: str = "indicative",
    max_pages: int | None = None,
    max_contracts: int | None = None,
    min_days_to_expiry: int | None = 1,
    include_spot: bool = True,
    cache_to_csv: bool = True,
    session: requests.Session | None = None,
    timeout_sec: float = 10.0,
    max_retries: int = 2,
    backoff_sec: float = 0.5,
) -> pd.DataFrame:
    """
    Download option snapshots from Alpaca Market Data API v1beta1.
    Returns a normalized DataFrame with columns: symbol, opra, K, T, S0, iv, type.

    Notes:
    - With the free/indicative feed, snapshots typically do NOT include greeks/IV.
      In that case, iv will be NaN.
    - Results are paginated; by default this helper pulls all pages.
      Use `max_pages` / `max_contracts` only to cap the fetch.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return pd.DataFrame()

    key, secret, _ = _load_alpaca_credentials()
    headers = {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
    }
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{quote_plus(sym)}"

    today = datetime.date.today()
    if include_spot:
        s0_val = fetch_spot_price(sym)
        s0_val = float(s0_val) if s0_val is not None else float("nan")
    else:
        s0_val = float("nan")

    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def _decode_from_opra(opra: str) -> tuple[float | None, datetime.date | None, str | None]:
        if not opra or len(opra) < 15:
            return None, None, None
        try:
            opra_str = str(opra)
            expiry = datetime.datetime.strptime(opra_str[-15:-9], "%y%m%d").date()
            strike = int(opra_str[-8:]) / 1000.0
            opt_type = "call" if opra_str[-9].upper() == "C" else "put"
            return strike, expiry, opt_type
        except Exception:
            return None, None, None

    records: list[dict] = []

    try:
        max_pages_int = int(max_pages) if max_pages is not None else None
    except Exception:
        max_pages_int = None
    if max_pages_int is not None and max_pages_int <= 0:
        max_pages_int = None

    try:
        max_contracts_int = int(max_contracts) if max_contracts is not None else None
    except Exception:
        max_contracts_int = None
    if max_contracts_int is not None and max_contracts_int <= 0:
        max_contracts_int = None

    try:
        min_days = int(min_days_to_expiry) if min_days_to_expiry is not None else None
    except Exception:
        min_days = None

    try:
        timeout_s = float(timeout_sec)
    except Exception:
        timeout_s = 10.0
    if timeout_s <= 0:
        timeout_s = 10.0

    try:
        max_retries_i = int(max_retries)
    except Exception:
        max_retries_i = 0
    if max_retries_i < 0:
        max_retries_i = 0

    try:
        backoff_s = float(backoff_sec)
    except Exception:
        backoff_s = 0.5
    if backoff_s < 0:
        backoff_s = 0.0

    requester = session.get if session is not None else requests.get

    page_token: str | None = None
    seen_page_tokens: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        if max_pages_int is not None and page_count > max_pages_int:
            break
        params: dict = {}
        if feed:
            params["feed"] = str(feed)
        if page_token:
            params["page_token"] = page_token

        payload: dict | None = None
        for attempt in range(max_retries_i + 1):
            try:
                resp = requester(url, headers=headers, params=params, timeout=timeout_s)
                status = int(getattr(resp, "status_code", 0) or 0)
                if status == 403:
                    # e.g. OPRA agreement not signed
                    try:
                        msg = (resp.json() or {}).get("message")
                    except Exception:
                        msg = None
                    logging.warning(f"[alpaca-options] forbidden for {sym} (feed={feed}): {msg or status}")
                    payload = None
                    break
                if status in (429, 500, 502, 503, 504):
                    if attempt < max_retries_i:
                        wait = backoff_s * (2 ** attempt)
                        retry_after = None
                        try:
                            retry_after = float(resp.headers.get("Retry-After") or 0.0)
                        except Exception:
                            retry_after = None
                        if retry_after is not None and retry_after > 0:
                            wait = max(wait, retry_after)
                        if wait > 0:
                            time.sleep(wait)
                        continue
                resp.raise_for_status()
                payload = resp.json() or {}
                if not isinstance(payload, dict):
                    payload = {}
                break
            except Exception as exc:
                if attempt < max_retries_i:
                    wait = backoff_s * (2 ** attempt)
                    if wait > 0:
                        time.sleep(wait)
                    continue
                logging.warning(f"[alpaca-options] download failed for {sym}: {exc}")
                payload = None
                break

        if payload is None:
            break

        snapshots = payload.get("snapshots") or payload.get("data") or {}
        if not isinstance(snapshots, dict) or not snapshots:
            break

        for opra, snap in snapshots.items():
            if max_contracts_int is not None and len(records) >= max_contracts_int:
                break
            if not isinstance(snap, dict):
                continue

            opra_sym = str(opra) if opra is not None else ""
            strike, expiry, opt_type = _decode_from_opra(opra_sym)
            if strike is None:
                strike = _first_not_none(
                    snap.get("strike"),
                    snap.get("strike_price"),
                    (snap.get("details") or {}).get("strike_price"),
                )
            if expiry is None:
                exp_val = _first_not_none(
                    snap.get("expiration_date"),
                    (snap.get("details") or {}).get("expiration_date"),
                )
                if exp_val:
                    try:
                        expiry = datetime.datetime.fromisoformat(str(exp_val)).date()
                    except Exception:
                        try:
                            expiry = datetime.datetime.strptime(str(exp_val), "%Y-%m-%d").date()
                        except Exception:
                            expiry = None
            if opt_type is None:
                right = _first_not_none(snap.get("type"), snap.get("right"))
                if right is not None:
                    opt_type = "call" if str(right).upper().startswith("C") else "put"

            if strike is None or expiry is None or opt_type is None:
                continue

            days_to_expiry = int((expiry - today).days)
            if min_days is not None and days_to_expiry < min_days:
                continue

            greeks = _first_not_none(snap.get("greeks"), snap.get("latestGreeks")) or {}
            if not isinstance(greeks, dict):
                greeks = {}
            iv_val = _first_not_none(
                greeks.get("iv"),
                greeks.get("impliedVolatility"),
                greeks.get("implied_volatility"),
                snap.get("iv"),
                snap.get("impliedVolatility"),
                snap.get("implied_volatility"),
            )
            iv_float = float("nan")
            if iv_val is not None:
                try:
                    iv_float = float(iv_val)
                except Exception:
                    iv_float = float("nan")

            T = days_to_expiry / 365.0
            records.append(
                {
                    "symbol": sym,
                    "opra": opra_sym,
                    "K": float(strike),
                    "T": float(T),
                    "S0": s0_val,
                    "iv": iv_float,
                    "type": opt_type,
                }
            )

        if max_contracts_int is not None and len(records) >= max_contracts_int:
            break

        next_token = payload.get("next_page_token")
        if not next_token:
            break
        next_token = str(next_token)
        if next_token == page_token or next_token in seen_page_tokens:
            logging.warning(f"[alpaca-options] repeated next_page_token for {sym}; stopping pagination")
            break
        seen_page_tokens.add(next_token)
        page_token = next_token

    df = pd.DataFrame(records, columns=["symbol", "opra", "K", "T", "S0", "iv", "type"])
    if cache_to_csv:
        try:
            CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
            out_path = CACHE_CSV_DIR / f"options_alpaca_{sym}.csv"
            df.to_csv(out_path, index=False)
        except Exception:
            pass
    return df
