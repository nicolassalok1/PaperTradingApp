import math

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bsm_price(S0, K, r, q, T, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        return max(S0 - K, 0.0) if option_type.lower().startswith("c") else max(K - S0, 0.0)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.lower().startswith("c"):
        return S0 * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * math.exp(-q * T) * _norm_cdf(-d1)


def price_forward_start(S0, r, q, T_start, T_end, sigma, option_type="call"):
    if T_end <= T_start:
        return 0.0
    tau = float(T_end) - float(T_start)
    forward = float(S0) * math.exp((float(r) - float(q)) * float(T_start))
    d1 = (0.5 * float(sigma) ** 2 * tau) / (float(sigma) * math.sqrt(tau))
    d2 = d1 - float(sigma) * math.sqrt(tau)

    disc = math.exp(-float(r) * tau)
    if str(option_type).lower().startswith("c"):
        return float(forward * _norm_cdf(d1) - forward * _norm_cdf(d2) * disc)
    return float(forward * _norm_cdf(-d2) * disc - forward * _norm_cdf(-d1))


def price_straddle(S0, K, r, q, T, sigma):
    return float(
        _bsm_price(S0, K, r, q, T, sigma, "call") + _bsm_price(S0, K, r, q, T, sigma, "put")
    )


def price_strangle(S0, K_put, K_call, r, q, T, sigma):
    return float(
        _bsm_price(S0, K_call, r, q, T, sigma, "call")
        + _bsm_price(S0, K_put, r, q, T, sigma, "put")
    )


def price_call_spread(S0, K_long, K_short, r, q, T, sigma):
    return float(
        _bsm_price(S0, K_long, r, q, T, sigma, "call")
        - _bsm_price(S0, K_short, r, q, T, sigma, "call")
    )


def price_put_spread(S0, K_long, K_short, r, q, T, sigma):
    return float(
        _bsm_price(S0, K_long, r, q, T, sigma, "put")
        - _bsm_price(S0, K_short, r, q, T, sigma, "put")
    )


def price_butterfly(S0, K1, K2, K3, r, q, T, sigma):
    c1 = _bsm_price(S0, K1, r, q, T, sigma, "call")
    c2 = _bsm_price(S0, K2, r, q, T, sigma, "call")
    c3 = _bsm_price(S0, K3, r, q, T, sigma, "call")
    return float(c1 - 2 * c2 + c3)


def price_condor(S0, K1, K2, K3, K4, r, q, T, sigma):
    c1 = _bsm_price(S0, K1, r, q, T, sigma, "call")
    c2 = _bsm_price(S0, K2, r, q, T, sigma, "call")
    c3 = _bsm_price(S0, K3, r, q, T, sigma, "call")
    c4 = _bsm_price(S0, K4, r, q, T, sigma, "call")
    return float(c1 - c2 - c3 + c4)


def price_iron_condor(S0, K_put_long, K_put_short, K_call_short, K_call_long, r, q, T, sigma):
    put_spread = price_put_spread(S0, K_put_long, K_put_short, r, q, T, sigma)
    call_spread = price_call_spread(S0, K_call_long, K_call_short, r, q, T, sigma)
    return float(put_spread + call_spread)


def price_iron_butterfly(S0, K_put_long, K_short, K_call_long, r, q, T, sigma):
    put_spread = price_put_spread(S0, K_put_long, K_short, r, q, T, sigma)
    call_spread = price_call_spread(S0, K_call_long, K_short, r, q, T, sigma)
    return float(put_spread + call_spread)


def price_calendar_spread(S0, K, r, q, T_short, T_long, sigma):
    c_long = _bsm_price(S0, K, r, q, T_long, sigma, "call")
    c_short = _bsm_price(S0, K, r, q, T_short, sigma, "call")
    return float(c_long - c_short)


def price_diagonal_spread(S0, K_short, K_long, r, q, T_short, T_long, sigma):
    c_long = _bsm_price(S0, K_long, r, q, T_long, sigma, "call")
    c_short = _bsm_price(S0, K_short, r, q, T_short, sigma, "call")
    return float(c_long - c_short)


def price_digital(S0, K, r, q, T, sigma, payout=1.0, option_type="call"):
    if T <= 0 or sigma <= 0:
        return payout if (S0 > K and option_type.lower().startswith("c")) else 0.0
    d2 = (math.log(S0 / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    disc = math.exp(-float(r) * float(T))
    if option_type.lower().startswith("c"):
        return float(payout * disc * _norm_cdf(d2))
    return float(payout * disc * _norm_cdf(-d2))


def price_asset_or_nothing(S0, K, r, q, T, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return float(S0 if (S0 > K and option_type.lower().startswith("c")) else 0.0)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if option_type.lower().startswith("c"):
        return float(S0 * math.exp(-float(q) * float(T)) * _norm_cdf(d1))
    return float(S0 * math.exp(-float(q) * float(T)) * _norm_cdf(-d1))


def price_chooser(S0, K, r, q, T, sigma):
    call_price = _bsm_price(S0, K, r, q, T, sigma, "call")
    put_price = _bsm_price(S0, K, r, q, T, sigma, "put")
    return float(max(call_price, put_price))


def price_quanto(
    S0_domestic, fx_rate, K, r_dom, r_for, sigma_dom, sigma_fx, rho, T, option_type="call"
):
    adj_r = float(r_dom) - float(r_for) - float(rho) * float(sigma_dom) * float(sigma_fx)
    return float(_bsm_price(S0_domestic * fx_rate, K, adj_r, 0.0, T, sigma_dom, option_type))


def price_rainbow(
    S1, S2, K, r, q, T, sigma1, sigma2, corr=0.0, option_type="call", n_paths=5000, seed=None
):
    rng = np.random.default_rng(seed)
    dt = float(T)
    z1 = rng.standard_normal(int(n_paths))
    z2 = rng.standard_normal(int(n_paths))
    z2_corr = corr * z1 + math.sqrt(max(0.0, 1 - corr**2)) * z2
    s1 = S1 * np.exp((r - q - 0.5 * sigma1**2) * dt + sigma1 * math.sqrt(dt) * z1)
    s2 = S2 * np.exp((r - q - 0.5 * sigma2**2) * dt + sigma2 * math.sqrt(dt) * z2_corr)
    if option_type.lower().startswith("c"):
        payoff = np.maximum(np.maximum(s1, s2) - K, 0.0)
    else:
        payoff = np.maximum(K - np.minimum(s1, s2), 0.0)
    disc = math.exp(-float(r) * float(T))
    return float(disc * payoff.mean())


def _constant_corr_matrix(n_assets: int, rho: float) -> np.ndarray:
    """
    Build an equicorrelation matrix with 1 on the diagonal and rho off-diagonal.
    Ensures rho is within the PSD range [-1/(n-1), 1).
    """
    n = int(n_assets)
    if n <= 1:
        return np.eye(max(n, 1), dtype=float)
    rho_min = -1.0 / float(n - 1) + 1e-9
    rho_max = 0.999999
    rho_clamped = float(min(max(float(rho), rho_min), rho_max))
    mat = np.full((n, n), rho_clamped, dtype=float)
    np.fill_diagonal(mat, 1.0)
    return mat


def _correlated_normals(
    rng: np.random.Generator, n_paths: int, n_assets: int, rho: float | None
) -> np.ndarray:
    """
    Return correlated standard normals (n_paths, n_assets) using an equicorrelation rho.
    Falls back to iid normals if Cholesky fails.
    """
    paths = int(max(1, n_paths))
    assets = int(max(1, n_assets))
    z = rng.standard_normal((paths, assets))
    if rho is None or assets <= 1:
        return z
    try:
        corr = _constant_corr_matrix(assets, float(rho))
        L = np.linalg.cholesky(corr)
        return z @ L.T
    except Exception:
        return z


def price_basket_call(
    weights,
    spots,
    r,
    q,
    T,
    sigma,
    *,
    strike: float | None = None,
    rho: float | None = None,
    n_paths: int = 5000,
    seed=None,
):
    """
    Basket call option price via Monte Carlo (log-normal, common sigma, optional equicorrelation rho).

    Notes:
      - If strike is None, we price an ATM basket with K = sum(w_i * S_i0) after weight normalization.
      - weights are normalized to sum to 1.0.
    """
    weights_arr = np.array(weights, dtype=float)
    spots_arr = np.array(spots, dtype=float)
    if weights_arr.size == 0 or spots_arr.size == 0 or weights_arr.size != spots_arr.size:
        return 0.0

    if float(T) <= 0 or float(sigma) <= 0:
        w_sum = float(weights_arr.sum()) if weights_arr.size else 0.0
        if not np.isfinite(w_sum) or abs(w_sum) <= 1e-12:
            return 0.0
        w_norm = weights_arr / w_sum
        basket0 = float(np.sum(w_norm * spots_arr))
        K = float(basket0 if strike is None else strike)
        return float(max(basket0 - K, 0.0))

    w_sum = float(weights_arr.sum())
    if not np.isfinite(w_sum) or abs(w_sum) <= 1e-12:
        return 0.0
    w_norm = weights_arr / w_sum

    dt = float(T)
    K = float(np.sum(w_norm * spots_arr)) if strike is None else float(strike)
    rng = np.random.default_rng(seed)
    z = _correlated_normals(rng, int(n_paths), int(len(w_norm)), rho)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    diff = float(sigma) * math.sqrt(dt) * z
    sims = spots_arr * np.exp(drift + diff)
    basket = (sims * w_norm).sum(axis=1)
    payoff = np.maximum(basket - K, 0.0)
    return float(math.exp(-float(r) * dt) * payoff.mean())


def price_basket_put(
    weights,
    spots,
    r,
    q,
    T,
    sigma,
    *,
    strike: float | None = None,
    rho: float | None = None,
    n_paths: int = 5000,
    seed=None,
):
    """
    Basket put option price via Monte Carlo (log-normal, common sigma, optional equicorrelation rho).

    Notes:
      - If strike is None, we price an ATM basket with K = sum(w_i * S_i0) after weight normalization.
      - weights are normalized to sum to 1.0.
    """
    weights_arr = np.array(weights, dtype=float)
    spots_arr = np.array(spots, dtype=float)
    if weights_arr.size == 0 or spots_arr.size == 0 or weights_arr.size != spots_arr.size:
        return 0.0

    if float(T) <= 0 or float(sigma) <= 0:
        w_sum = float(weights_arr.sum()) if weights_arr.size else 0.0
        if not np.isfinite(w_sum) or abs(w_sum) <= 1e-12:
            return 0.0
        w_norm = weights_arr / w_sum
        basket0 = float(np.sum(w_norm * spots_arr))
        K = float(basket0 if strike is None else strike)
        return float(max(K - basket0, 0.0))

    w_sum = float(weights_arr.sum())
    if not np.isfinite(w_sum) or abs(w_sum) <= 1e-12:
        return 0.0
    w_norm = weights_arr / w_sum

    dt = float(T)
    K = float(np.sum(w_norm * spots_arr)) if strike is None else float(strike)
    rng = np.random.default_rng(seed)
    z = _correlated_normals(rng, int(n_paths), int(len(w_norm)), rho)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    diff = float(sigma) * math.sqrt(dt) * z
    sims = spots_arr * np.exp(drift + diff)
    basket = (sims * w_norm).sum(axis=1)
    payoff = np.maximum(K - basket, 0.0)
    return float(math.exp(-float(r) * dt) * payoff.mean())


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0
) -> float:
    """
    Black-Scholes price for a European option.

    Falls back to intrinsic if inputs are degenerate.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return float(S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))
    return float(K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1))
