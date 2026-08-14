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
    # Rubinstein: the strike is fixed at T_start to the spot observed then, so by
    # scaling the value is S0*exp(-q*T_start) times an at-the-money vanilla over the
    # remaining maturity tau. Both legs are discounted, and the (r-q) drift belongs
    # in d1 — dropping either is invisible at r = q = 0 and wrong everywhere else.
    tau = float(T_end) - float(T_start)
    sigma = float(sigma)
    root = sigma * math.sqrt(tau)
    d1 = (float(r) - float(q) + 0.5 * sigma * sigma) * tau / root
    d2 = d1 - root

    prefactor = float(S0) * math.exp(-float(q) * float(T_start))
    disc_r = math.exp(-float(r) * tau)
    disc_q = math.exp(-float(q) * tau)
    if str(option_type).lower().startswith("c"):
        return float(prefactor * (disc_q * _norm_cdf(d1) - disc_r * _norm_cdf(d2)))
    return float(prefactor * (disc_r * _norm_cdf(-d2) - disc_q * _norm_cdf(-d1)))


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
