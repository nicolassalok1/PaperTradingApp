import math

import numpy as np
import pandas as pd

from app.model.options.core import pricing_lib
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
