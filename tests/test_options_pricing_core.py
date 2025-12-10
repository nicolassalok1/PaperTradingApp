import numpy as np
from app.model.options.core.pricing_lib import price_european_call_bs
from app.model.options.core.payoff import call_payoff


def test_bs_call_price_monotonic_in_spot():
    S1, S2 = 90.0, 110.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    T = 1.0

    p1 = price_european_call_bs(S1, K, r, sigma, T)
    p2 = price_european_call_bs(S2, K, r, sigma, T)

    assert p2 > p1  # plus le spot monte, plus le call vaut cher


def test_call_payoff_sanity():
    S = np.array([80, 100, 120], dtype=float)
    K = 100.0
    payoff = call_payoff(S, K)
    assert payoff.tolist() == [0.0, 0.0, 20.0]
