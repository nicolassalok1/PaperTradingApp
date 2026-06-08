"""Tests quant pour le pricing Black-Scholes/Merton (européen).

Oracles INDEPENDANTS du code testé :
- valeur closed-form de litterature (S=K=100, r=0.05, q=0, sigma=0.2, T=1 -> call ~ 10.4506),
- put-call parity : C - P = S e^{-qT} - K e^{-rT},
- bornes no-arbitrage : max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT},
- monotonies (croissant en sigma, croissant en S pour un call),
- limite sigma -> 0 (valeur du forward actualise).

Cibles :
- app/model/options/engines/black_scholes.py :: black_scholes_price(S, K, T, r, sigma, option_type, q)
- app/model/options/core/pricing_lib.py :: bs_price_call / bs_price_put
"""

import math

import pytest

from app.model.options.engines.black_scholes import black_scholes_price
from app.model.options.core.pricing_lib import (
    bs_price_call,
    bs_price_put,
    price_european_call_bs,
)

pytestmark = pytest.mark.unit


# --- Oracle independant : Black-Scholes closed-form recalcule a la main ---------


def _ref_bsm(S, K, T, r, sigma, q, option_type):
    """Reference BSM independante (formule analytique reimplementee)."""
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    nd = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if option_type == "call":
        return S * math.exp(-q * T) * nd(d1) - K * math.exp(-r * T) * nd(d2)
    return K * math.exp(-r * T) * nd(-d2) - S * math.exp(-q * T) * nd(-d1)


# --- 1. Valeur closed-form connue de la litterature ----------------------------


def test_known_atm_call_value():
    # Hull / valeur de reference universelle : ~10.4506
    price = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_known_atm_put_value():
    # Deduit par parite a partir de la valeur de litterature du call : ~5.5735
    price = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_pricing_lib_call_matches_literature():
    price = bs_price_call(100.0, 100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_pricing_lib_put_matches_literature():
    price = bs_price_put(100.0, 100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_price_european_call_bs_alias_matches():
    # Le wrapper doit reproduire bs_price_call sur les memes inputs.
    p1 = price_european_call_bs(100.0, 95.0, r=0.03, sigma=0.25, T=0.5, q=0.01)
    p2 = bs_price_call(100.0, 95.0, r=0.03, q=0.01, sigma=0.25, T=0.5)
    assert p1 == pytest.approx(p2, rel=1e-12)


# --- 2. Put-call parity : C - P == S e^{-qT} - K e^{-rT} -----------------------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (120.0, 100.0, 0.5, 0.03, 0.35, 0.02),
        (80.0, 100.0, 2.0, 0.07, 0.15, 0.04),
        (100.0, 110.0, 0.25, 0.01, 0.5, 0.0),
    ],
)
def test_put_call_parity_engine(S, K, T, r, sigma, q):
    call = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", q=q)
    put = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put", q=q)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (call - put) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (120.0, 100.0, 0.5, 0.03, 0.35, 0.02),
        (80.0, 100.0, 2.0, 0.07, 0.15, 0.04),
    ],
)
def test_put_call_parity_pricing_lib(S, K, T, r, sigma, q):
    call = bs_price_call(S, K, r=r, q=q, sigma=sigma, T=T)
    put = bs_price_put(S, K, r=r, q=q, sigma=sigma, T=T)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (call - put) == pytest.approx(expected, abs=1e-9)


# --- 3. Bornes no-arbitrage -----------------------------------------------------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (60.0, 100.0, 1.0, 0.05, 0.3, 0.0),
        (150.0, 100.0, 1.0, 0.05, 0.3, 0.02),
        (100.0, 100.0, 1.0, 0.05, 0.8, 0.0),
    ],
)
def test_call_no_arbitrage_bounds(S, K, T, r, sigma, q):
    call = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", q=q)
    upper = S * math.exp(-q * T)
    lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    assert call >= lower - 1e-9
    assert call <= upper + 1e-9
    assert call >= -1e-12


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (150.0, 100.0, 1.0, 0.05, 0.3, 0.0),
        (60.0, 100.0, 1.0, 0.05, 0.3, 0.02),
    ],
)
def test_put_no_arbitrage_bounds(S, K, T, r, sigma, q):
    put = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put", q=q)
    upper = K * math.exp(-r * T)
    lower = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    assert put >= lower - 1e-9
    assert put <= upper + 1e-9


# --- 4. Monotonies --------------------------------------------------------------


def test_call_increasing_in_sigma():
    sigmas = [0.05, 0.1, 0.2, 0.4, 0.8]
    prices = [
        black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=s, option_type="call")
        for s in sigmas
    ]
    for a, b in zip(prices, prices[1:]):
        assert b > a


def test_put_increasing_in_sigma():
    sigmas = [0.05, 0.1, 0.2, 0.4, 0.8]
    prices = [
        black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=s, option_type="put")
        for s in sigmas
    ]
    for a, b in zip(prices, prices[1:]):
        assert b > a


def test_call_increasing_in_spot():
    spots = [60.0, 80.0, 100.0, 120.0, 140.0]
    prices = [
        black_scholes_price(S=S, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
        for S in spots
    ]
    for a, b in zip(prices, prices[1:]):
        assert b > a


def test_put_decreasing_in_spot():
    spots = [60.0, 80.0, 100.0, 120.0, 140.0]
    prices = [
        black_scholes_price(S=S, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="put")
        for S in spots
    ]
    for a, b in zip(prices, prices[1:]):
        assert b < a


# --- 5. Limite sigma -> 0 : valeur deterministe (forward actualise) -------------


def test_call_limit_sigma_to_zero_itm():
    # Forward F = S e^{(r-q)T} > K -> call -> e^{-rT}(F - K) = S e^{-qT} - K e^{-rT}.
    S, K, T, r, q = 120.0, 100.0, 1.0, 0.05, 0.0
    # sigma tres petit (mais > 0 pour rester dans la branche analytique)
    price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=1e-6, option_type="call")
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert price == pytest.approx(expected, abs=1e-3)


def test_call_limit_sigma_to_zero_otm():
    # Forward < K -> call -> 0.
    price = black_scholes_price(S=80.0, K=100.0, T=1.0, r=0.05, sigma=1e-6, option_type="call")
    assert price == pytest.approx(0.0, abs=1e-6)


def test_put_limit_sigma_to_zero_itm():
    # Forward < K -> put ITM -> e^{-rT}(K - F) = K e^{-rT} - S e^{-qT}.
    S, K, T, r, q = 80.0, 100.0, 1.0, 0.05, 0.0
    price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=1e-6, option_type="put")
    expected = K * math.exp(-r * T) - S * math.exp(-q * T)
    assert price == pytest.approx(expected, abs=1e-3)


# --- 6. Coherence avec l'oracle analytique independant sur une grille -----------


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q, opt",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call"),
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put"),
        (110.0, 90.0, 0.75, 0.02, 0.4, 0.03, "call"),
        (90.0, 110.0, 1.5, 0.06, 0.25, 0.01, "put"),
    ],
)
def test_matches_independent_closed_form(S, K, T, r, sigma, q, opt):
    got = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=opt, q=q)
    expected = _ref_bsm(S, K, T, r, sigma, q, opt)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


# --- 7. Effet du dividende q : un call diminue, un put augmente avec q ----------


def test_call_decreasing_in_dividend_yield():
    qs = [0.0, 0.02, 0.05, 0.1]
    prices = [
        black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call", q=q)
        for q in qs
    ]
    for a, b in zip(prices, prices[1:]):
        assert b < a


def test_put_increasing_in_dividend_yield():
    qs = [0.0, 0.02, 0.05, 0.1]
    prices = [
        black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="put", q=q)
        for q in qs
    ]
    for a, b in zip(prices, prices[1:]):
        assert b > a


# --- 8. Branche degeneree (fallback intrinseque vs zero) ------------------------


def test_engine_degenerate_returns_intrinsic():
    # black_scholes_price retombe sur l'intrinseque (non actualise) si T<=0 ou sigma<=0.
    assert black_scholes_price(S=120.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type="call") == pytest.approx(20.0)
    assert black_scholes_price(S=80.0, K=100.0, T=1.0, r=0.05, sigma=0.0, option_type="put") == pytest.approx(20.0)
    assert black_scholes_price(S=120.0, K=100.0, T=1.0, r=0.05, sigma=0.0, option_type="put") == pytest.approx(0.0)


def test_pricing_lib_degenerate_returns_zero():
    # bs_price_call/put renvoient 0.0 sur inputs degeneres (convention du module).
    assert bs_price_call(100.0, 100.0, sigma=0.0, T=1.0) == 0.0
    assert bs_price_put(100.0, 100.0, sigma=0.2, T=0.0) == 0.0
