"""
Tests for the rough-vol forward curve / discounting / IV recomputation.

Target: app/model/calibration/rough_vol/forward_curve.py (spec 4.2)

ORACLES (all independent of the code under test):
  - The synthetic chain is built forward: a KNOWN spot, rate, dividend yield and
    flat vol produce call prices through the repo's `bs_call_price`, and put
    prices through exact put-call parity. The forward F = S0 exp((r-q)T) and the
    discount D = exp(-rT) are therefore known *by construction*, never read back
    from the module under test. A non-zero dividend keeps F != S0 so a module
    that silently returned the spot would fail.
  - Black-76 is restated inline with scipy.stats.norm as an independent pricer,
    and cross-checked against the module's `black76_call_price`.
  - The implied dividend yield is checked against the q used to build the chain.

Numerical note: `implied_vol_call` inverts with Brent at xtol=1e-6, so IV
round-trips are asserted at 1e-6, not at machine precision.

Determinism: no RNG, no network, no Monte-Carlo. Discounting is always injected
(explicit D / r, or a fake curve object), so the yield-curve loader is never
touched.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
from scipy.stats import norm

from app.model.calibration.implied_vol import bs_call_price
from app.model.calibration.rough_vol.chain_cleaning import CleaningConfig, clean_expiry_chain
from app.model.calibration.rough_vol.forward_curve import (
    FLAG_PARITY_FALLBACK,
    FLAG_PARITY_SLOPE_OFF,
    METHOD_PARITY_REGRESSION,
    METHOD_PARITY_SINGLE_STRIKE,
    REASON_NO_OTM_QUOTE,
    SOURCE_ACTIVE_CURVE,
    SOURCE_CURVE_OBJECT,
    SOURCE_EXPLICIT_D,
    SOURCE_EXPLICIT_R,
    ForwardConfig,
    SurfaceConfig,
    black76_call_price,
    build_forward_curve,
    build_forward_point,
    build_otm_surface,
    collect_parity_pairs,
    estimate_forward,
    forward_curve_report,
    implied_dividend_yield,
    implied_vol_from_forward,
    resolve_discount,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Known market: a NON-ZERO dividend yield, so F != S0 by construction.
# ---------------------------------------------------------------------------

S0 = 100.0
T = 0.5
R = 0.03
Q = 0.015
VOL = 0.22
D = math.exp(-R * T)
F = S0 * math.exp((R - Q) * T)
STRIKES = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]


def test_known_market_is_self_consistent():
    """Guard the oracle itself: F != S0, and exp(-rT) == D exactly."""
    assert F == pytest.approx(100.7528, abs=1e-3)
    assert abs(F - S0) > 0.5
    assert math.exp(-(-math.log(D) / T) * T) == pytest.approx(D, rel=0, abs=1e-15)


def call_price(K: float) -> float:
    return bs_call_price(S0, float(K), T, R, Q, VOL)


def put_price(K: float) -> float:
    """Exact put-call parity on the forward: P = C - D (F - K)."""
    return call_price(K) - D * (F - float(K))


def make_row(option_type: str, K: float, price: float, *, spread_rel: float) -> dict:
    half = 0.5 * float(spread_rel)
    return {
        "underlying": "TEST",
        "contractSymbol": f"TEST-{option_type}-{K:g}",
        "expiry": "2026-07-01",
        "expiry_ts": 1_782_864_000,
        "T": T,
        "strike": float(K),
        # Deliberately wrong vendor IV: the pipeline must never read it.
        "iv": 0.99,
        "bid": price * (1.0 - half),
        "ask": price * (1.0 + half),
        "lastPrice": price,
        "openInterest": 25,
        "volume": 7,
        "inTheMoney": False,
        "type": option_type,
        "S0": S0,
    }


def synthetic_chain(
    *,
    call_strikes=None,
    put_strikes=None,
    spread_rel: float = 0.02,
):
    """Cleaned chain built from the known market above."""
    calls = STRIKES if call_strikes is None else call_strikes
    puts = STRIKES if put_strikes is None else put_strikes
    rows = [make_row("call", K, call_price(K), spread_rel=spread_rel) for K in calls]
    rows += [make_row("put", K, put_price(K), spread_rel=spread_rel) for K in puts]
    chain = clean_expiry_chain(rows, config=CleaningConfig())
    assert chain.removals == (), f"le chantier de test doit être propre: {chain.removals}"
    return chain


def black76_reference(K: float, vol: float) -> float:
    """Independent Black-76 call price: D * (F N(d1) - K N(d2))."""
    sqrt_t = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / sqrt_t
    d2 = d1 - sqrt_t
    return float(D * (F * norm.cdf(d1) - K * norm.cdf(d2)))


# ---------------------------------------------------------------------------
# Discounting: injected, lazy, never from the chain fetch
# ---------------------------------------------------------------------------


class FakeCurve:
    """Minimal YieldCurve duck-type; counts how often it is consulted."""

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self.n_discount_calls = 0
        self.n_zero_calls = 0

    def discount_factor(self, T_years: float) -> float:
        self.n_discount_calls += 1
        return math.exp(-self.rate * float(T_years))

    def zero_rate(self, T_years: float) -> float:
        self.n_zero_calls += 1
        return self.rate


def test_resolve_discount_precedence_and_internal_consistency():
    explicit_d = resolve_discount(T, D=D)
    assert explicit_d.source == SOURCE_EXPLICIT_D
    assert explicit_d.D == pytest.approx(D, rel=0, abs=1e-15)
    assert explicit_d.r == pytest.approx(R, abs=1e-12)
    assert math.exp(-explicit_d.r * T) == pytest.approx(explicit_d.D, abs=1e-15)

    explicit_r = resolve_discount(T, r=R)
    assert explicit_r.source == SOURCE_EXPLICIT_R
    assert explicit_r.D == pytest.approx(D, abs=1e-15)

    curve = FakeCurve(R)
    from_curve = resolve_discount(T, curve=curve)
    assert from_curve.source == SOURCE_CURVE_OBJECT
    assert from_curve.D == pytest.approx(D, abs=1e-15)
    assert from_curve.r == pytest.approx(R, abs=1e-12)
    assert from_curve.r_curve == pytest.approx(R)
    assert from_curve.zero_rate_gap == pytest.approx(0.0, abs=1e-12)
    assert curve.n_discount_calls == 1

    # Explicit values must NOT consult the curve at all.
    curve2 = FakeCurve(R)
    resolve_discount(T, curve=curve2, D=D)
    resolve_discount(T, curve=curve2, r=R)
    assert curve2.n_discount_calls == 0

    assert from_curve.to_dict()["source"] == SOURCE_CURVE_OBJECT


def test_resolve_discount_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_discount(0.0, D=D)
    with pytest.raises(ValueError):
        resolve_discount(T, D=0.0)
    with pytest.raises(ValueError):
        resolve_discount(T, r=float("nan"))


def test_curve_loader_is_never_reached_when_discounting_is_injected(monkeypatch):
    """
    The `get_active_curve` import lives inside the function; with (r, D) or a
    curve injected it must never be executed. A booby-trapped module proves it.
    """
    boom = types.ModuleType("app.model.yieldcurve.service")

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("get_active_curve ne doit pas être appelé")

    boom.get_active_curve = _explode
    monkeypatch.setitem(sys.modules, "app.model.yieldcurve.service", boom)

    assert resolve_discount(T, D=D).source == SOURCE_EXPLICIT_D
    assert resolve_discount(T, r=R).source == SOURCE_EXPLICIT_R
    assert resolve_discount(T, curve=FakeCurve(R)).source == SOURCE_CURVE_OBJECT
    point = build_forward_point(synthetic_chain(), D=D)
    assert point is not None and point.discount_source == SOURCE_EXPLICIT_D
    # The default path would have used the (trapped) active curve.
    assert SOURCE_ACTIVE_CURVE == "active_curve"


# ---------------------------------------------------------------------------
# Put-call parity forward
# ---------------------------------------------------------------------------


def test_parity_regression_recovers_the_known_forward_and_slope():
    chain = synthetic_chain()
    point = build_forward_point(chain, D=D)
    assert point is not None

    assert point.parity_slope_diag.method == METHOD_PARITY_REGRESSION
    assert point.F == pytest.approx(F, abs=1e-8)
    assert point.D == pytest.approx(D, abs=1e-15)
    assert point.r == pytest.approx(R, abs=1e-12)
    assert point.flags == ()

    diag = point.parity_slope_diag
    assert diag.slope == pytest.approx(-D, abs=1e-10)
    assert diag.slope_expected == pytest.approx(-D, abs=1e-15)
    assert diag.slope_abs_error == pytest.approx(0.0, abs=1e-10)
    assert diag.slope_rel_error == pytest.approx(0.0, abs=1e-10)
    assert diag.implied_discount == pytest.approx(D, abs=1e-10)
    assert diag.r_squared == pytest.approx(1.0, abs=1e-12)
    assert diag.residual_rmse == pytest.approx(0.0, abs=1e-10)
    # The free-slope forward agrees too, since the data is exactly parity-consistent.
    assert diag.forward_free_slope == pytest.approx(F, abs=1e-6)
    assert diag.n_pairs == ForwardConfig().max_pairs

    # Diagnostics survive the controller `_json_safe` (plain floats / lists).
    as_dict = point.to_dict()
    assert isinstance(as_dict["parity_slope_diag"]["strikes"], list)
    assert all(isinstance(x, float) for x in as_dict["parity_slope_diag"]["strikes"])


def test_parity_regression_uses_the_strikes_nearest_the_money():
    chain = synthetic_chain()
    config = ForwardConfig(max_pairs=4)
    pairs = collect_parity_pairs(chain, config=config)
    assert len(pairs) == len(STRIKES)

    estimate = estimate_forward(pairs, D=D, config=config)
    assert estimate is not None
    forward, diag = estimate
    assert forward == pytest.approx(F, abs=1e-8)
    assert diag.n_pairs == 4
    # F ~ 100.75 -> the four nearest strikes are 95, 100, 105, 110.
    assert list(diag.strikes) == [95.0, 100.0, 105.0, 110.0]


def test_q_implied_recovers_the_dividend_yield_used_to_build_the_chain():
    point = build_forward_point(synthetic_chain(), D=D)
    assert point is not None
    assert point.S0 == pytest.approx(S0)
    assert point.q_implied == pytest.approx(Q, abs=1e-9)

    # Direct restatement of the definition, on the known values.
    assert implied_dividend_yield(F=F, S0=S0, T=T, r=R) == pytest.approx(Q, abs=1e-12)
    assert math.isnan(implied_dividend_yield(F=F, S0=0.0, T=T, r=R))
    assert math.isnan(implied_dividend_yield(F=-1.0, S0=S0, T=T, r=R))


def test_single_pair_fallback_recovers_the_forward():
    """Only one strike is quoted on both sides -> single-strike parity fallback."""
    chain = synthetic_chain(call_strikes=[100.0, 105.0, 110.0], put_strikes=[100.0])
    pairs = collect_parity_pairs(chain)
    assert [p.strike for p in pairs] == [100.0]

    point = build_forward_point(chain, D=D)
    assert point is not None
    assert point.parity_slope_diag.method == METHOD_PARITY_SINGLE_STRIKE
    assert FLAG_PARITY_FALLBACK in point.flags
    assert point.F == pytest.approx(F, abs=1e-8)
    assert point.q_implied == pytest.approx(Q, abs=1e-9)
    # No slope can be fitted from a single point: the diagnostic says so.
    assert math.isnan(point.parity_slope_diag.slope)
    assert list(point.parity_slope_diag.strikes) == [100.0]

    # Hand-computed oracle: F = K* + (C(K*) - P(K*)) / D.
    manual = 100.0 + (call_price(100.0) - put_price(100.0)) / D
    assert point.F == pytest.approx(manual, abs=1e-12)


def test_no_parity_pair_yields_no_forward_point():
    chain = synthetic_chain(call_strikes=[110.0, 115.0], put_strikes=[85.0, 90.0])
    assert collect_parity_pairs(chain) == []
    assert build_forward_point(chain, D=D) is None
    assert build_forward_curve([chain], discounts={T: D}) == []


def test_slope_diagnostic_exposes_inconsistent_discounting():
    """
    Feed a discount factor that contradicts the chain: the fitted free slope
    still recovers the TRUE discount, and the reported discrepancy flags it.
    """
    wrong_D = 0.9 * D
    point = build_forward_point(synthetic_chain(), D=wrong_D)
    assert point is not None

    diag = point.parity_slope_diag
    assert diag.implied_discount == pytest.approx(D, abs=1e-10)
    assert diag.slope_expected == pytest.approx(-wrong_D)
    assert diag.slope_rel_error == pytest.approx(abs(D - wrong_D) / wrong_D, rel=1e-8)
    assert diag.slope_rel_error > ForwardConfig().slope_tolerance_rel
    assert FLAG_PARITY_SLOPE_OFF in point.flags
    # Nothing is dropped: the forward is still reported, just flagged.
    assert point.F > 0.0


def test_build_forward_curve_sorts_and_reports():
    chain = synthetic_chain()
    points = build_forward_curve([chain], discounts={T: D})
    assert len(points) == 1
    assert points[0].F == pytest.approx(F, abs=1e-8)

    report = forward_curve_report(points)
    assert report["n_expiries"] == 1
    assert report["n_parity_fallbacks"] == 0
    assert report["n_slope_off"] == 0
    assert isinstance(report["points"][0]["F"], float)


# ---------------------------------------------------------------------------
# Black-76 inversion
# ---------------------------------------------------------------------------


def test_black76_price_matches_an_independent_oracle():
    for K in STRIKES:
        for vol in (0.10, VOL, 0.45):
            assert black76_call_price(F=F, K=K, T=T, D=D, vol=vol) == pytest.approx(
                black76_reference(K, vol), rel=1e-12, abs=1e-12
            )
    # The repo's spot-parametrised call must agree with Black-76 on the same market.
    for K in STRIKES:
        assert call_price(K) == pytest.approx(black76_reference(K, VOL), rel=1e-12, abs=1e-12)


def test_black76_inversion_round_trip():
    """Price a known-vol Black-76 call, invert it, recover the vol."""
    for K in STRIKES:
        for vol in (0.10, VOL, 0.45):
            price = black76_call_price(F=F, K=K, T=T, D=D, vol=vol)
            recovered = implied_vol_from_forward(price, K=K, T=T, F=F, D=D, option_type="call")
            assert recovered == pytest.approx(vol, abs=1e-6)


def test_put_side_ivs_recovered_via_parity_match_the_call_side():
    for K in STRIKES:
        iv_call = implied_vol_from_forward(
            call_price(K), K=K, T=T, F=F, D=D, option_type="call"
        )
        iv_put = implied_vol_from_forward(put_price(K), K=K, T=T, F=F, D=D, option_type="put")
        assert iv_call == pytest.approx(VOL, abs=1e-6)
        # The parity conversion is exact, so both sides invert the same price.
        assert iv_put == pytest.approx(iv_call, abs=1e-9)


def test_implied_vol_from_forward_guards():
    assert math.isnan(implied_vol_from_forward(1.0, K=100.0, T=T, F=F, D=0.0))
    assert math.isnan(implied_vol_from_forward(1.0, K=100.0, T=0.0, F=F, D=D))
    assert math.isnan(implied_vol_from_forward(float("nan"), K=100.0, T=T, F=F, D=D))
    # A put deep below the forward converts to a positive call price...
    assert implied_vol_from_forward(
        put_price(80.0), K=80.0, T=T, F=F, D=D, option_type="put"
    ) == pytest.approx(VOL, abs=1e-6)
    # ...but a nonsensical price has no implied vol rather than a silent guess.
    assert math.isnan(implied_vol_from_forward(-1.0, K=100.0, T=T, F=F, D=D))
    with pytest.raises(ValueError):
        implied_vol_from_forward(1.0, K=100.0, T=T, F=F, D=D, option_type="straddle")


# ---------------------------------------------------------------------------
# OTM market surface (recomputed IVs, log-forward moneyness)
# ---------------------------------------------------------------------------


def test_otm_surface_uses_puts_below_the_forward_and_calls_above():
    chain = synthetic_chain()
    point = build_forward_point(chain, D=D)
    assert point is not None

    points, rejections = build_otm_surface(chain, point)

    assert rejections == []
    assert [p.K for p in points] == sorted(STRIKES)
    assert [p.k for p in points] == sorted(p.k for p in points)
    for p in points:
        assert p.k == pytest.approx(math.log(p.K / point.F), abs=1e-15)
        assert p.option_type == ("put" if p.K < point.F else "call")
        # Recomputed IV, NOT the (deliberately wrong) vendor IV of 0.99.
        assert p.iv == pytest.approx(VOL, abs=1e-6)
        assert p.vendor_iv == pytest.approx(0.99)
        assert p.vendor_iv_gap == pytest.approx(p.iv - 0.99, abs=1e-12)
        assert isinstance(p.to_dict()["k"], float)

    # The call-equivalent price of a put leg is its parity image.
    put_points = [p for p in points if p.option_type == "put"]
    assert put_points
    for p in put_points:
        assert p.call_equivalent_price == pytest.approx(call_price(p.K), abs=1e-10)


def test_otm_surface_rejects_a_strike_missing_its_otm_leg():
    # Strike 90 < F is only quoted as a call: the OTM (put) leg is missing.
    chain = synthetic_chain(call_strikes=STRIKES, put_strikes=[95.0, 100.0, 105.0])
    point = build_forward_point(chain, D=D)
    assert point is not None

    points, rejections = build_otm_surface(chain, point)

    rejected_strikes = sorted(r.strike for r in rejections)
    assert rejected_strikes == [80.0, 85.0, 90.0]
    assert {r.reason for r in rejections} == {REASON_NO_OTM_QUOTE}
    assert all(r.option_type == "put" for r in rejections)
    assert 90.0 not in [p.K for p in points]
    assert isinstance(rejections[0].to_dict()["detail"]["k"], float)


def test_otm_surface_moneyness_window_is_configurable():
    chain = synthetic_chain()
    point = build_forward_point(chain, D=D)
    assert point is not None

    points, rejections = build_otm_surface(
        chain, point, config=SurfaceConfig(max_abs_log_moneyness=0.10)
    )
    assert points
    assert all(abs(p.k) <= 0.10 for p in points)
    assert all(abs(math.log(r.strike / point.F)) > 0.10 for r in rejections)
