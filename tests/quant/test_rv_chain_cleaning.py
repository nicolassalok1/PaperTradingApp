"""
Tests for the rough-vol market-data cleaner.

Target: app/model/calibration/rough_vol/chain_cleaning.py (spec 4.1)

ORACLES (all independent of the code under test):
  - Quote-validity outcomes are decided by hand-built rows whose expected
    verdict is obvious from the bid/ask themselves (crossed, zero bid, stale,
    wide spread).
  - The arbitrage checks are pinned against textbook static no-arbitrage
    conditions restated inline: call mids non-increasing in K, put mids
    non-decreasing, and mid(K2) <= w*mid(K1) + (1-w)*mid(K3) with
    w = (K3-K2)/(K3-K1). The violating chains are built by *perturbing one mid*
    of an arbitrage-free Black-Scholes chain, so the violation is known by
    construction, not by asking the cleaner.
  - Arbitrage-free reference chains come from the repo's own
    `bs_call_price` with puts derived by exact put-call parity: any removal on
    such a chain would itself be a bug.

Determinism: no RNG, no network, no Monte-Carlo.
"""

from __future__ import annotations

import math

import pytest

from app.model.calibration.implied_vol import bs_call_price
from app.model.calibration.rough_vol.chain_cleaning import (
    AMERICAN_EXERCISE_CAVEAT,
    FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN,
    FLAG_EMPTY_CHAIN,
    FLAG_ONE_SIDED,
    FLAG_VENDOR_IV_MISSING,
    FLAG_VENDOR_IV_SENTINEL,
    REASON_BUTTERFLY_ARBITRAGE,
    REASON_CROSSED_QUOTE,
    REASON_STALE_QUOTE,
    REASON_VERTICAL_ARBITRAGE,
    REASON_WIDE_SPREAD,
    REASON_ZERO_BID_TAIL,
    CleaningConfig,
    clean_expiry_chain,
    clean_option_chains,
    cleaning_report,
    evaluate_viability,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture-free row builders (the raw fetch_options_details_yahoo schema)
# ---------------------------------------------------------------------------

S0 = 100.0
T_TEST = 0.25
R_TEST = 0.02
Q_TEST = 0.0
VOL_TEST = 0.20
D_TEST = math.exp(-R_TEST * T_TEST)
F_TEST = S0 * math.exp((R_TEST - Q_TEST) * T_TEST)


def make_row(
    *,
    option_type: str,
    strike: float,
    bid: float,
    ask: float,
    iv: float = VOL_TEST,
    volume: float | None = 5.0,
    open_interest: float | None = 10.0,
    T: float = T_TEST,
    spot: float = S0,
    expiry_ts: int = 1_767_225_600,
) -> dict:
    """One raw chain row, mirroring the Yahoo schema exactly."""
    return {
        "underlying": "TEST",
        "contractSymbol": f"TEST-{option_type or '?'}-{strike!r}",
        "expiry": "2026-01-01",
        "expiry_ts": expiry_ts,
        "T": T,
        "strike": float(strike),
        "iv": iv,
        "bid": bid,
        "ask": ask,
        "lastPrice": 0.5 * (bid + ask),
        "openInterest": open_interest,
        "volume": volume,
        "inTheMoney": bool(strike < spot) if option_type == "call" else bool(strike > spot),
        "type": option_type,
        "S0": spot,
    }


def arbitrage_free_rows(
    strikes,
    *,
    spread_rel: float = 0.02,
    iv: float = VOL_TEST,
    with_calls: bool = True,
    with_puts: bool = True,
) -> list[dict]:
    """
    Calls from Black-Scholes, puts from exact put-call parity.

    The resulting mids satisfy every static no-arbitrage condition, so this chain
    must survive the cleaner untouched. `spread_rel` is applied multiplicatively
    around the mid, which leaves the mid exactly equal to the model price.
    """
    rows: list[dict] = []
    half = 0.5 * float(spread_rel)
    for K in strikes:
        call = bs_call_price(S0, float(K), T_TEST, R_TEST, Q_TEST, VOL_TEST)
        put = call - D_TEST * (F_TEST - float(K))
        if with_calls:
            rows.append(
                make_row(
                    option_type="call",
                    strike=float(K),
                    bid=call * (1.0 - half),
                    ask=call * (1.0 + half),
                    iv=iv,
                )
            )
        if with_puts:
            rows.append(
                make_row(
                    option_type="put",
                    strike=float(K),
                    bid=put * (1.0 - half),
                    ask=put * (1.0 + half),
                    iv=iv,
                )
            )
    return rows


def mids_by_strike(quotes) -> dict[float, float]:
    return {float(q.strike): float(q.mid) for q in quotes}


# ---------------------------------------------------------------------------
# Baseline: a clean chain must survive untouched
# ---------------------------------------------------------------------------


def test_arbitrage_free_chain_survives_untouched():
    strikes = [85, 90, 95, 100, 105, 110, 115]
    chain = clean_expiry_chain(arbitrage_free_rows(strikes))

    assert chain.removals == ()
    assert len(chain.calls) == len(strikes)
    assert len(chain.puts) == len(strikes)
    assert chain.T == pytest.approx(T_TEST)
    assert chain.S0 == pytest.approx(S0)
    # The American-exercise assumption is stamped, never hidden.
    assert FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN in chain.flags
    assert "américain" in AMERICAN_EXERCISE_CAVEAT

    # Independent restatement of the no-arbitrage conditions the cleaner enforces.
    call_mids = [q.mid for q in chain.calls]
    put_mids = [q.mid for q in chain.puts]
    assert all(a >= b for a, b in zip(call_mids, call_mids[1:]))
    assert all(a <= b for a, b in zip(put_mids, put_mids[1:]))


# ---------------------------------------------------------------------------
# Quote validity
# ---------------------------------------------------------------------------


def test_crossed_quote_is_dropped_with_reason_code():
    rows = arbitrage_free_rows([95, 100, 105])
    rows.append(make_row(option_type="call", strike=97.5, bid=4.0, ask=3.0))

    chain = clean_expiry_chain(rows)

    crossed = [rec for rec in chain.removals if rec.reason == REASON_CROSSED_QUOTE]
    assert len(crossed) == 1
    assert crossed[0].strike == pytest.approx(97.5)
    assert crossed[0].option_type == "call"
    assert crossed[0].detail["bid"] == pytest.approx(4.0)
    assert crossed[0].detail["ask"] == pytest.approx(3.0)
    assert 97.5 not in mids_by_strike(chain.calls)
    # The record is JSON-friendly and carries a French label for the report.
    assert crossed[0].to_dict()["reason"] == REASON_CROSSED_QUOTE
    assert isinstance(crossed[0].to_dict()["reason_fr"], str)


def test_zero_bid_is_flagged_one_sided_not_dropped():
    rows = arbitrage_free_rows([100, 105])
    rows.append(make_row(option_type="call", strike=130.0, bid=0.0, ask=0.05))

    chain = clean_expiry_chain(rows)

    assert all(rec.strike != pytest.approx(130.0) for rec in chain.removals)
    tail = [q for q in chain.calls if q.strike == pytest.approx(130.0)]
    assert len(tail) == 1
    assert tail[0].one_sided is True
    assert FLAG_ONE_SIDED in tail[0].flags
    assert tail[0].mid == pytest.approx(0.025)
    assert chain.diagnostics()["n_one_sided"] == 1

    # A zero bid makes (ask - bid) / mid identically 2, so the ratio test is
    # uninformative for one-sided quotes and is skipped by default. Turning the
    # exemption off drops them all -- that is the documented alternative.
    assert tail[0].spread_rel == pytest.approx(2.0)
    strict = clean_expiry_chain(
        rows, config=CleaningConfig(apply_spread_filter_to_one_sided=True)
    )
    assert [rec.reason for rec in strict.removals] == [REASON_WIDE_SPREAD]
    assert 130.0 not in mids_by_strike(strict.calls)


def test_negative_or_zero_ask_rows_are_dropped():
    rows = [
        make_row(option_type="call", strike=100.0, bid=0.0, ask=0.0),
        make_row(option_type="call", strike=105.0, bid=-1.0, ask=2.0),
    ]
    chain = clean_expiry_chain(rows)

    assert chain.calls == ()
    assert {rec.reason for rec in chain.removals} == {"nonpositive_ask", "negative_bid"}


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_stale_quote_dropped_or_kept_according_to_config():
    rows = arbitrage_free_rows([95, 100])
    rows.append(
        make_row(
            option_type="call",
            strike=120.0,
            bid=0.10,
            ask=0.12,
            volume=0,
            open_interest=0,
        )
    )

    dropped = clean_expiry_chain(rows, config=CleaningConfig(drop_stale=True))
    stale = [rec for rec in dropped.removals if rec.reason == REASON_STALE_QUOTE]
    assert len(stale) == 1
    assert stale[0].strike == pytest.approx(120.0)
    assert 120.0 not in mids_by_strike(dropped.calls)

    kept = clean_expiry_chain(rows, config=CleaningConfig(drop_stale=False))
    assert not [rec for rec in kept.removals if rec.reason == REASON_STALE_QUOTE]
    assert 120.0 in mids_by_strike(kept.calls)

    # A contract with activity on either leg is never stale.
    alive = clean_expiry_chain(
        [make_row(option_type="call", strike=120.0, bid=0.10, ask=0.12, volume=0, open_interest=7)],
        config=CleaningConfig(drop_stale=True),
    )
    assert len(alive.calls) == 1


# ---------------------------------------------------------------------------
# Spread filter
# ---------------------------------------------------------------------------


def test_wide_spread_beyond_s_max_is_dropped():
    config = CleaningConfig()
    rows = arbitrage_free_rows([95, 100])
    # OTM tail: (ask - bid) / mid = 1.2 / 1.0 = 1.2 > s_max_otm (0.5).
    rows.append(make_row(option_type="call", strike=120.0, bid=0.4, ask=1.6))

    chain = clean_expiry_chain(rows, config=config)

    wide = [rec for rec in chain.removals if rec.reason == REASON_WIDE_SPREAD]
    assert len(wide) == 1
    assert wide[0].strike == pytest.approx(120.0)
    assert wide[0].detail["spread_rel"] == pytest.approx(1.2)
    assert wide[0].detail["s_max"] == pytest.approx(config.s_max_otm)


def test_atm_band_uses_the_tighter_spread_ceiling():
    config = CleaningConfig(s_max_otm=0.5, s_max_atm=0.25, atm_band_log=0.05)
    # Relative spread 0.4: allowed in the tail, rejected at the money.
    atm = clean_expiry_chain(
        [make_row(option_type="call", strike=100.0, bid=4.0, ask=6.0)], config=config
    )
    otm = clean_expiry_chain(
        [make_row(option_type="call", strike=120.0, bid=0.8, ask=1.2)], config=config
    )

    assert [rec.reason for rec in atm.removals] == [REASON_WIDE_SPREAD]
    assert atm.removals[0].detail["s_max"] == pytest.approx(config.s_max_atm)
    assert otm.removals == ()
    assert len(otm.calls) == 1


# ---------------------------------------------------------------------------
# Arbitrage repair
# ---------------------------------------------------------------------------


def test_vertical_monotonicity_violation_on_calls_is_detected_and_logged():
    rows = [
        make_row(option_type="call", strike=90.0, bid=11.9, ask=12.1),   # mid 12.0
        make_row(option_type="call", strike=95.0, bid=7.9, ask=8.1),     # mid 8.0
        make_row(option_type="call", strike=100.0, bid=8.5, ask=9.5),    # mid 9.0 -> violation
    ]
    # Oracle: a call mid must not increase with the strike.
    assert 9.0 > 8.0

    chain = clean_expiry_chain(rows)

    vertical = [rec for rec in chain.removals if rec.reason == REASON_VERTICAL_ARBITRAGE]
    assert len(vertical) == 1
    # The wider-spread (worse) quote of the violating pair is the one removed.
    assert vertical[0].strike == pytest.approx(100.0)
    assert vertical[0].detail["mid"] == pytest.approx(9.0)
    assert vertical[0].detail["neighbour_strike"] == pytest.approx(95.0)

    mids = [q.mid for q in chain.calls]
    assert mids == [pytest.approx(12.0), pytest.approx(8.0)]
    assert all(a >= b for a, b in zip(mids, mids[1:]))


def test_vertical_monotonicity_violation_on_puts_is_detected_and_logged():
    rows = [
        make_row(option_type="put", strike=90.0, bid=1.95, ask=2.05),    # mid 2.0
        make_row(option_type="put", strike=95.0, bid=3.5, ask=4.5),      # mid 4.0
        make_row(option_type="put", strike=100.0, bid=2.95, ask=3.05),   # mid 3.0 -> violation
    ]
    # Oracle: a put mid must not decrease with the strike.
    assert 3.0 < 4.0

    chain = clean_expiry_chain(rows)

    vertical = [rec for rec in chain.removals if rec.reason == REASON_VERTICAL_ARBITRAGE]
    assert len(vertical) == 1
    assert vertical[0].strike == pytest.approx(95.0)
    mids = [q.mid for q in chain.puts]
    assert all(a <= b for a, b in zip(mids, mids[1:]))


def test_butterfly_convexity_violation_is_detected_and_logged():
    K1, K2, K3 = 90.0, 95.0, 100.0
    m1, m2, m3 = 12.0, 9.0, 5.0
    w = (K3 - K2) / (K3 - K1)
    # Independent oracle: the butterfly w*C(K1) + (1-w)*C(K3) - C(K2) must be >= 0.
    assert w * m1 + (1.0 - w) * m3 - m2 == pytest.approx(-0.5)

    rows = [
        make_row(option_type="call", strike=K1, bid=m1 - 0.1, ask=m1 + 0.1),
        make_row(option_type="call", strike=K2, bid=m2 - 0.1, ask=m2 + 0.1),
        make_row(option_type="call", strike=K3, bid=m3 - 0.1, ask=m3 + 0.1),
    ]
    chain = clean_expiry_chain(rows)

    butterfly = [rec for rec in chain.removals if rec.reason == REASON_BUTTERFLY_ARBITRAGE]
    assert len(butterfly) == 1
    # The middle strike is the expensive one, so it is the one removed.
    assert butterfly[0].strike == pytest.approx(K2)
    assert butterfly[0].detail["convex_bound"] == pytest.approx(8.5)
    assert butterfly[0].detail["butterfly"] == pytest.approx(-0.5)
    assert sorted(mids_by_strike(chain.calls)) == [K1, K3]

    # Surviving triples are convex (there are none left here, but the invariant
    # is what matters): re-checking the kept mids finds no violation.
    kept = sorted(mids_by_strike(chain.calls).items())
    for (ka, ma), (kb, mb), (kc, mc) in zip(kept, kept[1:], kept[2:]):
        wi = (kc - kb) / (kc - ka)
        assert mb <= wi * ma + (1.0 - wi) * mc + 1e-9


# ---------------------------------------------------------------------------
# Minimum viability
# ---------------------------------------------------------------------------


def test_expiry_short_of_otm_quotes_is_skew_only():
    config = CleaningConfig()
    # 5 strikes bracketing the forward, but only 2 OTM quotes on each side.
    chain = clean_expiry_chain(arbitrage_free_rows([90, 95, 100, 105, 110]), config=config)

    viability = chain.viability
    assert viability.n_otm_calls == 2 < config.n_min_otm_per_side
    assert viability.n_otm_puts == 2 < config.n_min_otm_per_side
    assert viability.usable_for_kvar is False
    assert "insufficient_otm_calls" in viability.reasons
    assert "insufficient_otm_puts" in viability.reasons

    assert viability.n_strikes_near_atm == 5 >= config.n_min_skew_strikes
    assert viability.usable_for_skew is True


def test_wide_expiry_is_usable_for_kvar():
    config = CleaningConfig()
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    chain = clean_expiry_chain(arbitrage_free_rows(strikes), config=config)

    viability = chain.viability
    assert viability.n_otm_calls >= config.n_min_otm_per_side
    assert viability.n_otm_puts >= config.n_min_otm_per_side
    assert viability.usable_for_kvar is True
    assert viability.usable_for_skew is True
    assert viability.reasons == ()


def test_viability_can_be_recomputed_against_the_true_forward():
    """Spec 4.2 re-runs the test on the parity forward, not the spot proxy."""
    config = CleaningConfig(n_min_otm_per_side=2)
    chain = clean_expiry_chain(arbitrage_free_rows([90, 95, 100, 105, 110]), config=config)
    assert chain.viability.usable_for_kvar is True

    # Push the reference forward above every call strike: no OTM call remains.
    shifted = evaluate_viability(chain.calls, chain.puts, forward_ref=200.0, config=config)
    assert shifted.n_otm_calls == 0
    assert shifted.usable_for_kvar is False


def test_no_forward_reference_yields_an_unusable_verdict():
    report = evaluate_viability((), (), forward_ref=float("nan"))
    assert report.usable_for_kvar is False
    assert report.usable_for_skew is False
    assert report.reasons == ("no_forward_reference",)


# ---------------------------------------------------------------------------
# Vendor IV is a cross-check, never an input
# ---------------------------------------------------------------------------


def test_vendor_iv_sentinel_is_flagged_and_never_gates_a_quote():
    config = CleaningConfig()
    strikes = [95, 100, 105]
    sentinel_iv = config.vendor_iv_sentinel  # exactly at the sentinel threshold

    normal = clean_expiry_chain(arbitrage_free_rows(strikes, iv=VOL_TEST), config=config)
    sentinel = clean_expiry_chain(arbitrage_free_rows(strikes, iv=sentinel_iv), config=config)
    missing = clean_expiry_chain(arbitrage_free_rows(strikes, iv=None), config=config)

    # 1. The sentinel is flagged, on every quote, and nothing is dropped for it.
    assert sentinel.removals == ()
    assert all(FLAG_VENDOR_IV_SENTINEL in q.flags for q in sentinel.quotes)
    assert all(FLAG_VENDOR_IV_SENTINEL not in q.flags for q in normal.quotes)
    assert all(FLAG_VENDOR_IV_MISSING in q.flags for q in missing.quotes)
    assert sentinel.diagnostics()["n_vendor_iv_sentinel"] == len(sentinel.quotes)

    # 2. The cleaned output is independent of the vendor IV: identical quotes,
    #    identical mids, identical removals — only the vendor column differs.
    def fingerprint(chain):
        return [
            (
                q.option_type,
                q.strike,
                q.bid,
                q.ask,
                q.mid,
                q.spread_rel,
                q.one_sided,
                tuple(
                    f
                    for f in q.flags
                    if f not in (FLAG_VENDOR_IV_SENTINEL, FLAG_VENDOR_IV_MISSING)
                ),
            )
            for q in chain.quotes
        ]

    assert fingerprint(normal) == fingerprint(sentinel) == fingerprint(missing)
    assert normal.viability == sentinel.viability == missing.viability
    assert len(normal.removals) == len(sentinel.removals) == len(missing.removals)

    # 3. The vendor value is still carried, for cross-checking only.
    assert all(q.vendor_iv == pytest.approx(sentinel_iv) for q in sentinel.quotes)
    assert all(math.isnan(q.vendor_iv) for q in missing.quotes)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_nan_riddled_chain_does_not_raise_and_yields_a_flagged_empty_chain():
    nan = float("nan")
    rows = [
        make_row(option_type="call", strike=nan, bid=1.0, ask=1.1),
        make_row(option_type="call", strike=100.0, bid=nan, ask=nan),
        make_row(option_type="put", strike=95.0, bid=1.0, ask=nan),
        make_row(option_type="put", strike=nan, bid=nan, ask=nan, T=nan, spot=nan),
        make_row(option_type="", strike=100.0, bid=1.0, ask=1.1),
        {"type": "call", "strike": "not-a-number", "bid": 1.0, "ask": 1.1, "T": 0.25},
        {},
    ]

    chain = clean_expiry_chain(rows)  # must not raise

    assert chain.is_empty
    assert chain.calls == ()
    assert chain.puts == ()
    assert FLAG_EMPTY_CHAIN in chain.flags
    assert len(chain.removals) == len(rows)
    assert chain.viability.usable_for_kvar is False
    assert chain.viability.usable_for_skew is False
    # Diagnostics stay plain-python (controller `_json_safe` friendly).
    diag = chain.diagnostics()
    assert isinstance(diag["removals_by_reason"], dict)
    assert all(isinstance(v, int) for v in diag["removals_by_reason"].values())


def test_empty_input_is_handled():
    chain = clean_expiry_chain(None)
    assert chain.is_empty
    assert chain.removals == ()
    assert clean_option_chains(None, None) == []


def test_duplicate_strike_keeps_the_tighter_quote():
    rows = [
        make_row(option_type="call", strike=100.0, bid=4.9, ask=5.1),   # rel spread 0.04
        make_row(option_type="call", strike=100.0, bid=4.0, ask=6.0),   # rel spread 0.40
    ]
    chain = clean_expiry_chain(rows)

    assert len(chain.calls) == 1
    assert chain.calls[0].spread_rel == pytest.approx(0.04)
    assert [rec.reason for rec in chain.removals] == ["duplicate_strike"]


# ---------------------------------------------------------------------------
# Multi-expiry entry point + report
# ---------------------------------------------------------------------------


def test_clean_option_chains_groups_by_expiry_and_sorts_by_maturity():
    near = arbitrage_free_rows([95, 100, 105])
    far = [
        {**row, "T": 1.0, "expiry_ts": 1_798_761_600, "expiry": "2027-01-01"}
        for row in arbitrage_free_rows([95, 100, 105])
    ]
    calls = [r for r in far + near if r["type"] == "call"]
    puts = [r for r in far + near if r["type"] == "put"]

    chains = clean_option_chains(calls, puts)

    assert [c.T for c in chains] == [pytest.approx(T_TEST), pytest.approx(1.0)]
    assert all(c.n_quotes == 6 for c in chains)

    report = cleaning_report(chains)
    assert report["n_expiries"] == 2
    assert report["n_quotes_kept"] == 12
    assert report["n_quotes_removed"] == 0
    assert report["exercise_style_caveat"] == AMERICAN_EXERCISE_CAVEAT
    assert len(report["expiries"]) == 2


# ---------------------------------------------------------------------------
# CBOE zero-bid wall (the second gate on one-sided quotes)
#
# Oracle: the CBOE VIX white-paper rule, restated inline -- walking outward from
# the money, once two consecutive strikes quote a zero bid, that run and every
# strike beyond it leave the replication set. The expected survivor list is
# therefore readable straight off the hand-built bids, never from the cleaner.
# ---------------------------------------------------------------------------
def test_two_consecutive_zero_bids_truncate_the_call_tail():
    rows = arbitrage_free_rows([95, 100, 105])
    # 110 keeps a genuine two-way market; 115 and 120 are the consecutive wall;
    # 125 and 130 lie beyond it and must go even though they quote an ask.
    rows.append(make_row(option_type="call", strike=110.0, bid=0.30, ask=0.40))
    for strike in (115.0, 120.0, 125.0, 130.0):
        rows.append(make_row(option_type="call", strike=strike, bid=0.0, ask=0.05))

    chain = clean_expiry_chain(rows)

    survivors = sorted(mids_by_strike(chain.calls))
    assert 110.0 in survivors
    assert [s for s in survivors if s >= 115.0] == []
    truncated = {
        rec.strike for rec in chain.removals if rec.reason == REASON_ZERO_BID_TAIL
    }
    assert truncated == {115.0, 120.0, 125.0, 130.0}


def test_two_consecutive_zero_bids_truncate_the_put_tail_downward():
    rows = arbitrage_free_rows([95, 100, 105])
    rows.append(make_row(option_type="put", strike=90.0, bid=0.25, ask=0.35))
    for strike in (85.0, 80.0, 75.0):
        rows.append(make_row(option_type="put", strike=strike, bid=0.0, ask=0.05))

    chain = clean_expiry_chain(rows)

    survivors = sorted(mids_by_strike(chain.puts))
    assert 90.0 in survivors
    assert [s for s in survivors if s <= 85.0] == []
    truncated = {
        rec.strike for rec in chain.removals if rec.reason == REASON_ZERO_BID_TAIL
    }
    assert truncated == {85.0, 80.0, 75.0}


def test_an_isolated_zero_bid_survives_so_spec_4_1_stays_alive():
    """A single zero-bid strike is not a wall: it stays, flagged one-sided."""
    rows = arbitrage_free_rows([95, 100, 105])
    rows.append(make_row(option_type="call", strike=115.0, bid=0.0, ask=0.05))
    rows.append(make_row(option_type="call", strike=120.0, bid=0.10, ask=0.20))

    chain = clean_expiry_chain(rows)

    assert not [rec for rec in chain.removals if rec.reason == REASON_ZERO_BID_TAIL]
    lone = [q for q in chain.calls if q.strike == pytest.approx(115.0)]
    assert len(lone) == 1
    assert lone[0].one_sided is True
    assert FLAG_ONE_SIDED in lone[0].flags


def test_zero_bid_stop_count_zero_disables_truncation():
    rows = arbitrage_free_rows([95, 100, 105])
    for strike in (115.0, 120.0, 125.0):
        rows.append(make_row(option_type="call", strike=strike, bid=0.0, ask=0.05))

    chain = clean_expiry_chain(rows, config=CleaningConfig(zero_bid_stop_count=0))

    assert not [rec for rec in chain.removals if rec.reason == REASON_ZERO_BID_TAIL]
    assert {115.0, 120.0, 125.0} <= set(mids_by_strike(chain.calls))


def test_itm_zero_bids_never_amputate_the_chain():
    """The walk is OTM-only: a zero-bid pair below the forward cannot cut calls."""
    rows = arbitrage_free_rows([95, 100, 105])
    # Deep ITM calls with (implausibly) zero bids, i.e. strictly below S0=100.
    rows.append(make_row(option_type="call", strike=80.0, bid=0.0, ask=20.5))
    rows.append(make_row(option_type="call", strike=85.0, bid=0.0, ask=15.5))

    chain = clean_expiry_chain(rows, config=CleaningConfig(arb_tol_abs=0.75))

    assert not [rec for rec in chain.removals if rec.reason == REASON_ZERO_BID_TAIL]
    assert 105.0 in mids_by_strike(chain.calls)


def test_truncation_removes_the_short_maturity_k_var_bias():
    """
    The quantitative reason the wall exists.

    Worthless far strikes quoted at a half-tick ask contribute
    sum_i Q(K_i) * dK_i / K_i**2 to the log-contract replication. Here that
    spurious mass is computed inline from the surviving quotes and must vanish
    once the wall is applied -- otherwise K_var, hence xi0(T), hence the
    short-maturity slope that yields H, inherits a maturity-dependent bias.
    """
    strikes = [70.0 + 2.5 * i for i in range(9)]  # 70 .. 90, all worthless
    rows = arbitrage_free_rows([95, 100, 105])
    rows.append(make_row(option_type="put", strike=92.5, bid=0.02, ask=0.06))
    for strike in strikes:
        rows.append(make_row(option_type="put", strike=strike, bid=0.0, ask=0.01))

    def spurious_mass(chain) -> float:
        return sum(
            q.mid * 2.5 / (q.strike * q.strike)
            for q in chain.puts
            if q.strike <= 90.0
        )

    unwalled = clean_expiry_chain(rows, config=CleaningConfig(zero_bid_stop_count=0))
    walled = clean_expiry_chain(rows)

    assert spurious_mass(unwalled) > 0.0
    assert spurious_mass(walled) == pytest.approx(0.0, abs=1e-15)
