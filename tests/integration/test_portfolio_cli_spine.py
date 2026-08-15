"""
Contract net for the part of the JSON-portfolio cluster that actually executes.

The pass-3 handoff flagged `app/model/{trading,portfolio,dashboard,backtesting}/`
as "outside the safety net": no test imports them, the Streamlit app never loads
them, and they survive only through the hand-launched CLI scripts. Mapping the
call graph turned that into a much smaller problem than the file listing suggests.

Exactly two roots run this cluster, and both are launched by hand:

    app/model/market_data/scripts/update_balance.py         -> process_matured_forwards
    app/model/market_data/scripts/update_portfolio_value.py -> recompute_portfolio_value

Everything else in those four packages is *imported* but never *called*. The
import happens as a side effect: reaching `app.model.dashboard.cache` executes
`app/model/dashboard/__init__.py`, which re-exports `dashboard.service`, which
drags in the backtesting engine, `trading.execution`, `trading.systems` and
`portfolio.positions`. So `buy_asset`, `sell_asset`, `auto_execute_trading_levels`,
`compute_spot_pnl`, `compute_spot_totals`, `dashboard_price` and the
trading-system writers are all loaded and none of them is ever invoked — the live
buy/sell path is Alpaca, in `app/controller/trading_controller.py`, not this
JSON-file layer. Covering that dormant surface would pin a superseded generation
in place and make it harder to retire, so this module tests the executed spine
only.

Both roots move money: one adjusts `balance` when a forward settles, the other
rewrites `portfolio_value` wholesale. An error there corrupts the dashboard
silently, which is precisely what "outside the safety net" was costing.

Every expected number below is derived by hand from the rule being asserted,
never by running the code and recording what it returned.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest

from app.model.dashboard import cache as cache_mod
from app.model.portfolio import forwards as forwards_mod
from app.model.portfolio import settlement as settlement_mod
from app.model.portfolio import valuation as valuation_mod
from app.model.trading import logs as logs_mod

pytestmark = pytest.mark.integration


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect every module-level JSON path onto tmp_path.

    These modules resolve their file constants at call time, so patching the
    constant is enough and the real I/O still runs. `valuation` keeps its own
    copies of the paths, hence the duplicate patches onto the same files.
    """
    vars_file = tmp_path / "dashboard_vars.json"
    spot_file = tmp_path / "spot_portfolio.json"
    forwards_file = tmp_path / "forwards_portfolio.json"
    trades_file = tmp_path / "trades_log.json"

    monkeypatch.setattr(cache_mod, "DASHBOARD_VARS_FILE", vars_file)
    monkeypatch.setattr(valuation_mod, "DASHBOARD_VARS_FILE", vars_file)
    monkeypatch.setattr(valuation_mod, "SPOT_FILE", spot_file)
    monkeypatch.setattr(valuation_mod, "FORWARDS_FILE", forwards_file)
    monkeypatch.setattr(forwards_mod, "FORWARDS_FILE", forwards_file)
    monkeypatch.setattr(logs_mod, "TRADES_LOG_FILE", trades_file)

    class _Store:
        vars = vars_file
        spot = spot_file
        forwards = forwards_file
        trades = trades_file

    return _Store


# ---------------------------------------------------------------------------
# update_portfolio_value.py -> recompute_portfolio_value
# ---------------------------------------------------------------------------


def test_portfolio_value_signs_the_spot_legs_by_side(store):
    # 12 AAA long  @ 7.50 -> +90.00
    #  4 BBB short @ 2.50 -> -10.00
    # total = 80.00
    _write(store.vars, {"prices": {"AAA": 7.5, "BBB": 2.5}})
    _write(
        store.spot,
        {
            "AAA": {"quantity": 12, "side": "long"},
            "BBB": {"quantity": 4, "side": "short"},
        },
    )
    _write(store.forwards, {})

    assert valuation_mod.recompute_portfolio_value() == pytest.approx(80.0)


def test_a_forward_leg_is_worth_what_settling_it_would_pay(store):
    """Carried at (mark - forward_price) x qty, not at notional.

    The notional reading would count the underlying as owned outright before the
    strike is paid, and would contradict `settlement`, which credits only the
    price difference at maturity.
    """
    # 3 CCC forward long, struck at 8.00, marked at 9.00 -> (9 - 8) x 3 = 3.00
    _write(store.vars, {"prices": {"CCC": 9.0}})
    _write(store.spot, {})
    _write(
        store.forwards,
        {"f1": {"symbol": "CCC", "quantity": 3, "side": "long", "forward_price": 8.0}},
    )

    assert valuation_mod.recompute_portfolio_value() == pytest.approx(3.0)


def test_an_unquoted_forward_leg_is_worth_nothing(store):
    # No mark for CCC -> the leg falls back to its own strike, so it is worth
    # (8 - 8) x 3 = 0.00 rather than contributing a full 24.00 of notional.
    _write(store.vars, {"prices": {}})
    _write(store.spot, {})
    _write(
        store.forwards,
        {"f1": {"symbol": "CCC", "quantity": 3, "side": "long", "forward_price": 8.0}},
    )

    assert valuation_mod.recompute_portfolio_value() == pytest.approx(0.0)


def test_a_short_forward_leg_takes_the_opposite_sign(store):
    # Same 1.00 move against a short of 3 -> -(9 - 8) x 3 = -3.00
    _write(store.vars, {"prices": {"CCC": 9.0}})
    _write(store.spot, {})
    _write(
        store.forwards,
        {"f1": {"symbol": "CCC", "quantity": 3, "side": "short", "forward_price": 8.0}},
    )

    assert valuation_mod.recompute_portfolio_value() == pytest.approx(-3.0)


def test_a_position_with_no_quote_contributes_nothing(store):
    # AAA is quoted (2 x 5.00 = 10.00), ZZZ is not and must not blow up.
    _write(store.vars, {"prices": {"AAA": 5.0}})
    _write(
        store.spot,
        {
            "AAA": {"quantity": 2, "side": "long"},
            "ZZZ": {"quantity": 999, "side": "long"},
        },
    )
    _write(store.forwards, {})

    assert valuation_mod.recompute_portfolio_value() == pytest.approx(10.0)


def test_recompute_persists_the_value_without_clobbering_the_rest_of_the_cache(store):
    _write(store.vars, {"prices": {"AAA": 5.0}, "balance": 1234.5, "last_refresh": "2026-08-14"})
    _write(store.spot, {"AAA": {"quantity": 2, "side": "long"}})
    _write(store.forwards, {})

    valuation_mod.recompute_portfolio_value()

    cache = _read(store.vars)
    assert cache["portfolio_value"] == pytest.approx(10.0)
    assert cache["balance"] == pytest.approx(1234.5)  # untouched
    assert cache["prices"] == {"AAA": 5.0}
    assert cache["last_refresh"] == "2026-08-14"


# ---------------------------------------------------------------------------
# update_balance.py -> process_matured_forwards / check_and_settle_forward
# ---------------------------------------------------------------------------


@pytest.fixture
def no_market(monkeypatch):
    """Cut both price sources; each test re-arms the one it is asserting on."""

    def _no_history(*args, **kwargs):
        return pd.DataFrame(), None, False

    def _no_quote(symbol):
        raise AssertionError(f"get_data must not be reached for {symbol}")

    monkeypatch.setattr(settlement_mod, "load_or_fetch_closing_history", _no_history)
    monkeypatch.setattr(settlement_mod, "get_data", _no_quote)
    return monkeypatch


def test_a_forward_maturing_later_is_left_alone(store, no_market):
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "long",
        "forward_price": 100.0,
        "maturity": "2026-12-31",
    }

    settled = settlement_mod.check_and_settle_forward(
        "f1", fwd, datetime.date(2026, 8, 15)
    )

    assert settled is False
    assert _read(store.vars)["balance"] == pytest.approx(1000.0)


def test_matured_long_forward_credits_the_gain_to_the_balance(store, no_market):
    # Settles at 110.00 against a forward price of 100.00, 2 units, long:
    # delta = (110 - 100) x 2 = +20.00 -> balance 1000 -> 1020
    no_market.setattr(settlement_mod, "get_data", lambda symbol: {"price": 110.0})
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "long",
        "forward_price": 100.0,
        "maturity": "2026-08-10",
    }

    settled = settlement_mod.check_and_settle_forward(
        "f1", fwd, datetime.date(2026, 8, 15)
    )

    assert settled is True
    assert _read(store.vars)["balance"] == pytest.approx(1020.0)


def test_matured_short_forward_debits_the_same_amount(store, no_market):
    # Same move, short side: delta = (100 - 110) x 2 = -20.00 -> balance 980
    no_market.setattr(settlement_mod, "get_data", lambda symbol: {"price": 110.0})
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "short",
        "forward_price": 100.0,
        "maturity": "2026-08-10",
    }

    settled = settlement_mod.check_and_settle_forward(
        "f1", fwd, datetime.date(2026, 8, 15)
    )

    assert settled is True
    assert _read(store.vars)["balance"] == pytest.approx(980.0)


def test_settlement_uses_the_last_close_on_or_before_maturity(store, no_market):
    """History wins over the live quote, and the post-maturity close is ignored.

    `get_data` is left armed to raise: reaching it would mean the history branch
    was skipped.
    """
    history = pd.DataFrame(
        {
            "Date": ["2026-08-08", "2026-08-10", "2026-08-13"],
            "Close": [95.0, 104.0, 130.0],
        }
    )
    no_market.setattr(
        settlement_mod,
        "load_or_fetch_closing_history",
        lambda *a, **k: (history.copy(), None, False),
    )
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "long",
        "forward_price": 100.0,
        "maturity": "2026-08-10",
    }

    # Settles on the 2026-08-10 close (104.00), not the 2026-08-13 one (130.00):
    # delta = (104 - 100) x 2 = +8.00
    settled = settlement_mod.check_and_settle_forward(
        "f1", fwd, datetime.date(2026, 8, 15)
    )

    assert settled is True
    assert _read(store.vars)["balance"] == pytest.approx(1008.0)


@pytest.mark.parametrize(
    "broken, why",
    [
        ({"symbol": "", "quantity": 2, "forward_price": 100.0}, "no symbol"),
        ({"symbol": "AAA", "quantity": 0, "forward_price": 100.0}, "zero size"),
        ({"symbol": "AAA", "quantity": -2, "forward_price": 100.0}, "negative size"),
        ({"symbol": "AAA", "quantity": 2, "forward_price": 0.0}, "no strike"),
    ],
)
def test_an_unusable_forward_never_touches_the_balance(store, no_market, broken, why):
    _write(store.vars, {"balance": 1000.0})
    fwd = {"side": "long", "maturity": "2026-08-10", **broken}

    settled = settlement_mod.check_and_settle_forward(
        "f1", fwd, datetime.date(2026, 8, 15)
    )

    assert settled is False, why
    assert _read(store.vars)["balance"] == pytest.approx(1000.0), why


def test_an_unreadable_maturity_is_treated_as_not_due(store, no_market):
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "long",
        "forward_price": 100.0,
        "maturity": "not-a-date",
    }

    assert (
        settlement_mod.check_and_settle_forward("f1", fwd, datetime.date(2026, 8, 15))
        is False
    )
    assert _read(store.vars)["balance"] == pytest.approx(1000.0)


def test_settling_a_forward_records_the_trade(store, no_market):
    no_market.setattr(settlement_mod, "get_data", lambda symbol: {"price": 110.0})
    _write(store.vars, {"balance": 1000.0})
    fwd = {
        "symbol": "AAA",
        "quantity": 2,
        "side": "long",
        "forward_price": 100.0,
        "maturity": "2026-08-10",
    }

    settlement_mod.check_and_settle_forward("f1", fwd, datetime.date(2026, 8, 15))

    log = _read(store.trades)
    assert len(log) == 1
    assert log[0]["symbol"] == "AAA"
    assert log[0]["side"] == "long"
    assert log[0]["quantity"] == pytest.approx(2.0)
    assert log[0]["price"] == pytest.approx(110.0)
    assert log[0]["source"] == "forwards"


def test_process_drops_the_settled_forward_and_keeps_the_rest(store, no_market):
    no_market.setattr(settlement_mod, "get_data", lambda symbol: {"price": 110.0})
    _write(store.vars, {"balance": 1000.0})
    _write(
        store.forwards,
        {
            "due": {
                "symbol": "AAA",
                "quantity": 2,
                "side": "long",
                "forward_price": 100.0,
                "maturity": "2020-01-01",
            },
            "later": {
                "symbol": "BBB",
                "quantity": 5,
                "side": "long",
                "forward_price": 50.0,
                "maturity": "2099-12-31",
            },
        },
    )

    settlement_mod.process_matured_forwards()

    remaining = _read(store.forwards)
    assert list(remaining) == ["later"]
    assert remaining["later"]["symbol"] == "BBB"
    assert _read(store.vars)["balance"] == pytest.approx(1020.0)


def test_settling_a_forward_moves_equity_without_creating_or_destroying_any(
    store, no_market
):
    """The invariant that ties the two CLI roots together.

    `update_balance.py` settles a matured forward into `balance`;
    `update_portfolio_value.py` rebuilds `portfolio_value` from the remaining book.
    Settlement moves value between those two buckets — it must not change their
    sum. Stated with the settlement price equal to the last mark, so that the only
    thing under test is the bookkeeping and not a price move between the two runs.

    Nothing here is derived from the code: the invariant comes from what a
    settlement *is*.
    """
    no_market.setattr(settlement_mod, "get_data", lambda symbol: {"price": 110.0})
    _write(store.vars, {"balance": 1000.0, "prices": {"AAA": 110.0}})
    _write(store.spot, {})
    _write(
        store.forwards,
        {
            "due": {
                "symbol": "AAA",
                "quantity": 2,
                "side": "long",
                "forward_price": 100.0,
                "maturity": "2020-01-01",
            }
        },
    )

    equity_before = _read(store.vars)["balance"] + valuation_mod.recompute_portfolio_value()

    settlement_mod.process_matured_forwards()

    equity_after = _read(store.vars)["balance"] + valuation_mod.recompute_portfolio_value()

    assert equity_after == pytest.approx(equity_before)


def test_nothing_due_leaves_the_book_untouched(store, no_market):
    _write(store.vars, {"balance": 1000.0})
    book = {
        "later": {
            "symbol": "BBB",
            "quantity": 5,
            "side": "long",
            "forward_price": 50.0,
            "maturity": "2099-12-31",
        }
    }
    _write(store.forwards, book)

    settlement_mod.process_matured_forwards()

    assert _read(store.forwards) == book
    assert _read(store.vars)["balance"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Persistence contracts the two roots depend on
# ---------------------------------------------------------------------------


def test_saving_a_forward_normalizes_it_and_drops_unknown_fields(store):
    forwards_mod.save_forwards(
        {
            "f1": {
                "symbol": "  aapl  ",
                "maturity": "2026-09-30",
                "forward_price": "12.5",
                "quantity": "3",
                "side": "SHORT",
                "created_at": "2026-08-01",
                "note": "dropped",
            }
        }
    )

    stored = _read(store.forwards)["f1"]
    assert stored == {
        "symbol": "AAPL",
        "maturity": "2026-09-30",
        "forward_price": 12.5,
        "quantity": 3.0,
        "side": "short",
        "created_at": "2026-08-01",
    }


def test_a_missing_forwards_file_reads_as_an_empty_book(store):
    assert not store.forwards.exists()
    assert forwards_mod.load_forwards() == {}


def test_logging_a_trade_appends_instead_of_replacing(store):
    logs_mod.log_trade("AAA", "long", 1.0, 10.0, source="forwards")
    logs_mod.log_trade("BBB", "short", 2.0, 20.0, source="manual")

    log = _read(store.trades)
    assert [e["symbol"] for e in log] == ["AAA", "BBB"]
    assert [e["source"] for e in log] == ["forwards", "manual"]


def test_a_corrupt_dashboard_cache_reads_as_defaults_instead_of_raising(store):
    store.vars.parent.mkdir(parents=True, exist_ok=True)
    store.vars.write_text("{ this is not json", encoding="utf-8")

    assert cache_mod.load_dashboard_cache() == {"prices": {}, "last_refresh": None}
