"""
Probe = the 5 proposed unit tests for app/model/iv_dashboard/service.py, written as
pytest tests so the sketches in the review are proven to run green against the code
under review (not collected by the suite: file name does not match test_*.py).

Run: python -m pytest scripts/review_iv_dashboard/p1_tests_service_sketches.py -q
All network is monkeypatched; --disable-socket is active via pyproject addopts.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.model.iv_dashboard import service as svc

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# 1. _fetch_atm_snapshots: pagination, params, page cap, credentials
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_atm_snapshots_paginates_and_caps(monkeypatch):
    monkeypatch.setattr(svc, "_alpaca_data_headers", lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"})
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params))
        page = len(calls)
        return _Resp({"snapshots": {f"SPY260918C0045{page}000": {"p": page}}, "next_page_token": f"tok{page}"})

    monkeypatch.setattr(svc.requests, "get", fake_get)
    out = svc._fetch_atm_snapshots("spy", feed="indicative", spot=450.0, dte_min=15, dte_max=60)
    assert len(calls) == svc._SNAPSHOT_MAX_PAGES  # token never exhausted -> cap
    assert "page_token" not in calls[0] and calls[1]["page_token"] == "tok1" and calls[2]["page_token"] == "tok2"
    assert calls[0]["strike_price_gte"] == pytest.approx(405.0) and calls[0]["strike_price_lte"] == pytest.approx(495.0)
    assert calls[0]["feed"] == "indicative" and calls[0]["limit"] == 1000
    assert len(out) == 3  # merged across pages


def test_fetch_atm_snapshots_stops_without_token_and_raises_on_http(monkeypatch):
    monkeypatch.setattr(svc, "_alpaca_data_headers", lambda: {"x": "y"})
    monkeypatch.setattr(svc.requests, "get", lambda *a, **k: _Resp({"snapshots": {"A": {}}, "next_page_token": None}))
    assert list(svc._fetch_atm_snapshots("SPY", feed="opra", spot=None, dte_min=15, dte_max=60)) == ["A"]
    monkeypatch.setattr(svc.requests, "get", lambda *a, **k: _Resp({}, status=403))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        svc._fetch_atm_snapshots("SPY", feed="opra", spot=None, dte_min=15, dte_max=60)


def test_fetch_atm_snapshots_requires_keys(monkeypatch):
    monkeypatch.setattr(svc, "_alpaca_data_headers", lambda: None)
    monkeypatch.setattr(svc.requests, "get", lambda *a, **k: pytest.fail("must not hit the network"))
    with pytest.raises(EnvironmentError):
        svc._fetch_atm_snapshots("SPY", feed="indicative", spot=100.0, dte_min=15, dte_max=60)


# --------------------------------------------------------------------------- #
# 2. OPRA decoding / snapshot mid / snapshot iv (pure parsers)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "opra, expected",
    [
        ("SPY260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
        ("SPY260918P00450500", (450.5, dt.date(2026, 9, 18), "put")),
        ("BRKB260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
        ("garbage", (None, None, None)),
        ("", (None, None, None)),
    ],
)
def test_decode_opra(opra, expected):
    assert svc._decode_opra(opra) == expected


@pytest.mark.parametrize(
    "snap, expected",
    [
        ({"latestQuote": {"bp": 1.0, "ap": 1.2}}, 1.1),
        ({"latestQuote": {"bp": 0.0, "ap": 1.2}}, 1.2),  # no bid -> ask
        ({"latestQuote": {"bp": 0.0, "ap": 0.0}, "latestTrade": {"p": 0.9}}, 0.9),  # -> last trade
        ({"latest_quote": {"bp": "2", "ap": "4"}}, 3.0),  # snake_case + strings
        ({"latestQuote": {"bp": "x", "ap": "y"}, "latestTrade": {"price": 0.5}}, 0.5),
        ({}, None),
    ],
)
def test_snapshot_mid(snap, expected):
    assert svc._snapshot_mid(snap) == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "snap, expected",
    [
        ({"impliedVolatility": 0.21}, 0.21),
        ({"greeks": {"iv": 0.19}}, 0.19),
        ({"latestGreeks": {"impliedVolatility": "0.3"}}, 0.3),
        ({"greeks": "not-a-dict"}, float("nan")),
        ({}, float("nan")),
    ],
)
def test_snapshot_iv(snap, expected):
    got = svc._snapshot_iv(snap)
    assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 3. fetch_current_atm_iv: greeks path, parity inversion of a put, median, method tag
# --------------------------------------------------------------------------- #
def _bs_put(S, K, T, sigma, r=0.0):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def test_fetch_current_atm_iv_mixes_greeks_and_parity_inversion(monkeypatch):
    today = dt.date.today()
    expiry = today + dt.timedelta(days=30)
    T = 30 / 365.0
    spot, sigma_put = 100.0, 0.25
    tag = expiry.strftime("%y%m%d")
    put_mid = _bs_put(spot, 100.0, T, sigma_put)
    snaps = {
        f"SPY{tag}C00100000": {"greeks": {"iv": 0.21}},  # direct greeks
        f"SPY{tag}P00100000": {"latestQuote": {"bp": put_mid - 0.01, "ap": put_mid + 0.01}},  # parity + BS
        f"SPY{tag}C00130000": {"greeks": {"iv": 0.9}},  # 30% OTM -> outside the ATM band
    }
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: spot)
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: snaps)
    monkeypatch.delenv("ALPACA_OPTION_DATA_FEED", raising=False)

    info, log = svc.fetch_current_atm_iv("SPY")
    assert info is not None, log
    assert info["method"] == "mixte (greeks + inversion BS)"
    assert info["n_contracts"] == 2 and info["dte"] == 30 and info["expiry"] == expiry
    assert info["iv"] == pytest.approx(np.median([0.21, sigma_put]), abs=2e-3)
    assert info["feed"] == svc.DEFAULT_OPTION_FEED


def test_fetch_current_atm_iv_degrades_to_none(monkeypatch):
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: None)
    info, log = svc.fetch_current_atm_iv("SPY")
    assert info is None and "Spot indisponible" in log[-1]

    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 100.0)
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 403")))
    import app.model.options.logic as logic
    monkeypatch.setattr(logic, "download_options_alpaca", lambda *a, **k: pd.DataFrame())
    info, log = svc.fetch_current_atm_iv("SPY")
    assert info is None and "Aucun contrat" in log[-1]


# --------------------------------------------------------------------------- #
# 4. record_iv_observation / load_iv_history round trip (tmp_path, service global patched)
# --------------------------------------------------------------------------- #
def test_iv_history_upsert_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "CACHE_IV_HISTORY_DIR", tmp_path)  # NOT app.utils.paths (captured at import)
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    pd.DataFrame([{"date": yesterday, "iv": 0.19, "dte": 29, "n_contracts": 5, "method": "x", "spot": 99.0}]).to_csv(
        tmp_path / "iv_daily_SPY.csv", index=False
    )
    svc.record_iv_observation(" spy ", {"iv": 0.21, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 100.0})
    svc.record_iv_observation("SPY", {"iv": 0.25, "dte": 31, "n_contracts": 4, "method": "mixte", "spot": 101.0})
    hist = svc.load_iv_history("spy")
    assert hist["iv"].tolist() == [0.19, 0.25]  # same-day upsert overwrote 0.21, older row kept
    assert hist["date"].is_monotonic_increasing and str(hist["date"].dtype).startswith("datetime64")
    assert not (svc._iv_history_path("SPY").parent != tmp_path)
    assert list(svc.load_iv_history("ZZZZ").columns) == ["date", "iv"]  # missing file
    (tmp_path / "iv_daily_BAD.csv").write_text("not,a,csv\n\x00", encoding="utf-8")
    assert svc.load_iv_history("BAD").empty  # corrupt file -> empty, no raise
    svc.record_iv_observation("SPY", {"iv": None})  # best-effort: must not raise


# --------------------------------------------------------------------------- #
# 5. fetch_daily_closes fallback chain + get_iv_dashboard_data degradation
# --------------------------------------------------------------------------- #
def _closes_df(n=600):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"Date": idx, "Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def test_fetch_daily_closes_falls_back_to_stooq_after_both_alpaca_feeds(monkeypatch):
    feeds = []

    def boom(sym, start, *, feed=None):
        feeds.append(feed)
        raise RuntimeError("alpaca down")

    monkeypatch.setattr(svc, "_fetch_closes_alpaca", boom)
    monkeypatch.setattr(svc, "fetch_ohlc_history", lambda sym, period, interval: _closes_df())
    df, source, log = svc.fetch_daily_closes("spy", years=2.0)
    assert feeds == [None, "iex"]
    assert source == "fallback (Stooq/Yahoo)" and not df.empty and list(df.columns) == ["Date", "Close"]
    assert sum("indisponibles" in m for m in log) == 2

    monkeypatch.setattr(svc, "fetch_ohlc_history", lambda *a, **k: None)
    df, source, log = svc.fetch_daily_closes("spy", years=2.0)
    assert df.empty and source == "none"


def test_get_iv_dashboard_data_degrades_without_iv_and_raises_without_prices(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "CACHE_IV_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(svc, "fetch_daily_closes", lambda sym, **k: (_closes_df(), "alpaca", ["ok"]))
    monkeypatch.setattr(svc, "fetch_current_atm_iv", lambda sym: (None, ["Spot indisponible (test)"]))
    out = svc.get_iv_dashboard_data("spy")
    assert out["current_iv"] is None and out["iv_error"] == "Spot indisponible (test)"
    assert out["analysis"] is not None and out["analysis_error"] is None
    assert out["iv_history"].empty and np.isnan(out["iv_vs_series_percentile"])
    assert out["symbol"] == "SPY" and out["source"] == "alpaca" and not out["series"].empty

    monkeypatch.setattr(svc, "fetch_daily_closes", lambda sym, **k: (pd.DataFrame(), "none", ["nope"]))
    with pytest.raises(RuntimeError, match="Aucune donnée de prix"):
        svc.get_iv_dashboard_data("SPY")
