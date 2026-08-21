"""
Unit tests for app/model/iv_dashboard/service.py (the Alpaca plumbing behind the
🌡️ Vol Implicite tab). Everything is offline: requests / alpaca client / cache dir
are monkeypatched, --disable-socket is active via pyproject addopts.

Covers the five findings of docs/review-2026-08-iv-dashboard-alpaca.md:
  M1 no stale chain cache served as today's IV,
  M2 no mean-reversion signal derived from an IV-within-RV percentile,
  M3 Black-Scholes inversion on the parity-implied forward (not spot, r = q = 0),
  M4 exchange (New York) date for the cache key and the DTE,
  M5 direct coverage of the fetchers / parsers / cache / fallback chain.
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

_UTC = dt.timezone.utc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _bs(S, K, T, r, q, sigma, kind):
    """Independent Black-Scholes oracle with continuous r and q."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def _quote(mid, half_spread=0.01):
    return {"latestQuote": {"bp": mid - half_spread, "ap": mid + half_spread}}


def _opra(root, expiry, kind, strike):
    return f"{root}{expiry.strftime('%y%m%d')}{'C' if kind == 'call' else 'P'}{int(round(strike * 1000)):08d}"


def _closes_df(n=600):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"Date": idx, "Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def _freeze_clock(monkeypatch, when_utc: dt.datetime) -> None:
    """Pin the service clock (M4): the exchange date derives from this UTC instant."""
    monkeypatch.setattr(svc, "_utc_now", lambda: when_utc, raising=False)


# --------------------------------------------------------------------------- #
# M1 — a failing snapshot call must NOT fall back to a stale cached chain
# --------------------------------------------------------------------------- #
def test_snapshot_failure_never_serves_cached_chain_as_current_iv(monkeypatch):
    import app.model.options.logic as logic

    calls = []
    stale = pd.DataFrame(
        [{"symbol": "SPY", "opra": "SPY260918C00640000", "K": 640.0, "T": 30 / 365, "S0": 640.0, "iv": 0.25, "type": "call"}]
    )
    monkeypatch.setattr(logic, "download_options_alpaca", lambda *a, **k: calls.append(k) or stale)
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 640.0)
    monkeypatch.setattr(
        svc, "_fetch_atm_snapshots", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 403 Forbidden"))
    )

    info, log = svc.fetch_current_atm_iv("SPY")

    assert info is None, "a days-old cached chain must never become today's IV"
    assert calls == [], "the chain downloader (and its ageless CSV cache) must not be consulted"
    assert "HTTP 403" in log[-1], log


# --------------------------------------------------------------------------- #
# M2 — IV-within-RV percentile carries no mean-reversion signal
# --------------------------------------------------------------------------- #
def test_iv_percentile_within_rv_has_no_regime_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "CACHE_IV_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(svc, "fetch_daily_closes", lambda sym, **k: (_closes_df(), "alpaca", ["ok"]))
    iv_info = {"iv": 0.90, "spot": 100.0, "expiry": dt.date(2030, 1, 1), "dte": 30, "n_contracts": 4, "method": "x", "feed": "indicative"}
    monkeypatch.setattr(svc, "fetch_current_atm_iv", lambda sym: (iv_info, ["iv ok"]))

    out = svc.get_iv_dashboard_data("SPY")

    # the (honest) percentile metric stays, the (VRP-biased) signal goes
    assert out["iv_vs_series_percentile"] == pytest.approx(1.0)
    assert out["iv_regime"] is None


# --------------------------------------------------------------------------- #
# M3 — inversion on the parity-implied forward: no ±1 vol-pt bias per side
# --------------------------------------------------------------------------- #
def _spy_like_chain(expiry, *, S=640.0, r=0.04, q=0.013, sigma=0.16, dte=30, strikes_call=None, strikes_put=None):
    """Quotes only (no greeks) priced with real r/q, so the inversion path is exercised."""
    T = dte / 365.0
    ks = [610.0, 621.0, 634.0, 640.0, 646.0, 659.0, 670.0]  # all strictly inside the ±5 % band
    snaps = {}
    for K in strikes_call if strikes_call is not None else ks:
        snaps[_opra("SPY", expiry, "call", K)] = _quote(_bs(S, K, T, r, q, sigma, "call"))
    for K in strikes_put if strikes_put is not None else ks:
        snaps[_opra("SPY", expiry, "put", K)] = _quote(_bs(S, K, T, r, q, sigma, "put"))
    return snaps


@pytest.mark.parametrize(
    "strikes_call, strikes_put, label",
    [
        (None, [640.0, 646.0, 659.0, 670.0], "calls everywhere, puts only at/above spot"),
        ([610.0, 621.0, 634.0, 640.0], None, "puts everywhere, calls only at/below spot"),
    ],
)
def test_bs_inversion_is_unbiased_when_call_put_mix_is_unbalanced(monkeypatch, strikes_call, strikes_put, label):
    now = dt.datetime(2026, 8, 21, 14, 0, tzinfo=_UTC)  # 10:00 New York
    _freeze_clock(monkeypatch, now)
    expiry = dt.date(2026, 9, 20)  # 30 DTE
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 640.0)
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: _spy_like_chain(expiry, strikes_call=strikes_call, strikes_put=strikes_put))
    monkeypatch.delenv("ALPACA_OPTION_DATA_FEED", raising=False)

    info, log = svc.fetch_current_atm_iv("SPY")

    assert info is not None, log
    assert info["method"] == "inversion Black-Scholes (mid)"
    # true sigma 16 %; tolerance 15 bp (quote half-spread 1 ct + r-discounting residual ≈ 5 bp)
    assert abs(info["iv"] - 0.16) < 0.0015, f"{label}: IV {info['iv']:.4f} vs 0.1600"


def test_bs_inversion_keeps_one_otm_contract_per_strike_in_low_vol(monkeypatch):
    """
    r = q = 0 on spot pushed ITM puts below intrinsic (silently rejected as NaN) in
    low vol. Only the OTM side of each strike is inverted now (the ITM side is
    redundant by parity and carries no vol information): 7 strikes -> 7 contracts,
    none rejected, median unbiased.
    """
    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 21, 14, 0, tzinfo=_UTC))
    expiry = dt.date(2026, 9, 20)
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 640.0)
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: _spy_like_chain(expiry, sigma=0.10))
    monkeypatch.delenv("ALPACA_OPTION_DATA_FEED", raising=False)

    info, log = svc.fetch_current_atm_iv("SPY")

    assert info is not None, log
    assert info["n_contracts"] == 7, log
    assert abs(info["iv"] - 0.10) < 0.0015, f"IV {info['iv']:.4f} vs 0.1000"


# --------------------------------------------------------------------------- #
# M4 — exchange date (America/New_York), not the machine's local date
# --------------------------------------------------------------------------- #
def test_cache_row_is_keyed_by_new_york_date(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "CACHE_IV_HISTORY_DIR", tmp_path)
    # 2026-08-20 15:30 New York == 2026-08-20 19:30 UTC == 2026-08-21 03:30 Singapore
    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 20, 19, 30, tzinfo=_UTC))

    svc.record_iv_observation("SPY", {"iv": 0.18, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 640.0})

    hist = pd.read_csv(tmp_path / "iv_daily_SPY.csv")
    assert hist["date"].tolist() == ["2026-08-20"]


def test_dte_uses_new_york_date(monkeypatch):
    # 2026-08-20 19:30 UTC is still Aug-20 in New York (Aug-21 east of UTC+4h30)
    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 20, 19, 30, tzinfo=_UTC))
    expiry = dt.date(2026, 9, 19)
    snaps = {_opra("SPY", expiry, "call", 100.0): {"greeks": {"iv": 0.20}}}
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 100.0)
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: snaps)
    monkeypatch.delenv("ALPACA_OPTION_DATA_FEED", raising=False)

    info, log = svc.fetch_current_atm_iv("SPY")

    assert info is not None, log
    assert info["dte"] == 30


# --------------------------------------------------------------------------- #
# M5.1 — _fetch_atm_snapshots: pagination, params, page cap, credentials
# --------------------------------------------------------------------------- #
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
# M5.2 — pure parsers: OPRA decoding, snapshot mid, snapshot iv
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "opra, expected",
    [
        ("SPY260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
        ("SPY260918P00450500", (450.5, dt.date(2026, 9, 18), "put")),
        ("SPXW260918C05000000", (5000.0, dt.date(2026, 9, 18), "call")),
        ("BRKB260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
        ("AAPL1260918C00150000", (150.0, dt.date(2026, 9, 18), "call")),  # adjusted root
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
# M5.3 — fetch_current_atm_iv: greeks path + parity inversion, median, method tag
# --------------------------------------------------------------------------- #
def test_fetch_current_atm_iv_mixes_greeks_and_parity_inversion(monkeypatch):
    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 21, 14, 0, tzinfo=_UTC))
    expiry = dt.date(2026, 9, 20)
    T = 30 / 365.0
    spot, sigma_put = 100.0, 0.25
    snaps = {
        _opra("SPY", expiry, "call", 100.0): {"greeks": {"iv": 0.21}},  # direct greeks
        _opra("SPY", expiry, "put", 100.0): _quote(_bs(spot, 100.0, T, 0.0, 0.0, sigma_put, "put")),  # inversion
        _opra("SPY", expiry, "call", 130.0): {"greeks": {"iv": 0.9}},  # 30 % OTM -> outside the ATM band
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

    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 21, 14, 0, tzinfo=_UTC))
    monkeypatch.setattr(svc, "fetch_spot_price", lambda s: 100.0)
    far = dt.date(2026, 12, 18)  # 119 DTE: outside [15, 60]
    monkeypatch.setattr(svc, "_fetch_atm_snapshots", lambda *a, **k: {_opra("SPY", far, "call", 100.0): {"greeks": {"iv": 0.2}}})
    info, log = svc.fetch_current_atm_iv("SPY")
    assert info is None and "Aucun contrat" in log[-1]


# --------------------------------------------------------------------------- #
# M5.4 — record_iv_observation / load_iv_history round trip
# --------------------------------------------------------------------------- #
def test_iv_history_upsert_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "CACHE_IV_HISTORY_DIR", tmp_path)  # NOT app.utils.paths (captured at import)
    _freeze_clock(monkeypatch, dt.datetime(2026, 8, 21, 14, 0, tzinfo=_UTC))
    pd.DataFrame([{"date": "2026-08-20", "iv": 0.19, "dte": 29, "n_contracts": 5, "method": "x", "spot": 99.0}]).to_csv(
        tmp_path / "iv_daily_SPY.csv", index=False
    )
    svc.record_iv_observation(" spy ", {"iv": 0.21, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 100.0})
    svc.record_iv_observation("SPY", {"iv": 0.25, "dte": 31, "n_contracts": 4, "method": "mixte", "spot": 101.0})
    hist = svc.load_iv_history("spy")
    assert hist["iv"].tolist() == [0.19, 0.25]  # same-day upsert overwrote 0.21, older row kept
    assert hist["date"].is_monotonic_increasing and str(hist["date"].dtype).startswith("datetime64")
    assert svc._iv_history_path("SPY").parent == tmp_path
    assert list(svc.load_iv_history("ZZZZ").columns) == ["date", "iv"]  # missing file
    (tmp_path / "iv_daily_BAD.csv").write_text("not,a,csv\n\x00", encoding="utf-8")
    assert svc.load_iv_history("BAD").empty  # corrupt file -> empty, no raise
    svc.record_iv_observation("SPY", {"iv": None})  # best-effort: must not raise


# --------------------------------------------------------------------------- #
# M5.5 — fetch_daily_closes fallback chain + get_iv_dashboard_data degradation
# --------------------------------------------------------------------------- #
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
