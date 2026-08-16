"""The Calibration avancée tab must offer the Yahoo IV surfaces it can load, not a blank
"Ticker" box the user has to guess into.

Yahoo Finance publishes no catalogue of optionable underlyings, so the offer is built from
two local sources: the surfaces already downloaded into cache/YahooOptionChains (loadable
offline) and the optionable universe scanned into data/alpaca_optionable_tickers.csv.
These tests pin the cache-side discovery; the universe reader is the tab's existing
`_load_optionable_tickers`. No network, no real cache dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("K,T,S0,iv,type\n", encoding="utf-8")


def test_cached_tickers_come_from_both_yahoo_file_families(tmp_path):
    from app.model.options.data.iv_surface import list_cached_iv_surface_tickers

    _touch(tmp_path / "iv_surface_yahoo_AAPL.csv")
    _touch(tmp_path / "yahoo_chain_MSFT_Y2p0_E12.csv")
    _touch(tmp_path / "yahoo_chain_MSFT_Y1p0_E12.csv")  # same ticker, other max_years
    _touch(tmp_path / "yahoo_chain_brk-b_Y2p0_E12.csv")  # lower case on disk
    (tmp_path / "yahoo_chain_MSFT_Y2p0_E12.json").write_text("{}", encoding="utf-8")  # meta, not a surface
    _touch(tmp_path / "iv_surface_yahoo_.csv")  # empty ticker stem
    _touch(tmp_path / "something_else.csv")

    assert list_cached_iv_surface_tickers(tmp_path) == ["AAPL", "BRK-B", "MSFT"]


def test_cached_tickers_missing_dir_is_empty(tmp_path):
    from app.model.options.data.iv_surface import list_cached_iv_surface_tickers

    assert list_cached_iv_surface_tickers(tmp_path / "nope") == []


def test_controller_exposes_cached_tickers(tmp_path, monkeypatch):
    import app.model.options.data.iv_surface as ivs
    from app.controller.calibration_controller import CalibrationController

    _touch(tmp_path / "iv_surface_yahoo_SPY.csv")
    monkeypatch.setattr(ivs, "CACHE_YAHOO_OPTION_CHAINS_DIR", tmp_path)

    assert CalibrationController().list_cached_yahoo_surface_tickers() == ["SPY"]
