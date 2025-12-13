import json

import pandas as pd

from app.model.market_data import market_data as md
from app.model.options import logic as opt_logic


def test_yahoo_option_chain_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(md, "CACHE_YAHOO_OPTION_CHAINS_DIR", tmp_path)

    csv_path, meta_path = md._yahoo_chain_cache_paths(
        "AAPL", max_maturity_years=2.0, max_expiries=12
    )
    tmp_path.mkdir(parents=True, exist_ok=True)

    chain_df = pd.DataFrame(
        [
            {"type": "call", "strike": 100.0, "T": 1.0, "iv": 0.2, "S0": 123.0},
            {"type": "put", "strike": 100.0, "T": 1.0, "iv": 0.25, "S0": 123.0},
        ]
    )
    chain_df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({"spot": 123.0, "rf": 0.0, "div": 0.0}), encoding="utf-8")

    def _boom(*_a, **_k):
        raise AssertionError("Yahoo API should not be called on cache hit")

    monkeypatch.setattr(md, "_fetch_yahoo_options_json", _boom)

    calls_df, puts_df, spot, rf, div = md.fetch_options_details_yahoo(
        "AAPL",
        max_maturity_years=2.0,
        max_expiries=12,
        use_cache=True,
        cache_ttl_sec=3600,
    )
    assert not calls_df.empty
    assert not puts_df.empty
    assert float(spot) == 123.0
    assert float(rf) == 0.0
    assert float(div) == 0.0


def test_alpaca_option_chain_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_OPTION_CHAIN_CACHE_TTL_SEC", "999999")
    monkeypatch.setattr(opt_logic, "CACHE_ALPACA_OPTION_CHAINS_DIR", tmp_path)
    monkeypatch.setattr(opt_logic, "CACHE_CSV_DIR", tmp_path / "legacy")

    tmp_path.mkdir(parents=True, exist_ok=True)
    cached_path = tmp_path / "options_alpaca_AAPL.csv"
    expected = pd.DataFrame(
        [{"symbol": "AAPL", "opra": "X", "K": 100.0, "T": 1.0, "S0": 123.0, "iv": 0.2, "type": "call"}]
    )
    expected.to_csv(cached_path, index=False)

    def _boom(*_a, **_k):
        raise AssertionError("Alpaca API should not be called on cache hit")

    monkeypatch.setattr(opt_logic.requests, "get", _boom)

    got = opt_logic.download_options_alpaca("AAPL", cache_to_csv=True)
    assert got is not None and not got.empty
    assert list(got.columns) == list(expected.columns)
    assert float(got.iloc[0]["K"]) == 100.0

