from app.utils import paths


def test_cache_subdirectories_exist():
    assert paths.CACHE_OHLC_DIR.is_dir()
    assert paths.CACHE_YAHOO_OPTION_CHAINS_DIR.is_dir()
    assert paths.CACHE_ALPACA_OPTION_CHAINS_DIR.is_dir()

