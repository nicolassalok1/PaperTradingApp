import pytest

from app.utils import paths

pytestmark = pytest.mark.unit


def test_cache_subdirectories_exist():
    assert paths.CACHE_OHLC_DIR.is_dir()
    assert paths.CACHE_YAHOO_OPTION_CHAINS_DIR.is_dir()
    assert paths.CACHE_ALPACA_OPTION_CHAINS_DIR.is_dir()

