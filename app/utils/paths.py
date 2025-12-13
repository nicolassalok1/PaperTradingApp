from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parents[2]
# Datastores live at repo root: JSON snapshots under data/, CSV caches under cache/
DATASETS_DIR = ROOT_DIR  # kept for backward compatibility if needed
JSON_DIR = ROOT_DIR / "data"
# Unified cache location for CSVs (market data, options, yield curves)
CACHE_CSV_DIR = ROOT_DIR / "cache"

# Structured cache sub-directories (kept under cache/ for backwards compatibility)
CACHE_OHLC_DIR = CACHE_CSV_DIR / "OHLC"
CACHE_YAHOO_OPTION_CHAINS_DIR = CACHE_CSV_DIR / "YahooOptionChains"
CACHE_ALPACA_OPTION_CHAINS_DIR = CACHE_CSV_DIR / "AlpacaOptionChains"

# Ensure expected directories exist
JSON_DIR.mkdir(parents=True, exist_ok=True)
CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
CACHE_OHLC_DIR.mkdir(parents=True, exist_ok=True)
CACHE_YAHOO_OPTION_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ALPACA_OPTION_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
