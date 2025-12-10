"""
Utility script to refresh cached spot prices in data/dashboard_vars.json.

It gathers all tickers from:
  - options_portfolio.json
  - spot_portfolio.json
  - forwards_portfolio.json
  - trading_systems.json

For each unique ticker, it fetches the current spot via app.model.market_data.realtime.get_data
and writes the consolidated prices map to dashboard_vars.json (preserving other keys).
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Set

from app.model.market_data.cache_refresh import (
    collect_tickers,
    fetch_prices,
    update_dashboard_prices,
    update_portfolio_files,
)


def main() -> None:
    tickers = collect_tickers()
    prices = fetch_prices(tickers)
    update_dashboard_prices(prices)
    update_portfolio_files(prices)
    print(f"Updated {len(prices)} prices and refreshed portfolio files.")


if __name__ == "__main__":
    main()
