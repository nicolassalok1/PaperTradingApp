"""
Small CLI helper to download price history via Stooq and emit CSV to stdout.
Usage:
    python fetch_history_cli.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --freq D
"""

import argparse
import sys

from app.model.market_data.service import fetch_historical_prices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (ex: AAPL)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--freq", default="D", help="Frequency (D/W/M)")
    args = parser.parse_args()

    try:
        df = fetch_historical_prices(args.ticker, start=args.start, end=args.end, freq=args.freq)
        if df is None or df.empty:
            print("", end="")
            return 2
        df.to_csv(sys.stdout, index=False)
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
