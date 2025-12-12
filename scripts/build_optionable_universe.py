from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from app.model.options.logic import fetch_alpaca_option_tickers, download_options_alpaca


DEFAULT_OUTPUT = Path("data/alpaca_optionable_tickers.csv")


def build_optionable_universe(limit_assets: int, min_contracts: int, output: Path) -> int:
    """
    Build a list of underlyings that have options available via Alpaca.

    1) Pull a universe of active US equities from Alpaca.
    2) For each symbol, call download_options_alpaca().
    3) Keep only symbols with a non-empty options DataFrame (and at least min_contracts rows).
    4) Save the result to a CSV file for use in the UI.
    """
    symbols = fetch_alpaca_option_tickers(limit=limit_assets)
    optionable: List[dict] = []

    for sym in symbols:
        df = download_options_alpaca(sym)
        if df is None or df.empty:
            continue
        if len(df) < min_contracts:
            continue
        optionable.append({"symbol": sym, "n_contracts": int(len(df))})

    output.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(optionable)
    if not df_out.empty:
        df_out = df_out.sort_values("symbol")
    df_out.to_csv(output, index=False)
    return int(len(df_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a precomputed universe of optionable tickers for Alpaca "
            "and save it to a CSV used by the Options (Alpaca) UI."
        )
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        default=500,
        help="Maximum number of Alpaca assets to scan (default: 500).",
    )
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=10,
        help="Minimum number of option contracts required to keep a ticker (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Path to the output CSV file (default: {DEFAULT_OUTPUT}).",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    n = build_optionable_universe(
        limit_assets=args.limit_assets,
        min_contracts=args.min_contracts,
        output=output_path,
    )
    print(f"Saved {n} optionable tickers to {output_path}")


if __name__ == "__main__":
    main()

