from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.model.options.logic import fetch_alpaca_option_tickers, download_options_alpaca
from app.utils.paths import CACHE_CSV_DIR
from app.utils.symbol_mapper import map_to_stooq


DEFAULT_OUTPUT = Path("data/alpaca_optionable_tickers.csv")


def _delete_downloaded_options(sym: str) -> None:
    """Best-effort removal of the per-symbol options cache file."""
    sym_norm = (sym or "").strip().upper()
    if not sym_norm:
        return
    cache_file = CACHE_CSV_DIR / f"options_alpaca_{sym_norm}.csv"
    try:
        cache_file.unlink(missing_ok=True)
    except Exception:
        pass


def _delete_stooq_cache(sym: str) -> None:
    """Remove Stooq price cache used transiently for spot lookup."""
    mapped = map_to_stooq(sym)
    if not mapped:
        return
    for suffix in ("D", "d"):
        cache_file = CACHE_CSV_DIR / f"stooq_{mapped}_start_end_{suffix}.csv"
        try:
            cache_file.unlink(missing_ok=True)
        except Exception:
            pass


def build_optionable_universe(limit_assets: int | None, min_contracts: int, output: Path) -> int:
    """
    Build a list of underlyings that have options available via Alpaca.

    1) Pull a universe of active US equities from Alpaca (unlimited by default).
    2) For each symbol, call download_options_alpaca().
    3) Keep only symbols with a non-empty options DataFrame (optional min_contracts filter).
    4) Save the result to a CSV file for use in the UI.
    5) Remove downloaded options and spot cache files so only the summary CSV remains.
    """
    limit = limit_assets if limit_assets and limit_assets > 0 else None
    symbols = fetch_alpaca_option_tickers(limit=limit)
    optionable: List[dict] = []

    for sym in symbols:
        df = download_options_alpaca(sym)
        _delete_downloaded_options(sym)
        _delete_stooq_cache(sym)
        if df is None or df.empty:
            continue
        if min_contracts and min_contracts > 0 and len(df) < min_contracts:
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
        default=0,
        help="Maximum number of Alpaca assets to scan (0 or negative means no limit; default: unlimited).",
    )
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=0,
        help="Minimum number of option contracts required to keep a ticker (0 or negative means no minimum).",
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
