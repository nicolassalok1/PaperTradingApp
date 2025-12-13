from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List
import logging

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.model.options.logic import fetch_alpaca_option_tickers, download_options_alpaca
from app.utils.paths import CACHE_CSV_DIR
from app.utils.symbol_mapper import map_to_stooq


DEFAULT_OUTPUT = Path("data/alpaca_optionable_tickers.csv")
logger = logging.getLogger(__name__)


def _append_optionable_row(output: Path, row: dict) -> None:
    """
    Append a single optionable row to CSV so progress is visible in real time.
    Headers are written only once, when the file does not yet exist.
    """
    try:
        header_needed = not output.exists()
        pd.DataFrame([row]).to_csv(output, mode="a", header=header_needed, index=False)
    except Exception:
        # Best-effort; don't fail the whole run on a single write issue.
        pass


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
    logger.info("Fetched %d tradable symbols from Alpaca (limit=%s).", len(symbols), limit or "none")
    optionable: List[dict] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    # Fresh file for this run; rows will be appended as soon as they qualify.
    output.unlink(missing_ok=True)

    ok_count = 0
    with tqdm(total=len(symbols), desc="Scanning Alpaca tickers", unit="ticker") as pbar:
        for sym in symbols:
            df = download_options_alpaca(sym)
            _delete_downloaded_options(sym)
            _delete_stooq_cache(sym)
            if df is None or df.empty:
                pbar.update(1)
                pbar.set_postfix({"ok": ok_count})
                logger.info("[alpaca-options] %s returned no contracts.", sym)
                continue
            if min_contracts and min_contracts > 0 and len(df) < min_contracts:
                pbar.update(1)
                pbar.set_postfix({"ok": ok_count})
                logger.info(
                    "[alpaca-options] %s skipped: %d contracts < min_contracts=%d.",
                    sym,
                    len(df),
                    min_contracts,
                )
                continue
            row = {"symbol": sym, "n_contracts": int(len(df))}
            optionable.append(row)
            _append_optionable_row(output, row)
            ok_count += 1
            pbar.update(1)
            pbar.set_postfix({"ok": ok_count})
            logger.info("[alpaca-options] %s OK: %d contracts (appended).", sym, len(df))

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

    def _configure_logging():
        log_dir = REPO_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "build_optionable_universe.log"
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler])
        return log_path

    log_path = _configure_logging()

    n = build_optionable_universe(
        limit_assets=args.limit_assets,
        min_contracts=args.min_contracts,
        output=output_path,
    )
    print(f"Saved {n} optionable tickers to {output_path}")
    print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    main()
