from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.model.options.logic import download_options_alpaca


DEFAULT_OUTPUT = Path("logs/alpaca_optionable_smoke.csv")
logger = logging.getLogger(__name__)


DEFAULT_CANDIDATES: list[str] = [
    # Broad / liquid ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "IVV",
    "VXX",
    "UVXY",
    "TLT",
    "IEF",
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLC",
    "XLU",
    "XLB",
    "XLRE",
    "SMH",
    "SOXX",
    "ARKK",
    # Mega / high-volume single names
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "GOOG",
    "META",
    "TSLA",
    "NFLX",
    "AMD",
    "INTC",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "CSCO",
    "QCOM",
    "TXN",
    "MU",
    # Financials / energy / staples / etc.
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "XOM",
    "CVX",
    "OXY",
    "KO",
    "PEP",
    "WMT",
    "COST",
    "DIS",
    "NKE",
    "BA",
    "CAT",
    "GE",
    "PFE",
    "JNJ",
    "UNH",
    "MRK",
    # Popular "retail" tickers
    "PLTR",
    "SOFI",
    "GME",
    "AMC",
    "COIN",
    "RIVN",
    "LCID",
    "HOOD",
]


def _resolve_repo_relative_path(path: Path) -> Path:
    try:
        if path.is_absolute():
            return path
    except Exception:
        pass
    return REPO_ROOT / path


def _parse_candidates(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []
    if args.tickers:
        for part in str(args.tickers).split(","):
            sym = part.strip().upper()
            if sym:
                tickers.append(sym)

    if args.tickers_file:
        try:
            raw = Path(args.tickers_file).read_text(encoding="utf-8")
        except Exception:
            raw = ""
        for line in raw.splitlines():
            sym = line.strip().upper()
            if not sym or sym.startswith("#"):
                continue
            tickers.append(sym)

    if not tickers:
        tickers = list(DEFAULT_CANDIDATES)

    # Uniques (keep order)
    uniq: list[str] = []
    seen = set()
    for s in tickers:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _configure_logging() -> Path:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "smoke_test_alpaca_optionable_candidates.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    return log_path


def _summarize_chain(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "ok": False,
            "n_contracts": 0,
            "T_min": None,
            "T_max": None,
            "sample_opra": "",
        }
    sample_opra = ""
    try:
        sample_opra = str(df["opra"].iloc[0])
    except Exception:
        sample_opra = ""

    t_min = None
    t_max = None
    try:
        t_min = float(pd.to_numeric(df["T"], errors="coerce").min())
        t_max = float(pd.to_numeric(df["T"], errors="coerce").max())
    except Exception:
        t_min = None
        t_max = None
    return {
        "ok": True,
        "n_contracts": int(len(df)),
        "T_min": t_min,
        "T_max": t_max,
        "sample_opra": sample_opra,
    }


def smoke_test(
    tickers: Iterable[str],
    *,
    max_pages: int,
    min_days: int | None,
    output: Path,
) -> pd.DataFrame:
    results: list[dict] = []
    ok_count = 0
    tickers_list = list(tickers)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    with tqdm(total=len(tickers_list), desc="Smoke testing Alpaca options", unit="ticker", file=sys.stdout) as pbar:
        for sym in tickers_list:
            try:
                df = download_options_alpaca(
                    sym,
                    include_spot=False,
                    cache_to_csv=False,
                    max_pages=int(max_pages),
                    min_days_to_expiry=min_days,
                )
                summary = _summarize_chain(df)
                row = {"symbol": sym, **summary}
                results.append(row)
                if summary.get("ok"):
                    ok_count += 1
                logger.info("[smoke] %s ok=%s n=%s", sym, summary.get("ok"), summary.get("n_contracts"))
            except Exception as exc:
                results.append(
                    {
                        "symbol": sym,
                        "ok": False,
                        "n_contracts": 0,
                        "T_min": None,
                        "T_max": None,
                        "sample_opra": "",
                        "error": str(exc),
                    }
                )
                logger.info("[smoke] %s error=%s", sym, exc)

            pbar.update(1)
            pbar.set_postfix({"ok": ok_count})

    df_out = pd.DataFrame(results)
    df_out.to_csv(output, index=False)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quick smoke test for Alpaca options snapshots on a curated list of "
            "high-likelihood optionable tickers (ETFs + mega caps)."
        )
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers to test (overrides defaults).",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default="",
        help="Path to a text file containing one ticker per line (comments with # supported).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max snapshot pages to pull per ticker (default: 10).",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=1,
        help="Minimum days to expiry to keep (default: 1; use 0 to include same-day, -1 to include expired).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"CSV output path (default: {DEFAULT_OUTPUT}; relative paths are repo-root relative).",
    )

    args = parser.parse_args()
    log_path = _configure_logging()

    tickers = _parse_candidates(args)
    out_path = _resolve_repo_relative_path(Path(args.output))
    min_days = int(args.min_days)
    if min_days < 0:
        min_days_to_expiry = None
    else:
        min_days_to_expiry = min_days

    df = smoke_test(
        tickers,
        max_pages=int(args.max_pages),
        min_days=min_days_to_expiry,
        output=out_path,
    )
    ok = int(df["ok"].sum()) if "ok" in df.columns else 0
    print(f"OK: {ok}/{len(df)} tickers. Output: {out_path}")
    print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    main()

