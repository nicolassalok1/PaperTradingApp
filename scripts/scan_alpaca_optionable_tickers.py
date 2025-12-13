from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from urllib.parse import quote_plus

import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.model.options.logic import _load_alpaca_credentials, fetch_alpaca_option_tickers


DEFAULT_OUTPUT = Path("data/scan_alpaca_optionable_tickers.csv")
logger = logging.getLogger(__name__)


def _resolve_repo_relative_path(path: Path) -> Path:
    """Interpret relative paths as repo-root relative (robust to being run from any CWD)."""
    try:
        if path.is_absolute():
            return path
    except Exception:
        pass
    return REPO_ROOT / path


def _configure_logging() -> Path:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "scan_alpaca_optionable_tickers.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])
    return log_path


def _append_row(output: Path, row: dict) -> None:
    """Append a single row to CSV so progress is visible in real time."""
    try:
        header_needed = not output.exists()
        pd.DataFrame([row]).to_csv(output, mode="a", header=header_needed, index=False)
    except Exception:
        pass


def _is_optionable_via_contracts_endpoint(
    symbol: str,
    *,
    base_url: str,
    headers: dict,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> bool:
    """
    Check if an underlying has at least one ACTIVE option contract via:
      GET {base_url}/v2/options/contracts?underlying_symbols=SYM&limit=1&status=active
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return False

    url = f"{(base_url or '').rstrip('/')}/v2/options/contracts"
    params = {"underlying_symbols": sym, "limit": 1, "status": "active"}

    last_exc: Exception | None = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=float(timeout_s))
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(max(0.0, float(backoff_s)) * (2**attempt))
                continue
            logger.info("[alpaca-contracts] %s request failed: %s", sym, exc)
            return False

        if resp.status_code == 200:
            try:
                payload = resp.json() or {}
            except Exception:
                payload = {}
            contracts = payload.get("option_contracts") or []
            return isinstance(contracts, list) and len(contracts) > 0

        if resp.status_code == 429 and attempt < retries:
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else float(backoff_s) * (2**attempt)
            except Exception:
                delay = float(backoff_s) * (2**attempt)
            time.sleep(min(60.0, max(1.0, delay)))
            continue

        # Invalid/unsupported symbols (e.g. warrants/units) often return 422.
        if resp.status_code in (400, 404, 422):
            logger.info("[alpaca-contracts] %s not optionable (%d).", sym, resp.status_code)
            return False

        if 500 <= resp.status_code < 600 and attempt < retries:
            time.sleep(max(0.0, float(backoff_s)) * (2**attempt))
            continue

        logger.info("[alpaca-contracts] %s unexpected status %d: %s", sym, resp.status_code, resp.text[:200])
        return False

    if last_exc is not None:
        logger.info("[alpaca-contracts] %s request failed after retries: %s", sym, last_exc)
    return False


def _snapshot_contract_count(
    symbol: str,
    *,
    headers: dict,
    feed: str,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> int:
    """
    Return number of option snapshots returned for an underlying on the FIRST page:
      GET https://data.alpaca.markets/v1beta1/options/snapshots/{SYM}?feed=...
    If 0, consider the underlying as not optionable (or invalid).
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return 0

    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{quote_plus(sym)}"
    params = {"feed": str(feed or "indicative")}

    last_exc: Exception | None = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=float(timeout_s))
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(max(0.0, float(backoff_s)) * (2**attempt))
                continue
            logger.info("[alpaca-snapshots] %s request failed: %s", sym, exc)
            return 0

        if resp.status_code == 200:
            try:
                payload = resp.json() or {}
            except Exception:
                payload = {}
            snapshots = payload.get("snapshots") or payload.get("data") or {}
            return int(len(snapshots)) if isinstance(snapshots, dict) else 0

        if resp.status_code == 429 and attempt < retries:
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else float(backoff_s) * (2**attempt)
            except Exception:
                delay = float(backoff_s) * (2**attempt)
            time.sleep(min(60.0, max(1.0, delay)))
            continue

        if resp.status_code in (400, 404, 422):
            logger.info("[alpaca-snapshots] %s not optionable (%d).", sym, resp.status_code)
            return 0

        if 500 <= resp.status_code < 600 and attempt < retries:
            time.sleep(max(0.0, float(backoff_s)) * (2**attempt))
            continue

        logger.info("[alpaca-snapshots] %s unexpected status %d: %s", sym, resp.status_code, resp.text[:200])
        return 0

    if last_exc is not None:
        logger.info("[alpaca-snapshots] %s request failed after retries: %s", sym, last_exc)
    return 0


def scan_alpaca_optionable_tickers(
    *,
    output: Path,
    limit: int | None,
    method: str,
    feed: str,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> int:
    key, secret, base_url = _load_alpaca_credentials()
    if not key or not secret or not base_url:
        logger.error("Missing Alpaca credentials (APCA_API_KEY_ID / APCA_API_SECRET_KEY / APCA_API_BASE_URL).")
        return 0

    symbols = fetch_alpaca_option_tickers(limit=limit)
    if not symbols:
        logger.error("No tickers returned from Alpaca assets endpoint; check credentials and connectivity.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.unlink(missing_ok=True)
    except Exception:
        pass

    # Create file early (useful if interrupted before the first optionable ticker).
    try:
        output.write_text("symbol,n_contracts\n", encoding="utf-8")
    except Exception:
        pass

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    method_norm = str(method or "snapshots").strip().lower()
    if method_norm not in {"snapshots", "contracts"}:
        raise ValueError("method must be one of: snapshots, contracts")

    optionable: list[dict] = []
    ok_count = 0
    try:
        disable_bar = not sys.stdout.isatty()
    except Exception:
        disable_bar = True

    with tqdm(total=len(symbols), desc="Scanning Alpaca underlyings", unit="ticker", disable=disable_bar) as pbar:
        for sym in symbols:
            if method_norm == "contracts":
                ok = _is_optionable_via_contracts_endpoint(
                    sym,
                    base_url=base_url,
                    headers=headers,
                    timeout_s=timeout_s,
                    retries=retries,
                    backoff_s=backoff_s,
                )
                n_contracts = 1 if ok else 0
            else:
                n_contracts = _snapshot_contract_count(
                    sym,
                    headers=headers,
                    feed=str(feed or "indicative"),
                    timeout_s=timeout_s,
                    retries=retries,
                    backoff_s=backoff_s,
                )
                ok = n_contracts > 0

            if ok:
                row = {"symbol": sym, "n_contracts": int(n_contracts)}
                optionable.append(row)
                ok_count += 1
                _append_row(output, row)
            pbar.update(1)
            pbar.set_postfix({"ok": ok_count})

    df_out = pd.DataFrame(optionable, columns=["symbol", "n_contracts"])
    if not df_out.empty:
        df_out = df_out.drop_duplicates(subset=["symbol"]).sort_values("symbol")
    df_out.to_csv(output, index=False)
    return int(len(df_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan all Alpaca tickers and output the list of optionable underlyings (CSV)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Path to the output CSV file (default: {DEFAULT_OUTPUT}; relative paths are repo-root relative).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of Alpaca underlyings scanned (0 = no limit).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="snapshots",
        choices=["snapshots", "contracts"],
        help=(
            "How to detect 'optionable'. "
            "'snapshots' uses Alpaca Market Data options snapshots (recommended for the options chain UI). "
            "'contracts' uses the trading API /v2/options/contracts."
        ),
    )
    parser.add_argument(
        "--feed",
        type=str,
        default="indicative",
        help="Alpaca market data feed for the snapshots method (default: indicative).",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout (seconds).")
    parser.add_argument("--retries", type=int, default=2, help="Retries per ticker on transient errors / 429.")
    parser.add_argument("--backoff", type=float, default=1.0, help="Base backoff (seconds) for retries.")

    args = parser.parse_args()
    output_path = _resolve_repo_relative_path(Path(args.output))
    limit = int(args.limit) if args.limit and int(args.limit) > 0 else None

    log_path = _configure_logging()
    n = scan_alpaca_optionable_tickers(
        output=output_path,
        limit=limit,
        method=str(args.method),
        feed=str(args.feed),
        timeout_s=float(args.timeout),
        retries=int(args.retries),
        backoff_s=float(args.backoff),
    )
    print(f"Saved {n} optionable tickers to {output_path}")
    print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    main()
