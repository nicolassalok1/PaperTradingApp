import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.io import load_json_file, save_json_file
from app.utils.paths import JSON_DIR

TRADES_LOG_FILE = JSON_DIR / "trades_log.json"


def load_trades_log() -> List[Dict]:
    return load_json_file(TRADES_LOG_FILE, [])


def save_trades_log(log: List[Dict]) -> None:
    save_json_file(TRADES_LOG_FILE, log)


def log_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    *,
    source: str = "manual",
    meta: Dict[str, Any] | None = None,
) -> None:
    log = load_trades_log()
    entry = {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "source": source,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta is not None:
        entry["meta"] = meta if isinstance(meta, dict) else {"info": meta}
    log.append(entry)
    save_trades_log(log)


def append_trade_log(
    symbol: str, side: str, qty: float, price: float, *, source: str, meta: Optional[Dict] = None
) -> None:
    """Best-effort trade logging: a failed write must not abort the caller.

    Moved here from `trading.buy_sell` when that module was retired — it was the
    only part of it still reached, and settlement calls it while adjusting a
    balance, where losing the log entry beats losing the settlement.
    """
    try:
        log_trade(symbol, side, qty, price, source=source, meta=meta)
    except Exception:
        pass
