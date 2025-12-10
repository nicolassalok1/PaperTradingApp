import json
from pathlib import Path
from typing import Dict

from app.utils.io import load_json_file, save_json_file
from app.utils.paths import JSON_DIR

FORWARDS_FILE = JSON_DIR / "forwards_portfolio.json"


def load_forwards() -> Dict[str, dict]:
    return load_json_file(FORWARDS_FILE, {})


def save_forwards(forwards: Dict[str, dict]) -> None:
    sanitized: Dict[str, dict] = {}
    allowed_keys = {"symbol", "maturity", "forward_price", "quantity", "side", "created_at"}
    for fid, f in (forwards or {}).items():
        if not isinstance(f, dict):
            continue
        sanitized[fid] = {
            "symbol": str(f.get("symbol", "")).strip().upper(),
            "maturity": f.get("maturity"),
            "forward_price": float(f.get("forward_price", 0.0) or 0.0),
            "quantity": float(f.get("quantity", 0.0) or 0.0),
            "side": str(f.get("side", "long")).lower(),
            "created_at": f.get("created_at"),
        }
        sanitized[fid] = {k: v for k, v in sanitized[fid].items() if k in allowed_keys}
    save_json_file(FORWARDS_FILE, sanitized)
