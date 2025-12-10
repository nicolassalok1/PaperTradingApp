# placeholder
# Portfolio persistence and normalization utilities.
import json
from pathlib import Path
from typing import Dict

from app.utils.paths import JSON_DIR


def load_portfolio(portfolio_file: Path) -> Dict[str, dict]:
    """Load the portfolio snapshot from disk; returns an empty dict on failure."""
    try:
        with open(portfolio_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    normalized: Dict[str, dict] = {}
    for sym, entry in obj.items():
        e = entry if isinstance(entry, dict) else {}
        normalized[sym] = {
            "quantity": float(e.get("quantity", 0.0) or 0.0),
            "side": str(e.get("side", "long")).lower(),
            "last_updated": e.get("last_updated"),
        }
    return normalized


def save_portfolio(portfolio_file: Path, portfolio: dict) -> None:
    """Persist the portfolio snapshot to disk."""
    payload: Dict[str, dict] = {}
    for sym, entry in (portfolio or {}).items():
        e = entry if isinstance(entry, dict) else {}
        payload[sym] = {
            "quantity": float(e.get("quantity", 0.0) or 0.0),
            "side": str(e.get("side", "long")).lower(),
            "last_updated": e.get("last_updated"),
        }
    portfolio_file.parent.mkdir(parents=True, exist_ok=True)
    with open(portfolio_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


PORTFOLIO_FILE = JSON_DIR / "spot_portfolio.json"


def load_portfolio_default() -> Dict[str, dict]:
    return load_portfolio(PORTFOLIO_FILE)


def save_portfolio_default(portfolio: dict) -> None:
    save_portfolio(PORTFOLIO_FILE, portfolio)
