"""
Portfolio repository helpers (moved from app.utils.repository).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.utils.paths import JSON_DIR

PORTFOLIO_FILE = JSON_DIR / "spot_portfolio.json"


def load_json(path: Path):
    with Path(path).open() as f:
        return json.load(f)


def save_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_portfolio(path: Path = PORTFOLIO_FILE):
    """
    Load portfolio JSON (options, quantities, structure).
    Pure data load: no computation.
    """
    return load_json(path)


def save_portfolio(data, path: Path = PORTFOLIO_FILE):
    """
    Save updated portfolio JSON.
    """
    save_json(path, data)


__all__ = ["load_portfolio", "save_portfolio", "PORTFOLIO_FILE"]
