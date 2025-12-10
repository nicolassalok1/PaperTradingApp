from typing import Dict

from app.utils.io import load_json_file, save_json_file
from app.utils.paths import JSON_DIR

OPTIONS_BOOK_FILE = JSON_DIR / "options_portfolio.json"


def load_options_book() -> Dict[str, dict]:
    """Load canonical options portfolio (no ancienne fallbacks)."""
    data = load_json_file(OPTIONS_BOOK_FILE, {})
    return data if isinstance(data, dict) else {}


def save_options_book(book: Dict[str, dict]) -> None:
    """Persist options portfolio to canonical path."""
    save_json_file(OPTIONS_BOOK_FILE, book if isinstance(book, dict) else {})
