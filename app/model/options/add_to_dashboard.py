import json
from datetime import datetime

from app.model.options.core.book import load_options_book, save_options_book


def add_option_to_dashboard_clean(payload: dict):
    """Append a priced option from the pricer into the dashboard store."""
    book = load_options_book()
    oid = f"opt_{len(book)+1}_{datetime.now().timestamp()}"
    book[oid] = payload
    save_options_book(book)
    return oid


__all__ = ["add_option_to_dashboard_clean"]
