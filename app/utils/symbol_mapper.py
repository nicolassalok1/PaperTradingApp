from __future__ import annotations


def map_to_stooq(symbol: str) -> str:
    """
    Map an input ticker to Stooq format.
    Defaults to US equities (.us). Keeps existing suffixes if provided.
    """
    sym = (symbol or "").strip()
    if not sym:
        return ""
    sym_lower = sym.lower()
    if "." in sym_lower:
        return sym_lower
    if sym_lower.startswith("^"):
        return sym_lower[1:] + ".us"
    return sym_lower + ".us"


__all__ = ["map_to_stooq"]
