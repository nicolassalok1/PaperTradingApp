"""Probe: _decode_opra right-anchored slicing against an attack set (offline)."""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.iv_dashboard.service import _decode_opra  # noqa: E402

# (symbol, expected (strike, expiry, type) or None if garbage)
CASES = [
    ("SPY260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
    ("SPY260918P00450000", (450.0, dt.date(2026, 9, 18), "put")),
    ("SPXW260918C05000000", (5000.0, dt.date(2026, 9, 18), "call")),
    ("XSP260918P00640000", (640.0, dt.date(2026, 9, 18), "put")),
    ("BRKB260918C00480000", (480.0, dt.date(2026, 9, 18), "call")),
    ("AAPL1260918C00150000", (150.0, dt.date(2026, 9, 18), "call")),   # adjusted contract root+digit
    ("SPX260918C12345500", (12345.5, dt.date(2026, 9, 18), "call")),  # 5-digit strike
    ("SPY7260918C00640000", (640.0, dt.date(2026, 9, 18), "call")),   # mini-style root
    ("SPY260918c00450000", (450.0, dt.date(2026, 9, 18), "call")),     # lowercase type
    ("SPY260918X00450000", None),      # invalid type letter -> should be None
    ("SPY260918C0045000", None),       # 7-digit strike (malformed)
    ("260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),        # no root
    ("SPY261318C00450000", None),      # month 13
    ("", None),
    ("garbage", None),
    (None, None),
    (123456789012345, None),           # int -> str -> digits only
    ("SPY260918C00450000 ", None),     # trailing space
    ("O:SPY260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),   # polygon-style prefix
]

wrong_silent = []
for sym, expected in CASES:
    got = _decode_opra(sym)
    got_n = None if got == (None, None, None) else got
    ok = got_n == expected
    tag = "OK " if ok else "BAD"
    if not ok:
        wrong_silent.append((sym, expected, got_n))
    print(f"{tag} {sym!r:28} -> {got_n}   (expected {expected})")

print("\nSilently wrong decodes (non-None but != expected):",
      [(s, g) for s, e, g in wrong_silent if g is not None])
print("Wrong None (valid but rejected):", [(s, e) for s, e, g in wrong_silent if g is None])
