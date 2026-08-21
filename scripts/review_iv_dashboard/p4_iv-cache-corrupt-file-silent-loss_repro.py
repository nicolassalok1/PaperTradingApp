"""p4 skeptic repro — iv-cache-corrupt-file-silent-loss.

Offline, deterministic. Redirects CACHE_IV_HISTORY_DIR to a temp dir, then
feeds record_iv_observation() with several pre-existing file states and
measures: bytes on disk before/after, rows returned by load_iv_history(),
and the warnings emitted. No tracked file is modified.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.iv_dashboard import service  # noqa: E402

INFO = {"iv": 0.1623, "dte": 30, "n_contracts": 8, "method": "greeks Alpaca", "spot": 645.2}


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def run_case(name: str, seed_bytes: bytes | None, n_calls: int = 2):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="p4_ivcache_"))
    service.CACHE_IV_HISTORY_DIR = tmp
    path = service._iv_history_path("SPY")
    if seed_bytes is not None:
        path.write_bytes(seed_bytes)
    before = path.stat().st_size if path.exists() else -1

    cap = _Capture()
    logging.getLogger().addHandler(cap)
    logging.getLogger().setLevel(logging.WARNING)
    for _ in range(n_calls):
        service.record_iv_observation("SPY", INFO)
    logging.getLogger().removeHandler(cap)

    after = path.stat().st_size if path.exists() else -1
    rows = len(service.load_iv_history("SPY"))
    content = path.read_bytes()[:80] if path.exists() else b"<absent>"
    print(f"[{name}]")
    print(f"  size before={before}  after={after}  load_iv_history rows={rows}")
    print(f"  warnings ({len(cap.msgs)}): {[m[:110] for m in cap.msgs]}")
    print(f"  head bytes: {content!r}")
    return after, rows, cap.msgs


if __name__ == "__main__":
    # Baseline: absent file -> must create and upsert (2 calls same day -> 1 row)
    run_case("absent file (baseline)", None)
    # 0-byte file (to_csv opens with 'w' => truncation happens before any write)
    run_case("0-byte file", b"")
    # header-less: columns without 'date'
    run_case("header foo,iv + 1 row", b"foo,iv\n1,0.2\n")
    # header-only CSV with a 'date' column (valid but empty) -> should be fine
    run_case("header-only date,iv", b"date,iv\n")
    # partially written last row (crash mid-write after header + 1.5 rows)
    run_case(
        "partial last row",
        b"date,iv,dte,n_contracts,method,spot\n2026-08-18,0.15,30,8,greeks Alpaca,640.0\n2026-08-19,0.1",
    )
    # garbage binary
    run_case("binary garbage", bytes(range(256)))

    # Does pandas to_csv really truncate at open (i.e. is a 0-byte file a plausible crash artefact)?
    import pandas as pd

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="p4_trunc_")) / "x.csv"
    tmp.write_text("old content\n")

    class Boom(Exception):
        pass

    class ExplodingFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return ExplodingFrame

    # Simulate a crash while pandas is formatting the body: patch the CSV formatter.
    import pandas.io.formats.csvs as csvs

    orig = csvs.CSVFormatter._save_body

    def _boom(self, *a, **k):
        raise Boom("simulated kill during write")

    csvs.CSVFormatter._save_body = _boom
    try:
        pd.DataFrame({"date": ["2026-08-20"], "iv": [0.16]}).to_csv(tmp, index=False)
    except Boom:
        pass
    finally:
        csvs.CSVFormatter._save_body = orig
    print(f"[to_csv truncation check] size after interrupted to_csv = {tmp.stat().st_size} bytes, "
          f"content={tmp.read_bytes()!r}")
