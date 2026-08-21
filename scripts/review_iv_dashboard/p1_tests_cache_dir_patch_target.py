"""
Probe: which attribute must a unit test patch to redirect the IV history cache to
tmp_path?  service.py does `from app.utils.paths import CACHE_IV_HISTORY_DIR`, so the
name is bound in service's namespace at import. Also runs a record/load round trip
(write to a temp dir, never to the real cache) and checks the upsert semantics.
No network.
"""
import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import app.utils.paths as paths  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402

real = svc._iv_history_path("spy")
print("default path:", real, "| exists before probe:", real.exists())

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    # (a) patching app.utils.paths does NOT redirect the service
    paths.CACHE_IV_HISTORY_DIR = tmp / "via_paths"
    print("patch app.utils.paths.CACHE_IV_HISTORY_DIR ->", svc._iv_history_path("spy").parent == tmp / "via_paths")
    # (b) patching the service's module-global DOES
    svc.CACHE_IV_HISTORY_DIR = tmp / "via_service"
    print("patch service.CACHE_IV_HISTORY_DIR          ->", svc._iv_history_path("spy").parent == tmp / "via_service")

    # round trip + upsert (same day twice -> one row; iv overwritten by the 2nd call)
    info1 = {"iv": 0.21, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 500.0}
    info2 = {"iv": 0.25, "dte": 31, "n_contracts": 4, "method": "mixte (greeks + inversion BS)", "spot": 501.0}
    svc.record_iv_observation(" spy ", info1)
    svc.record_iv_observation("SPY", info2)
    p = svc._iv_history_path("SPY")
    raw = pd.read_csv(p)
    print("csv path:", p.name, "| rows after 2 same-day upserts:", len(raw), "| iv stored:", raw["iv"].tolist())
    hist = svc.load_iv_history("spy")
    print("load_iv_history columns:", list(hist.columns), "| dtypes date:", hist["date"].dtype, "| iv:", hist["iv"].tolist())

    # pre-existing older row stays and sorts before today
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    pd.DataFrame([{"date": yesterday, "iv": 0.19, "dte": 29, "n_contracts": 5, "method": "x", "spot": 499.0}]).to_csv(p, index=False)
    svc.record_iv_observation("SPY", info1)
    hist = svc.load_iv_history("SPY")
    print("after upsert on a 1-row older file: dates=", [d.date().isoformat() for d in hist["date"]], "ivs=", hist["iv"].tolist())

    # corrupt file -> empty frame (no raise)
    p.write_text("not,a,csv\n\x00\x00", encoding="utf-8")
    print("corrupt csv -> load returns empty:", svc.load_iv_history("SPY").empty)

    # missing file -> empty frame with the expected columns
    print("missing symbol -> columns:", list(svc.load_iv_history("ZZZZ").columns))

    # record with iv=None must not raise (best-effort)
    try:
        svc.record_iv_observation("SPY", {"iv": None})
        print("record with iv=None: no raise (logged warning)")
    except Exception as exc:  # noqa: BLE001
        print("record with iv=None RAISED:", exc)

print("real cache untouched:", real.exists() is False or real.stat().st_mtime < __import__('time').time() - 60)
