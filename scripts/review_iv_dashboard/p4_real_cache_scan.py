"""p4 probe: scan locally cached OHLC CSVs (Stooq/Yahoo fallback material) for
duplicate dates and non-positive closes -- sizes the realistic likelihood of
findings duplicate-date-crash and rv-bad-close-silent-drop. No network."""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
roots = [
    os.path.join(WORKTREE, "cache"),
    os.path.abspath(os.path.join(WORKTREE, "..", "..", "..", "cache")),
]
files = []
for r in roots:
    files += glob.glob(os.path.join(r, "OHLC", "*.csv"))
    files += glob.glob(os.path.join(r, "*.csv"))
    files += glob.glob(os.path.join(r, "stooq*", "*.csv"))
    files += glob.glob(os.path.join(r, "**", "*.csv"), recursive=True)
files = sorted(set(files))
print(f"{len(files)} csv files")
tot_dup = tot_bad = n_price = 0
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:  # noqa: BLE001
        print("  skip", os.path.basename(f), e)
        continue
    dcol = next((c for c in df.columns if str(c).lower() in ("date", "timestamp", "datetime")), None)
    ccol = next((c for c in df.columns if str(c).lower() == "close"), None)
    if dcol is None or ccol is None:
        continue
    n_price += 1
    d = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    c = pd.to_numeric(df[ccol], errors="coerce")
    ndup = int(d.duplicated().sum())
    nbad = int(((c <= 0) | c.isna()).sum())
    tot_dup += ndup
    tot_bad += nbad
    flag = "  <--" if (ndup or nbad) else ""
    print(f"  {os.path.basename(f):50s} rows={len(df):5d} dupDates={ndup:3d} close<=0|nan={nbad:3d}{flag}")
print(f"price files={n_price} TOTAL duplicate dates={tot_dup}  bad closes={tot_bad}")
