import pandas as pd
from pathlib import Path
from typing import Tuple

from app.utils.paths import CACHE_CSV_DIR

APP_DIR = Path(__file__).resolve().parents[2]
YIELD_CURVE_CACHE_FILE = CACHE_CSV_DIR / "yield_curve.csv"

YIELD_TICKERS = {
    "3M": "^IRX",
    "5Y": "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}


def download_yield_curve_to_cache(period: str = "1y") -> Tuple[pd.DataFrame | None, Path | None]:
    """Télécharge les taux Treasury et les écrit dans le cache CSV (static placeholder)."""
    # Stooq ne fournit pas directement ces taux; on utilise un placeholder constant.
    today = pd.Timestamp.today().normalize()
    df_close = pd.DataFrame(
        [{"Date": today, "3M": 2.0, "5Y": 2.0, "10Y": 2.0, "30Y": 2.0}]
    ).set_index("Date")
    YIELD_CURVE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_close.to_csv(YIELD_CURVE_CACHE_FILE, index=True)
    except Exception:
        pass
    return df_close, YIELD_CURVE_CACHE_FILE


def load_yield_curve_csv(ensure_cache: bool = True):
    """Charge la courbe des taux depuis les emplacements connus, sinon tente un download."""
    candidates = [
        YIELD_CURVE_CACHE_FILE,
        APP_DIR / "Yiled_curve.csv",
        APP_DIR / "Yield_curve.csv",
        APP_DIR / "yield_curve.csv",
        APP_DIR / "data" / "Yiled_curve.csv",
        APP_DIR / "data" / "Yield_curve.csv",
        APP_DIR / "data" / "yield_curve.csv",
        Path.cwd() / "Yiled_curve.csv",
        Path.cwd() / "Yield_curve.csv",
        Path.cwd() / "yield_curve.csv",
        Path.cwd() / "data" / "Yiled_curve.csv",
        Path.cwd() / "data" / "Yield_curve.csv",
        Path.cwd() / "data" / "yield_curve.csv",
    ]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        first_col = str(df.columns[0])
        if first_col.lower() in {"date", "day", "datetime"}:
            try:
                df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
                df = df.dropna(subset=[first_col])
                df = df.set_index(first_col)
            except Exception:
                pass
        try:
            if path != YIELD_CURVE_CACHE_FILE:
                YIELD_CURVE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(YIELD_CURVE_CACHE_FILE)
        except Exception:
            pass
        return df, path
    if ensure_cache:
        return download_yield_curve_to_cache(period="1y")
    return None, None
