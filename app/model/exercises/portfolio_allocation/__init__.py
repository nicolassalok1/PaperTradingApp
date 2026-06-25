"""
SPX/VIX risk-targeted portfolio-allocation exercise.

`engine.py` is the validated take-home reference (numerically identical to the TS
engine kept under `reference/`); it is imported unmodified. `yahoo_data.py` ports
the TS server fetch to `requests` so the Streamlit server can pull ^GSPC/^VIX
without `yfinance`. The reference TS/React/_validate bundle stays under
`reference/` (dormant) so `tsx reference/_validate/validate.ts` still runs.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_CSV = PACKAGE_DIR / "reference" / "spx_vix_daily.csv"

__all__ = ["PACKAGE_DIR", "BUNDLED_CSV"]
