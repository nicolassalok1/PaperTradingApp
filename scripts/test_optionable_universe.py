import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_optionable_universe import build_optionable_universe


def _alpaca_creds_available() -> bool:
    return bool(
        os.getenv("APCA_API_KEY_ID")
        and os.getenv("APCA_API_SECRET_KEY")
        and os.getenv("APCA_API_BASE_URL")
    )


@pytest.mark.skipif(
    not _alpaca_creds_available(),
    reason="Alpaca credentials not configured; skipping optionable universe test.",
)
def test_build_optionable_universe_smoke(tmp_path: Path):
    """
    Smoke-test for the optionable universe builder.
    Ensures the helper runs end-to-end, creates a CSV, and row count is consistent.
    """
    out_path = tmp_path / "alpaca_optionable_tickers.csv"

    n = build_optionable_universe(
        limit_assets=50,
        min_contracts=1,
        output=out_path,
    )

    assert out_path.exists(), "Output CSV was not created."

    df = pd.read_csv(out_path)
    assert len(df) == n
    if n > 0:
        assert "symbol" in df.columns

