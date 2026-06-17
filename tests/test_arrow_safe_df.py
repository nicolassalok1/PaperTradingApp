"""Regression: display-time DataFrame normalisation avoids the Arrow mixed-column warning [E08].

Streamlit emits `UserWarning: The DataFrame has column names of mixed type. They will be
converted to strings and not roundtrip correctly.` when a DataFrame with heterogeneous
column-name types reaches Arrow conversion (st.dataframe). main_app._arrow_safe_df is the
display chokepoint that makes DataFrames Arrow-friendly; it must also normalise mixed-type
column names to str — on the display copy only, without mutating the caller's DataFrame.

Run: conda run -n papertrading python -m pytest tests/test_arrow_safe_df.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.mark.unit
def test_arrow_safe_df_stringifies_mixed_type_columns():
    from app.vue import main_app

    df = pd.DataFrame([[1, 2, 3]], columns=["label", 0, 1.5])  # str + int + float names
    safe = main_app._arrow_safe_df(df)

    # Display copy has uniform str column names -> no Arrow mixed-type warning.
    assert all(isinstance(c, str) for c in safe.columns)
    assert list(safe.columns) == ["label", "0", "1.5"]
    # The caller's DataFrame is untouched.
    assert list(df.columns) == ["label", 0, 1.5]


@pytest.mark.unit
def test_arrow_safe_df_leaves_uniform_columns_untouched():
    from app.vue import main_app

    df = pd.DataFrame([[1, 2]], columns=["a", "b"])
    safe = main_app._arrow_safe_df(df)
    assert list(safe.columns) == ["a", "b"]
