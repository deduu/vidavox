# ---------------- retrieval_system/utils/validation.py ----------------
"""Simple dataframe validation helpers."""

from typing import Sequence
import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str]):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
