# ---------------- retrieval_system/utils/helpers.py ----------------
"""Misc, lightweight helpers used across modules."""

import os
import re
from pathlib import Path
from typing import Union


def slugify(text: str) -> str:
    """Very small helper for filesystem-safe names."""
    text = re.sub(r"[^\w\-_. ]", "_", text)
    return re.sub(r"\s+", "_", text).strip("_")


def ensure_dir(path: Union[str, Path]) -> None:
    """Create directory (incl. parents) iff it doesn’t yet exist."""
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)
