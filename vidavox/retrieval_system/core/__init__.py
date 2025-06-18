# ---------------- retrieval_system/core/__init__.py ----------------
"""Core sub-package - shared data structures and main engine."""

from .components import SearchResult, BaseResultFormatter  # noqa: F401
from .engine import RetrievalEngine  # noqa: F401

__all__ = ["SearchResult", "BaseResultFormatter", "RetrievalEngine"]