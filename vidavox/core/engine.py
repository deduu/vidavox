"""
vidavox.core.engine
Just re-exports the real class that lives in retrieval_system.core.engine
so that IDEs and type checkers can resolve it unambiguously.
"""
from __future__ import annotations

from ..retrieval_system.core.engine import RetrievalEngine as RetrievalEngine

__version__ = "0.2"
__all__: list[str] = ["RetrievalEngine"]
