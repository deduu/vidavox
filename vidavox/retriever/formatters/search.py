# ---------------- retrieval_system/core/components.py ----------------
"""Core data structures and base classes for the retrieval system."""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SearchResult:
    """Data structure for search results."""
    doc_id: str
    text: str
    meta_data: Dict[str, Any]
    score: float


class BaseResultFormatter:
    """Base class for result formatters."""

    def format(self, result: "SearchResult") -> Dict[str, Any]:  # noqa: F821 – forward ref
        """Return a dict version of the dataclass (default impl)."""
        return asdict(result)
