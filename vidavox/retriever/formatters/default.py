# formatters/default.py
"""Default result formatter."""

from typing import Dict, Any
from .base import BaseResultFormatter
from ..formatters.search import SearchResult


class DefaultResultFormatter(BaseResultFormatter):
    """Default formatter that extracts basic fields."""

    def format(self, result: SearchResult) -> Dict[str, Any]:
        """Format result with default field extraction."""
        return {
            "id": result.doc_id,
            "url": result.meta_data.get('url', 'unknown_url'),
            "text": result.text,
            "page": result.meta_data.get('page', 'unknown'),
            "score": result.score,
        }
