# formatters/base.py
"""Base formatter class."""

from typing import Dict, Any
from dataclasses import asdict
from ..core.components import SearchResult


class BaseResultFormatter:
    """Base class for result formatters."""
    
    def format(self, result: SearchResult) -> Dict[str, Any]:
        """Format a search result into a dictionary.
        
        Args:
            result: SearchResult to format
            
        Returns:
            Formatted result as dictionary
        """
        return asdict(result)