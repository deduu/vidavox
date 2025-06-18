# formatters/custom.py
"""Custom result formatter with URL processing."""

from typing import Dict, Any
from .base import BaseResultFormatter
from ..core.components import SearchResult


class CustomResultFormatter(BaseResultFormatter):
    """Custom formatter that processes URLs and page numbers."""
    
    def format(self, result: SearchResult) -> Dict[str, Any]:
        """Format result with custom URL and page processing."""
        base_url = result.meta_data.get("url", "unknown_url")
        original_page = result.meta_data.get("page")
        source = result.meta_data.get("source")
        
        # Initialize with default values
        full_url = base_url
        final_page = original_page

        # Check if page has a usable value
        if original_page is not None and original_page != "unknown":
            try:
                # Attempt to convert page to an integer
                page_number = int(original_page)
                
                # Calculate the incremented page number
                incremented_page = page_number + 1
                
                # Build the full URL and set final page
                full_url = f"{base_url}#page={incremented_page}"
                final_page = incremented_page
                
            except (ValueError, TypeError):
                print(f"Warning: Could not convert page '{original_page}' to an integer.")

        return {
            "id": result.doc_id,
            "url": full_url,
            "text": result.text,
            "page": final_page,
            "source": source,
            "score": result.score,
        }
