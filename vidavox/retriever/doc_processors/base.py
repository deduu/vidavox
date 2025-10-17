
# processors/base.py
"""Base processor interface."""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path

from vidavox.document_store.models import Document
from vidavox.document import ProcessingConfig

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Abstract base class for file processors."""

    @abstractmethod
    def can_process(self, file_path: str, **kwargs) -> bool:
        """Check if this processor can handle the given file."""
        pass

    @abstractmethod
    def process(
        self,
        file_path: str,
        doc_id: str,
        folder_id: str,
        existing_docs: Dict[str, Document],
        **kwargs
    ) -> List[Tuple[str, str, Dict]]:
        """Process the file and return list of (doc_id, text, metadata) tuples."""
        pass
