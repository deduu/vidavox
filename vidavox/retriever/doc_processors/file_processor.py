# processors/file_processor.py
"""Standard file processor for regular documents."""

import logging
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path

from .base import BaseProcessor
from vidavox.document import DocumentSplitter, ProcessingConfig
from vidavox.document.file_processor import FileProcessor
from vidavox.document_store.models import Document

logger = logging.getLogger(__name__)


class StandardFileProcessor(BaseProcessor):
    """Processor for standard document files (PDF, DOCX, etc.)."""

    def __init__(self):
        self.file_processor = FileProcessor()
        self._file_mtime_index: Dict[str, str] = {}

    def can_process(self, file_path: str, **kwargs) -> bool:
        """Check if this is a standard document file."""
        # Process any file that's not CSV or Excel
        ext = Path(file_path).suffix.lower()
        return ext not in ['.csv', '.xlsx', '.xls']

    def process(
        self,
        file_path: str,
        doc_id: str,
        folder_id: str,
        existing_docs: Dict[str, Document],
        file_url: Optional[str] = None,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        use_recursive: bool = True,
        **kwargs
    ) -> List[Tuple[str, str, Dict]]:
        """Process a standard document file."""
        config = config or ProcessingConfig()
        file_name = Path(file_path).name

        # Check file modification time for incremental processing
        base_meta = self.file_processor.get_file_metadata(Path(file_path))
        current_mtime = base_meta.get("modification_time")

        # Skip if file hasn't changed
        if self._file_mtime_index.get(file_name) == current_mtime:
            logger.info(
                f"{file_name} unchanged (mtime={current_mtime}); skipping.")
            return []

        batch_docs: List[Tuple[str, str, Dict]] = []
        try:
            # Process the document into chunks
            nodes = DocumentSplitter(config, use_recursive=use_recursive).run(
                file_path, chunker
            )

            for idx, doc in enumerate(nodes):
                file_doc_id = f"{doc_id}_{file_name}_chunk{idx}"
                meta = dict(base_meta)  # copy base metadata
                meta["file_type"] = Path(file_path).suffix.lower().lstrip(".")
                meta["url"] = file_url
                meta["folder_id"] = folder_id

                # Combine with document-specific metadata
                combined_meta = {**meta, **doc.metadata}
                batch_docs.append(
                    (file_doc_id, doc.page_content, combined_meta))

            # Update mtime index
            self._file_mtime_index[file_name] = current_mtime
            logger.info(
                f"Successfully processed {len(nodes)} chunks from {file_name}")

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")

        return batch_docs
