import logging
from typing import List, Tuple, Dict, Optional, Callable, Union, Any
from pathlib import Path


from vidavox.schemas.common import DocItem
from vidavox.retriever.schema.data import EngineConfig
from vidavox.document import ProcessingConfig
from vidavox.document.doc_manager import DocumentManager

logger = logging.getLogger(__name__)


class FileProcessor:
    """Coordinates file processing with proper error handling."""

    def __init__(self, processors: List[Any], config: EngineConfig):
        self.processors = processors
        self.config = config

    def select_processor(self, file_path: str):
        """Select appropriate processor for file."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        raise ValueError(f"No processor available for {file_path}")

    async def process_file(
        self,
        item: Union[str, DocItem],
        existing_docs: Dict,
        chunker: Optional[Callable],
        config: Optional[ProcessingConfig],
        **kwargs,
    ) -> Tuple[List[Tuple[str, str, Dict]], Optional[str]]:
        """
        Process a single file.

        Returns:
            (processed_docs, error_message)
        """
        try:
            # Normalize input
            if isinstance(item, str):
                path_str = item
                doc_id = Path(path_str).stem
                file_url = None
                folder_id = None
            else:
                path_str = item.path
                doc_id = item.doc_id
                file_url = item.url
                folder_id = item.folder_id

            # Select and run processor
            processor = self.select_processor(path_str)

            process_kwargs = {
                "doc_id": doc_id,
                "folder_id": folder_id,
                "existing_docs": existing_docs,
                "file_url": file_url,
                "config": config,
                "chunker": chunker,
                **kwargs,
            }

            docs = processor.process(path_str, **process_kwargs)
            return docs or [], None

        except Exception as e:
            error_msg = f"Failed to process {getattr(item, 'path', item)}: {e}"
            logger.error(error_msg, exc_info=True)
            return [], error_msg
