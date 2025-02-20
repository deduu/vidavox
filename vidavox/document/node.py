import os
import logging
from typing import List
from dataclasses import dataclass
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkInfo:
    content: str
    metadata: dict

class DocumentNode(Document):
    pass
    
class DocumentNodes:
    """
    A container for a list of document chunks with extra helper methods.
    """
    def __init__(self, nodes: List[Document]):
        self.nodes = nodes

    def get_chunk_content_and_metadata(self, index: int) -> ChunkInfo:
        """
        Retrieve the content and metadata of the chunk at the given index.

        Args:
            index: Index of the chunk.

        Returns:
            A ChunkInfo instance containing the content and metadata.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.nodes):
            raise IndexError("Chunk index out of range")
        chunk = self.nodes[index]
        return ChunkInfo(content=chunk.page_content, metadata=chunk.metadata)

    # Optionally, implement list-like behavior:
    def __iter__(self):
        return iter(self.nodes)
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]
