import os
import logging
from tqdm import tqdm
from typing import List, Optional, Callable, Dict
from langchain.docstore.document import Document
from dataclasses import dataclass, field
from threading import Lock
from vidavox.document.config import ProcessingConfig, SplitterConfig
from vidavox.document.loader import LoaderFactory
from vidavox.document.node import DocumentNodes

from vidavox.utils.script_tracker import log_processing_time

@dataclass
class DocumentProcessingContext:
    """Context for document processing pipeline"""
    file_path: str
    documents: List[Document] = field(default_factory=list)
    split_documents: List[Document] = field(default_factory=list)
    config: Optional[ProcessingConfig] = None
    custom_chunker: Optional[Callable[[Document], List[Document]]] = None
    show_progress: bool = False
    use_recursive: bool = True
    metadata: Dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)  # Thread safety
    
    def add_metadata(self, key: str, value):
        with self._lock:
            self.metadata[key] = value
    
    def add_split_document(self, doc: Document):
        """Thread-safe method to add split documents"""
        with self._lock:
            self.split_documents.append(doc)
    
    def extend_split_documents(self, docs: List[Document]):
        """Thread-safe method to extend split documents"""
        with self._lock:
            self.split_documents.extend(docs)